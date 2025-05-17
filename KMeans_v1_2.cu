#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <numeric>
#include <cuda_runtime.h>

using namespace std;

int N, d, k, max_iters = 1000;
float *d_data, *d_centroids, *d_distances, *d_min_distances;
int *d_labels_odd, *d_labels_even, *d_counts;
int *d_flag;

__global__ void compute_distances_kernel(const float *data, const float *centroids,
                                         float *distances, int N, int d, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * k)
        return;

    int point_idx = idx / k;
    int centroid_idx = idx % k;

    float sum = 0.0f;
    for (int i = 0; i < d; ++i)
    {
        float diff = data[point_idx * d + i] - centroids[centroid_idx * d + i];
        sum += diff * diff;
    }
    distances[idx] = sum;
}

__global__ void assign_clusters_kernel(const float *distances, int *labels,
                                       float *min_distances, int N, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float minm_dist = distances[idx * k];
    int minm_index = 0;

    for (int j = 1; j < k; ++j)
    {
        float dist = distances[idx * k + j];
        if (dist < minm_dist)
        {
            minm_dist = dist;
            minm_index = j;
        }
    }

    labels[idx] = minm_index;
    min_distances[idx] = minm_dist;
}

__global__ void update_centroids_kernel(const float *data, const int *labels,
                                        float *centroids, int *counts,
                                        int N, int d, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * d)
        return;

    int point_idx = idx / d;
    int dim_idx = idx % d;
    int cluster = labels[point_idx];

    atomicAdd(&centroids[cluster * d + dim_idx], data[point_idx * d + dim_idx]);
    if (dim_idx == 0)
    {
        atomicAdd(&counts[cluster], 1);
    }
}

__global__ void normalize_centroids_kernel(float *centroids, const int *counts,
                                           int d, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * d)
        return;

    int cluster = idx / d;
    int dim = idx % d;

    if (counts[cluster] > 0)
    {
        centroids[cluster * d + dim] /= counts[cluster];
    }
}

__global__ void initialise_centroids(int k, int d, float *d_data, float *d_centroids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * d)
        return;
    d_centroids[idx] = d_data[idx];
}

__global__ void detect_change(int *d_labels_even, int *d_labels_odd, int N, int *d_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    if (*d_flag == 1)
        return;

    if (d_labels_even[idx] != d_labels_odd[idx])
    {
        atomicExch(d_flag, 1);
    }
}

int main()
{
    ifstream infile("input2.txt");
    infile >> N >> d >> k;

    float data_arr[N * d];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < d; ++j)
        {
            float val;
            infile >> val;
            data_arr[i * d + j] = val;
        }

    //////////////////// INITIALISATION ////////////////

    // device mem and data cpy
    cudaMalloc(&d_data, N * d * sizeof(float));
    cudaMalloc(&d_centroids, k * d * sizeof(float));
    cudaMalloc(&d_distances, N * k * sizeof(float));
    cudaMemcpy(d_data, data_arr, N * d * sizeof(float), cudaMemcpyHostToDevice);
    initialise_centroids<<<((k * d) + 255) / 256, 256>>>(k, d, d_data, d_centroids);
    cudaDeviceSynchronize();
    float *h_centroids = new float[k * d];
    cudaMemcpy(h_centroids, d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost);
    // cout << "Initial centroids:" << endl;
    // for (int i = 0; i < k; ++i)
    // {
    //     for (int j = 0; j < d; ++j)
    //     {
    //         cout << h_centroids[i * d + j] << " ";
    //     }
    //     cout << endl;
    // }
    ////////////////////////////////////////////////////

    /////// K-MEANS ////////////////////////////////////
    float *distances;
    float *min_distances;
    int *counts;

    cudaMalloc(&distances, N * k * sizeof(float));
    cudaMalloc(&min_distances, N * sizeof(float));
    cudaMalloc(&counts, k * sizeof(int));
    cudaMalloc(&d_labels_odd, N * sizeof(int));
    cudaMalloc(&d_labels_even, N * sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));

    int blockSize = 256;
    int iter = 0;
    for (; iter < max_iters; ++iter)
    {
        int numBlocks = (N * k + blockSize - 1) / blockSize;
        compute_distances_kernel<<<numBlocks, blockSize>>>(
            d_data, d_centroids, distances, N, d, k);
        cudaDeviceSynchronize();

        numBlocks = (N + blockSize - 1) / blockSize;
        int *d_labels;
        if (iter % 2)
        {
            d_labels = d_labels_odd;
        }
        else
        {
            d_labels = d_labels_even;
        }
        assign_clusters_kernel<<<numBlocks, blockSize>>>(
            distances, d_labels, min_distances, N, k);
        cudaDeviceSynchronize();

        cudaMemset(d_centroids, 0, k * d * sizeof(float));
        cudaMemset(counts, 0, k * sizeof(int));
        numBlocks = (N * d + blockSize - 1) / blockSize;
        update_centroids_kernel<<<numBlocks, blockSize>>>(
            d_data, d_labels, d_centroids, counts, N, d, k);
        cudaDeviceSynchronize();

        numBlocks = (k * d + blockSize - 1) / blockSize;
        normalize_centroids_kernel<<<numBlocks, blockSize>>>(
            d_centroids, counts, d, k);
        cudaDeviceSynchronize();

        cudaMemset(d_flag, 0, sizeof(int));
        detect_change<<<((N + blockSize - 1) / blockSize), blockSize>>>(
            d_labels_even, d_labels_odd, N, d_flag);
        int *flag = new int[1];
        cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (!flag[0])
            break;
    }
    cout << "iterations " << iter << endl;

    // freee
    cudaFree(distances);
    cudaFree(min_distances);
    cudaFree(counts);
    cudaFree(d_flag);
    cudaFree(d_labels_odd);
    cudaFree(d_labels_even);
    ////////////////////////////////////////////////////

    vector<int> h_labels(N);
    int *d_labels;
    if (max_iters % 2)
    {
        d_labels = d_labels_odd;
    }
    else
    {
        d_labels = d_labels_even;
    }
    cudaMemcpy(h_labels.data(), d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

    // freee
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_labels);

    for (int i = 0; i < N; ++i)
        cout << h_labels[i] << " ";

    cout << endl;
    return 0;
}
