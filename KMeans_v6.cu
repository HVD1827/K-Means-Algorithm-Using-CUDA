#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <numeric>
#include <cuda_runtime.h>
#include <set>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>

#include <cub/cub.cuh>

using namespace std;

#define PRINT_CENTROIDS                                                                           \
    cudaMemcpy(print_h_centroid_help, d_centroids, k *d * sizeof(float), cudaMemcpyDeviceToHost); \
    cout << "Initial centroids:" << endl;                                                         \
    for (int i = 0; i < k; ++i)                                                                   \
    {                                                                                             \
        for (int j = 0; j < d; ++j)                                                               \
        {                                                                                         \
            cout << h_centroids[i * d + j] << " ";                                                \
        }                                                                                         \
        cout << endl;                                                                             \
    }

#define ERROR(n)                                                    \
    e = cudaGetLastError();                                         \
    if (e != cudaSuccess)                                           \
    {                                                               \
        fprintf(stderr, "Error:%d %s\n", n, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }

int N, d, k, max_iters = 200;
float *d_data, *d_centroids, *d_distances, *d_min_distances;
int *d_labels_odd, *d_labels_even;
int *d_flag;

__global__ void setup_curand_states(curandState *states, unsigned long seed, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;
    curand_init(seed, idx, 0, &states[idx]); // Initialize CURAND
}

__global__ void compute_distances(float *data, float *centroids,
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

__global__ void compute_distances_kmp(float *data,
                                      float *distances, int *centroid_set, int N, int d, int size, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * size)
        return;

    int point_idx = idx / size;
    int centroid_idx = centroid_set[idx % size];

    float sum = 0.0f;
    for (int i = 0; i < d; ++i)
    {
        float diff = data[point_idx * d + i] - data[centroid_idx * d + i];
        sum += diff * diff;
    }
    distances[(k * point_idx) + (idx % size)] = sum;
}

__global__ void assign_clusters(float *distances, int *labels,
                                float *min_distances, int size, int N, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float minm_dist = distances[idx * k];
    int minm_index = 0;

    for (int j = 1; j < size; ++j)
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

__global__ void update_centroids(float *data, int *labels,
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

__global__ void normalize_centroids(float *centroids, int *counts,
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

__global__ void detect_change(int *d_labels_even, int *d_labels_odd, int N, int *d_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    if (*d_flag == 1)
        return;

    if (d_labels_even[idx] != d_labels_odd[idx])
    {
        // atomicExch(d_flag, 1); // can be removed
        *d_flag = 1;
    }
}

__global__ void compute_probabs(float *min_distances,
                                float *probabilities,
                                int *in_centroids_set_flag,
                                float *total_cost,
                                int *centroid_set,
                                int *centroid_count,
                                curandState *states,
                                int l, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float val = (float)l * (min_distances[idx] / (float)*total_cost);
    probabilities[idx] = val;

    // for checking if we should add it in the set
    float probab_random = curand_uniform(&states[idx]);
    if (probab_random < val)
    {
        // int index = atomicInc((unsigned int *)centroid_count, INT_MAX);
        in_centroids_set_flag[idx] = 1;
        // can't avoid atomics ig. try if possible.
        // centroid_set[idx] = 1;
    }
}

__global__ void compute_probabs_2(int *in_centroids_set, int *in_centroids_set_flag,
                                  int *centroid_set, int *centroid_count, int N, int k, int l_kmp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    if (in_centroids_set_flag[idx] == 1)
    {
        int index = *centroid_count - 1 + in_centroids_set[idx];
        if (index < (k + 2 * l_kmp))
            centroid_set[index] = idx;
    }
}

void compute_total_cost(float *min_distances, float *total_cost, int N)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           min_distances, total_cost, N);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           min_distances, total_cost, N);

    cudaFree(d_temp_storage);
}

__global__ void assign_centroids(float *data, float *centroids, int d, int centroid_indx, int point_indx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d)
        return;
    centroids[(centroid_indx * d) + idx] = data[(point_indx * d) + idx];
}

__global__ void assign_centroids_fast(float *data, float *centroids, int d, int *centroid_set, int centroid_indx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d)
        return;
    centroids[(centroid_indx * d) + idx] = data[(centroid_set[centroid_indx] * d) + idx];
}

int main(int argc, char *argv[])
{

    cudaError_t e = cudaGetLastError();
    if (argc != 2)
    {
        cout << "Enter the file name too" << endl;
        return 1;
    }

    ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        cerr << "Error: Could not open file " << argv[1] << endl;
        return 1;
    }

    infile >> N >> d >> k;

    float *print_h_centroid_help = new float[k * d];

    float data_arr[N * d];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            float val;
            infile >> val;
            data_arr[i * d + j] = val;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_data, N * d * sizeof(float));
    cudaMemcpy(d_data, data_arr, N * d * sizeof(float), cudaMemcpyHostToDevice);

    int *counts;
    cudaMalloc(&counts, k * sizeof(int));

    int *d_labels;
    cudaMalloc(&d_labels, N * sizeof(int));

    cudaMalloc(&d_centroids, k * d * sizeof(float));
    cudaMalloc(&d_distances, N * k * sizeof(float));
    cudaMalloc(&d_min_distances, N * sizeof(float));

    cudaMalloc(&d_labels_odd, N * sizeof(int));
    cudaMalloc(&d_labels_even, N * sizeof(int));

    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));

    /////////////////////////// KMEANS PARALLEL INIT./////////
    int l_kmp = 5;
    int rounds = 200; // default, will break if k centroids are found, not more than this

    int first_center_idx = 0; // initialise the frst center randomly to the first point(bcz it is random)
    cudaMemcpy(d_centroids, &data_arr[first_center_idx * d], d * sizeof(float), cudaMemcpyHostToDevice);

    float *d_probabs;
    cudaMalloc(&d_probabs, N * sizeof(float));
    // float *h_probabs = new float[N];

    set<int> C_indices_kmp;
    C_indices_kmp.insert(first_center_idx);

    float *d_total_cost;
    cudaMalloc(&d_total_cost, sizeof(float));
    int size;

    int *d_centroid_set;
    cudaMalloc(&d_centroid_set, (k + (2 * l_kmp)) * sizeof(int));
    // cudaMemset(d_centroid_set, 1, sizeof(int)); doesnt work, use memcpy ;/ this is because it sets bytes not the specified size
    cudaMemset(d_centroid_set, 0, (k + (2 * l_kmp)) * sizeof(int));
    int val = 0;
    cudaMemcpy(d_centroid_set, &val, sizeof(int), cudaMemcpyHostToDevice);

    // int *h1 = new int[N];
    // cudaMemcpy(h1, d_centroid_set, N * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < N; ++i)
    // {
    //     cout << h1[i] << " ";
    // }
    // cout << endl;
    int *d_centroid_count;
    cudaMalloc(&d_centroid_count, sizeof(int));
    val = 1;
    cudaMemcpy(d_centroid_count, &val, sizeof(int), cudaMemcpyHostToDevice);

    curandState *d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
    setup_curand_states<<<(N + 255) / 256, 256>>>(d_states, 2025, N);

    float *d_distances_kmp;
    cudaMalloc(&d_distances_kmp, N * k * sizeof(float));
    cudaMemset(d_distances_kmp, 0, N * k * sizeof(float));

    size = 1;

    int *d_in_centroids_set, *d_in_centroids_set_flag;
    cudaMalloc(&d_in_centroids_set, N * sizeof(int));
    cudaMalloc(&d_in_centroids_set_flag, N * sizeof(int));

    for (int r = 0; r < rounds; ++r)
    {
        // cout << "Round " << r + 1 << endl;
        // distances to nearest cntroid or Dsq.
        // size = C_indices_kmp.size();
        // cudaMemcpy(&size, d_centroid_count, sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "Current centroid count: " << size << endl;
        if (size >= k)
        {
            // cout << "k centroids done, breaking" << endl;
            cout << "rounds done " << r << endl;
            break;
        }
        compute_distances_kmp<<<((N * (size)) + 255) / 256, 256>>>(d_data, d_distances_kmp, d_centroid_set, N, d, size, k);
        // float *h_distances_kmp = new float[N * size];
        // cudaMemcpy(h_distances_kmp, d_distances_kmp, N * size * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < size; i++)
        // {
        //     cout << "Centroid number " << i << " : ";
        //     for (int j = 0; j < N; j++)
        //     {
        //         cout << h_distances_kmp[i * N + j] << " ";
        //     }
        //     cout << endl;
        // }

        ERROR(1);
        cudaDeviceSynchronize();

        // nearest centroid assignment

        assign_clusters<<<(N + 255) / 256, 256>>>(
            d_distances_kmp, d_labels, d_min_distances, size, N, k);
        ERROR(2);
        cudaDeviceSynchronize();
        // float *h_min_distances = new float[N];
        // cudaMemcpy(h_min_distances, d_min_distances, N * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < N; i++)
        // {
        //     cout << "Point number " << i << " : ";
        //     cout << h_min_distances[i] << endl;
        // }

        // compute probabs
        cudaMemset(d_total_cost, 0, sizeof(float));
        compute_total_cost(d_min_distances, d_total_cost, N);

        // float *tot_cst = new float[1];
        // cudaMemcpy(tot_cst, d_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
        // cout << "tot_cst " << *tot_cst << endl;

        ERROR(3);
        cudaDeviceSynchronize();

        // int *d_in_centroids_set, *d_in_centroids_set_flag;
        // cudaMalloc(&d_in_centroids_set, N * sizeof(int));
        // cudaMalloc(&d_in_centroids_set_flag, N * sizeof(int));
        cudaMemset(d_in_centroids_set, 0, N * sizeof(int));
        cudaMemset(d_in_centroids_set_flag, 0, N * sizeof(int));

        compute_probabs<<<((N) + 255) / 256, 256>>>(d_min_distances, d_probabs, d_in_centroids_set_flag, d_total_cost, d_centroid_set, d_centroid_count, d_states, l_kmp, N);

        /////////////// DEBUG PRINTING ////////////////
        // int *in_centroids_set = new int[N];
        // int *in_centroids_set_flag = new int[N];
        // cudaMemcpy(in_centroids_set_flag, d_in_centroids_set_flag, N * sizeof(int), cudaMemcpyDeviceToHost);

        // cout << "flags:" << endl;
        // for (int i = 0; i < N; ++i)
        // {
        //     cout << i << " : " << in_centroids_set_flag[i] << endl;
        // }
        ////////////////////////////////////////////////

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      d_in_centroids_set_flag, d_in_centroids_set, N);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      d_in_centroids_set_flag, d_in_centroids_set, N);

        ///////////////// DEBUG PRINTING ////////////////
        // cudaMemcpy(in_centroids_set, d_in_centroids_set, N * sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "prefix sum" << endl;
        // for (int i = 0; i < N; ++i)
        // {
        //     cout << i << " : " << in_centroids_set[i] << endl;
        // }
        /////////////////////////////////////////////////

        compute_probabs_2<<<((N) + 255) / 256, 256>>>(d_in_centroids_set, d_in_centroids_set_flag, d_centroid_set,
                                                      d_centroid_count, N, k, l_kmp);

        // int *cent_count = new int[1];
        int offset;
        // offset = last element in the prefx sm array, which is d_in_centroids_set + N - 1

        cudaMemcpy(&offset, d_in_centroids_set + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "offset " << offset << endl;
        size += offset;
        // cout << "size " << size << endl;
        cudaMemcpy(d_centroid_count, &size, sizeof(int), cudaMemcpyHostToDevice);

        ERROR(4);
        cudaDeviceSynchronize();

        // float *h_probabs = new float[N];
        // cudaMemcpy(h_probabs, d_probabs, N * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < N; ++i)
        // {
        //     cout << h_probabs[i] << endl;
        // }
        //////// the above part is bad, sequential code on the CPU is the bottlneck //////////////
        // possible options: launch a kernel to do this, maintaining the count and an array for this -- looks good, consider all the limitations
        // merge with the compute_probabs kernel onlyy
    }
    cout << "size " << size << endl;

    // int *h_centr_set = new int[k + (2 * l_kmp)];
    // cudaMemcpy(h_centr_set, d_centroid_set, (k + (2 * l_kmp)) * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < k + (2 * l_kmp); ++i)
    // {
    //     cout << h_centr_set[i] << " ";
    // }
    // cout << endl;

    // we have now more thann k points, reduce it ko k, but how?
    // one way is to weigh them according to kparallel but timetakin

    ///////////////////////// !!!! BUGGY T_T !!!! ////////////////
    // int *h = new int[N];
    // cudaMemcpy(h, d_centroid_set, N * sizeof(int), cudaMemcpyDeviceToHost);
    // int count = 0;
    // for (int i = 0; i < N; i++)
    // {
    //     if (count == k)
    //         break;
    //     if (h[i])
    //     {
    //         assign_centroids<<<(d + 255) / 256, 256>>>(d_data, d_centroids, d, count, i);
    //         count++;
    //     }
    // }
    ///////////////////////// !!!!!!!!!!!!!!!!!!! ////////////////

    // int *h_centroid_set = new int[k + (2 * l_kmp)];
    // cudaMemcpy(h_centroid_set, d_centroid_set, (k + (2 * l_kmp)) * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < k; i++)
    // {
    //     assign_centroids<<<(d + 255) / 256, 256>>>(d_data, d_centroids, d, i, h_centroid_set[i]);
    // }

    for (int i = 0; i < k; i++)
    {
        assign_centroids_fast<<<(d + 255) / 256, 256>>>(d_data, d_centroids, d, d_centroid_set, i);
    }

    /////// K-MEANS ////////////////////////////////////
    int iter = 0;
    for (; iter < max_iters; ++iter)
    {
        compute_distances<<<(N * k + 255) / 256, 256>>>(
            d_data, d_centroids, d_distances, N, d, k);
        cudaDeviceSynchronize();

        if (iter % 2)
        {
            d_labels = d_labels_odd;
        }
        else
        {
            d_labels = d_labels_even;
        }
        assign_clusters<<<(N + 255) / 256, 256>>>(
            d_distances, d_labels, d_min_distances, k, N, k);
        cudaDeviceSynchronize();

        cudaMemset(d_centroids, 0, k * d * sizeof(float));
        cudaMemset(counts, 0, k * sizeof(int));
        update_centroids<<<(N * d + 255) / 256, 256>>>(
            d_data, d_labels, d_centroids, counts, N, d, k);

        cudaDeviceSynchronize();

        normalize_centroids<<<(k * d + 255) / 256, 256>>>(
            d_centroids, counts, d, k);

        cudaDeviceSynchronize();

        cudaMemset(d_flag, 0, sizeof(int));
        detect_change<<<((N + 255) / 256), 256>>>(
            d_labels_even, d_labels_odd, N, d_flag);

        int *flag = new int[1];
        cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (!flag[0])
            break;
    }
    // cout << "iterations " << iter << endl;

    ////////////////////////////////////////////////////

    vector<int> h_labels(N);
    if (iter % 2)
    {
        d_labels = d_labels_odd;
    }
    else
    {
        d_labels = d_labels_even;
    }
    cudaMemcpy(h_labels.data(), d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    ofstream outfile("time_v6.txt", std::ios::app);
    outfile << fixed << setprecision(2);
    outfile << "Time taken for file " << argv[1] << ": " << duration.count() << " ms" << endl;
    // freee
    cudaFree(d_min_distances);
    cudaFree(counts);
    cudaFree(d_labels_odd);
    cudaFree(d_labels_even);
    cudaFree(d_labels);
    cudaFree(d_flag);
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_probabs);
    cudaFree(d_centroids);

    for (int i = 0; i < N; ++i)
        cout << h_labels[i] << " ";

    cout << endl;
    // cout << "iterations " << iter << endl;

    return 0;
}