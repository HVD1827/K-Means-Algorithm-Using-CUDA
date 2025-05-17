#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <numeric>
#include <cuda_runtime.h>


using namespace std;

// Global variables
int N, d, k, max_iters = 100;
thrust::device_vector<float> d_data;
thrust::device_vector<float> d_centroids;
thrust::device_vector<int> d_labels;
thrust::device_vector<float> d_distances;
thrust::device_vector<float> d_min_distances;
thrust::device_vector<int> d_counts;

// Random number generator
random_device rd;
mt19937 gen(rd());

// CUDA kernel for computing squared distances between all points and centroids
__global__ void compute_distances_kernel(const float* data, const float* centroids, 
                                        float* distances, int N, int d, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * k) return;
    
    int point_idx = idx / k;
    int centroid_idx = idx % k;
    
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = data[point_idx * d + i] - centroids[centroid_idx * d + i];
        sum += diff * diff;
    }
    distances[idx] = sum;
}

// CUDA kernel for assigning points to nearest centroid
__global__ void assign_clusters_kernel(const float* distances, int* labels, 
                                      float* min_distances, int N, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float min_dist = distances[idx * k];
    int best_cluster = 0;
    
    for (int j = 1; j < k; ++j) {
        float dist = distances[idx * k + j];
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = j;
        }
    }
    
    labels[idx] = best_cluster;
    min_distances[idx] = min_dist;
}

// CUDA kernel for updating centroids
__global__ void update_centroids_kernel(const float* data, const int* labels, 
                                      float* centroids, int* counts, 
                                      int N, int d, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * d) return;
    
    int point_idx = idx / d;
    int dim_idx = idx % d;
    int cluster = labels[point_idx];
    
    atomicAdd(&centroids[cluster * d + dim_idx], data[point_idx * d + dim_idx]);
    if (dim_idx == 0) {
        atomicAdd(&counts[cluster], 1);
    }
}

// CUDA kernel for normalizing centroids
__global__ void normalize_centroids_kernel(float* centroids, const int* counts, 
                                          int d, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * d) return;
    
    int cluster = idx / d;
    int dim = idx % d;
    
    if (counts[cluster] > 0) {
        centroids[cluster * d + dim] /= counts[cluster];
    }
}

// K-means++ initialization using CUDA
void kmeans_plusplus_init(const vector<vector<float>>& data_vec) {
    thrust::host_vector<float> h_centroids(k * d, 0);
    thrust::host_vector<float> h_data(N * d);
    
    // Flatten data
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            h_data[i * d + j] = data_vec[i][j];
        }
    }
    
    // 1. Choose first centroid randomly
    uniform_int_distribution<> dis(0, N-1);
    int first_idx = dis(gen);
    for (int j = 0; j < d; ++j) {
        h_centroids[j] = h_data[first_idx * d + j];
    }
    
    // Copy data to device
    d_data = h_data;
    d_centroids.resize(k * d);
    d_distances.resize(N * k);
    thrust::copy(h_centroids.begin(), h_centroids.end(), d_centroids.begin());
    
    // 2. For each remaining centroid
    for (int c = 1; c < k; c++) {
        // Compute distances to nearest centroid for all points
        int blockSize = 256;
        int numBlocks = (N * k + blockSize - 1) / blockSize;
        compute_distances_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_centroids.data()),
            thrust::raw_pointer_cast(d_distances.data()),
            N, d, k
        );
        cudaDeviceSynchronize();
        
        // Find min distances for each point
        thrust::device_vector<float> min_distances(N);
        thrust::device_vector<int> indices(N);
        for (int i = 0; i < N; ++i) {
            auto start = d_distances.begin() + i * k;
            auto end = start + k;
            auto min_it = thrust::min_element(start, end);
            min_distances[i] = *min_it;
        }
        
        // Convert distances to probabilities
        float total_distance = thrust::reduce(min_distances.begin(), min_distances.end());
        thrust::device_vector<float> probabilities(N);
        thrust::transform(min_distances.begin(), min_distances.end(),
                         probabilities.begin(),
                         [total_distance] __device__ (float x) { return x / total_distance; });
        
        // Select next centroid using weighted probability
        discrete_distribution<> weighted_dist(probabilities.begin(), probabilities.end());
        int next_idx = weighted_dist(gen);
        
        // Add new centroid
        for (int j = 0; j < d; ++j) {
            h_centroids[c * d + j] = h_data[next_idx * d + j];
        }
        thrust::copy(h_centroids.begin(), h_centroids.end(), d_centroids.begin());
    }
}

void kmeans() {
    thrust::device_vector<float> distances(N * k);
    thrust::device_vector<float> min_distances(N);
    thrust::device_vector<int> counts(k, 0);
    
    int blockSize = 256;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        // Assign clusters
        int numBlocks = (N * k + blockSize - 1) / blockSize;
        compute_distances_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_centroids.data()),
            thrust::raw_pointer_cast(distances.data()),
            N, d, k
        );
        
        numBlocks = (N + blockSize - 1) / blockSize;
        assign_clusters_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(distances.data()),
            thrust::raw_pointer_cast(d_labels.data()),
            thrust::raw_pointer_cast(min_distances.data()),
            N, k
        );
        
        // Update centroids
        thrust::fill(d_centroids.begin(), d_centroids.end(), 0.0f);
        thrust::fill(counts.begin(), counts.end(), 0);
        
        numBlocks = (N * d + blockSize - 1) / blockSize;
        update_centroids_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(d_labels.data()),
            thrust::raw_pointer_cast(d_centroids.data()),
            thrust::raw_pointer_cast(counts.data()),
            N, d, k
        );
        
        numBlocks = (k * d + blockSize - 1) / blockSize;
        normalize_centroids_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_centroids.data()),
            thrust::raw_pointer_cast(counts.data()),
            d, k
        );
    }
}

int main() {
    ifstream infile("input2.txt");
    infile >> N >> d >> k;

    vector<vector<float>> data_vec(N, vector<float>(d));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < d; ++j)
            infile >> data_vec[i][j];

    // Initialize device vectors
    d_labels.resize(N);
    
    // Initialize centroids using K-means++
    kmeans_plusplus_init(data_vec);

    // Run K-means
    kmeans();

    // Copy results back to host
    thrust::host_vector<int> h_labels = d_labels;
    
    // Output labels
    for (int i = 0; i < N; ++i)
        cout << h_labels[i] << " ";

    cout << endl;
    return 0;
}
