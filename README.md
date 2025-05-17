# K-Means Clustering using CUDA

This project implements a parallel version of the **K-Means Clustering** algorithm using **CUDA**.

## Objective

To cluster `N` d-dimensional points into `k` groups on the GPU by minimizing the **sum of squared Euclidean distances** between data points and their assigned centroids.

---

##  Techniques and Optimizations Used

### Versions Implemented
- `v1 & v2`: Sequential + Random/KMeans++ Initialization.
- `v3`: Basic CUDA parallelization.
- `v4`: Accuracy improvements with parallel updates.
- `v5`: Merged kernels and removed sequential bottlenecks via profiling (Nsight).
- `v6`: Used **CUB library** for atomic-free reductions (prefix sum, reduce).
- `v7`: Tried memory coalescing and CUDA streams.
- `v8`: Final improvements and stabilization.

### 🚀 Key CUDA Techniques
- **CUB library**: Used `DeviceReduce`, `ReduceByKey` for lock-free centroid updates.
- **Memory Coalescing**: Optimized global memory access for speedup.
- **Kernel Fusion**: Merged distance calculation and assignment into a single kernel to reduce overhead.
- **CURAND**: Used for probabilistic seeding during KMeans++ initialization.
- **CUDA Streams**: Attempted concurrent execution (limited by kernel dependencies).
- **Early Stopping**: Convergence-aware termination improved performance.
- **Dynamic Parallelism**: Hybrid design used for smaller problem branches.

---

## 📊 Performance Comparison

| Input | Sequential (ms) | Our CUDA (ms) | Nvidia Sample (ms) | Our Accuracy | Nvidia Accuracy |
|-------|------------------|----------------|---------------------|---------------|------------------|
| 1     | 41305            | 121            | 165                 | 0.8099        | 0.8992           |
| 2     | 4846             | 96             | 122                 | 0.7380        | 0.8208           |
| 3     | 192              | 111            | 111                 | 0.5158        | 0.8910           |
| 4     | 1442             | 110            | 141                 | 0.7273        | 0.8766           |
| 5     | 3159             | 101            | 121                 | 0.7569        | 0.8657           |

---

## 🏗️ Project Structure

```
.
├── KMeans.cu                # Early CUDA version
├── KMeans_v3_4.cu           # First parallel CUDA attempt
├── KMeans_v5.cu             # Merged kernels, reduced overhead
├── KMeans_v5_test.cu        # Testing utilities
├── KMeans_v6.cu             # Atomic-free version using CUB
├── KMeans_v6_stable.cu      # Stable tested version
├── KMeans_v7.cu             # With streams & coalescing
├── KMeans_v8.cu             # Final version
├── KMeans_Sequential.cpp    # Baseline sequential version
├── KMeanspp_Sequential.cpp # KMeans++ Sequential
├── README.md                # This file
```

---

## How to Build and Run

### Prerequisites
- CUDA Toolkit 11.x or above
- C++14 compatible compiler
- NVIDIA GPU with compute capability 6.0+
- CUB library (bundled with recent CUDA)

### Compilation Example

To compile any CUDA version:

```bash
nvcc -ccbin g++-10 -std=c++14 -o kmeans_vx KMeans_vx.cu
```
Where 'x' is the version number.

To compile the sequential version:

```bash
g++ -O3 -o kmeans_seq KMeans_Sequential.cpp
```

### Running

```bash
./kmeans_vx <input_file> 
```

### Inpu File
Input file should have the input in the following format:
```
N
d
k
point_1(d values correcponding to each dimension)
point_2
point_3
....
```
---

## 📃 License

This project is for academic and educational use. Attribution required for reproduction or derivative works.

---

## 🙏 Acknowledgements

Thanks to the CS6023 course instructor and the Nsight Systems profiler for helping us optimize CUDA code effectively.
