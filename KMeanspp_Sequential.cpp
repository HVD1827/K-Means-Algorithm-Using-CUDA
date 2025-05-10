#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <random>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace std;

int N, d, k, max_iters = 10;
vector<vector<float>> data_vec;
vector<vector<float>> centroids;
vector<int> labels;

// Random number generator
random_device rd;
mt19937 gen(rd());

// Compute squared Euclidean distance between two points (no sqrt for efficiency)
float squared_distance(const vector<float> &a, const vector<float> &b)
{
    float sum = 0.0;
    for (int i = 0; i < d; ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sum;
}

// Compute Euclidean distance between two points
float euclidean_distance(const vector<float> &a, const vector<float> &b)
{
    return sqrt(squared_distance(a, b));
}

// K-means++ initialization
// Improved K-means++ initialization
void kmeans_plusplus_init()
{
    centroids.resize(k, vector<float>(d, 0));

    // 1. Choose first centroid randomly
    uniform_int_distribution<int> first_centroid(0, N - 1);
    int first_idx = first_centroid(gen);
    centroids[0] = data_vec[first_idx];

    // 2. Initialize distances and probabilities
    vector<float> distances(N, numeric_limits<float>::max());
    vector<float> probabilities(N, 0.0f);

    // 3. For each remaining centroid
    for (int c = 1; c < k; c++)
    {
        float total_distance = 0.0f;

        // Compute distances to nearest existing centroid
        for (int i = 0; i < N; i++)
        {
            float min_dist = numeric_limits<float>::max();
            for (int j = 0; j < c; j++)
            {
                float dist = squared_distance(data_vec[i], centroids[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                }
            }
            distances[i] = min_dist;
            total_distance += min_dist;
        }

        // Handle case where all distances are zero (unlikely but possible)
        if (total_distance == 0.0f)
        {
            // Fall back to random selection
            uniform_int_distribution<int> random_point(0, N - 1);
            centroids[c] = data_vec[random_point(gen)];
            continue;
        }

        // Convert distances to probabilities
        for (int i = 0; i < N; i++)
        {
            probabilities[i] = distances[i] / total_distance;
        }

        // Select next centroid using weighted probability
        discrete_distribution<int> weighted_dist(probabilities.begin(), probabilities.end());
        centroids[c] = data_vec[weighted_dist(gen)];
    }
}

// Assign each point to the nearest centroid
void assign_clusters()
{
    for (int i = 0; i < N; ++i)
    {
        float min_dist = numeric_limits<float>::max();
        int best_cluster = 0;
        for (int j = 0; j < k; ++j)
        {
            float dist = euclidean_distance(data_vec[i], centroids[j]);
            if (dist < min_dist)
            {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
}

// Recompute centroids as the mean of assigned points
void update_centroids()
{
    vector<vector<float>> new_centroids(k, vector<float>(d, 0));
    vector<int> counts(k, 0);

    for (int i = 0; i < N; ++i)
    {
        int cluster = labels[i];
        for (int j = 0; j < d; ++j)
            new_centroids[cluster][j] += data_vec[i][j];
        counts[cluster]++;
    }

    for (int i = 0; i < k; ++i)
    {
        if (counts[i] == 0)
            continue; // avoid division by zero
        for (int j = 0; j < d; ++j)
            new_centroids[i][j] /= counts[i];
    }

    centroids = new_centroids;
}

void kmeans()
{
    for (int iter = 0; iter < max_iters; ++iter)
    {
        assign_clusters();
        update_centroids();
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Enter the file name too" << endl;
        return 1;
    }
    // format ./a.out file_path
    // Read input from file
    cout << argv[1] << endl;

    ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        cerr << "Error: Could not open file " << argv[1] << endl;
        return 1;
    }
    infile >> N >> d >> k;

    data_vec.resize(N, vector<float>(d));
    labels.resize(N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < d; ++j)
            infile >> data_vec[i][j];

    // Initialize centroids using K-means++
    auto start = chrono::high_resolution_clock::now();
    kmeans_plusplus_init();

    kmeans();

    // Output labels
    for (int i = 0; i < N; ++i)
        cout << labels[i] << " ";
    cout << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    ofstream outfile("time_sequential.txt", std::ios::app);
    outfile << fixed << setprecision(2);
    outfile << "Time taken for file " << argv[1] << ": " << duration.count() << " ms" << endl;

    return 0;
}