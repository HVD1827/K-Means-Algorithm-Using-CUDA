#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <limits>

using namespace std;

int N, d, k, max_iters = 200;
vector<vector<float>> data_vec;
vector<vector<float>> centroids;
vector<int> labels;

float euclidean_distance(const vector<float>& a, const vector<float>& b) {
    float sum = 0.0;
    for (int i = 0; i < d; ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

void assign_clusters() {
    for (int i = 0; i < N; ++i) {
        float min_dist = numeric_limits<float>::max();
        int best_cluster = 0;
        for (int j = 0; j < k; ++j) {
            float dist = euclidean_distance(data_vec[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
}

void update_centroids() {
    vector<vector<float>> new_centroids(k, vector<float>(d, 0));
    vector<int> counts(k, 0);

    for (int i = 0; i < N; ++i) {
        int cluster = labels[i];
        for (int j = 0; j < d; ++j)
            new_centroids[cluster][j] += data_vec[i][j];
        counts[cluster]++;
    }

    for (int i = 0; i < k; ++i) {
        if (counts[i] == 0) continue; 
        for (int j = 0; j < d; ++j)
            new_centroids[i][j] /= counts[i];
    }

    centroids = new_centroids;
}

void kmeans() {
    for (int iter = 0; iter < max_iters; ++iter) {
        assign_clusters();
        update_centroids();
    }
}

int main() {
    ifstream infile("input.txt");
    infile >> N >> d >> k;

    data_vec.resize(N, vector<float>(d));
    centroids.resize(k, vector<float>(d));
    labels.resize(N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < d; ++j)
            infile >> data_vec[i][j];

    for (int i = 0; i < k; ++i)
        centroids[i] = data_vec[i];

    kmeans();

    // labels
    for (int i = 0; i < N; ++i)
        cout << labels[i] << " ";

    cout << endl;
    return 0;
}
