#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

// just for debugging purposes to catch errors and stuff
#define CUDA_CHECK(call) do {                                \
    cudaError_t err = (call);                                \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error: %s at line %d\n",       \
                cudaGetErrorString(err), __LINE__);           \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)


// naive similarity computation kernel
__global__ void similarity_kernel(
    const float* queries,
    const float* candidates,
    const int* query_labels,
    const int* candidate_labels,
    float* similarities,
    int B, int N, int D
) {
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= B || c >= N) return;

    // exclude candidates with matching labels
    if (query_labels[q] == candidate_labels[c]) {
        similarities[q * N + c] = -INFINITY;
        return;
    }

    // Compute dot product
    float score = 0.0f;

    for (int d = 0; d < D; d++) {
        score += queries[q * D + d] * candidates[c * D + d];
    }

    similarities[q * N + c] = score;
}


// naive top-k similarity kernel
__global__ void topk_kernel(
    float* similarities,
    float* topk_scores,
    int* topk_indices,
    int B, int N, int k
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= B) return;

    // Initialize top-k results
    for (int i = 0; i < k; i++) {
        topk_scores[q * k + i] = -INFINITY;
        topk_indices[q * k + i] = -1;
    }

    // Scan candidates only once
    for (int c = 0; c < N; c++) {
        float score = similarities[q * N + c];

        // Skip if not better than the current smallest top-k score
        if (score <= topk_scores[q * k + k - 1]) {
            continue;
        }

        // Insert score into sorted top-k array
        int pos = k - 1;

        while (pos > 0 && score > topk_scores[q * k + pos - 1]) {
            topk_scores[q * k + pos] =
                topk_scores[q * k + pos - 1];

            topk_indices[q * k + pos] =
                topk_indices[q * k + pos - 1];

            pos--;
        }

        topk_scores[q * k + pos] = score;
        topk_indices[q * k + pos] = c;
    }
}

// host side: setup, execute, benchmark, and verify
int main() {
    const int B = 32;
    const int N = 100000;
    const int D = 128;
    const int k = 10;

    const int iterations = 100;

    printf("Tiled with no Shared Memory CUDA Hard-Negative Mining\n");

    // host inputs
    std::vector<float> h_queries(B * D);
    std::vector<float> h_candidates(N * D);
    std::vector<int> h_query_labels(B);
    std::vector<int> h_candidate_labels(N);

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 99);

    // generate normalized query embeddings
    for (int q = 0; q < B; q++) {
        float norm_sq = 0.0f;

        for (int d = 0; d < D; d++) {
            float v = normal(rng);
            h_queries[q * D + d] = v;
            norm_sq += v * v;
        }

        float norm = std::sqrt(norm_sq);

        for (int d = 0; d < D; d++) {
            h_queries[q * D + d] /= norm;
        }

        h_query_labels[q] = label_dist(rng);
    }

    // generate normalized candidate embeddings
    for (int c = 0; c < N; c++) {
        float norm_sq = 0.0f;

        for (int d = 0; d < D; d++) {
            float v = normal(rng);
            h_candidates[c * D + d] = v;
            norm_sq += v * v;
        }

        float norm = std::sqrt(norm_sq);

        for (int d = 0; d < D; d++) {
            h_candidates[c * D + d] /= norm;
        }

        h_candidate_labels[c] = label_dist(rng);
    }

    // GPU pointers
    float *d_queries, *d_candidates, *d_similarities;
    float *d_topk_scores;
    int *d_query_labels, *d_candidate_labels, *d_topk_indices;

    // allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_queries, B * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidates, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_similarities, B * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_scores, B * k * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_query_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidate_labels, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_indices, B * k * sizeof(int)));

    // copy inputs from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(),
                          B * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_candidates, h_candidates.data(),
                          N * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_query_labels, h_query_labels.data(),
                          B * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_candidate_labels, h_candidate_labels.data(),
                          N * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel launch configurations
    dim3 threads(32, 8);
    dim3 blocks((N + 31) / 32, (B + 7) / 8);
    const int topk_threads = 256;
    const int topk_blocks = (B + topk_threads - 1) / topk_threads;

    // Execute both kernels
    auto run = [&]() {
        similarity_kernel<<<blocks, threads>>>(
            d_queries, d_candidates,
            d_query_labels, d_candidate_labels,
            d_similarities, B, N, D
        );
        CUDA_CHECK(cudaGetLastError());

        topk_kernel<<<topk_blocks, topk_threads>>>(
            d_similarities, d_topk_scores,
            d_topk_indices, B, N, k
        );
        CUDA_CHECK(cudaGetLastError());
    };

    // Warm-up
    for (int i = 0; i < 20; i++) {
        run();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark with CUDA events
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        run();
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));

    printf("B=%d N=%d D=%d k=%d\n", B, N, D, k);
    printf("Average latency: %.4f ms\n", total_ms / iterations);

    // Copy outputs to CPU
    std::vector<float> h_scores(B * k);
    std::vector<int> h_indices(B * k);

    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_topk_scores,
                          B * k * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_topk_indices,
                          B * k * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first query results
    printf("First query indices: ");
    for (int i = 0; i < k; i++) {
        printf("%d ", h_indices[i]);
    }
    printf("\n");

    printf("First query scores: ");
    for (int i = 0; i < k; i++) {
        printf("%.6f ", h_scores[i]);
    }
    printf("\n");

    // Basic correctness checks
    bool valid = true;

    for (int q = 0; q < B; q++) {
        for (int i = 0; i < k; i++) {
            int idx = h_indices[q * k + i];
            float score = h_scores[q * k + i];

            if (idx < 0 || idx >= N ||
                h_query_labels[q] == h_candidate_labels[idx]) {
                valid = false;
                break;
            }

            if (i > 0 && score > h_scores[q * k + i - 1]) {
                valid = false;
                break;
            }
        }
    }

    if (valid) {
        printf("Basic correctness checks: PASSED\n");
    } else {
        printf("Basic correctness checks: FAILED\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_similarities));
    CUDA_CHECK(cudaFree(d_topk_scores));
    CUDA_CHECK(cudaFree(d_query_labels));
    CUDA_CHECK(cudaFree(d_candidate_labels));
    CUDA_CHECK(cudaFree(d_topk_indices));

    if (valid) {
        return 0;
    }
    else return 1;
}