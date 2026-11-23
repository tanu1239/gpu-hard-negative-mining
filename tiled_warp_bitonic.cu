#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#define TILE 256
#include <algorithm>
#include <utility>
#define QUERIES_PER_BLOCK 8 // queries per block/warps per block
#define CANDS_PER_PASS 32 // candidates copied per pass
#define NUM_PASSES (TILE / CANDS_PER_PASS)
#define MAX_DIM 128 // max embedding size
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define MERGE_THREADS 256 //threads per merge block
#define MERGE_WARPS (MERGE_THREADS / WARP_SIZE) // 8 warps

// just for debugging purposes to catch errors and stuff
#define CUDA_CHECK(call) do {                                \
    cudaError_t err = (call);                                \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error: %s at line %d\n",       \
                cudaGetErrorString(err), __LINE__);           \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)


__global__ void tiled_fused_kernel(
    const float* queries,
    const float* candidates,
    const int* query_labels,
    const int* candidate_labels,
    float* partial_scores,
    int* partial_indices,
    int B, int N, int D, int k, int num_tiles
) {
    __shared__ float Queries[QUERIES_PER_BLOCK][MAX_DIM];

    //Holds the current pass's 32 candidates
    __shared__ float Candidates[CANDS_PER_PASS][MAX_DIM + 1]; //+1 pad to avoid bank conflicts between threads


    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE; //which query for this block
    int lane = tid % WARP_SIZE; //position within group of 32 candidates
    int tile = blockIdx.x; //which chunk
    int qBase = blockIdx.y * QUERIES_PER_BLOCK; //the first query
    int q = qBase + warp; //This thread's query

    //Copies this block's queries into shared memory
    for (int i = tid; i < QUERIES_PER_BLOCK * D; i += blockDim.x) {
        int row = i / D;
        int d = i % D;
        int qi  = qBase + row;
        Queries[row][d] = (qi < B) ? queries[qi * D + d] : 0.0f;
    }

    float my_scores[NUM_PASSES];

    //Scores this chunk
    for (int p = 0; p < NUM_PASSES; p++) {
        int cBase = tile * TILE + p * CANDS_PER_PASS; //first candidate
        __syncthreads();

        //Copy this passes's 32 candidate into shared memory (all 8 warps/queries reuses this)
        for (int i = tid; i < CANDS_PER_PASS * D; i += blockDim.x) {
            int row = i / D;
            int d   = i % D;
            int c   = cBase + row;
            Candidates[row][d] = (c < N) ? candidates[c * D + d] : 0.0f;
        }
        __syncthreads(); 

        //Computes score
        int c = cBase + lane;
        float s = -INFINITY;
        if (q < B && c < N && query_labels[q] != candidate_labels[c]) {
            s = 0.0f;
            for (int d = 0; d < D; d++)
                s += Queries[warp][d] * Candidates[lane][d]; //Dot product
        }
        my_scores[p] = s;
    }

    for (int r = 0; r < k; r++) {
        //finds best own score
        float best = -INFINITY;
        int best_p = 0;
        for (int p = 0; p < NUM_PASSES; p++) {
            if (my_scores[p] > best) {
                best = my_scores[p];
                best_p = p;
            }
        }
        int best_idx = tile * TILE + best_p * CANDS_PER_PASS + lane;

        // finds the best score/index across the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_score = __shfl_down_sync(FULL_MASK, best, offset);
            int other_idx = __shfl_down_sync(FULL_MASK, best_idx, offset);
            if (other_score > best) {
                best = other_score;
                best_idx = other_idx;
            }
        }
        best = __shfl_sync(FULL_MASK, best, 0); //broadcasts winner
        best_idx = __shfl_sync(FULL_MASK, best_idx, 0);

        // one thread per warp writes the winner
        if (lane == 0 && q < B) {
            int out = (q * num_tiles + tile) * k + r;
            partial_scores[out]  = best;
            partial_indices[out] = (best == -INFINITY) ? -1 : best_idx;
        }

        // The owner removes from own scores
        for (int p = 0; p < NUM_PASSES; p++) {
            if (tile * TILE + p * CANDS_PER_PASS + lane == best_idx)
                my_scores[p] = -INFINITY;
        }
    }
}



#define SELECT_K 16
#define BITONIC_MERGE_TILE 256

__global__ void bitonic_merge_kernel(
    const float* input_scores,
    const int* input_indices,
    float* output_scores,
    int* output_indices,
    int count,
    int groups_per_query,
    int k
) {
    int tid = threadIdx.x;
    int group = blockIdx.x;
    int q = blockIdx.y;

    int input_pos = group * BITONIC_MERGE_TILE + tid;

    __shared__ float scores[BITONIC_MERGE_TILE];
    __shared__ int indices[BITONIC_MERGE_TILE];

    // load 256 candidates, padding invalid entries.
    if (input_pos < count) {
        int offset = q * count + input_pos;
        int idx = input_indices[offset];

        scores[tid] = (idx >= 0)
            ? input_scores[offset] : -INFINITY;
        indices[tid] = idx;
    } else {
        scores[tid] = -INFINITY;
        indices[tid] = -1;
    }

    __syncthreads();

    // sort every group of 16 ascending.
    for (int size = 2; size <= SELECT_K; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            int partner = tid ^ stride;

            float my_score = scores[tid];
            float other_score = scores[partner];
            int other_idx = indices[partner];

            __syncthreads();

            bool ascending =
                (size == SELECT_K) || ((tid & size) == 0);

            bool swap =
                (ascending &&
                    ((tid < partner && my_score > other_score) ||
                     (tid > partner && my_score < other_score))) ||
                (!ascending &&
                    ((tid < partner && my_score < other_score) ||
                     (tid > partner && my_score > other_score)));

            if (swap) {
                scores[tid] = other_score;
                indices[tid] = other_idx;
            }

            __syncthreads();
        }
    }

    // hierarchically merge top-16 lists.
    for (int groupStride = SELECT_K;
         groupStride < BITONIC_MERGE_TILE;
         groupStride *= 2) {

        int group_id = tid / groupStride;
        int local = tid % groupStride;

        // Compare two sorted top-16 lists.
        if ((group_id % 2 == 0) && local < SELECT_K) {
            int partner =
                (group_id + 1) * groupStride
                + SELECT_K - 1 - local;

            float other_score = scores[partner];

            if (other_score > scores[tid]) {
                scores[tid] = other_score;
                indices[tid] = indices[partner];
            }
        }

        __syncthreads();

        // re-sort the surviving top-16 lists.
        int nextStride = groupStride * 2;

        for (int size = 2; size <= SELECT_K; size *= 2) {
            for (int stride = size / 2; stride > 0; stride /= 2) {
                int localPos = tid % nextStride;
                int partner = tid ^ stride;

                float my_score = -INFINITY;
                float other_score = -INFINITY;
                int other_idx = -1;

                if (localPos < SELECT_K) {
                    my_score = scores[tid];
                    other_score = scores[partner];
                    other_idx = indices[partner];
                }

                __syncthreads();

                if (localPos < SELECT_K) {
                    bool ascending =
                        ((localPos & size) == 0);

                    bool swap =
                        (ascending &&
                            ((tid < partner &&
                              my_score > other_score) ||
                             (tid > partner &&
                              my_score < other_score))) ||
                        (!ascending &&
                            ((tid < partner &&
                              my_score < other_score) ||
                             (tid > partner &&
                              my_score > other_score)));

                    if (swap) {
                        scores[tid] = other_score;
                        indices[tid] = other_idx;
                    }
                }

                __syncthreads();
            }
        }
    }

    // the top 16 are in positions 0-15, ascending, write the largest k in descending order.
    if (tid < k) {
        int out = (q * groups_per_query + group) * k + tid;
        int src = SELECT_K - 1 - tid;

        output_scores[out] = scores[src];
        output_indices[out] = indices[src];
    }
}


// host side: setup, execute, benchmark, and verify
int main() {
    const int B = 32;
    const int N = 100000;
    const int D = 128;
    const int k = 10;

    const int iterations = 100;
    const int num_tiles = (N + TILE - 1) / TILE;

    printf("Tiled Fused CUDA Hard-Negative Mining\n");


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
    float *d_queries, *d_candidates, *d_partial_scores, *d_topk_scores;
    int *d_query_labels, *d_candidate_labels;
    int *d_partial_indices, *d_topk_indices;

    float* d_merge_scores;
    int* d_merge_indices;

    const int merge_count = num_tiles * k;
    const int merge_groups =
        (merge_count + BITONIC_MERGE_TILE - 1) / BITONIC_MERGE_TILE;

    CUDA_CHECK(cudaMalloc(
        (void**)&d_merge_scores,
        B * merge_groups * k * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&d_merge_indices,
        B * merge_groups * k * sizeof(int)
    ));

    // allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_queries, B * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidates, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_scores, B * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_scores, B * num_tiles * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_indices, B * num_tiles * k * sizeof(int)));
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
    dim3 threads(QUERIES_PER_BLOCK * WARP_SIZE); //256 threads
    dim3 blocks(num_tiles, (B + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK);

    // Execute both kernels
    auto run = [&]() {
        tiled_fused_kernel<<<blocks, threads>>>(
            d_queries,
            d_candidates,
            d_query_labels,
            d_candidate_labels,
            d_partial_scores,
            d_partial_indices,
            B, N, D, k, num_tiles
        );
        CUDA_CHECK(cudaGetLastError());
    
        
        // Pass 1: 3910 finalists -> 160 finalists per query.
        dim3 merge_blocks_1(merge_groups, B);

        bitonic_merge_kernel<<<merge_blocks_1, BITONIC_MERGE_TILE>>>(
            d_partial_scores,
            d_partial_indices,
            d_merge_scores,
            d_merge_indices,
            merge_count,
            merge_groups,
            k
        );
        CUDA_CHECK(cudaGetLastError());

        // Pass 2: 160 finalists -> final top 10 per query.
        int remaining = merge_groups * k;

        // This version assumes remaining <= 256.
        bitonic_merge_kernel<<<dim3(1, B), BITONIC_MERGE_TILE>>>(
            d_merge_scores,
            d_merge_indices,
            d_topk_scores,
            d_topk_indices,
            remaining,
            1,
            k
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

    // CPU reference: compute all similarities and find exact top-k (checking to make sure it's correct)
    bool exact = true;

    for (int q = 0; q < B; q++) {
        std::vector<std::pair<float, int>> all_scores;

        for (int c = 0; c < N; c++) {
            if (h_query_labels[q] == h_candidate_labels[c])
                continue;

            float score = 0.0f;

            for (int d = 0; d < D; d++) {
                score += h_queries[q * D + d] *
                        h_candidates[c * D + d];
            }

            all_scores.push_back({score, c});
        }

        std::partial_sort(
            all_scores.begin(),
            all_scores.begin() + k,
            all_scores.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            }
        );

        for (int j = 0; j < k; j++) {
            int gpu_idx = h_indices[q * k + j];
            float gpu_score = h_scores[q * k + j];

            float cpu_score = all_scores[j].first;
            int cpu_idx = all_scores[j].second;

            if (gpu_idx != cpu_idx ||
                std::fabs(gpu_score - cpu_score) > 1e-4f) {

                printf("Mismatch q=%d rank=%d: GPU (%d, %.6f), CPU (%d, %.6f)\n",
                    q, j, gpu_idx, gpu_score, cpu_idx, cpu_score);

                exact = false;
                break;
            }
        }
    }

    printf("Exact top-k correctness: %s\n", exact ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_partial_scores));
    CUDA_CHECK(cudaFree(d_partial_indices));
    CUDA_CHECK(cudaFree(d_topk_scores));
    CUDA_CHECK(cudaFree(d_query_labels));
    CUDA_CHECK(cudaFree(d_candidate_labels));
    CUDA_CHECK(cudaFree(d_topk_indices));
    CUDA_CHECK(cudaFree(d_merge_scores));
    CUDA_CHECK(cudaFree(d_merge_indices));

    if (valid) {
        return 0;
    }
    else return 1;
}