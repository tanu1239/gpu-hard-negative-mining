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

    //holds the current pass's 32 candidates
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

    
    // shangbag local sort -> merge -> rebuild.
    // each lane initially owns one score from each of 8 passes.
    float vals[NUM_PASSES];
    int ids[NUM_PASSES];

    for (int p = 0; p < NUM_PASSES; ++p) {
        int c = tile * TILE + p * CANDS_PER_PASS + lane;
        vals[p] = my_scores[p];
        ids[p] = (c < N && my_scores[p] != -INFINITY) ? c : -1;
    }

    // sort groups of k within each 32-lane warp.
    // all groups end up in ascending order.
    for (int size = 2; size <= k; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int p = 0; p < NUM_PASSES; ++p) {
                float other = __shfl_xor_sync(
                    FULL_MASK, vals[p], stride);
                int other_id = __shfl_xor_sync(
                    FULL_MASK, ids[p], stride);

                bool ascending =
                    (size == k) || ((lane & size) == 0);

                bool take_other = ascending
                    ? (((lane & stride) == 0)
                        ? vals[p] > other
                        : vals[p] < other)
                    : (((lane & stride) == 0)
                        ? vals[p] < other
                        : vals[p] > other);

                if (take_other) {
                    vals[p] = other;
                    ids[p] = other_id;
                }
            }
        }
    }

    //MERGE + REBUILD WITHIN EACH PASS
    // reduce 32 candidates to k, each merge retains the larger k elements.
    // rebuild needs only log2(k) stages, not a full sort.
    for (int groupSize = 2 * k;
         groupSize <= WARP_SIZE;
         groupSize <<= 1) {

        int groupBase = (lane / groupSize) * groupSize;
        int upperStart = groupBase + groupSize - k;

        bool active = lane >= upperStart;

        int partner =
            groupBase + groupSize / 2 - k
            + groupBase + groupSize - 1 - lane;

        for (int p = 0; p < NUM_PASSES; ++p) {
            float other = __shfl_sync(
                FULL_MASK, vals[p], partner);
            int other_id = __shfl_sync(
                FULL_MASK, ids[p], partner);

            if (active && other > vals[p]) {
                vals[p] = other;
                ids[p] = other_id;
            }
        }

        // rebuild the surviving bitonic top-k, in ascending order.
        for (int stride = k >> 1;
             stride > 0;
             stride >>= 1) {

            for (int p = 0; p < NUM_PASSES; ++p) {
                float other = __shfl_xor_sync(
                    FULL_MASK, vals[p], stride);
                int other_id = __shfl_xor_sync(
                    FULL_MASK, ids[p], stride);

                bool take_other =
                    ((lane & stride) == 0)
                    ? vals[p] > other
                    : vals[p] < other;

                if (active && take_other) {
                    vals[p] = other;
                    ids[p] = other_id;
                }
            }
        }
    }

    // MERGE + REBUILD ACROSS PASSES
    // hierarchically combine the eight top-k lists. Only one surviving list remains at the end.
    bool active = lane >= WARP_SIZE - k;
    int reverseLane = (WARP_SIZE - k)
                      + (WARP_SIZE - 1) - lane;

    for (int groupStride = 1;
         groupStride < NUM_PASSES;
         groupStride <<= 1) {

        // merge pairs of sorted top-k lists.
        for (int p = 0;
             p < NUM_PASSES;
             p += 2 * groupStride) {

            float other = __shfl_sync(
                FULL_MASK,
                vals[p + groupStride],
                reverseLane);

            int other_id = __shfl_sync(
                FULL_MASK,
                ids[p + groupStride],
                reverseLane);

            if (active && other > vals[p]) {
                vals[p] = other;
                ids[p] = other_id;
            }
        }

        // rebuild the surviving bitonic sequences.
        for (int stride = k >> 1;
             stride > 0;
             stride >>= 1) {

            for (int p = 0;
                 p < NUM_PASSES;
                 p += 2 * groupStride) {

                float other = __shfl_xor_sync(
                    FULL_MASK, vals[p], stride);
                int other_id = __shfl_xor_sync(
                    FULL_MASK, ids[p], stride);

                bool take_other =
                    ((lane & stride) == 0)
                    ? vals[p] > other
                    : vals[p] < other;

                if (active && take_other) {
                    vals[p] = other;
                    ids[p] = other_id;
                }
            }
        }
    }

    // vals[0] contains the final top-k in ascending order in the last k lanes.
    // reverse them to output descending top-k.
    float selected_score = __shfl_sync(
        FULL_MASK, vals[0], WARP_SIZE - 1 - lane);

    int selected_idx = __shfl_sync(
        FULL_MASK, ids[0], WARP_SIZE - 1 - lane);

    if (lane < k && q < B) {
        int out = (q * num_tiles + tile) * k + lane;
        partial_scores[out] = selected_score;
        partial_indices[out] = selected_idx;
    }
}


__global__ void hierarchical_bitonic_merge(
    const float* in_scores,
    const int* in_indices,
    float* out_scores,
    int* out_indices,
    int num_lists,
    int input_width,
    int K)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int pair = blockIdx.x * 8 + warp;
    int q = blockIdx.y;

    int out_lists = (num_lists + 1) / 2;
    if (pair >= out_lists) return;

    int first = 2 * pair;
    int second = first + 1;

    int base_a = (q * num_lists + first) * input_width;
    int base_b = (q * num_lists + second) * input_width;
    int base_out = (q * out_lists + pair) * K;

    float a = -INFINITY;
    float b = -INFINITY;
    int ai = -1;
    int bi = -1;

    int offset = lane - (K - input_width);

    if (offset >= 0 && offset < input_width) {
        ai = in_indices[base_a + offset];
        a = ai >= 0 ? in_scores[base_a + offset] : -INFINITY;
    }

    if (second < num_lists && lane < input_width) {
        bi = in_indices[base_b + lane];
        b = bi >= 0 ? in_scores[base_b + lane] : -INFINITY;
    }

    // reverse the second descending list.
    float reversed_b = __shfl_sync(
        0xffffffff, b, K - 1 - lane, K);
    int reversed_bi = __shfl_sync(
        0xffffffff, bi, K - 1 - lane, K);

    // retain the larger element from each mirrored pair.
    float value = fmaxf(a, reversed_b);
    int index = (a >= reversed_b) ? ai : reversed_bi;

    // the retained sequence is bitonic, rebuild it into descending order.
    for (int stride = K / 2; stride > 0; stride >>= 1) {
        float other = __shfl_xor_sync(
            0xffffffff, value, stride, K);
        int other_index = __shfl_xor_sync(
            0xffffffff, index, stride, K);

        bool take_other = ((lane & stride) == 0)
            ? (other > value)
            : (other < value);

        if (take_other) {
            value = other;
            index = other_index;
        }
    }

    if (lane < K) {
        out_scores[base_out + lane] = value;
        out_indices[base_out + lane] = index;
    }
}

__global__ void extract_topk(
    const float* scores,
    const int* indices,
    float* final_scores,
    int* final_indices,
    int B,
    int input_width,
    int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * k) return;

    int q = i / k;
    int rank = i % k;

    final_scores[i] = scores[q * input_width + rank];
    final_indices[i] = indices[q * input_width + rank];
}

// host side: setup, execute, benchmark, and verify
int main() {
    const int B = 32;
    const int N = 48000;
    const int D = 128;
    const int k = 128;

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

    // allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_queries, B * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidates, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_scores, B * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_scores, B * num_tiles * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_indices, B * num_tiles * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_query_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidate_labels, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_indices, B * k * sizeof(int)));

    int K = 1;
    while (K < k) K <<= 1;

    if (K > 32) {
        fprintf(stderr, "This implementation supports k <= 32\n");
        return 1;
    }

    int max_merge_lists = (num_tiles + 1) / 2;
    size_t merge_count = (size_t)B * max_merge_lists * K;

    float *merge_scores_a, *merge_scores_b;
    int *merge_indices_a, *merge_indices_b;

    CUDA_CHECK(cudaMalloc((void**)&merge_scores_a,
                        merge_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&merge_scores_b,
                        merge_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&merge_indices_a,
                        merge_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&merge_indices_b,
                        merge_count * sizeof(int)));

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

    const int finalists_per_query = num_tiles * k;
    const size_t merge_smem = finalists_per_query * (sizeof(float) + sizeof(int));

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
    
        int lists = num_tiles;
        int input_width = k;

        const float* src_scores = d_partial_scores;
        const int* src_indices = d_partial_indices;

        float* dst_scores = merge_scores_a;
        int* dst_indices = merge_indices_a;

        while (lists > 1) {
            int next_lists = (lists + 1) / 2;

            dim3 merge_grid((next_lists + 7) / 8, B);

            hierarchical_bitonic_merge<<<merge_grid, 256>>>(
                src_scores,
                src_indices,
                dst_scores,
                dst_indices,
                lists,
                input_width,
                K
            );
            CUDA_CHECK(cudaGetLastError());

            lists = next_lists;
            input_width = K;

            src_scores = dst_scores;
            src_indices = dst_indices;

            if (dst_scores == merge_scores_a) {
                dst_scores = merge_scores_b;
                dst_indices = merge_indices_b;
            } else {
                dst_scores = merge_scores_a;
                dst_indices = merge_indices_a;
            }
        }

        extract_topk<<<(B * k + 255) / 256, 256>>>(
            src_scores,
            src_indices,
            d_topk_scores,
            d_topk_indices,
            B,
            input_width,
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
    CUDA_CHECK(cudaFree(merge_scores_a));
    CUDA_CHECK(cudaFree(merge_scores_b));
    CUDA_CHECK(cudaFree(merge_indices_a));
    CUDA_CHECK(cudaFree(merge_indices_b));

    if (valid) {
        return 0;
    }
    else return 1;
}