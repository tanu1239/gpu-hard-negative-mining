# GPU-Accelerated Hard-Negative Mining for Contrastive Learning

## Summary

We plan to implement and optimize hard-negative mining for contrastive learning on the NVIDIA GPUs in the lab. We will investigate shared-memory tiling, parallel top-k selection, and different GPU work partitioning methods to improve performance and reduce memory usage compared to naive implementations.

## Background


Contrastive learning is a machine learning technique that trains models to produce similar embeddings for related examples and dissimilar embeddings for unrelated examples. An embedding is a numerical vector representing an input, such as an image or a word, in a learned feature space. In hard-negative mining, a query is an input example for which we want to find challenging negatives, while candidates are other examples in a larger dataset that are evaluated for their similarity to the query. Hard-negative mining identifies negative examples among the candidates that are particularly similar to a given query, allowing models to learn to distinguish between challenging examples that are similar.

In our project, we will focus on exact hard-negative mining over a large collection of candidate embeddings. Given a batch of B query embeddings of dimension D and a collection containing N candidate embeddings, the algorithm computes the similarity between every query and every candidate using their dot product. This involves multiplying corresponding elements of the two embedding vectors and summing the results, producing a B × N matrix of similarity scores. Then, the algorithm selects the k candidates with the highest similarity scores for each query. Computing all these dot products requires O(BND) operations, while storing the similarity matrix requires O(BN) memory. However, since only k candidates are ultimately selected for each query, the vast majority of these similarity scores are discarded. For example, with 512 queries, 100,000 candidates, and k = 32, the algorithm computes and stores 51.2 million similarity scores (approximately 205 MB in 32-bit floating-point format), but ultimately retains only 16,384 (512 x 32) candidates. This motivates using GPU parallelism to accelerate the large number of independent similarity computations, while developing more memory-efficient algorithms that avoid storing the entire similarity matrix when only a small fraction of its scores are actually needed. 

We aim to develop efficient GPU implementations that exploit parallelism in both similarity computation and top-k selection. In particular, we will investigate processing candidate embeddings in tiles and selecting the highest-scoring candidates without storing the full similarity matrix.

## The Challenge

Hard-negative mining presents several opportunities and tradeoffs to explore:

Memory management - Although computing similarities between embeddings is highly parallelizable, efficiently mapping this computation onto a GPU requires careful memory management. A naive implementation may repeatedly load the same embeddings from global memory, while a tiled implementation can reuse embeddings through shared memory. We will investigate different tile sizes, memory layouts, and thread-block configurations to balance memory efficiency, data reuse, arithmetic intensity, and GPU utilization.

Top-k Selection - Unlike similarity computation, top-k selection may require communication between threads to identify the highest-scoring candidates if candidates are split amongst threads. We plan to investigate different parallel selection algorithms, including repeated reductions and bitonic top-k (Shanbhag et al., 2018). We will also explore integrating top-k selection directly into tiled similarity computation, retaining only the best candidates from each tile and merging partial results to obtain the exact global top-k. This could eliminate the need to store the entire similarity matrix, but introduces tradeoffs involving synchronization and intermediate memory usage.

Finally, we will investigate different ways to distribute the workload across GPU threads and blocks. Assigning one thread block per query simplifies selection but may underutilize the GPU when the number of queries is small. Distributing the candidates for each query across multiple blocks increases parallelism but requires additional communication to merge partial results for top-k selection. We hope to determine which strategies perform best under different workload sizes and how they scale.

## Resources

We plan to implement our algorithms primarily in using the NVIDIA GPUs available on the CMU GHC machines. We will use PyTorch to implement a reference hard-negative mining pipeline for correctness testing and performance comparisons along with a naive CUDA implementation (no parallelism). NVIDIA and machine profiling tools will be used to measure execution time, memory usage, and performance bottlenecks.

Our project will draw on two main papers. Contrastive Learning with Hard Negative Samples by Robinson et al. provides background on hard-negative sampling. Efficient Top-K Query Processing on Massively Parallel Hardware by Shanbhag et al. presents parallel GPU selection algorithms, including bitonic top-k, which we plan to study and adapt for our implementation.

We will implement our own CUDA kernels rather than relying on existing optimized mining libraries. We will initially use synthetic embedding matrices for controlled experiments and may later use embeddings from a small contrastive-learning training pipeline that we create or find online.


## Deliverables

Baseline implementation - Our first goal is to implement a straightforward hard-negative mining pipeline in PyTorch, consisting of similarity computation, invalid-candidate filtering, and top-k selection. We will then implement a naive CUDA version using separate kernels for similarity computation and selection. These implementations will serve as correctness references and performance baselines.

Similarity computations - We plan to optimize similarity computations using shared-memory tiling, coalesced memory accesses, and data reuse. We will experiment with different thread-block configurations, tile sizes, and memory layouts to reduce redundant global-memory accesses. 

Top-k selection strategies - We will then implement and compare different parallel top-k selection algorithms, including repeated parallel reductions and bitonic top-k. We will explore integrating selection directly into tiled similarity computation, maintaining local top-k candidates and merging partial results across tiles to obtain the exact global top-k. We will investigate the tradeoffs between separate and integrated implementations in terms of synchronization overhead, memory traffic, and GPU utilization.

Work partitioning strategies - We also plan to compare different parallel work partitioning strategies. We will investigate assigning one thread block per query versus distributing each query's candidate embeddings across multiple blocks as well as other potential strategies. We will evaluate how these strategies affect GPU utilization, load balancing, and the overhead of merging intermediate results.

Performance Analysis - Finally, we will conduct a detailed performance evaluation comparing all implementations. We will vary the number of queries, number of candidates, embedding dimension, and number of selected negatives (k), measuring execution time, GPU memory usage, and speedup relative to the naive CUDA implementations. We will use profiling tools to identify performance bottlenecks and analyze the effects of memory access patterns, synchronization, and work partitioning.

## Extra Work We Hope to Achieve

If time permits, we hope to integrate our optimized CUDA mining implementation into a small PyTorch contrastive-learning training pipeline that we implement or find online. This would allow us to evaluate whether improvements in mining performance translate into faster training iterations.

We may also implement an approximate hard-negative mining algorithm that examines only a subset of promising candidates rather than computing similarities against all candidates. We would then compare our approximate and exact implementations in terms of runtime, memory usage, and the fraction of true top-k negatives retrieved (candidates with highest similarity to query). 

## Fallback Plan

If integrating similarity computation and top-k selection (computing similarities within each tile, selecting its local top-k candidates, and merging these results to obtain the global top-k) proves too difficult in the given time, we will focus on optimizing them as separate CUDA kernels. We can still investigate shared-memory tiling, different parallel top-k algorithms, synchronization strategies, GPU work partitioning, and evaluate performance across different workloads. 

If integrating our implementation into a full contrastive-learning training pipeline becomes too time-consuming, we will instead benchmark the mining operation independently using randomly generated embedding vectors or vectors from some small pretrained model, without implementing the full training process.

## Platform Choice

We chose NVIDIA GPUs because hard-negative mining involves large numbers of independent similarity computations that are well suited to data parallelism. GPUs provide high memory bandwidth and fast shared memory, which can be exploited for tiled similarity computation and cooperative top-k selection. CUDA also offers explicit control over thread organization, memory access patterns, and synchronization. We have access to the GHC machines for development and benchmarking.

## Proposed Schedule

Week 1 - Continue literature review by continuing to read up on hard-negative mining and parallel GPU algorithms. Implement the PyTorch reference pipeline and a naive CUDA implementation for both similarity computation and top-k selection. Establish correctness tests and initial performance baselines.

Week 2 - Optimize similarity computation using shared-memory tiling, coalesced memory accesses, and different thread-block configurations. Compare the optimized implementation against the naive CUDA and PyTorch baselines, and prepare preliminary results for the November 23 milestone.

Week 3 - Implement and compare different parallel top-k selection algorithms, including repeated reductions and bitonic top-k. Work on integrating selection into tiled similarity computation by retaining local top-k candidates and merging them across tiles.

Week 4 - Finish optimizing top-k selection and compare different GPU work partitioning strategies. If time permits, integrate the optimized implementation into a small PyTorch contrastive-learning training pipeline to collect results and begin trying to implement approximate hard-negative mining.

Week 5 - Complete remaining experiments and approximate mining if feasible. Conduct final performance and scalability comparisons, analyze bottlenecks, and prepare the final report and poster.






