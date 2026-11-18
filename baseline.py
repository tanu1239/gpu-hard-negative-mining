
import torch
import torch.nn.functional as F


def hard_negative_mining(queries, candidates, query_labels, candidate_labels, k):
    # similarity scores between every query and candidate
    similarities = queries @ candidates.T  # shape (B, N)

    # mask out candidates with the same label as the query
    mask = query_labels[:, None] == candidate_labels[None, :]
    similarities.masked_fill_(mask, float("-inf"))

    # find the k hardest negatives for each query
    scores, indices = torch.topk(similarities, k, dim=1)

    return scores, indices


def main():
    device = "cuda"

    B = 32
    N = 100_000
    D = 128
    k = 10

    torch.manual_seed(42)

    # generate random normalized embeddings
    queries = F.normalize(torch.randn(B, D, device=device), dim=1)
    candidates = F.normalize(torch.randn(N, D, device=device), dim=1)

    # generate class labels
    query_labels = torch.randint(0, 100, (B,), device=device)
    candidate_labels = torch.randint(0, 100, (N,), device=device)

    # warm up the GPU so that benchmark doesn't include init time 
    for _ in range(20):
        hard_negative_mining(
            queries, candidates, query_labels, candidate_labels, k
        )

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    iterations = 100

    start.record()

    for _ in range(iterations):
        scores, indices = hard_negative_mining(
            queries, candidates, query_labels, candidate_labels, k
        )

    end.record()
    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / iterations

    print("GPU:", torch.cuda.get_device_name(0))
    print("Scores shape:", scores.shape)
    print("Indices shape:", indices.shape)
    print(f"Average latency: {latency_ms:.4f} ms")
    print("First query indices:", indices[0].tolist())
    print("First query scores:", scores[0].tolist())


if __name__ == "__main__":
    main()
