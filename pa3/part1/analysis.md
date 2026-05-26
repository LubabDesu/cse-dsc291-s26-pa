# Part 1.3 — Benchmark Analysis

## 1. Setup
- **Hardware:** MacBook (Apple Silicon)
- **World Size:** 4 Ranks
- **Configuration:** Feature Dim = 64, Hidden Dim = 256, Output Dim = 64, Top-k = 2
- **MPI Library:** mpi4py

## 2. Sweeps
We swept the **Batch Size** from 8 to 2048 to observe the scaling behavior of each MoE variant.

| Batch Size | SimpleMoE (ms) | MoE_TP (ms) | MoE_EP (ms) |
|------------|----------------|-------------|-------------|
| 8          | 0.23           | 1.48        | 0.27        |
| 16         | 0.23           | 2.38        | 0.42        |
| 32         | 0.46           | 4.89        | 1.14        |
| 64         | 1.08           | 9.54        | 1.01        |
| 128        | 1.77           | 17.85       | 2.09        |
| 256        | 3.51           | 35.87       | 3.77        |
| 512        | 6.76           | 70.87       | 8.85        |
| 1024       | 12.96          | 142.71      | 17.64       |
| 2048       | 27.26          | 318.53      | 45.02       |

## 3. Discussion

### Computation vs. Communication Bottlenecks
The benchmark highlights a significant transition between computation-bound and communication-bound regimes:

1.  **Small Batch Size (8-32)**: At low batch sizes, the actual math (matrix multiplication) is extremely fast. However, both TP and EP variants incur network overhead. `SimpleMoE` is the fastest here because it avoids any inter-process communication. The parallel versions are **Communication-Bound**.

2.  **MoE_TP Performance**: `MoE_TP` is significantly slower than both Simple and EP, especially as the batch size increases (reaching 318ms at batch 2048). This is because our implementation calls `ShardedExpert` inside a loop over every token. Each expert call triggers an MPI `Allreduce`. For a batch of 2048, we are invoking thousands of blocking collectives. This high frequency of small messages is highly inefficient and makes TP the bottleneck in this setup.

3.  **MoE_EP Performance**: `MoE_EP` performs much better than TP and stays relatively close to `SimpleMoE`. This is because EP aggregates all token routing into just two `Alltoall` calls per forward pass (one for tokens, one for results). By batching the communication, it minimizes the network latency penalty.

### Conclusion
For the specific hidden dimensions and world size in this benchmark, the overhead of distributed communication generally outweighs the speedup from parallelizing the computation. To see a speedup in the parallel variants, we would likely need much larger hidden dimensions (where computation dominates communication) or a more optimized TP implementation that batches collectives across the entire batch rather than token-by-token.
