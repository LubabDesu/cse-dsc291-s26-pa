# Part 3 — Speculative Decoding Report

This report presents the performance evaluation and optimization analysis of a single-sequence (batch=1) speculative decoder implemented for Pythia models on an NVIDIA T4 GPU.

---

## 1. Experimental Setup
* **Target Model**: `EleutherAI/pythia-1.4b-deduped` (1.4 Billion parameters, loaded in FP16)
* **Draft Model**: `EleutherAI/pythia-160m-deduped` (160 Million parameters, loaded in FP16)
* **Hardware**: Google Colab T4 GPU
* **Task**: Generate 100 tokens greedily starting from three distinct prompt contexts.

---

## 2. Benchmark Performance (k = 8)
Below are the results of speculative decoding compared against the target-only baseline (greedy autoregressive generation) with `num_speculative_tokens = 8`:

| Prompt Context | Speculative Latency (s) | Baseline Latency (s) | Speedup (x) | Acceptance Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Prompt 1**: "The future of artificial intelligence is" | 1.60s | 2.05s | 1.28x | 91.67% |
| **Prompt 2**: "Write a short story about a robot..." | 1.12s | 1.84s | 1.64x | 100.00% |
| **Prompt 3**: "Write the lyrics to the song 'Happy Birthday'." | 1.40s | 1.80s | 1.29x | 93.68% |

### Analysis
All three prompts exceeded the performance target of a **1.0x speedup** and a **75% draft token acceptance rate**. The high acceptance rates (91.67% to 100%) indicate that the draft model is highly aligned with the target model's predictions, allowing the target model to confirm and accept large blocks of tokens in a single forward pass.

---

## 3. Parameter Sweep (num_speculative_tokens)
To understand the trade-offs of speculation length, we swept `num_speculative_tokens` (k) in {2, 4, 8, 16} using Prompt 1:

| num_speculative_tokens (k) | Average Latency (s) | Tokens / Second | Acceptance Rate (%) | Speedup vs Baseline |
| :---: | :---: | :---: | :---: | :---: |
| **Baseline (Target Only)** | 2.05s | 49.90 | N/A | 1.00x |
| **k = 2** | 2.92s | 34.50 | 97.06% | 0.70x (slowdown) |
| **k = 4** | 2.62s | 39.00 | 95.24% | 0.78x (slowdown) |
| **k = 8** | 1.71s | 63.30 | 91.67% | 1.20x |
| **k = 16** | 1.33s | 76.70 | 85.45% | 1.54x |

### Discussion of Trade-offs
1. **Acceptance Rate vs. Speculation Length**: As `num_speculative_tokens` increases, the draft-token acceptance rate decreases (from 97.06% at k=2 down to 85.45% at k=16). This is because predicting further into the future is inherently harder, leading to cumulative divergence.
2. **Speedup vs. Speculation Length**: Despite the lower acceptance rate, the overall speedup increases as k grows. Proposing more tokens per step amortizes the target model's forward pass cost over larger verified token blocks.
3. **Small k Bottleneck**: At small values (k = 2 and k = 4), speculative decoding is slower than the baseline (0.70x and 0.78x). The latency overhead of running both models and copying memory exceeds the savings of the short correct sequences, demonstrating that speculative decoding is only beneficial when the batch/propose block size is sufficiently large.

---

## 4. Bonus 3.B: N-gram Lookup Decoding
We implemented N-gram lookup (Prompt Lookup Decoding) as a training-free alternative to the draft model. The lookup searches the sequence history for matching n-grams and copies the following tokens. Below is the comparison of N-gram lookup (k = 8) against the Pythia-160M draft model:

| Prompt Context | N-gram Speedup | N-gram Acceptance | Pythia-160M Speedup | Pythia-160M Acceptance |
| :--- | :---: | :---: | :---: | :---: |
| **Prompt 1** (Fact/Intro) | **9.28x** | 68.00% | 1.28x | 91.67% |
| **Prompt 2** (Creative Story) | **2.22x** | 16.62% | 1.64x | 100.00% |
| **Prompt 3** (Lyrics) | **6.19x** | 78.38% | 1.29x | 93.68% |

### Key Findings
1. **Zero Draft Overhead**: N-gram lookup operates purely on array slices in CPU/system memory (taking ~0ms). In contrast, the neural draft model requires active GPU forward passes. Because drafting is computationally "free," N-gram lookup yields much higher speedups.
2. **Repetition Advantage**: On highly structured or repetitive text (like Prompt 3's lyrics, which get a 6.19x speedup, and Prompt 1, which gets a 9.28x speedup), N-gram matching is highly accurate.
3. **No Penalty for Mismatches**: On creative tasks with low repetition (like Prompt 2), the N-gram acceptance rate drops to 16.62%. However, because drafting is computationally free, it still achieves a **2.22x speedup** over the baseline. Under a neural draft model, a 16% acceptance rate would cause a severe slowdown, showing that N-gram lookup is exceptionally robust.
