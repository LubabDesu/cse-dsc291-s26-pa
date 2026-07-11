# CSE 291 / DSC 291: Machine Learning Systems

Coursework from UC San Diego's ML Systems course (Spring 2026). The theme across all three assignments is building the pieces underneath a deep learning framework by hand instead of leaning on PyTorch's built-ins: no `loss.backward()`, no `torch.distributed`, no library MoE routing. Just the primitives.

## PA1: Autograd Engine

A reverse-mode automatic differentiation engine built from scratch (`pa1/auto_diff.py`), roughly 20 ops deep: `MatMul`, `Softmax`, `LayerNorm`, `ReLU`, `Sqrt`, `Power`, broadcasting, transpose, and the elementwise arithmetic, each with a hand-derived forward and backward pass wired into a computation graph (`Node`, `Op`, `Evaluator`). The engine trains a full transformer (`pa1/transformer.py`) end to end, gradients included, without ever touching PyTorch's autograd.

## PA2: GPU Kernels + Distributed Communication

Two parts, both close to the metal:

- A fused Triton kernel (`pa2/student_kernel.py`) computing `D = ReLU(A @ B + C)` in fp16 with fp32 accumulation, tuned across tile assignment, shared memory staging, and register-level accumulation, with an autotuning grid search over block configs.
- MPI collective primitives implemented by hand (`pa2/mpi_wrapper/comm.py`): `myAllreduce` and `myAlltoall`, plus the data-parallel data splitting and naive model-parallel forward/backward communication built on top of them (`pa2/model`, `pa2/data`).

## PA3: MoE, Scaling Laws, Speculative Decoding

- Mixture-of-Experts implemented two ways (`pa3/part1/moe.py`): tensor-parallel (`MoE_TP`) and expert-parallel (`MoE_EP`), benchmarked against each other.
- Training cost analysis under scaling laws (`pa3/part2`): deriving Llama-3 8B's exact parameter count from its config, then picking the compute-optimal model size and GPU under a fixed budget, plus a bonus cost breakdown for DeepSeek-V3's MoE architecture.
- Speculative decoding implemented and benchmarked (`pa3/part3`), with a bonus tree/multi-branch speculation variant.

## Stack

Python, PyTorch (tensors only, autograd disabled), Triton, MPI (`mpi4py`), NumPy.

---

Originally coursework for CSE 291 / DSC 291 (UCSD, Spring 2026), forked here to track my own submissions.
