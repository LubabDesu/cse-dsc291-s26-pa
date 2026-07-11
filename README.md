# Machine Learning Systems from Scratch

Coursework and implementations from UC San Diego's CSE 291 / DSC 291 Machine Learning Systems course, Spring 2026. The repository explores the systems underneath modern deep-learning frameworks, including automatic differentiation, GPU kernels, distributed communication, Mixture-of-Experts parallelism, model-scaling analysis, and inference.

*Note: This repository contains coursework implementations and assignment scaffolding.*

## [PA1: Autodiff Engine and Transformer](./pa1)
- Built a reverse-mode automatic-differentiation engine supporting 20+ operations (including matrix multiplication, softmax, LayerNorm, ReLU, broadcasting, transpose, and elementwise arithmetic) by implementing the forward and backward passes of these operations on top of a provided graph scaffolding.
- Constructed the backward computational graph and static evaluator.
- Trained a small decoder-only Transformer end-to-end without calling PyTorch autograd, with gradients validated against PyTorch within test tolerances.

## [PA2: Triton Kernels and Distributed Communication](./pa2)
- Implemented a fused Triton kernel computing `ReLU(A @ B + C)` with FP32 accumulation.
- Achieved a verified 1.1033x speedup over the PyTorch reference on an NVIDIA A10G: 0.3380 ms versus 0.3730 ms.
- Implemented MPI all-reduce and all-to-all using point-to-point communication.
- Implemented data-parallel splitting and model-parallel forward/backward communication.

## [PA3: Distributed MoE and LLM Systems](./pa3)
- Implemented and benchmarked tensor-parallel and expert-parallel Mixture-of-Experts prototypes using MPI.
- Analysed the communication bottlenecks of TP versus EP across batch sizes.
- Implemented Llama-style parameter, FLOP, and memory analysis where completed.

## Speculative Decoding
The speculative-decoding implementation and benchmark notebook is available separately:

[Open the speculative-decoding notebook in Google Colab](https://colab.research.google.com/drive/1ExLTbff6hp_7D530fZ9ht3onNWNwG_rd)

## Technologies
Python, PyTorch tensors with autograd disabled, Triton, MPI/mpi4py, NumPy.
