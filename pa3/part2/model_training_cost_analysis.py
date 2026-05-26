"""Model training cost analysis for Part 2.

You will implement three functions:

  - `model_training_cost_analysis_llama(config_path)`
  - `model_training_cost_analysis_deepseek(config_path)`
  - `get_optimal_N_D_from_cost(cost_budget)`

Run from the command line:

  python model_training_cost_analysis.py --model_config llama3_8b_config.json
  python model_training_cost_analysis.py --model_config deepseek_v3_config.json
  python model_training_cost_analysis.py --training_budget 5000000
"""
import argparse
import json
import math


def model_training_cost_analysis_llama(model_config_path):
    """Analyze training cost of a dense Llama-style model.

    Returns:
        total_params:   total trainable parameter count (int)
        flops_layer_TF: forward FLOPs of a single transformer layer in TFLOPs
                        (use a representative sequence length)
        peak_memory_GB: peak forward memory of a single transformer layer
                        under bf16 / fp16 with rematerialization at layer
                        boundaries (in GB)
    """
    with open(model_config_path, "r") as f:
        config = json.load(f)

    # Extract architecture parameters
    h = config["hidden_size"]
    i = config["intermediate_size"]
    L = config["num_hidden_layers"]
    n_q = config["num_attention_heads"]
    n_kv = config["num_key_value_heads"]
    v = config["vocab_size"]
    s = config["max_position_embeddings"]  # Grading uses max_position_embeddings
    tie = config.get("tie_word_embeddings", False)

    # Helper for projections
    head_dim = h // n_q

    embeddings = v * h
    attn = h * h + 2 * (h * n_kv / n_q * h) + h * h
    mlp = 3 * (h * i)
    params_in_blocks = (attn + mlp) * L
    norms = 2 * h * L + h
    lm_head = v * h
    total_params = params_in_blocks + norms + embeddings
    if not tie : 
        total_params += lm_head
    
    B = 1
    flops_layer_TF = 0 
    # projections :  Q + KV + output
    flops_layer_TF += (2 * B * s * h * h) + (2 * 2 * B * s * h * n_kv / n_q * head_dim) + (2 * B * s * h * h)
    # attention score
    flops_layer_TF += 2 * B * n_q * s * s * head_dim
    # context 
    flops_layer_TF += 2 * B * n_q * s * s * head_dim
    # MLP SwiGLU
    flops_layer_TF += 2 * B * s * h * i * 2 + 2 * B * s * i * h

    flops_layer_TF /= 1e12

    # memory
    attention_peak = (B * n_q * s * s) + (3 * B * s * h)

    mlp_peak = (2 * B * s * i)

    
    peak_memory_GB = max(mlp_peak, attention_peak) * 2 / 1e9
    return (total_params, flops_layer_TF, peak_memory_GB)


def model_training_cost_analysis_deepseek(model_config_path):
    """Analyze training cost of a DeepSeek-V3-style MoE model.

    Same return signature as the Llama version. Take care to:
      - account for the MLA attention parameters (q/kv low-rank projections
        plus the rope/no-rope head splits)
      - distinguish dense layers (the first `first_k_dense_replace`) from
        MoE layers (use `n_routed_experts`, `moe_intermediate_size`, plus
        `n_shared_experts`)
      - report *activated* FLOPs per token in the MoE layers (i.e. only the
        `num_experts_per_tok` experts that fire)
    """
    with open(model_config_path, "r") as f:
        config = json.load(f)

    # Global
    h = config["hidden_size"]
    L = config["num_hidden_layers"]
    v = config["vocab_size"]
    s = config["max_position_embeddings"]
    
    # MLA Attention
    n_heads = config["num_attention_heads"]
    q_rank = config["q_lora_rank"]
    kv_rank = config["kv_lora_rank"]
    d_nope = config["qk_nope_head_dim"]
    d_rope = config["qk_rope_head_dim"]
    d_v = config["v_head_dim"]
    
    # MLP / MoE
    k_dense = config["first_k_dense_replace"]
    dense_i = config["intermediate_size"]
    n_routed = config["n_routed_experts"]
    n_shared = config["n_shared_experts"]
    top_k = config["num_experts_per_tok"]
    moe_i = config["moe_intermediate_size"]

    # TODO: Implement parameter, FLOPs, and memory calculations.

    # params
    embeddings, lm_head = v * h, v * h
    norms = (2 * L + 1) * h
    # mla attn pper layer
    mla_attn = 0
    mla_attn += h * q_rank + q_rank * (n_heads * (d_rope + d_nope))
    mla_attn += h * kv_rank + kv_rank * n_heads * (d_nope + d_v) + h * d_rope
    mla_attn += n_heads * d_v * h

    # mlp and moe layers

    # dense layers
    dense_mlp = 3 * h * dense_i

    # moe layers
    mlp_moe = (n_routed + n_shared) * 3 * h * moe_i
    total = embeddings + lm_head + norms + L * mla_attn + k_dense * dense_mlp + ( L - k_dense ) * mlp_moe

    activated_moe = (top_k + n_shared) * 3 * h * moe_i


    # FLOPS
    B = 1


def get_optimal_N_D_from_cost(cost_budget):
    """Pick the GPU and (N, D) that minimize loss under a $ training budget.

    Use the scaling law:

        L(N, D) = 406.4 / N^0.34 + 410.7 / D^0.29 + 1.69

    GPU options (assume MFU = 40% across all three):

        - H100 (SXM, fp16):   $3.0/hour, peak FP16 = 989 TFLOPs
        - H200 (SXM, fp16):   $4.0/hour, peak FP16 = 989 TFLOPs
        - B200 (SXM, fp16):   $6.0/hour, peak FP16 = 2250 TFLOPs

    The training-flops budget for a given GPU at price `p` $/h is

        F_total = cost_budget / p * 3600 * peak_TFLOPs * 1e12 * MFU

    Use the standard 6 N D ≈ F_total approximation to convert budget into a
    constraint, then solve the (N, D) optimization implied by the scaling law.

    Returns:
        N: optimal model parameter count (absolute number)
        D: optimal training token count (absolute number)
        training_budget_flops: effective total training FLOPs
        best_gpu: name of the selected GPU, one of {'H100', 'H200', 'B200'}
    """
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training cost analysis")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--training_budget", type=float, default=None,
                        help="Training budget in dollars")
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_deepseek(args.model_config)
            )
        elif "llama" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_llama(args.model_config)
            )
        else:
            print("Unknown model type — name your config llama*.json or deepseek*.json")
            raise SystemExit(1)
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(
            args.training_budget
        )
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
