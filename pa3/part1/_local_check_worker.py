
"""Worker run inside mpirun -n 4 from local_check.py. Writes a JSON report."""
import json
import os
import sys
import numpy as np

from mpi_wrapper import mpi
from rng import register_rng
from moe import SimpleMoE, MoE_TP, MoE_EP


def _reset_rngs(rank):
    register_rng("expert", np.random.RandomState(0))
    register_rng("router", np.random.RandomState(0))
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))


def _broadcast(seed, batch, dim):
    if mpi.Get_rank() == 0:
        X = np.random.RandomState(seed).randn(batch, dim)
    else:
        X = None
    return mpi.bcast(X, root=0)


def main():
    rank = mpi.Get_rank()
    ws = mpi.Get_size()
    dim = 8 * ws

    report = {"world_size": ws}

    # Draw the input ONCE, before any model construction, so the rng state
    # at SimpleMoE init matches the rng state at MoE_TP init exactly.
    X = _broadcast(42, 6, dim)

    # ----- SimpleMoE oracle -----
    _reset_rngs(rank)
    simple = SimpleMoE(dim, dim * 2, dim, num_experts=ws, topk=2)
    out_simple = simple(X)
    report["simple_shape"] = list(out_simple.shape)

    # ----- TP -----
    try:
        _reset_rngs(rank)
        tp = MoE_TP(dim, dim * 2, dim, num_experts=ws, topk=2)
        # Sanity-check that the router weights match SimpleMoE's, since both
        # were constructed under the same rng state.
        router_match = bool(np.allclose(simple.router.linear.weight, tp.router.linear.weight))
        report["tp_router_match_simple"] = router_match
        out_tp = tp(X)
        report["tp_shape"] = list(out_tp.shape)
        report["tp_vs_simple_max_diff"] = float(np.abs(out_simple - out_tp).max())
        report["tp_output_max_abs"] = float(np.abs(out_tp).max())
    except Exception as e:
        report["tp_error"] = repr(e)

    # ----- EP -----
    try:
        _reset_rngs(rank)
        ep = MoE_EP(dim, dim * 2, dim, num_experts=ws, topk=2)
        out_ep = ep(X)
        report["ep_shape"] = list(out_ep.shape)
        gathered = mpi.allgather(out_ep)
        if rank == 0:
            diffs = [float(np.abs(gathered[0] - gathered[r]).max()) for r in range(ws)]
            report["ep_cross_rank_max_diff"] = max(diffs)
            # Magnitude check: reject all-zero outputs (rejects the empty-stub
            # case that would otherwise pass cross-rank consistency trivially).
            report["ep_output_max_abs"] = float(np.abs(gathered[0]).max())
            # Order-of-magnitude sanity vs SimpleMoE: EP uses rank-specific
            # expert weights so values won't equal SimpleMoE, but they should
            # at least be on the same order of magnitude.
            simple_mag = float(np.abs(out_simple).max()) + 1e-12
            report["ep_vs_simple_log_ratio"] = float(
                np.log10(max(report["ep_output_max_abs"], 1e-30) / simple_mag)
            )
    except Exception as e:
        report["ep_error"] = repr(e)

    if rank == 0:
        with open(os.environ["PART1_REPORT"], "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
