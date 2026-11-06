'''
Micro-benchmark script for MatRoute Funnel Decoder and Sparse Encoder (Stage 7).
Measures:
1. Decoder step execution time (ms) and peak GPU memory (MB): FULL vs FUNNEL.
2. Encoder forward pass time (ms) and peak memory (MB): Dense vs Sparse kNN (K=100).
Evaluates across problem scales N in [200, 500, 1000, 2000].
Catches OOM gracefully if GPU memory limit is reached.
'''

import os
import sys
from pathlib import Path

# Add root repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import time
import argparse
from typing import Dict, List, Any

import torch
import torch.nn.functional as F

from matroute.layout import BlockLayout
from matroute.spec import get_variant_blocks
from matroute.policy import MatRoutePolicy
from matroute.features import (
    build_raw_block_features,
    build_dynamic_query_features,
)


def benchmark_decoder_step(
    policy: MatRoutePolicy,
    S: Any,
    N: int,
    batch_size: int = 16,
    pomo_size: int = 16,
    num_warmup: int = 5,
    num_repeat: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Benchmarks single decoder step execution time and peak memory for FULL vs FUNNEL.
    """
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Simulated encoded node embeddings
    active_dim = policy.layout.active_dim(S)
    encoded_nodes = torch.randn(batch_size, N, active_dim, device=device)
    keys = policy.decoder.compute_keys(encoded_nodes, S=S)

    h_first = torch.randn(batch_size, pomo_size, active_dim, device=device)
    h_last = torch.randn(batch_size, pomo_size, active_dim, device=device)
    dyn_feats = {}
    queries = policy.decoder.compute_queries(h_first, h_last, dyn_feats, S=S)
    feasible = torch.ones(batch_size, pomo_size, N, dtype=torch.bool, device=device)

    # 1. Benchmark FULL
    for _ in range(num_warmup):
        _ = policy.decoder.forward_step(queries, keys, feasible, S=S, level="FULL")
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_t = time.perf_counter()
    for _ in range(num_repeat):
        _ = policy.decoder.forward_step(queries, keys, feasible, S=S, level="FULL")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    full_step_ms = (time.perf_counter() - start_t) / num_repeat * 1000.0
    full_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    # 2. Benchmark FUNNEL
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    topk = {"CORE": min(50, N // 2), "DEM": min(20, N // 4), "TW": min(20, N // 4), "RTE": min(16, N // 4)}
    for _ in range(num_warmup):
        _ = policy.decoder.forward_step(queries, keys, feasible, S=S, level="FUNNEL", topk=topk)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_t = time.perf_counter()
    for _ in range(num_repeat):
        _ = policy.decoder.forward_step(queries, keys, feasible, S=S, level="FUNNEL", topk=topk)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    funnel_step_ms = (time.perf_counter() - start_t) / num_repeat * 1000.0
    funnel_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    speedup = full_step_ms / (funnel_step_ms + 1e-9)

    return {
        "N": N,
        "full_step_ms": round(full_step_ms, 3),
        "funnel_step_ms": round(funnel_step_ms, 3),
        "decoder_speedup": round(speedup, 2),
        "full_peak_mb": round(full_peak_mb, 2),
        "funnel_peak_mb": round(funnel_peak_mb, 2),
    }


def benchmark_encoder_pass(
    policy: MatRoutePolicy,
    S: Any,
    N: int,
    batch_size: int = 8,
    sparse_k: int = 100,
    num_warmup: int = 3,
    num_repeat: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Benchmarks Encoder forward pass execution time and peak memory for Dense vs Sparse kNN.
    """
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Simulated synthetic coordinates & distance matrix
    xy = torch.rand(batch_size, N, 2, device=device)
    dist = torch.cdist(xy, xy)
    problems = torch.zeros(batch_size, N, 6, device=device)
    problems[:, :, :2] = xy

    raw_feats = build_raw_block_features(problems=problems, variant="cvrp", S=S)
    init_emb = policy.input_embedder.concat_embedded(policy.input_embedder(raw_feats, S=S), S=S)

    # 1. Dense Encoder
    dense_ms = 0.0
    dense_mb = 0.0
    dense_oom = False
    try:
        for _ in range(num_warmup):
            _ = policy.encoder(init_emb, S=S, dist=dist, sparse_knn=None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_t = time.perf_counter()
        for _ in range(num_repeat):
            _ = policy.encoder(init_emb, S=S, dist=dist, sparse_knn=None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dense_ms = (time.perf_counter() - start_t) / num_repeat * 1000.0
        dense_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    except torch.cuda.OutOfMemoryError:
        dense_oom = True
        torch.cuda.empty_cache()

    # 2. Sparse kNN Encoder
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sparse_ms = 0.0
    sparse_mb = 0.0
    sparse_oom = False
    try:
        for _ in range(num_warmup):
            _ = policy.encoder(init_emb, S=S, dist=dist, sparse_knn=sparse_k)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_t = time.perf_counter()
        for _ in range(num_repeat):
            _ = policy.encoder(init_emb, S=S, dist=dist, sparse_knn=sparse_k)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sparse_ms = (time.perf_counter() - start_t) / num_repeat * 1000.0
        sparse_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    except torch.cuda.OutOfMemoryError:
        sparse_oom = True
        torch.cuda.empty_cache()

    return {
        "N": N,
        "dense_ms": round(dense_ms, 2) if not dense_oom else "OOM",
        "sparse_ms": round(sparse_ms, 2) if not sparse_oom else "OOM",
        "dense_peak_mb": round(dense_mb, 2) if not dense_oom else "OOM",
        "sparse_peak_mb": round(sparse_mb, 2) if not sparse_oom else "OOM",
    }


def main():
    parser = argparse.ArgumentParser(description="MatRoute Micro-benchmark (Funnel & Sparse)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000], help="Node sizes N")
    parser.add_argument("--preset", type=str, default="d192", help="Layout preset (d192, d128, d64)")
    parser.add_argument("--layers", type=int, default=6, help="Encoder layers for benchmark")
    parser.add_argument("--output_csv", type=str, default="results/bench_funnel.csv", help="CSV output path")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    print(f"=== MatRoute Micro-benchmark on {device} (preset={args.preset}, layers={args.layers}) ===")

    layout = BlockLayout.from_preset(args.preset)
    policy = MatRoutePolicy(layout=layout, num_encoder_layers=args.layers, zeta=50.0).to(device)
    policy.eval()

    S = get_variant_blocks("cvrp")  # CORE + DEM + RTE

    dec_results = []
    enc_results = []

    for N in args.sizes:
        print(f"\n--- Benchmarking Scale N={N} ---")
        try:
            d_res = benchmark_decoder_step(policy, S=S, N=N, device=device)
            dec_results.append(d_res)
            print(
                f"  Decoder Step: FULL = {d_res['full_step_ms']}ms ({d_res['full_peak_mb']}MB) | "
                f"FUNNEL = {d_res['funnel_step_ms']}ms ({d_res['funnel_peak_mb']}MB) | "
                f"Speedup = {d_res['decoder_speedup']}x"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  Decoder Step N={N}: OOM encountered on local GPU")
            dec_results.append({"N": N, "full_step_ms": "OOM", "funnel_step_ms": "OOM", "decoder_speedup": "-", "full_peak_mb": "OOM", "funnel_peak_mb": "OOM"})
            torch.cuda.empty_cache()

        try:
            e_res = benchmark_encoder_pass(policy, S=S, N=N, sparse_k=min(100, N // 2), device=device)
            enc_results.append(e_res)
            print(
                f"  Encoder Pass: Dense = {e_res['dense_ms']}ms ({e_res['dense_peak_mb']}MB) | "
                f"Sparse = {e_res['sparse_ms']}ms ({e_res['sparse_peak_mb']}MB)"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  Encoder Pass N={N}: OOM encountered on local GPU")
            enc_results.append({"N": N, "dense_ms": "OOM", "sparse_ms": "OOM", "dense_peak_mb": "OOM", "sparse_peak_mb": "OOM"})
            torch.cuda.empty_cache()

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "N", "Metric_A (Dense/Full)", "Metric_B (Sparse/Funnel)", "Speedup/Ratio", "Peak_Mem_A_MB", "Peak_Mem_B_MB"])
        for r in dec_results:
            writer.writerow(["Decoder", r["N"], r["full_step_ms"], r["funnel_step_ms"], r["decoder_speedup"], r["full_peak_mb"], r["funnel_peak_mb"]])
        for r in enc_results:
            writer.writerow(["Encoder", r["N"], r["dense_ms"], r["sparse_ms"], "-", r["dense_peak_mb"], r["sparse_peak_mb"]])

    print(f"\n[DONE] Micro-benchmark complete. Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
