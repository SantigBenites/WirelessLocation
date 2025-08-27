from __future__ import annotations
import json
import time
from datetime import datetime, timezone
import multiprocessing as mp
from typing import List, Tuple

import torch, os
from pymongo import MongoClient

from config import Config
from data import discover_numeric_feature_keys, accumulate_statistics
from pca import compute_pca_from_cov, select_components, feature_importance_from_pca, tolist
from dist_utils import (
    pick_free_port, setup_distributed_env, init_process_group, cleanup_distributed,
    broadcast_object, allreduce_tensor,
)
from logging_wandb import log_to_wandb
from writeback import write_results


def worker(rank: int, cfg: Config, world_size: int, master_addr: str, master_port: int,
           return_dict):
    torch.manual_seed(cfg.seed + rank)

    is_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}" if (is_cuda and world_size > 1) else ("cuda:0" if is_cuda else "cpu"))
    backend = "nccl" if is_cuda else "gloo"

    # Initialize distributed if multi-process
    if world_size > 1:
        setup_distributed_env(world_size, master_addr, master_port)
        init_process_group(rank, backend)

    if rank == 0:
        print(f"[rank {rank}] device={device} CUDA={is_cuda} world_size={world_size}")

    client = MongoClient(cfg.mongo_uri)

    # Rank 0 discovers features; broadcast to others
    if rank == 0:
        feat_keys = discover_numeric_feature_keys(client, cfg.db_name, cfg.collections,
                                                  set(cfg.exclude_keys), cfg.sample_per_collection)
        if not feat_keys:
            print("No numeric features discovered after exclusions. Exiting.")
            if world_size > 1:
                cleanup_distributed()
            return
    else:
        feat_keys = []

    feat_keys = broadcast_object(feat_keys)

    # Accumulate global statistics
    t0 = time.time()
    local_sum, local_XtX, local_N = accumulate_statistics(client, cfg.db_name, cfg.collections,
                                                          feat_keys, device, rank, world_size, cfg.batch_size)
    global_sum = allreduce_tensor(local_sum)
    global_XtX = allreduce_tensor(local_XtX)
    global_N = torch.tensor([local_N], dtype=torch.int64, device=device)
    global_N = allreduce_tensor(global_N)
    N = int(global_N.item())

    if rank == 0:
        print(f"Accumulation done in {time.time() - t0:.2f}s | N={N} D={len(feat_keys)}")
        eigvals, eigvecs, mu = compute_pca_from_cov(global_sum, global_XtX, N, device=device)
        D = eigvals.numel()
        ratios = eigvals / (eigvals.sum() + 1e-12)
        cum_ratios = torch.cumsum(ratios, dim=0)
        k = select_components(eigvals, variance_threshold=cfg.variance_threshold, n_components=cfg.n_components)
        importance = feature_importance_from_pca(eigvals, eigvecs, k)
        top_m = min(cfg.top_m_features, D)
        top_vals, top_idx = torch.topk(importance, k=top_m, largest=True, sorted=True)
        selected_features = [feat_keys[i] for i in tolist(top_idx.long())]

        results = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "db_name": cfg.db_name,
            "collections": cfg.collections,
            "excluded_keys": list(cfg.exclude_keys),
            "num_rows": N,
            "num_features": D,
            "selected_num_components": k,
            "variance_threshold": cfg.variance_threshold,
            "explained_variance_ratio": tolist(ratios),
            "cumulative_explained_variance": tolist(cum_ratios),
            "feature_keys": feat_keys,
            "feature_importance": {feat_keys[i]: float(top_vals[j]) for j, i in enumerate(tolist(top_idx.long()))},
            "selected_features": selected_features,
        }

        # Save JSON
        out_json = "pca_feature_selection_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_json}")

        # W&B (optional)
        log_to_wandb(
            enabled=cfg.wandb_enabled,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=cfg.wandb_run_name,
            notes=cfg.wandb_notes,
            mode=cfg.wandb_mode,
            config={
                "db_name": cfg.db_name,
                "collections": cfg.collections,
                "excluded_keys": list(cfg.exclude_keys),
                "variance_threshold": cfg.variance_threshold,
                "n_components_override": cfg.n_components,
                "top_m_features": cfg.top_m_features,
                "num_rows": N,
                "num_features": D,
                "device": str(device),
                "world_size": world_size,
            },
            ratios=results["explained_variance_ratio"],
            cum_ratios=results["cumulative_explained_variance"],
            k=k,
            feature_importance=results["feature_importance"],
            loadings_table=None,               # Optional: add loadings if you want
            loading_headers=None,
            outfile=out_json,
        )

        # Write-back
        if cfg.write_back_results:
            write_results(client, cfg.db_name, cfg.results_collection, {
                **results,
                "method": "PCA_feature_selection",
                "version": 1,
            })

        # Return small summary via shared dict (optional)
        return_dict.update({
            "selected_features": selected_features,
            "num_rows": N,
            "num_features": D,
            "selected_components": k,
        })

    if world_size > 1:
        cleanup_distributed()


def main():
    cfg = Config()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

    # Determine world_size
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = cfg.world_size or (n_cuda if n_cuda > 0 else 1)
    world_size = max(1, world_size)

    # If single process (CPU or 1 GPU), just run worker(0)
    if (not cfg.use_spawn) or world_size == 1:
        from collections import defaultdict
        rd = defaultdict(lambda: None)
        port = cfg.master_port or pick_free_port()
        worker(0, cfg, world_size, cfg.master_addr, port, rd)
        if rd.get("selected_features"):
            print("Top features:", rd["selected_features"])
        return

    # Multi-process spawn with mp
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    master_port = cfg.master_port or pick_free_port()
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, cfg, world_size, cfg.master_addr, master_port, return_dict))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    if "selected_features" in return_dict:
        print("Top features:", list(return_dict["selected_features"]))


if __name__ == "__main__":
    main()