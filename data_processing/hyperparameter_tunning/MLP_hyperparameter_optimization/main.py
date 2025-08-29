
#!/usr/bin/env python3
"""
ray_launcher.py
Run multiple end-to-end training jobs (your `main`) in parallel with Ray, one GPU per job.

Usage (local machine with N GPUs):
    python ray_launcher.py

On a Ray cluster:
    ray start --head  # or connect to an existing cluster
    # then inside this script, set ray.init(address="auto") instead of ray.init()
"""

from __future__ import annotations
import os
import json
import random
import time
from typing import Dict, List, Any, Tuple

import ray
from gpu_fucntion import ray_function
from runs import runs

# Optional: make logs shorter from libraries
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    # If you're on a Ray cluster, use: ray.init(address="auto")
    ray.init()

    futures = [
        ray_function.remote(r["model_name"],r["collections"], r["database"], seed_offset=i)
        for i, r in enumerate(runs)
    ]

    results: List[Dict[str, Any]] = ray.get(futures)

    # Pretty-print a compact JSON report
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
