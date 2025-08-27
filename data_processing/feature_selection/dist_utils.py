from __future__ import annotations
import os
import socket
from typing import Any, Dict, List, Tuple
import torch
import torch.distributed as dist


def pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def setup_distributed_env(world_size: int, master_addr: str, master_port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("WORLD_SIZE", str(world_size))


def init_process_group(rank: int, backend: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def broadcast_object(obj: Any) -> Any:
    if dist.is_available() and dist.is_initialized():
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]
    return obj


def allreduce_tensor(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t