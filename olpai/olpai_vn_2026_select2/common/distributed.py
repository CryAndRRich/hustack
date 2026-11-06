from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist

from common.paths import PROJECT_ROOT


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def gather(self, obj) -> list:
        if not self.enabled:
            return [obj]
        out = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            tensor = tensor.clone()
            dist.all_reduce(tensor)
        return tensor

    def broadcast_buffers(self, model: torch.nn.Module) -> None:
        if self.enabled:
            for buffer in model.buffers():
                dist.broadcast(buffer, 0)

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def init_distributed() -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        if device.type == "cuda":
            try:
                dist.init_process_group("nccl", device_id=device)
            except TypeError:
                dist.init_process_group("nccl")
        else:
            dist.init_process_group("gloo")
        return DistContext(dist.get_rank(), world_size, local_rank, device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistContext(device=device)


def launch_torchrun(nproc: int, config_path: Path, cwd: Path, module: str) -> None:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           "-m", module, "--config", str(config_path)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(cwd), env.get("PYTHONPATH", "")] if p)
    env.setdefault("OMP_NUM_THREADS", "1")
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"torchrun failed (exit code {process.returncode}), see log above")


RESULT_FILE = "result.json"


def run_or_launch(cfg, module: str, execute, log) -> dict:
    n_gpu = torch.cuda.device_count()
    if not cfg.ddp or n_gpu < 2:
        return execute(cfg)
    output_dir = Path(cfg.output_dir)
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / RESULT_FILE).unlink(missing_ok=True)
    log(f"[ddp] launching torchrun on {n_gpu} GPUs")
    launch_torchrun(n_gpu, config_path, cwd=PROJECT_ROOT, module=module)
    return json.loads((output_dir / RESULT_FILE).read_text(encoding="utf-8"))


def run_ranks(body, cfg):
    ctx = init_distributed()
    try:
        return body(cfg, ctx)
    finally:
        ctx.close()


def cli(config_cls, run, execute) -> None:
    args = sys.argv[1:]
    if args[:1] == ["--config"]:
        execute(config_cls(**json.loads(Path(args[1]).read_text(encoding="utf-8"))))
    else:
        run(config_cls().with_overrides(args))
