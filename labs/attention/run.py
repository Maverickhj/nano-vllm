"""python -m labs.attention.run --backend nanovllm --suite full"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

import torch

from labs.attention.checks import core_case, layer_case, skipped_slot_case
from labs.attention.fixtures import single_rank_group
from labs.attention.metrics import TOLERANCES


ROOT = Path(__file__).resolve().parents[2]


def _command(*args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    versions = {}
    for name in ("torch", "triton", "transformers", "flash-attn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    props = torch.cuda.get_device_properties(0)
    return {"python": sys.version, "executable": sys.executable, "versions": versions,
            "cuda": torch.version.cuda, "gpu": props.name, "sm": list(torch.cuda.get_device_capability(0)),
            "driver": _command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            "git_sha": _command("git", "rev-parse", "HEAD"),
            "git_dirty": bool(_command("git", "status", "--porcelain")),
            "command": shlex.join([sys.executable, "-m", "labs.attention.run", *sys.argv[1:]]),
            "torch_compile_disabled": os.environ.get("TORCHDYNAMO_DISABLE") == "1"}


def run_suite(backend="nanovllm", suite="full", dtype_name="bfloat16", seed=20261028):
    torch.manual_seed(seed)
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "backend": backend, "suite": suite, "dtype": dtype_name,
              "seed": seed, "environment": environment(), "tolerances": TOLERANCES[dtype_name],
              "scope": "correctness only; no latency or memory-traffic measurements",
              "deferred": ["MLA equivalence", "MLA kernels", "shared-prefix lifecycle", "performance sweep"],
              "cases": []}
    cases = []
    for kv in (8, 2, 1):
        matrix = [(2, 7)] if suite == "smoke" else [(b, s) for b in (1, 2) for s in (2, 7, 33)]
        for batch, length in matrix:
            cases.append((f"layer Hkv={kv} B={batch} S={length}",
                          lambda kv=kv, batch=batch, length=length: layer_case(
                              backend, kv, [length] * batch, seed=seed, dtype_name=dtype_name)))
        if suite == "full":
            cases.append((f"unequal paged prefix Hkv={kv}", lambda kv=kv: layer_case(
                backend, kv, [258, 259], starts=[5, 71], prefix_lengths=[255, 258],
                seed=seed, dtype_name=dtype_name)))
            cases.append((f"biased QKV Hkv={kv}", lambda kv=kv: layer_case(
                backend, kv, [7, 5], qkv_bias=True, seed=seed, dtype_name=dtype_name)))
        if backend == "nanovllm":
            cases.append((f"core extreme/mask Hkv={kv}", lambda kv=kv: core_case(kv, dtype_name, seed)))
    if backend == "nanovllm":
        cases.append(("slot_mapping=-1", skipped_slot_case))
    for label, execute in cases:
        print(f"RUN {label}", flush=True)
        try:
            result = execute()
        except NotImplementedError as error:
            result = {"passed": False, "status": "not_implemented", "error": str(error)}
            report["cases"].append({"label": label, **result})
            break  # Student TODOs must never be reported as passed/skipped.
        except Exception:
            result = {"passed": False, "status": "error", "error": traceback.format_exc()}
        report["cases"].append({"label": label, **result})
        print(f"{'PASS' if result['passed'] else 'FAIL'} {label}", flush=True)
    report["passed"] = len(report["cases"]) == len(cases) and all(c["passed"] for c in report["cases"])
    report["summary"] = {"passed": sum(c["passed"] for c in report["cases"]),
                         "executed": len(report["cases"]), "planned": len(cases)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nanovllm", "student"), default="nanovllm")
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output", type=Path, default=Path("labs/attention/results/baseline.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the production baseline; run CPU Oracle tests separately")
    # The caller's working directory is expected to be the NAS-backed repository.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with single_rank_group(args.output.parent / ".scratch"):
        with torch.inference_mode():
            report = run_suite(args.backend, args.suite, args.dtype, args.seed)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['summary']} -> {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
