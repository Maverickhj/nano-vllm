import torch


# Fixed before the baseline run. BF16 includes projection/norm/RoPE rounding.
# FlashAttention 2 does not accept FP32; FP32/FP64 are checked in Oracle tests.
TOLERANCES = {
    "bfloat16": {"projection": (0.008, 0.02, 0.012), "norm_rope": (0.04, 0.025, 0.02),
                 "output": (0.02, 0.03, 0.03)},
    "float16": {"projection": (0.001, 0.003, 0.002), "norm_rope": (0.005, 0.004, 0.004),
                "output": (0.004, 0.006, 0.006)},
}


def compare(actual, expected, stage, dtype):
    if actual.shape != expected.shape:
        raise AssertionError(f"{stage}: shape {actual.shape} != {expected.shape}")
    actual, expected = actual.detach().double(), expected.detach().double()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{stage}: non-finite values")
    delta = (actual - expected).abs().flatten()
    relative_l2 = (torch.linalg.vector_norm(delta) /
                   torch.linalg.vector_norm(expected).clamp_min(1e-12)).item()
    result = {
        "max_abs_error": delta.max().item(),
        "p99_abs_error": torch.quantile(delta, 0.99).item(),
        "relative_L2": relative_l2,
        "shape": list(actual.shape),
    }
    category = "projection" if stage.endswith("_proj") else (
        "norm_rope" if stage.endswith(("_norm", "_rope")) else "output"
    )
    atol, rtol, l2_limit = TOLERANCES[dtype][category]
    result["tolerance"] = {"atol": atol, "rtol": rtol, "relative_L2": l2_limit}
    close = torch.all((actual - expected).abs() <= atol + rtol * expected.abs()).item()
    result["passed"] = bool(close and relative_l2 <= l2_limit)
    return result
