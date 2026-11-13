from __future__ import annotations

from typing import Any

import numpy as np

GLUE_TASK_NAMES = frozenset(
    {"cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"}
)

METRIC_DECIMALS = 3


def _round3(x: float) -> float:
    return round(float(x), METRIC_DECIMALS)


def _to_leaderboard_scale(x: float) -> float:
    xf = float(x)
    if abs(xf) <= 1.000001:
        return xf * 100.0
    return xf


def _is_scalar_number(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (float, np.floating)):
        return True
    if isinstance(v, int):
        return True
    return False


def extract_official_glue_metrics(task: str, split: str, computed: dict[str, Any]) -> dict[str, float]:
    t = task.lower().replace("-", "")
    if t == "sts-b":
        t = "stsb"
    if t not in GLUE_TASK_NAMES:
        raise RuntimeError(f"Unknown GLUE task for metrics: {task!r}")

    lower_map = {str(k).lower(): v for k, v in computed.items()}

    def getf(key: str) -> float | None:
        if key not in lower_map:
            return None
        v = lower_map[key]
        if not _is_scalar_number(v):
            return None
        return float(v)

    def scaled(v: float | None) -> float | None:
        if v is None:
            return None
        return _round3(_to_leaderboard_scale(v))

    if t == "cola":
        v = getf("matthews_correlation")
        s = scaled(v)
        return {"mcc": s} if s is not None else {}

    if t == "stsb":
        out: dict[str, float] = {}
        p = scaled(getf("pearson"))
        if p is not None:
            out["pearson"] = p
        s_raw = getf("spearmanr")
        if s_raw is None:
            s_raw = getf("spearman_correlation")
        s = scaled(s_raw)
        if s is not None:
            out["spearman"] = s
        return out

    if t == "mnli":
        v = getf("accuracy")
        if v is None:
            return {}
        s = scaled(v)
        if s is None:
            return {}
        if split == "validation_matched":
            return {"matched_accuracy": s}
        if split == "validation_mismatched":
            return {"mismatched_accuracy": s}
        return {"accuracy": s}

    if t in ("sst2", "qnli", "rte", "wnli"):
        v = getf("accuracy")
        s = scaled(v)
        return {"accuracy": s} if s is not None else {}

    if t in ("mrpc", "qqp"):
        out = {}
        a = scaled(getf("accuracy"))
        f = scaled(getf("f1"))
        if a is not None:
            out["accuracy"] = a
        if f is not None:
            out["f1"] = f
        return out

    return {}


def add_mnli_mean_if_complete(metrics: dict[str, float]) -> None:
    if "matched_accuracy" in metrics and "mismatched_accuracy" in metrics:
        metrics["mnli_accuracy_mean"] = _round3(
            0.5 * (metrics["matched_accuracy"] + metrics["mismatched_accuracy"])
        )


def format_official_metrics_line(task: str, m: dict[str, float]) -> str:
    t = task.lower().replace("-", "")
    if t == "sts-b":
        t = "stsb"
    if t == "cola":
        return f"Mcc={m.get('mcc', float('nan')):.{METRIC_DECIMALS}f}"
    if t == "sst2" or t in ("qnli", "rte", "wnli"):
        return f"Acc={m.get('accuracy', float('nan')):.{METRIC_DECIMALS}f}"
    if t in ("mrpc", "qqp"):
        return (
            f"Acc={m.get('accuracy', float('nan')):.{METRIC_DECIMALS}f} "
            f"F1={m.get('f1', float('nan')):.{METRIC_DECIMALS}f}"
        )
    if t == "stsb":
        return (
            f"Pearson={m.get('pearson', float('nan')):.{METRIC_DECIMALS}f} "
            f"Spearman={m.get('spearman', float('nan')):.{METRIC_DECIMALS}f}"
        )
    if t == "mnli":
        a, b = m.get("matched_accuracy"), m.get("mismatched_accuracy")
        mean = m.get("mnli_accuracy_mean")
        d = METRIC_DECIMALS
        if a is not None and b is not None:
            if mean is not None:
                return f"Acc matched/mismatched={a:.{d}f}/{b:.{d}f} mean={mean:.{d}f}"
            return f"Acc matched/mismatched={a:.{d}f}/{b:.{d}f}"
        return str(m)
    return str(m)
