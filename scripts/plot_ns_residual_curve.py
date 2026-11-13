#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _bootstrap_args(ifile: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.chdir(repo_root)
    sys.argv = ["plot_ns_residual_curve.py", "--ifile", ifile]

    import models
    import optimizer
    import utility.LRScheduler
    import utility.rhoScheduler
    import utility.data
    import utility.dataLogger
    import utility.modelSaver
    import utility.utils
    from utility.args import Args

    Args.add_argument("--logDir", type=str, help="main directory to store logs")
    Args.add_argument("--logSubDir", type=str, help="subdir in logDir")
    Args.add_argument("--epochs", type=int, help="Total number of epochs")
    Args.add_argument("--contin", type=bool, help="continue from checkpoint")
    Args.parse_args()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ifile", default="configs/CIFAR100_WRN16_4.ini")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="logs_k/ns_residual_curve")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli, _ = parser.parse_known_args()

    _bootstrap_args(cli.ifile)

    from models import getModel
    from optimizer.orthosam import (
        _ns_project,
        _orthogonality_residual,
        _should_use_orthosam,
    )
    from utility.args import Args

    torch.manual_seed(cli.seed)
    device = torch.device(cli.device)

    model = getModel()(num_classes=100).to(device)
    model.train()

    inputs = torch.randn(cli.batch_size, 3, 32, 32, device=device)
    targets = torch.randint(0, 100, (cli.batch_size,), device=device)
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()

    matrices = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not _should_use_orthosam(param, name):
            continue
        matrices.append((name, param.grad.detach().clone()))

    if not matrices:
        raise RuntimeError("No OrthoSAM-eligible gradients found.")

    out_dir = Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ns_residual_vs_k.csv"

    rows = []
    print(f"Measuring residual on {len(matrices)} matrices, K=1..{cli.max_k}")
    for k in range(1, cli.max_k + 1):
        residuals = []
        for _, grad in matrices:
            projected = _ns_project(grad, steps=k, apply_matrix_scale=False)
            residuals.append(_orthogonality_residual(projected))
        mean_r = sum(residuals) / len(residuals)
        max_r = max(residuals)
        min_r = min(residuals)
        rows.append({"K": k, "residual_mean": mean_r, "residual_min": min_r, "residual_max": max_r})
        print(f"K={k:2d}  mean={mean_r:.6e}  min={min_r:.6e}  max={max_r:.6e}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "residual_mean", "residual_min", "residual_max"])
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Wrote {csv_path} (matplotlib not available, skip PNG)")
        return

    ks = [r["K"] for r in rows]
    means = [r["residual_mean"] for r in rows]
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(ks, means, marker="o")
    plt.xlabel("Newton-Schulz iterations K")
    plt.ylabel(r"mean residual $\|O^\top O - I\|_F / \sqrt{n}$")
    plt.title("OrthoSAM NS residual vs K (one minibatch)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = out_dir / "ns_residual_vs_k.png"
    plt.savefig(png_path, dpi=160)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
