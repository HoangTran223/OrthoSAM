#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from nlp.glue_metrics import (
    add_mnli_mean_if_complete,
    extract_official_glue_metrics,
    format_official_metrics_line,
)
from nlp.glue_model import BertGlueWrapper
from nlp.glue_tasks import GLUE_TASK_KEYS, glue_num_labels, is_glue_regression, mnli_eval_split_names
from optimizer import getOptimizer
from utility.args import Args
from utility.nlp_glue_args import register_nlp_glue_args
from utility.utils import initialize

register_nlp_glue_args()
import main as _vision_main

try:
    import evaluate
except ImportError as exc:
    raise RuntimeError(
        "Install NLP deps: pip install transformers datasets evaluate scikit-learn"
    ) from exc

EXCLUDED_OPTIMIZERS = frozenset(
    {
        "BayesianSAM",
        "AdamW_MSAM",
        "AdamW_SAM",
        "AdamW_FriendlySAM",
        "AdamW",
        "Adam",
    }
)

MSAM_VARIANT = frozenset({"MSAM"})

SAM_SGD_VARIANTS = frozenset(
    {
        "SAM",
        "OrthoSAM",
        "ASAM",
        "ESAM",
        "FriendlySAM",
        "FisherSAM",
        "MongeSAM",
        "GSAM",
    }
)

SAM_REGRESSION_STEP_VARIANTS = frozenset({"SAM", "OrthoSAM"})


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _tokenize_builder(task: str, tokenizer):
    k1, k2 = GLUE_TASK_KEYS[task.lower()]

    def _tok(examples):
        if k2 is None:
            enc = tokenizer(examples[k1], truncation=True, max_length=128)
        else:
            enc = tokenizer(examples[k1], examples[k2], truncation=True, max_length=128)
        enc["labels"] = examples["label"]
        return enc

    return _tok


def _build_metric(task: str):
    return evaluate.load("glue", task.lower())


def _sanitize_run_segment(s: str) -> str:
    s = str(s).strip().replace(" ", "")
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def _run_directory(task: str) -> Path:
    repo = Path(__file__).resolve().parents[1]
    root = Path(Args.glue_output_root).expanduser()
    if not root.is_absolute():
        root = (repo / root).resolve()
    lr_tag = _sanitize_run_segment(f"{float(Args.learningRate):.6g}")
    if Args.optimizer == "SGD":
        rho_tag = "rho_na"
    else:
        rho_tag = _sanitize_run_segment(f"rho{_vision_main._resolve_optimizer_rho()}")
    name = "__".join(
        [
            _sanitize_run_segment(task),
            _sanitize_run_segment(Args.optimizer),
            f"lr{lr_tag}",
            f"bs{int(Args.batchSize)}",
            f"ep{int(Args.epochs)}",
            rho_tag,
        ]
    )
    return root / name


def _format_seconds(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    hh = int(total_seconds // 3600)
    mm = int((total_seconds % 3600) // 60)
    ss = total_seconds - (hh * 3600 + mm * 60)
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


_GLUE_SAM_RHO_ATTRS = (
    "rho",
    "samRho",
    "asamRho",
    "esamRho",
    "friendlysamRho",
    "fishersamRho",
    "mongesamRho",
    "gsamRho",
    "bayesiansamRho",
    "msamRho",
    "adamwSamRho",
    "adamSamRho",
    "adamwMsamRho",
    "adamwFriendlysamRho",
    "orthosamRho",
    "adamwOrthosamRho",
)


def _glue_sam_rho_for_task(task: str) -> float:
    t = task.lower().replace("-", "")
    if t == "sts-b":
        t = "stsb"
    if t == "cola":
        return 0.002
    if t in ("rte", "stsb", "wnli"):
        return 0.004
    return 0.005


def _apply_glue_sam_rho_overrides(task: str) -> float:
    env = os.environ.get("SAM_GLUE_RHO_OVERRIDE", "").strip()
    if env:
        rho = float(env)
        if rho <= 0:
            raise RuntimeError(f"SAM_GLUE_RHO_OVERRIDE must be > 0, got {rho!r}")
    else:
        rho = float(_glue_sam_rho_for_task(task))
    for name in _GLUE_SAM_RHO_ATTRS:
        setattr(Args.data, name, rho)
    return rho


def main() -> None:
    Args.parse_args()

    task = Args.glue_task.lower().replace("-", "")
    if task == "sts-b":
        task = "stsb"
    if task not in GLUE_TASK_KEYS:
        raise RuntimeError(f"Unknown glue_task={Args.glue_task!r}. Choose from {sorted(GLUE_TASK_KEYS)}")

    sam_rho_applied = _apply_glue_sam_rho_overrides(task)

    if Args.optimizer in EXCLUDED_OPTIMIZERS:
        raise RuntimeError(
            f"Optimizer {Args.optimizer} is excluded on this benchmark. "
            f"Use SGD / MSAM / SAM / ASAM / ESAM / FriendlySAM / FisherSAM / MongeSAM / GSAM / OrthoSAM."
        )

    is_reg = is_glue_regression(task)
    if is_reg and Args.optimizer in SAM_SGD_VARIANTS and Args.optimizer not in SAM_REGRESSION_STEP_VARIANTS:
        raise RuntimeError(
            f"STS-B is regression (MSE). Optimizer {Args.optimizer} still uses cross-entropy in optimizer.step. "
            f"Use SGD, MSAM, SAM, or OrthoSAM for glue_task=stsb."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("nlp/run_glue_bert.py expects CUDA + NCCL (same as main.py).")

    rank = int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % torch.cuda.device_count())))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    initialize()

    device = torch.device(f"cuda:{local_rank}")
    num_labels = glue_num_labels(task)

    raw = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_fn = _tokenize_builder(task, tokenizer)

    train_ds = raw["train"].map(tok_fn, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer)

    per_device_bs = max(1, Args.batchSize // world_size)
    if Args.batchSize % world_size != 0 and rank == 0:
        print(f"[WARN] batchSize {Args.batchSize} not divisible by world_size {world_size}; using per_device {per_device_bs}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_bs,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=Args.dataThreads,
        collate_fn=collator,
        pin_memory=True,
    )

    hf = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    wrapper = BertGlueWrapper(hf).to(device)
    ddp = DDP(wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    n_train = len(train_ds)
    if Args.optimizer == "OrthoSAM":
        optimizer = getOptimizer(_unwrap(ddp).named_parameters(), n_data=n_train)
    else:
        optimizer = getOptimizer(_unwrap(ddp).parameters(), n_data=n_train)

    total_steps = max(1, len(train_loader)) * Args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    run_dir = _run_directory(task)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[GLUE] task={task} SAM-family ρ (all rhos)={sam_rho_applied}")
        print(f"[RUN_DIR] {run_dir}")

    history: list[dict] = []
    epoch_train_times_sec: list[float] = []
    total_train_time_sec = 0.0

    for epoch in range(1, Args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        ddp.train()
        running_loss = 0.0
        n_batches = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_train_start = time.perf_counter()

        if Args.optimizer in MSAM_VARIANT:
            optimizer.move_up_to_momentumAscent()

        for batch in train_loader:
            batch = _to_device(batch, device)
            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)

            if Args.optimizer in SAM_SGD_VARIANTS:
                if Args.optimizer in SAM_REGRESSION_STEP_VARIANTS:
                    loss, _ = optimizer.step(ddp, batch, labels, regression=is_reg)
                else:
                    loss, _ = optimizer.step(ddp, batch, labels)
            else:
                logits = ddp(batch)
                if is_reg:
                    loss = F.mse_loss(logits.squeeze(-1), labels.float())
                else:
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                if Args.grad_clip_norm and Args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(ddp.parameters(), Args.grad_clip_norm)
                optimizer.step()

            scheduler.step()
            running_loss += float(loss.detach().mean() if loss.dim() > 0 else loss)
            n_batches += 1

        if Args.optimizer in MSAM_VARIANT:
            optimizer.move_back_from_momentumAscent()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_train_elapsed = time.perf_counter() - epoch_train_start
        total_train_time_sec += epoch_train_elapsed
        epoch_train_times_sec.append(epoch_train_elapsed)

        if rank == 0 and Args.verbose:
            print(
                f"[train] epoch {epoch}/{Args.epochs} loss={running_loss / max(1, n_batches):.4f} "
                f"| train_time={epoch_train_elapsed:.3f}s"
            )

        if rank == 0:
            _unwrap(ddp).eval()
            metrics_out: dict[str, float] = {}
            for split in mnli_eval_split_names(task):
                val_raw = raw[split]
                val_tok = val_raw.map(tok_fn, batched=True, remove_columns=val_raw.column_names)
                val_loader = DataLoader(
                    val_tok,
                    batch_size=per_device_bs,
                    shuffle=False,
                    num_workers=Args.dataThreads,
                    collate_fn=collator,
                    pin_memory=True,
                )
                metric = _build_metric(task)
                preds_all: list = []
                refs_all: list = []

                with torch.no_grad():
                    for vb in val_loader:
                        vb = _to_device(vb, device)
                        lab = vb["labels"]
                        logits = _unwrap(ddp)(vb)
                        if is_reg:
                            preds_all.extend(logits.squeeze(-1).cpu().numpy().tolist())
                            refs_all.extend(lab.float().cpu().numpy().tolist())
                        else:
                            preds_all.extend(torch.argmax(logits, -1).cpu().numpy().tolist())
                            refs_all.extend(lab.cpu().numpy().tolist())

                m = metric.compute(predictions=preds_all, references=refs_all)
                sub = extract_official_glue_metrics(task, split, m)
                metrics_out.update(sub)

            if task == "mnli":
                add_mnli_mean_if_complete(metrics_out)

            history.append({"epoch": epoch, **metrics_out})
            print(
                f"[val] epoch {epoch} official: {format_official_metrics_line(task, metrics_out)} "
                f"| json_keys={list(metrics_out.keys())}"
            )
            with open(run_dir / "metrics_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            with open(run_dir / "params.json", "w", encoding="utf-8") as f:
                json.dump(vars(Args.data), f, indent=2)

            ckpt = {
                "epoch": epoch,
                "model_state_dict": _unwrap(ddp).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "glue_task": task,
                "optimizer_name": Args.optimizer,
            }
            torch.save(ckpt, run_dir / f"checkpoint_epoch{epoch}.pt")

        dist.barrier()

    if rank == 0:
        avg_train_time_sec = total_train_time_sec / max(1, len(epoch_train_times_sec))
        with open(run_dir / "training_time.txt", "w", encoding="utf-8") as f:
            f.write(f"optimizer: {Args.optimizer}\n")
            f.write(f"glue_task: {task}\n")
            f.write(f"epochs: {Args.epochs}\n")
            f.write(f"total_train_time_seconds: {total_train_time_sec:.6f}\n")
            f.write(f"total_train_time_hhmmss: {_format_seconds(total_train_time_sec)}\n")
            f.write(f"avg_train_time_per_epoch_seconds: {avg_train_time_sec:.6f}\n")
            f.write(f"avg_train_time_per_epoch_hhmmss: {_format_seconds(avg_train_time_sec)}\n")
            f.write("epoch_train_times_seconds: " + ", ".join(f"{x:.6f}" for x in epoch_train_times_sec) + "\n")

        torch.save(
            {
                "model_state_dict": _unwrap(ddp).state_dict(),
                "glue_task": task,
                "optimizer_name": Args.optimizer,
            },
            run_dir / "checkpoint_final.pt",
        )
        print(f"[done] task={task} optimizer={Args.optimizer} artifacts -> {run_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
