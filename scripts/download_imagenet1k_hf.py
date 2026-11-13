#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

SPLITS = {
    "train": ("train", 1281167),
    "validation": ("val", 50000),
}

REPO_ID = "ILSVRC/imagenet-1k"


def _fail(message: str) -> "None":
    raise SystemExit(f"[ERROR] {message}")


def _ensure_deps() -> None:
    try:
        import datasets
    except Exception as exc:
        _fail(
            "Missing dependency 'datasets'. Install the repo requirements first:\n"
            "    pip install -r requirements.txt"
        )
        raise exc


def _check_auth() -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        HfApi().dataset_info(REPO_ID, token=token)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            _fail(
                f"No access to the gated dataset '{REPO_ID}' (HTTP {status}).\n"
                f"  1. Accept the terms at https://huggingface.co/datasets/{REPO_ID}\n"
                "  2. Authenticate with `huggingface-cli login`, or export HF_TOKEN=hf_...\n"
                "Then re-run this script."
            )
        raise


def _count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.JPEG"))


def _export_shard(job: tuple) -> tuple:
    split, out_dir, cache_dir, shard_index, num_shards = job

    from datasets import Image, load_dataset

    dataset = load_dataset(REPO_ID, split=split, cache_dir=cache_dir)
    dataset = dataset.cast_column("image", Image(decode=False))
    shard = dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)

    out_path = Path(out_dir)
    written = 0
    for row_index, example in enumerate(shard):
        label = int(example["label"])
        class_dir = out_path / f"class_{label:04d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        target = class_dir / f"{shard_index:05d}_{row_index:07d}.JPEG"
        if target.exists():
            written += 1
            continue

        payload = example["image"]["bytes"]
        if payload is None:
            source = example["image"].get("path")
            if not source:
                _fail(f"Row {row_index} of shard {shard_index} has neither bytes nor a path.")
            payload = Path(source).read_bytes()

        staging = target.with_suffix(".partial")
        staging.write_bytes(payload)
        staging.rename(target)
        written += 1

    return shard_index, written


def _export_split(split: str, out_dir: Path, cache_dir: str, workers: int, num_shards: int) -> None:
    dir_name, expected = SPLITS[split]
    target_dir = out_dir / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    progress_dir = out_dir / ".progress" / dir_name
    progress_dir.mkdir(parents=True, exist_ok=True)

    pending = [i for i in range(num_shards) if not (progress_dir / f"{i:05d}.done").exists()]
    if not pending:
        print(f"[{dir_name}] all {num_shards} shards already exported, skipping")
        return

    print(f"[{dir_name}] exporting {len(pending)}/{num_shards} shards with {workers} workers")
    print(f"[{dir_name}] target: {target_dir}")

    jobs = [(split, str(target_dir), cache_dir, i, num_shards) for i in pending]
    done = 0
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for shard_index, written in pool.imap_unordered(_export_shard, jobs):
            (progress_dir / f"{shard_index:05d}.done").touch()
            done += 1
            print(f"[{dir_name}] shard {shard_index:5d} done ({written} images) "
                  f"- {done}/{len(pending)} shards", flush=True)

    total = _count_images(target_dir)
    if total != expected:
        _fail(
            f"[{dir_name}] exported {total} images but expected {expected}.\n"
            "Re-run the script; completed shards are skipped and missing ones are retried."
        )
    print(f"[{dir_name}] complete: {total} images")


def _check_only(out_dir: Path) -> int:
    ok = True
    for _, (dir_name, expected) in SPLITS.items():
        found = _count_images(out_dir / dir_name)
        classes = len(list((out_dir / dir_name).glob("class_*"))) if (out_dir / dir_name).is_dir() else 0
        status = "OK" if found == expected and classes == 1000 else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"[{dir_name}] images={found} (expected {expected}), classes={classes} (expected 1000) -> {status}")
    if ok:
        print(f"\nDataset looks complete. Point the config at it with:\n"
              f"  export SAM_DATASET_ROOT={out_dir.parent}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path.home() / ".datasets" / "ImageNet"),
                        help="output directory that will contain train/ and val/")
    parser.add_argument("--cache-dir", default=None,
                        help="Hugging Face parquet cache directory (defaults to the HF default)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8),
                        help="parallel export processes")
    parser.add_argument("--train-shards", type=int, default=256,
                        help="number of shards for the train split; more shards means finer resume granularity")
    parser.add_argument("--val-shards", type=int, default=16, help="number of shards for the val split")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS.keys()), choices=list(SPLITS.keys()),
                        help="which splits to export")
    parser.add_argument("--check-only", action="store_true",
                        help="only verify an existing export, download nothing")
    args = parser.parse_args()

    out_dir = Path(os.path.expanduser(args.out)).resolve()

    if args.check_only:
        sys.exit(_check_only(out_dir))

    _ensure_deps()
    _check_auth()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {REPO_ID} to {out_dir}")
    print("The first run downloads roughly 150 GB of parquet files; this takes a while.\n")

    for split in args.splits:
        shards = args.train_shards if split == "train" else args.val_shards
        _export_split(split, out_dir, args.cache_dir, args.workers, shards)

    print("\nDone.")
    print(f"Dataset root: {out_dir}")
    print(f"Point the training config at it with:\n  export SAM_DATASET_ROOT={out_dir.parent}")
    print("The Hugging Face parquet cache is no longer needed and can be removed to reclaim ~150 GB "
          f"({args.cache_dir or '~/.cache/huggingface/datasets'}).")


if __name__ == "__main__":
    main()
