import argparse
import csv
import os
import subprocess
import sys
import time
from itertools import product

from omegaconf import OmegaConf


def parse_list(raw: str, cast_type):
    return [cast_type(x.strip()) for x in raw.split(",") if x.strip()]


def build_config(base_cfg, lr, max_iter):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    cfg.lr = float(lr)
    cfg.max_iter = int(max_iter)

    if "lr_scheduler" in cfg and "kwargs" in cfg.lr_scheduler:
        cfg.lr_scheduler.kwargs.total_iter = int(max_iter)
        cfg.lr_scheduler.kwargs.warmup_steps = max(50, int(max_iter * 0.1))

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run finetune grid search for GSFixer")
    parser.add_argument(
        "--base_config",
        type=str,
        default="config/finetune_gsfixer.yaml",
        help="Base finetune config path",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for grid configs, runs and summary",
    )
    parser.add_argument(
        "--lrs",
        type=str,
        default="5e-5,8e-5",
        help="Comma-separated learning rates",
    )
    parser.add_argument(
        "--max_iters",
        type=str,
        default="700,1000",
        help="Comma-separated max_iter values",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python executable to run train.py",
    )
    parser.add_argument(
        "--extra_train_args",
        type=str,
        default="",
        help="Extra args appended to train.py command, e.g. '--no_wandb'",
    )

    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.base_config)
    lrs = parse_list(args.lrs, float)
    max_iters = parse_list(args.max_iters, int)

    configs_dir = os.path.join(args.output_root, "grid_configs")
    runs_dir = os.path.join(args.output_root, "grid_runs")
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    summary_csv = os.path.join(args.output_root, "grid_summary.csv")
    summary_rows = []

    for lr, max_iter in product(lrs, max_iters):
        tag = f"lr{lr:.0e}_iter{max_iter}".replace("+", "")
        cfg_path = os.path.join(configs_dir, f"finetune_{tag}.yaml")

        cfg = build_config(base_cfg, lr=lr, max_iter=max_iter)
        OmegaConf.save(cfg, cfg_path)

        cmd = [
            args.python_bin,
            "scripts/gsfixer/train.py",
            "--config",
            cfg_path,
            "--output_dir",
            runs_dir,
        ]

        if args.extra_train_args.strip():
            cmd.extend(args.extra_train_args.strip().split())

        print("=" * 80)
        print(f"[GRID] Running: {tag}")
        print("[GRID] Command:", " ".join(cmd))

        start = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - start

        row = {
            "tag": tag,
            "lr": lr,
            "max_iter": max_iter,
            "warmup_steps": int(cfg.lr_scheduler.kwargs.warmup_steps),
            "total_iter": int(cfg.lr_scheduler.kwargs.total_iter),
            "return_code": proc.returncode,
            "elapsed_sec": round(elapsed, 2),
            "elapsed_min": round(elapsed / 60.0, 2),
            "config_path": cfg_path,
        }
        summary_rows.append(row)

        print(f"[GRID] Finished: {tag}, return_code={proc.returncode}, elapsed_min={row['elapsed_min']}")

    fieldnames = [
        "tag",
        "lr",
        "max_iter",
        "warmup_steps",
        "total_iter",
        "return_code",
        "elapsed_sec",
        "elapsed_min",
        "config_path",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 80)
    print(f"[GRID] Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
