from __future__ import annotations

import argparse
from pathlib import Path

from is3d_native.config import load_train_config
from is3d_native.training import run_training_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training from YAML config.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--train-shards", type=str, default=None, help="Override train shard glob path.")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/is3d_latest.pt"),
        help="Checkpoint path for trained weights and optimizer/scaler state.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume training from a previous checkpoint.",
    )
    parser.add_argument(
        "--switch-to-deploy",
        action="store_true",
        help="Run FastViT re-parameterization after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)

    if args.train_shards is not None:
        cfg.train_shards = args.train_shards
    if args.steps is not None:
        cfg.steps = args.steps
    if args.num_workers is not None:
        cfg.dataloader_workers = args.num_workers

    model = run_training_loop(
        cfg,
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_checkpoint=args.output_checkpoint,
    )

    if args.switch_to_deploy:
        model.switch_to_deploy()
        print("FastViT switched to deploy mode (float64 merge -> float32 weights).")


if __name__ == "__main__":
    main()

