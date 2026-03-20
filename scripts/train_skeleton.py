from __future__ import annotations

import argparse

from is3d_native.training import TrainConfig, run_training_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal training skeleton.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--token-mask-ratio", type=float, default=0.15)
    parser.add_argument("--switch-to-deploy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        log_every=args.log_every,
        image_size=args.image_size,
        triplane_token_mask_ratio=args.token_mask_ratio,
    )
    model = run_training_loop(cfg)
    if args.switch_to_deploy:
        model.switch_to_deploy()
        print("FastViT switched to deploy mode (float64 merge -> float32 weights).")


if __name__ == "__main__":
    main()
