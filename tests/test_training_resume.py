from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import torch

import is3d_native.training as training
from is3d_native.inference import safe_torch_load
from is3d_native.training import TrainConfig


def _tiny_train_cfg(snapshot_dir: Path, steps: int) -> TrainConfig:
    return TrainConfig(
        steps=steps,
        batch_size=1,
        grad_accum_steps=2,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip_norm=0.0,
        log_every=1000,
        use_bf16_amp=False,
        use_cosine_lr=False,
        checkpoint_segments=False,
        image_size=32,
        fastvit_embed_dim=32,
        fastvit_depth=1,
        fastvit_num_heads=4,
        fastvit_patch_size=8,
        triplane_depth=1,
        triplane_num_heads=4,
        triplane_channels=4,
        triplane_resolution=8,
        triplane_token_mask_ratio=0.1,
        snapshot_dir=snapshot_dir,
    )


def _new_test_dir() -> Path:
    base = Path("artifacts/test_tmp")
    base.mkdir(parents=True, exist_ok=True)
    work_dir = base / f"resume_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    return work_dir


def test_resume_checkpoint_matches_full_run(monkeypatch) -> None:
    monkeypatch.setattr(training, "_select_device", lambda: torch.device("cpu"))

    work_dir = _new_test_dir()
    try:
        full_ckpt = work_dir / "full.pt"
        phase1_ckpt = work_dir / "phase1.pt"
        resumed_ckpt = work_dir / "resumed.pt"

        torch.manual_seed(2026)
        cfg_full = _tiny_train_cfg(snapshot_dir=work_dir / "nan_full", steps=4)
        training.run_training_loop(cfg_full, output_checkpoint=full_ckpt)

        torch.manual_seed(2026)
        cfg_phase1 = _tiny_train_cfg(snapshot_dir=work_dir / "nan_phase1", steps=2)
        training.run_training_loop(cfg_phase1, output_checkpoint=phase1_ckpt)

        cfg_phase2 = _tiny_train_cfg(snapshot_dir=work_dir / "nan_phase2", steps=4)
        training.run_training_loop(
            cfg_phase2,
            resume_from_checkpoint=phase1_ckpt,
            output_checkpoint=resumed_ckpt,
        )

        full_payload = safe_torch_load(full_ckpt)
        resumed_payload = safe_torch_load(resumed_ckpt)

        assert "optimizer" in resumed_payload
        assert "scaler" in resumed_payload
        assert resumed_payload["train_state"]["step"] == 4
        assert resumed_payload["train_state"]["update_step"] == 2

        full_state = full_payload["state_dict"]
        resumed_state = resumed_payload["state_dict"]

        assert full_state.keys() == resumed_state.keys()
        for key in full_state:
            torch.testing.assert_close(full_state[key], resumed_state[key], rtol=1e-6, atol=1e-6)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_resume_from_legacy_state_dict_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(training, "_select_device", lambda: torch.device("cpu"))

    work_dir = _new_test_dir()
    try:
        cfg = _tiny_train_cfg(snapshot_dir=work_dir / "nan_legacy", steps=1)
        model = training.build_model(cfg)
        legacy_ckpt = work_dir / "legacy.pt"
        torch.save({"state_dict": model.state_dict()}, legacy_ckpt)

        output_ckpt = work_dir / "legacy_resumed.pt"
        training.run_training_loop(
            cfg,
            resume_from_checkpoint=legacy_ckpt,
            output_checkpoint=output_ckpt,
        )

        payload = safe_torch_load(output_ckpt)
        assert payload["train_state"]["step"] == 1
        assert "state_dict" in payload
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

