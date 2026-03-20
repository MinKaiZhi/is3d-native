# Roadmap

## Phase 1 (done)
- Windows native bootstrap.
- CUDA check script.
- WebDataset conversion script.
- DataLoader throughput benchmark.

## Phase 2 (done)
- Integrated `FastViTBackbone + TriPlaneDecoder` training path.
- Added gradient checkpointing in both backbone and decoder.
- Enabled CUDA runtime flags (TF32) and BF16 AMP in training.
- Added CUDA-optimized attention path with non-CUDA fallback.

## Phase 3 (done)
- Implemented FastViT `switch_to_deploy()` with float64 branch merge.
- Added TriPlane token masking strategy in training.

## Phase 4 (done)
- Added `export_model.py` with state_dict-only export and fixed Mean/Std metadata.
- Added `check_cross_platform_consistency.py` for Windows CUDA vs Mac CPU/MPS error checks.
- Added `infer_triplane.py` with `chunk_size` inference mode for unified-memory safety.
