# is3d-native

Windows 原生环境下的 is3d（FastViT + TriPlane）训练工程起步版本。

## 当前状态
- 阶段 1: CUDA 校验、WebDataset 打包、DataLoader 吞吐测试已完成。
- 阶段 2: FastViT + TriPlane 训练主干、checkpoint、AMP/TF32、CUDA 回退策略已完成。
- 阶段 3: FastViT `switch_to_deploy()`（float64 合并）与 TriPlane token masking 已完成。
- 阶段 4: `state_dict` 导出、跨平台一致性校验脚本、`chunk_size` 推理模式已完成。

## 目录结构
- `src/is3d_native/models/` 模型定义（FastViT/TriPlane）
- `src/is3d_native/training.py` 训练循环
- `src/is3d_native/inference.py` 导出模型加载与 chunk 推理工具
- `scripts/` 训练、导出、验收脚本
- `configs/` 训练配置
- `tests/` 自动化测试

## 快速开始
```powershell
# 1) 创建并激活 conda 环境
conda env create -f .\environment.yml
conda activate is3d-native

# 2) 安装项目
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Dev

# 3) CUDA 环境检查
python .\scripts\check_cuda_env.py --strict --expect-cuda-major 12

# 3.1) 下载 CO3Dv2（默认单序列子集，约 8.9GB）
python .\scripts\download_co3d_dataset.py --single-sequence-subset --install-requirements --download-folder D:\datasets\co3dv2

# 4) 训练（可选训练后切 deploy）
$env:PYTHONPATH = "src"
python .\scripts\train_from_config.py --config .\configs\train_stable_v2_16g.yaml --output-checkpoint .\artifacts\checkpoints\is3d_latest.pt --switch-to-deploy

# 5) 导出（仅 state_dict + 硬编码 Mean/Std）
python .\scripts\export_model.py --config .\configs\train.yaml --output .\artifacts\export\is3d_state_dict.pt --switch-to-deploy

# 6) 推理（chunk 模式）
python .\scripts\infer_triplane.py --export-path .\artifacts\export\is3d_state_dict.pt --chunk-size 1 --synthetic-batch 4 --output .\artifacts\infer\triplane.pt
```

## 跨平台一致性验收
```powershell
# Windows CUDA: 生成对比基线
python .\scripts\check_cross_platform_consistency.py --export-path .\artifacts\export\is3d_state_dict.pt --device cuda --save-input .\artifacts\consistency\shared_input.pt --save-output .\artifacts\consistency\win_cuda_output.pt

# Mac CPU/MPS: 使用同一输入做比对
python .\scripts\check_cross_platform_consistency.py --export-path ./artifacts/export/is3d_state_dict.pt --device cpu --input-tensor ./artifacts/consistency/shared_input.pt --reference-tensor ./artifacts/consistency/win_cuda_output.pt --threshold 1e-5 --save-output ./artifacts/consistency/mac_cpu_output.pt
```

## 测试
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
```


## Resume from checkpoint
```powershell
# Continue training to a larger --steps target
python .\scripts\train_from_config.py --config .\configs\train_stable_v2_16g.yaml --steps 4000 --resume-from-checkpoint .\artifacts\checkpoints\is3d_latest.pt --output-checkpoint .\artifacts\checkpoints\is3d_latest.pt
```
