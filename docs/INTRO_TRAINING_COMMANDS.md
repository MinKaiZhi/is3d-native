# 项目介绍文档：训练命令步骤（Windows 原生）

> 目标：在 Windows 原生环境（非 WSL）下，使用 RTX 5080 16G 完成 is3d（FastViT + TriPlane）训练，并支持断点续训。

## 0. 前提要求
- 操作系统：Windows 原生
- 环境隔离：conda
- Python 环境名：`is3d-native`
- 数据格式建议：WebDataset `.tar`（避免 NTFS 小文件瓶颈）

## 1. 初始化环境
```powershell
conda env create -f .\environment.yml
conda activate is3d-native
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Dev
```

## 2. CUDA 环境检查（建议先做）
```powershell
python .\scripts\check_cuda_env.py --strict --expect-cuda-major 12
```

## 3. 数据准备（可选）

### 3.1 下载 CO3Dv2（单序列子集）
```powershell
python .\scripts\download_co3d_dataset.py --single-sequence-subset --install-requirements --download-folder D:\datasets\co3dv2
```

### 3.2 打包为 WebDataset
```powershell
python .\scripts\pack_webdataset.py --input-dir D:\datasets\co3dv2 --output-dir D:\datasets\co3dv2_wds --maxcount 5000
```

## 4. 数据吞吐测试（可选但推荐）
```powershell
python .\scripts\data_speed_test.py --source D:\datasets\co3dv2_wds\shard-*.tar --batch-size 8 --num-workers 4 --max-batches 200 --log-every 20
```

## 5. 启动训练（从 0 开始）
```powershell
$env:PYTHONPATH = "src"
python .\scripts\train_from_config.py --config .\configs\train_stable_v2_16g.yaml --output-checkpoint .\artifacts\checkpoints\is3d_latest.pt
```

## 6. 断点续训（resume-from-checkpoint）
> 注意：`--steps` 是“总步数目标”，不是“追加步数”。

示例：把训练从现有 checkpoint 继续到 4000 step
```powershell
$env:PYTHONPATH = "src"
python .\scripts\train_from_config.py --config .\configs\train_stable_v2_16g.yaml --steps 4000 --resume-from-checkpoint .\artifacts\checkpoints\is3d_latest.pt --output-checkpoint .\artifacts\checkpoints\is3d_latest.pt
```

## 7. 训练后切换 Deploy 权重（可选）
```powershell
$env:PYTHONPATH = "src"
python .\scripts\train_from_config.py --config .\configs\train_stable_v2_16g.yaml --output-checkpoint .\artifacts\checkpoints\is3d_latest.pt --switch-to-deploy
```

## 8. 常用覆盖参数
- 覆盖训练步数：`--steps 8000`
- 覆盖 dataloader worker：`--num-workers 4`
- 覆盖数据 shard：`--train-shards "D:\datasets\co3dv2_wds\shard-*.tar"`

示例：
```powershell
$env:PYTHONPATH = "src"
python .\scripts\train_from_config.py --config .\configs\train_stable_v2_16g.yaml --train-shards "D:\datasets\co3dv2_wds\shard-*.tar" --num-workers 4 --steps 2000 --output-checkpoint .\artifacts\checkpoints\is3d_latest.pt
```

## 9. 训练日志判读（当前项目）
- 关注 `loss` 是否稳定下降。
- 关注 `max_mem_mb` 是否长期低于 16G 显存上限。
- 确认 `tf32=True`（CUDA 路径）。
- 若中断，优先使用第 6 节命令继续训练。
