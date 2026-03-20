# 2026-03-20 进度与下一步

## 今日完成

### 1) 训练主流程已跑通到 4000 step
- 最新训练日志：
  - `[step 3950] loss=0.186682 | lr=6.90e-05 | token_mask_ratio=0.120 | max_mem_mb=379.72 | tf32=True`
  - `[step 4000] loss=0.168535 | lr=7.00e-05 | token_mask_ratio=0.120 | max_mem_mb=379.72 | tf32=True`
- 训练完成并保存：`artifacts/checkpoints/is3d_latest.pt`
- 当前 checkpoint 训练状态：`step=4000, updates=500`

### 2) v3 断点续训（resume-from-checkpoint）已完成
- 已支持命令行参数：`--resume-from-checkpoint`
- 训练保存格式升级为“可续训”checkpoint，包含：
  - `state_dict`
  - `optimizer`
  - `scaler`
  - `train_state`（step / update_step / current_lr）
  - `rng_state`
  - `train_config`
- 兼容旧版仅 `state_dict` 的 checkpoint（按 warm-start 继续训练）

### 3) 验证已完成
- 断点续训测试：`tests/test_training_resume.py`（2 passed）
- 导出加载回归测试：`tests/test_export_loading.py`（2 passed）
- 静态检查：`ruff check` 通过

## 当前可交付物
- 最新可续训 checkpoint：`artifacts/checkpoints/is3d_latest.pt`
- 续训入口脚本：`scripts/train_from_config.py`
- 核心训练与状态恢复：`src/is3d_native/training.py`
- 断点续训测试：`tests/test_training_resume.py`

## 下一步建议（按优先级）
1. 做一次模型导出（deploy）并固化可推理包。
2. 立刻跑跨平台一致性验收（Windows CUDA vs Mac CPU/MPS，阈值 `< 1e-5`）。
3. 增加周期性 checkpoint（例如每 200/500 step 保存一次），避免只保留 latest。
4. 基于真实验证集补充质量指标（不仅看训练 loss），用于判断 4000 step 后是否继续训练。

## 明天继续时的最短路径
1. 从 `artifacts/checkpoints/is3d_latest.pt` 继续训练或导出。
2. 若继续训练，先把 `--steps` 提升到目标步数，再用 `--resume-from-checkpoint`。
3. 若做交付，先 `--switch-to-deploy` 导出，再跑一致性脚本。
