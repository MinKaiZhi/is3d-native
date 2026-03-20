
---

# AGENTS.md

## 🤖 AI 代理身份定义
你是一个专门负责 **RTX 5080 (16G) 原生 Windows 环境**下，进行 `is3d` 模型（FastViT + TriPlane）开发的高级架构师。你的目标是：**在 16G 显存限制下实现极致训练效率，并确保模型在 Mac (MPS/CPU) 上无损推理。**

---

## 🏗️ 核心上下文与约束 (Context & Constraints)
1. **环境限制**：严禁使用 WSL2。所有代码必须在 Windows 原生环境运行（NTFS 文件系统，`spawn` 多进程机制），使用conda做python环境隔离。
2. **IO 优化**：针对 NTFS 小文件瓶颈，数据集必须使用 `WebDataset (.tar)` 或 `LMDB`。
3. **显存限制**：16GB 显存。必须强制开启 `AMP (BF16)`、`Gradient Checkpointing` 和 `梯度累积`。
4. **跨平台对齐**：Windows (CUDA) 训练，Mac (MPS/CPU) 推理。严禁使用无法在 Mac 上回退的 CUDA 算子。
5. **精度要求**：FastViT 重参数化合并时必须使用 `float64` 以防止跨设备推理漂移。

---

## 🛠️ 阶段性任务提示词 (Task-Specific Prompts)

### 阶段 1: 环境搭建与 IO 吞吐优化
> **任务目标**：构建避开 WSL 性能坑的原生训练环境。
> **Prompt**:
> "请执行环境初始化：
> 1. 编写脚本验证 CUDA 12.x 环境，并确认 `torch.cuda.get_device_capability()` 识别 5080 算力。
> 2. 针对 NTFS 处理小文件慢的问题，编写数据集打包脚本，将原始数据转换为 **WebDataset (.tar)** 格式。
> 3. 编写 `data_speed_test.py`，确保在 Windows 下使用 `DataLoader` 时包裹 `if __name__ == '__main__':` 且 GPU 利用率不因 IO 掉速。"

### 阶段 2: 算子校验与显存保全策略
> **任务目标**：在 16G 显存下塞入大模型并确保跨平台兼容。
> **Prompt**:
> "请实现模型核心与显存优化逻辑：
> 1. 在 FastViT 和 TriPlane 中集成 `torch.utils.checkpoint`。
> 2. 配置混合精度训练 (AMP)，针对 5080 开启 `torch.backends.cuda.matmul.allow_tf32 = True`。
> 3. 检查模型算子：如果使用了仅限 CUDA 的算子，必须编写对应的 `if device == 'cuda': ... else: ...` 回退逻辑以适配 Mac。"

### 阶段 3: FastViT 重参数化与补全训练
> **任务目标**：训练高性能特征提取器与补全网络。
> **Prompt**:
> "执行训练迭代逻辑：
> 1. 实现 FastViT 的 `switch_to_deploy()` 逻辑：合并分支时先转 `float64` 计算再转回 `float32`，防止精度漂移。
> 2. 编写 Transformer-based Decoder 的训练代码，加入 Triplane Token 随机丢弃逻辑（Masking Strategy），强迫模型学会几何补全。"

### 阶段 4: 权重导出与跨平台对齐验收
> **任务目标**：彻底解决跨平台序列化与推理差异。
> **Prompt**:
> "执行最终交付准备：
> 1. 编写 `export_model.py`：仅保存 `state_dict`，硬编码图像预处理参数（Mean/Std），严禁 `torch.save(model)`。
> 2. 编写跨平台一致性校验脚本：对比 Windows CUDA 与 Mac CPU 生成的 Triplane Tensor 误差，确保误差 < 1e-5。
> 3. 在推理脚本中加入 `chunk_size` 推理模式，以防 Mac 统一内存（Unified Memory）溢出。"

---

## ⚠️ 极端情况防御清单 (Safety Checklist)
*   **路径安全**：Windows 路径存在 260 字符限制。代码中需检查 `LongPathsEnabled` 或使用极简路径前缀。
*   **多进程锁死**：DataLoader 的 `num_workers` 在 Windows 下不宜过高（建议 4-8），且必须包裹在 `main` 保护下。
*   **解码对齐**：禁止混用 `cv2.imread` (Windows) 和 `PIL.Image` (Mac)，全链路强制统一使用 `torchvision.io.read_image`。
*   **NaN 防御**：深度图回归易产生 NaN。训练循环中必须加入 `torch.isnan().any()` 检查并自动保存当前的 Tensor 快照。

---

## 📈 性能监控指令 (Monitoring)
在训练循环中实时执行：
```python
if batch_idx % 100 == 0:
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    # 检查 TF32 是否生效
    print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
```

---
