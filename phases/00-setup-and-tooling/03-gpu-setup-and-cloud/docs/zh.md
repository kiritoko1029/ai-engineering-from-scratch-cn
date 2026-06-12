# GPU 设置与云平台

> 学习阶段使用 CPU 进行训练即可。实际应用则需要 GPU。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 0，课程 01
**耗时：** 约 45 分钟

## 学习目标

- 使用 `nvidia-smi` 及 PyTorch 的 CUDA API 验证本地 GPU 是否可用
- 配置 Google Colab 中的 T4 GPU，以便进行免费的云端实验
- 对比 CPU 与 GPU 下的矩阵乘法性能，并测量加速比
- 根据 fp16 经验法则估算可放入 VRAM 的最大模型规模

## 问题所在

第 1 至 3 阶段的大多数课程在 CPU 上即可正常运行。但一旦开始训练 CNN、Transformer 或 LLM（第 4 阶段及以后），就需要 GPU 加速。原本在 CPU 上需要 8 小时才能完成的训练，在 GPU 上仅需 10 分钟。

你有三种选择：本地 GPU、云端 GPU，或免费的 Google Colab。

## 原理说明

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 开始构建

### 方案 1：本地 NVIDIA GPU

检查您是否拥有该硬件：

```bash
nvidia-smi
```

安装带 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 方案 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 选择“Runtime” > “Change runtime type” > “T4 GPU”
3. 运行 `!nvidia-smi` 命令进行验证

可直接将本课程中的笔记本上传至 Colab。

### 方案 3：云端 GPU

适用于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没问题。

大多数课程都可以在 CPU 上运行。需要 GPU 的课程会明确说明，并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 实践构建：GPU 与 CPU 性能基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习题

1. 运行上述基准测试，并比较 CPU 与 GPU 的运行时间。
2. 如果没有 GPU，可在 Google Colab 上运行并做对比。
3. 查看你的 GPU 内存容量，估算能够加载的最大模型规模（经验法则：fp16 格式下每个参数占用 2 字节内存）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| CUDA | “GPU 编程” | NVIDIA 的并行计算平台，用于在 GPU 上运行代码。 |
| VRAM | “GPU 内存” | GPU 上的显存，与系统 RAM 分开，决定了模型的大小上限。 |
| fp16 | “半精度” | 16 位浮点数格式，其内存占用仅为 fp32 的一半，且精度损失极小。 |
| Tensor Core | “快速矩阵运算硬件” | 专为矩阵乘法设计的 GPU 核心，运算速度比普通核心快 4 到 8 倍。 |
