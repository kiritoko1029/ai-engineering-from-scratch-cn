# OCR与文档理解

> OCR 是一个包含三个阶段的处理流程——首先检测文本框，接着识别字符，最后进行排版。所有现代 OCR 系统都会对这些阶段进行重新排序或合并。

**类型：** 学习 + 实践
**语言：** Python
**先修知识：** 第 4 阶段第 06 课（检测），第 7 阶段第 02 课（自注意力机制）
**时长：** 约 45 分钟

## 学习目标

- 梳理传统的 OCR 处理流程（检测 -> 识别 -> 排版）以及现代的端到端替代方案（Donut、Qwen-VL-OCR）
- 实现用于序列到序列 OCR 训练的 CTC（连接主义时间分类）损失函数
- 使用 PaddleOCR 或 EasyOCR 在无需训练的情况下进行生产环境文档解析
- 区分 OCR、排版解析与文档理解——并根据具体任务选择合适的工具

## 问题所在

到处都是包含大量文字的图像：收据、发票、身份证件、扫描后的书籍、表格、白板、标识牌以及屏幕截图。从这些图像中提取结构化数据——不仅仅是字符，还包括“这是总金额”这类信息——是应用视觉技术领域中价值最高的任务之一。

该领域可划分为三个技能层级：

1. **传统 OCR 技术**：将像素转换为文本。
2. **布局解析**：将 OCR 输出的内容划分成不同区域（标题、正文、表格、表头等）。
3. **文档理解**：从已解析的布局中提取结构化字段（如“invoice_total = $42.50”）。

每个层级都有传统方法与现代方法之分，而且从“我需要从图像中获取文本”到“我需要从这张收据中得到总金额”的需求差距，往往比大多数团队意识到的要大。

## 概念概述

### 经典的流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**会为每一行或每一个单词生成四边形区域。  
- **识别阶段**会将每个检测到的区域裁剪至固定高度，随后通过 CNN + BiLSTM + CTC 模型生成字符序列。  
- **排版阶段**负责重建阅读顺序（拉丁文为从上到下、从左到右；阿拉伯文和日文则有所不同）。

### CTC，即条件随机场（Conditional Random Field），是一种用于序列建模的统计模型，它通过定义变量之间的条件概率分布来捕捉数据中的依赖结构，广泛应用于自然语言处理、语音识别及生物信息学等领域。

OCR识别技术能够从固定长度的特征图中生成长度可变的序列。CTC算法（Graves等人，2006年）允许在无需进行字符级对齐的情况下训练此类模型。该模型会在每个时间步输出属于（词汇表 + 空白符）的分布；CTC损失函数则会通过对所有在对重复字符进行合并并删除空白符后能够得到目标文本的对齐方式求边际值，从而确定最优损失。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC正是让CRNN在2015年得以成功应用，并且在2026年依然被用于训练大多数生产环境中的OCR模型的关键原因。

### 现代端到端模型

- **Donut**（Kim 等人，2022）——由 ViT 编码器与文本解码器组成；可直接读取图像并输出 JSON。该模型不包含文本检测器，也没有布局分析模块。
- **TrOCR**——用于行级 OCR 的 ViT 加上变换器解码器组合。
- **Qwen-VL-OCR / InternVL**——专为 OCR 任务微调的完整视觉语言模型；在 2026 年针对复杂文档的处理中取得了最佳准确率。
- **PaddleOCR**——基于经典的数据库与 CRNN 流水线的成熟生产级软件包；依然是开源领域的主流选择。

端到端模型虽然需要更多的数据与计算资源，但能够避免多阶段处理流程中出现的误差累积问题。

### 布局解析

对于结构化文档，需运行布局检测器（LayoutLMv3、DocLayNet），为每个区域标注标签：标题、段落、图表、表格、脚注。此时的读取顺序即为“按布局顺序遍历各区域，并将内容拼接起来”。

对于表单类文档，则应使用**键值对提取**模型（视觉信息丰富的文档可使用Donut，普通扫描件可使用LayoutLMv3）。这些模型会结合图像、检测到的文本以及位置信息，预测出结构化的键值对。

### 评估指标

- **字符错误率（CER）** —— 莱文斯坦距离除以参考文本长度。数值越低越好。生产环境的目标值：在干净数据扫描下的误差率需低于 2%。
- **词错误率（WER）** —— 在词级别上采用相同的计算方式。
- **结构化字段的 F1 分数** —— 用于键值对处理任务；用于衡量 `{invoice_total: 42.50}` 等格式是否正确呈现。
- **JSON 的编辑距离** —— 用于端到端的文档解析；Donut 论文提出了标准化的树形编辑距离算法。

## 构建它

### 步骤 1：CTC 损失函数 + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时会使用高效的 CuDNN 实现。贪心解码器比束搜索更简单，其平均字符错误率（CER）通常仅比束搜索高 1% 左右。

### 步骤 2：Tiny CRNN 识别器

用于行级OCR的最小化CNN + BiLSTM模型。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（将CNN最大池化层的高度设为1）。宽度则对应CTC算法中的时间维度。

### 步骤 3：合成 OCR

生成黑白对比的数字字符串，用于端到端的冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实的OCR数据集会包含字体、噪声、旋转、模糊以及颜色元素。上述处理流程完全相同。

### 第 4 步：训练概要

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这段简单的合成数据上，经过200步训练后，损失值应从约3降至约0.2。

## 使用它

三种生产环境可用方案：

- **PaddleOCR** — 成熟稳定、处理速度快，支持多语言。单行使用方式：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** — 原生 Python 开发，支持多语言，基于 PyTorch 框架。
- **Tesseract** — 传统 OCR 工具；在模型难以处理的旧扫描文档处理中仍十分实用。

如需实现端到端的文档解析，可使用 Donut 或大型语言模型（VLM）。

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于具有重复结构的收据、发票和表格，可对 Donut 进行微调。而对于结构不固定的文档或需要推理能力的 OCR 任务，目前默认选择的是类似 Qwen-VL-OCR 的 VLM 模型。

## 发布它

本课程将生成以下内容：

- `outputs/prompt-ocr-stack-picker.md` — 一个根据文档类型、语言及结构来选择 Tesseract / PaddleOCR / Donut / VLM-OCR 的提示模板。
- `outputs/skill-ctc-decoder.md` — 一种从零开始实现贪婪算法与束搜索 CTC 解码器的技能，包含长度归一化处理。

## 练习题

1. **（简单）** 使用5位随机数字字符串训练TinyCRNN，迭代500步。在保留的测试集上报告CER值。
2. **（中等）** 用束搜索（beam_width=5）替代贪婪解码法。报告CER的变化量。在哪些输入数据下束搜索的表现更优？
3. **（困难）** 对20张收据应用PaddleOCR进行文本提取，提取出每行的商品名称和价格，并针对{item_name, price}对计算与手工标注的真实值的F1分数。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OCR | “从像素中提取文本” | 将图像区域转换为字符序列 |
| CTC | “无需对齐的损失函数” | 一种在无需逐时间步标签的情况下训练序列模型的损失函数；通过对齐进行边际化处理 |
| CRNN | “传统OCR模型” | 卷积特征提取器 + 双向LSTM + CTC；2015年的基准模型，至今仍在生产环境中使用 |
| Donut | “端到端OCR” | ViT编码器 + 文本解码器；可直接从图像生成JSON |
| 布局解析 | “识别区域” | 检测并标记文档中的标题/表格/图表/段落等区域 |
| 阅读顺序 | “文本序列” | 将已识别的各区域按顺序组合成句子；对于拉丁文而言较为简单，而对于混合布局则较为复杂 |
| CER / WER | “错误率” | 以字符或单词为单位计算的莱文斯坦距离/参考长度 |
| VLM-OCR | “能够进行阅读的LLM” | 专为OCR任务训练或提示的视觉语言模型；在处理复杂文档时为当前的最先进技术 |

## 延伸阅读

- [CRNN（Shi 等人，2015）](https://arxiv.org/abs/1507.05717) —— 最初的 CNN+RNN+CTC 架构  
- [CTC（Graves 等人，2006）](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 算法的原始论文；蕴含了大量算法设计思路  
- [Donut（Kim 等人，2022）](https://arxiv.org/abs/2111.15664) —— 无需 OCR 的文档理解 Transformer 模型  
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) —— 开源的商用级 OCR 技术栈
