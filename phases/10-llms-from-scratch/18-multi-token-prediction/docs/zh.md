# 多 token 预测（MTP）

> 从 GPT-2 到 Llama 3 的每个自回归 LLM 都在一个损失上训练：预测下一个 token。DeepSeek-V3 在每个位置添加了第二个损失：预测下下个 token。额外的 14B 参数（在 671B 模型上）通过梯度流蒸馏回主模型，训练好的 MTP 头在推理时被重新用作推测解码草稿器，接受率 80% 以上。1.8 倍生成吞吐免费获得。本课程从 DeepSeek 技术报告构建顺序 MTP 模块，计算损失和共享头参数布局，并解释为什么 MTP 保持因果链而 Gloeckle 等人的原始并行 MTP 打破了它。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 10 阶段 · 04（预训练 mini GPT），第 10 阶段 · 15（推测解码）
**所需时间：** 约60分钟

## 学习目标

- 说出 MTP 训练目标并推导跨预测深度的联合损失。
- 解释 Gloeckle 等人的并行 MTP 头（2024）与 DeepSeek-V3 的顺序 MTP 模块之间的区别，以及为什么顺序设计保持了因果链。
- 计算向预训练运行添加 MTP 模块的参数和内存开销。
- 从零实现一个 MTP 模块：共享 embedding、每深度 Transformer 块、投影和共享输出头。

## 问题所在

下一个 token 预测是标准的 LLM 训练目标。每个隐藏状态被监督来预测恰好一个东西：紧随其后的 token。这是一个令人惊讶的弱信号。序列中的大多数信息都延伸到一个 token 之外——结构、连贯性、事实性、算术流。模型必须通过在数万亿 token 上积累许多单 token 信号来学习这些。

MTP 提出：如果每个隐藏状态同时被监督来预测多个未来 token 会怎样？Gloeckle 等人（Meta，2024）证明这有帮助。他们的实现在骨干网络之上放置了几个独立的输出头，每个预测不同的偏移。并行、简单，但头看到的是相同的隐藏状态，没有任何层级细化——而且预测不是因果链接的，因此不能用于推测解码。

DeepSeek-V3（2024 年 12 月）重新设计了 MTP，使其成为在每个预测深度保持因果链的顺序模块。模型从 `h_i^(0)` 预测 `t+1`，然后从新的隐藏状态 `h_i^(1)` 预测 `t+2`，`h_i^(1)` 结合了 `h_i^(0)` 和 `E(t+1)` 的 embedding，依此类推。每个深度都是自己的小型 Transformer 块。共享 embedding 和共享输出头保持参数开销适中。在 DeepSeek-V3 的规模下，MTP 模块在 671B 主模型权重之上额外增加 14B 参数。这 2% 的开销换来了更密集的训练信号和现成的推测解码草稿。

本课程从零构建一个 MTP 模块和 D 深度损失。数学很整洁，实现是 150 行。

## 概念说明

### 顺序 MTP 配方

DeepSeek-V3 在主模型之上添加 D 个 MTP 模块。每个模块 k（k = 1..D）预测深度 k 的 token——即给定到位置 i 的前缀，预测 `t_{i+k}`。

模块 k 包含：

- 一个 Transformer 块 `T_k`，有自己的注意力和 MLP。
- 一个投影矩阵 `M_k`，将前一深度的隐藏状态与下一深度真实 token 的 embedding 结合。
- 共享 embedding `E`（与主模型相同）。
- 共享输出头 `Out`（与主模型相同）。

训练时，对于到位置 i 的前缀，每深度隐藏状态为：

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

每深度预测为：

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

每深度损失是与真实 `t_{i+k}` 的交叉熵：

```
L_k = CE(logits_{i+k}, t_{i+k})
```

跨深度的联合损失：

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个小的加权因子——DeepSeek-V3 在前 10% 训练使用 0.3，之后使用 0.1。总训练损失为 `L_main + L_MTP`。

### 为什么是顺序而非并行

Gloeckle 的原始并行 MTP 有 D 个输出头，每个直接应用于 `h_i^(0)`。每个头从相同的骨干隐藏状态预测 `t_{i+k}`。这能训练好，但预测之间没有条件依赖。你不能用 `head_1` 的输出来帮助 `head_2`——头是并行触发的。

DeepSeek-V3 的顺序设计从 `h_i^(k-1)` 加实际下一个 token 的 embedding `E(t_{i+k})` 构建 `h_i^(k)`。这保持了因果链：要预测 `t_{i+k+1}`，深度 k+1 的模块看到了 `t_{i+k}` 处的内容。这在结构上与自回归解码器消费自身输出的方式相同——使 MTP 模块可以直接用作推测解码草稿器。

推理时：将 `h_i^(k-1)` 和草拟的 `t_{i+k}` 输入模块 k+1，得到 `t_{i+k+1}` 的预测。重复。这正是 EAGLE 风格的草稿，使用训练好的 MTP 模块作为草稿网络。DeepSeek-V3 报告第一个 MTP 模块 80% 以上的接受率和约 1.8 倍加速。

### 参数核算

对于隐藏维度 `h`、词表 `V` 的模型：

- 主模型：数十亿参数，加一个大小为 `V * h` 的输出头。
- 共享输出头：复用主模型的头。无额外参数。
- 共享 embedding：复用主模型的 embedding。无额外参数。
- 每个 MTP 模块：
  - 投影 `M_k`：`(2h) * h = 2h^2`。
  - Transformer 块 `T_k`：注意力（MHA 为 `4h^2`）加 MLP（SwiGLU 比率 8/3 时通常 `8h^2`）。每块约 `12h^2`。

每模块总额外：约 `14h^2`。对于 DeepSeek-V3 的 `h = 7168`，D = 1 个模块：理论上约 `14 * 7168^2 ≈ 7.2 亿`参数。DeepSeek-V3 报告 14B——差异主要来自 MTP 模块中的专家层也是 MoE。

### 推测解码的回报

预训练期间，MTP 模块使训练减慢约 10%（更多前向计算、额外损失）。回报是双重的：

1. 更密集的训练信号。每个隐藏状态看到 D+1 个监督目标。在 DeepSeek-V3 的消融中，MMLU、GSM8K、MATH、HumanEval 的效果：一致的几个百分点提升。

2. 推理时免费的推测解码草稿。MTP 模块已经训练好预测接下来的几个 token。作为草稿网络重新使用，它提供 80% 以上的接受率。在这个水平上，N=3 或 N=5 的推测解码给出 1.8 倍吞吐。10% 的训练时间成本在你第一次运行推理时就回本了。

### 与 EAGLE 的关系

EAGLE 在预训练之后单独训练一个小型草稿模型。MTP 将草稿烘焙进预训练。两种方法收敛到相似的接受率，但通过不同的流水线：

| 维度 | EAGLE-3 | MTP（DeepSeek-V3） |
|------|---------|-------------------|
| 训练时机 | 预训练后 | 预训练期间 |
| 向后兼容现有权重 | 是 | 否（需要重新训练） |
| 草稿参数 | 1-2 个 Transformer 层 | 1 个 Transformer 块 + 投影 |
| 接受率 | 0.88-0.92 | 深度 1 时 0.80+ |
| 加速之外的收益 | 仅推测解码 | 更密集训练信号 + 加速 |

## 开始构建

`code/main.py` 端到端构建一个 MTP 模块：共享 embedding、投影、Transformer 块、共享输出头。然后在短合成序列上计算每深度交叉熵损失并按组件打印参数量。32 token 的玩具词表使数字可读。

### 第 1 步：共享 embedding 表

单个 `vocab_size x hidden` 表被主模型和每个 MTP 模块在每个深度使用。不是第二个副本——是同一个张量。

### 第 2 步：每深度组合

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真正的 DeepSeek-V3 将两个 RMSNorm 向量拼接为 `[2h]` 并用 `h x 2h` 矩阵投影。玩具使用向量加法以保持标准库简洁。

### 第 3 步：深度 k 的 Transformer 块

自注意力加 MLP。玩具中，一层线性注意力块和 SwiGLU MLP 保持结构可见，无需 numpy。

### 第 4 步：共享输出头

复用主模型的输出投影。词表上的 logits。

### 第 5 步：每深度损失

softmax(logits) 与偏移 `k` 处真实 token 的交叉熵。用 `lambda / D` 缩放因子跨深度聚合。

### 第 6 步：参数核算

打印总参数量、共享（embedding、头）数量和每模块额外数量。展示 MTP 额外量与主模型大小的比率。

## 使用它

MTP 已集成到 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列中。推理时：

- DeepSeek 自己的服务栈开箱即用地将 MTP 模块用作推测解码器。
- vLLM 和 SGLang 在 2026 年 4 月已有 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang 教程展示了在 V3 检查点上实测 1.8 倍加速的具体 MTP 推测解码配置。

在新预训练运行中何时使用 MTP：

- 你控制整个预训练流水线，想要存储更密集的训练信号。
- 你知道你将大规模服务模型，想要免费的推测解码。
- 你的隐藏维度至少 4096。在 1B 规模下，开销的伤害大于收益的帮助。

何时不用：

- 微调现有的预训练稠密模型。MTP 模块没有训练。
- 你想要干净基线来比较的研究模型。MTP 改变了架构。

## 交付它

本课程产出 `outputs/skill-mtp-planner.md`。给定预训练运行规范（模型大小、数据、计算），它返回集成 MTP 的计划：深度数 D、`lambda` 调度、内存开销和推理时推测解码的接入方式。

## 练习

1. 运行 `code/main.py`。展示随着合成信号增强，每深度损失单调递减。修改合成数据使用固定模式，验证深度 1 和深度 2 损失都收敛。

2. 计算稠密 70B 模型（hidden 8192，80 层）带 D=1 MTP 模块的参数开销。与 DeepSeek-V3 报告的 14B 开销比较。解释为什么 DeepSeek 的数字更高：MTP Transformer 块继承了相同的 MoE 结构，膨胀了每模块参数量。

3. 在玩具中实现 D=2：添加第二个 MTP 模块接收 h^(1) 并预测 `t_{i+2}`。验证联合损失和参数核算与 DeepSeek 论文的公式 19-21 匹配。

4. 将玩具切换为并行 MTP（Gloeckle 风格）：在主隐藏状态之上添加 D 个输出头，每个预测不同偏移。度量每深度损失与顺序版本在同一合成信号上的对比。顺序版本应产生更低的深度 k 损失（k > 1），因为它以中间预测为条件。

5. 将训练好的 MTP 模块用作 EAGLE 风格草稿：在推理时调用模块 k 提出 `t_{i+k}`。在保留序列上度量这些草稿 token 相对于主模型预测的接受率。如果你在玩具上达到 50% 以上，你就复现了 MTP 作为草稿的经验特性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MTP 模块 | "额外损失块" | 一个小型 Transformer 块加投影，预测主模型之后 k 个位置的 token |
| 预测深度 | "哪个偏移" | 整数 k，模块 k 从到位置 i 的前缀预测 `t_{i+k}` |
| 并行 MTP | "Gloeckle 风格" | 同一骨干隐藏状态上的 D 个独立头，无条件链 |
| 顺序 MTP | "DeepSeek-V3 风格" | 每个模块以先前深度的隐藏状态加下一 token 的 embedding 为条件；保持因果链 |
| 共享输出头 | "复用主头" | MTP 模块调用主模型的 LM 头，而非单独的输出投影 |
| 共享 embedding | "复用主表" | 到处使用同一个词表 embedding 表；无重复参数 |
| 投影矩阵 M_k | "组合隐藏 + 下一 token" | 将先前隐藏状态和目标 token embedding 折叠为下一深度输入的 `h x 2h` 线性层 |
| 联合损失 L_MTP | "平均额外损失" | 每深度交叉熵损失的算术平均，乘以 `lambda` |
| 深度 1 接受率 | "MTP 草稿正确的频率" | D=1 MTP 模块的 top-1 预测等于主模型 top-1 预测的比率；DeepSeek-V3 上 80%+ |
| Lambda 加权 | "额外损失重要性" | 每深度缩放因子；DeepSeek-V3 训练开始时 0.3，之后 0.1 |

## 延伸阅读

- [DeepSeek-AI -- DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) -- 完整的顺序 MTP 描述（第 2.2 节），包括联合损失公式和推理时 1.8 倍加速
- [Gloeckle et al. -- Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) -- DeepSeek 设计改进的并行 MTP 基线
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) -- 685B 总计（671B 主 + 14B MTP），部署说明
- [Leviathan et al. -- Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) -- MTP 适配的推测解码框架
- [Li et al. -- EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) -- EAGLE 2025 年的草稿架构，MTP 的竞争对手
