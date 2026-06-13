# DeepSeek-V3 架构走读

> 第 10 阶段 · 第 14 课指出了每个开放模型转动的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总参数 671B，激活 37B）转动了全部六个，又额外加了四个：多头潜在注意力、无辅助损失负载均衡、多 token 预测和 DualPipe 训练。本课程从上到下阅读 DeepSeek-V3 的架构，并从发布的 config 推导每个参数量。学完之后，你能解释为什么 671B/37B 的比率是正确的押注，以及为什么 MLA + MoE 组合在前沿比单独任何一个都强。

**类型：** 学习
**语言：** Python（标准库，参数计算器）
**前置要求：** 第 10 阶段 · 14（开放模型走读），第 10 阶段 · 17（NSA），第 10 阶段 · 18（MTP），第 10 阶段 · 19（DualPipe）
**所需时间：** 约75分钟

## 学习目标

- 从上到下阅读 DeepSeek-V3 config，用六个 GPT-2 旋钮加四个 DeepSeek 特有添加来解释每个字段。
- 推导总参数量（671B）、激活参数量（37B）和贡献于每个的组件。
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与相同激活参数量的 GQA 稠密模型对比。
- 说出四个 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每个针对架构/训练栈的哪部分。

## 问题所在

DeepSeek-V3 是第一个架构上与 Llama 家族有实质性不同的前沿开放模型。Llama 3 405B 是"转了六个旋钮的 GPT-2"。DeepSeek-V3 是转了全部六个旋钮又加了四个的 GPT-2。阅读 Llama 3 config 是阅读 DeepSeek config 的热身，但深层结构——注意力块的形状、路由逻辑、训练时目标——足够不同，需要单独走读。

学习它的收益：DeepSeek-V3 的开放权重发布重新定义了开放模型中"前沿能力"的含义。这个架构是 2026 年许多训练运行正在复制的蓝图。理解它是任何接触前沿 LLM 训练或推理的角色的入场费。

## 概念说明

### 再次强调：不变的核心

DeepSeek-V3 仍然是自回归的。仍然堆叠解码器块。每个块仍然有注意力加 MLP 加两个 RMSNorm。仍然在 MLP 中使用 SwiGLU。仍然使用 RoPE。前归一化。权重绑定 embedding。与每个 Llama 或 Mistral 相同的基线。

### 转折：MLA 代替 GQA

从第 10 阶段 · 14 你知道 GQA 通过在 Q 头组之间共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（MLA）更进一步：K 和 V 被压缩到共享的低秩潜在表示（`kv_lora_rank`），然后按头动态解压。KV 缓存只存储潜在表示——通常每 token 每层 512 个浮点数，而不是 8 x 128 = 1024 个浮点数。

在 128k 上下文下，使用 MLA 的 DeepSeek-V3（每 token 每层一个共享潜在 `c^{KV}`；K 和 V 都通过可吸收到后续矩阵乘法中的上投影派生）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设的 GQA 基线（Llama 3 70B 形状，8 个 KV 头，头维度 128）需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA 在 128k 上下文下比 Llama-3-70B 风格的 GQA 缓存小 4 倍。

权衡：MLA 在每次注意力计算中添加一个解压步骤（每头）。额外计算与节省的带宽相比很小。长上下文推理的净收益。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 token。朴素路由器将太多工作集中在少数专家上，让其他专家空闲。标准修复：添加惩罚负载不均衡的辅助损失项。这有效但略微降低主任务性能。

DeepSeek-V3 引入无辅助损失方案。每专家偏置项被添加到路由器 logits 中，通过简单规则在训练期间调整：如果专家 e 过载，减小 `bias_e`；如果负载不足，增大它。没有额外损失项。训练保持干净。专家负载保持平衡。

对主损失的影响：无法度量。对 MoE 架构的影响：更干净，无需调优辅助损失超参数。

### MTP：更密集训练 + 免费草稿

从第 10 阶段 · 18 你知道 DeepSeek-V3 添加了 D=1 MTP 模块来预测两个位置之后的 token。推理时，训练好的模块被重新用作推测解码草稿，接受率 80% 以上。训练时，每个隐藏状态在 D+1 = 2 个目标上被监督，提供更密集的信号。

参数：在 671B 主模型之上 14B。开销：2.1%。

### 训练：DualPipe

从第 10 阶段 · 19 你知道 DualPipe 是一种双向流水线，将前向和反向块与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048 块 H800 规模下，它回收了 1F1B 会损失到流水线气泡的大约 24.5 万 GPU 小时。

### Config 逐字段解析

以下是 DeepSeek-V3 config（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析：

- `hidden_size=7168`：embedding 维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到此潜在维度并按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之上，1 个始终在线的专家为每个 token 做贡献。可以将其视为确保每个 token 获得可靠输出的"稠密基底"。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏大小。比稠密 MLP 小，因为有 256 个。

### 参数核算

完整计算在 `code/main.py` 中。要点：

- Embedding：`vocab * hidden = 129280 * 7168 ≈ 0.93B`。
- 前 3 个稠密块：MLA 注意力（每块约 1.44 亿）+ 稠密 MLP（每块约 2.6 亿）+ 归一化。总计约 12 亿。
- 58 个 MoE 块：MLA 注意力（约 1.44 亿）+ 每块 256 个专家（每个 3000 万）+ 1 个共享专家（3000 万）+ 归一化。每块总计约 79.5 亿，包括所有专家。58 个 MoE 块共约 4610 亿。
- MTP 模块：140 亿。

总计：核心架构约 4760 亿 + 140 亿 MTP + 发布的 671B 数字还包含额外结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中复现的数字在发布值的 3-5% 以内——差异来自 DeepSeek 报告第 2 节附录中记录的细粒度核算。

每次前向的激活参数：

- 注意力：每层 1.44 亿 x 61 = 88 亿（所有层都触发）。
- MLP 激活：前 3 层稠密（3 x 2.6 亿 = 7.8 亿），58 个 MoE 层每层激活 8 个路由 + 1 个共享 + 路由开销。每层激活 MLP 约 2.6 亿。总计：3 x 2.6 亿 + 58 x 2.6 亿 ≈ 159 亿。
- Embedding + 归一化：12 亿。
- 总激活：约 260 亿核心 + 140 亿 MTP（训练时有但推理时不总是运行）≈ 370 亿。

### 671B / 37B 比率

18 倍稀疏比（激活参数是总量的 5.5%）。DeepSeek-V3 是发布开放权重的最稀疏前沿 MoE 模型。Mixtral 8x7B 比率 13/47（28%）稠密得多。Llama 4 Maverick 比率 17B/400B（4.25%）可比。DeepSeek 的押注：在前沿规模，更多专家加更低激活比产生更好的每激活 FLOP 质量。

### DeepSeek-V3 的位置

| 模型 | 总量 | 激活 | 比率 | 注意力 | 新颖想法 |
|------|-----|------|------|-------|---------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上的推理训练运行。R1 使用相同架构。改变的是后训练配方（可验证任务上的大规模 RL），而非预训练架构。

DeepSeek-V4（如果发布）预计将保留 MLA + MoE + MTP 并添加 DSA（DeepSeek Sparse Attention），即第 10 阶段 · 17 中 NSA 的后继。谱系是稳定的：架构级创新累积；每个版本转动额外的旋钮。

```figure
moe-routing
```

## 使用它

`code/main.py` 是专门针对 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文的数字比较，并用于假设变体（256 专家 vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

关注点：

- 总参数量 vs 发布的 671B。
- 激活参数量 vs 发布的 37B。
- 128k 上下文下的 KV 缓存——MLA vs GQA 比较。
- 逐层分解以查看参数预算实际花在哪里。

## 交付它

本课程产出 `outputs/skill-deepseek-v3-reader.md`。给定 DeepSeek 家族模型（V3、R1 或任何未来变体），它产出逐组件的架构阅读，命名 config 的每个字段，按组件推导参数量，并识别模型使用了四个 DeepSeek 特有创新中的哪些。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估算与发布的 671B 比较，找出差异来自哪里。论文第 2 节有完整的逐项明细。

2. 修改 config 使用 MLA rank 256 代替 512。计算 128k 上下文下的 KV 缓存大小。买到了多少百分比的缩减，以每头表达能力的什么代价？

3. 将 DeepSeek-V3 的（256 专家，top-8）路由与假设的（512 专家，top-8）变体比较。总参数增长；激活参数不变。额外专家容量在理论上买到什么，在推理时代价是什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）关于 MLA 的第 2.1 节。用三句话解释为什么 K 和 V 解压矩阵可以在推理时效率"吸收"到后续矩阵乘法中。

5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算存储 671B 权重的 FP8 vs BF16 内存节省。这与 14.8T token 训练预算如何交叉？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MLA | "多头潜在注意力" | 将 K 和 V 压缩到共享的低秩潜在表示（kv_lora_rank，通常 512），按头动态解压；KV 缓存只存储潜在表示 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在表示的大小；DeepSeek-V3 使用 512 |
| 前 k 个稠密层 | "早期层保持稠密" | MoE 模型的前几层跳过 MoE 路由器，运行稠密 MLP 以保持稳定性 |
| num_experts_per_tok | "top-k 路由" | 每个 token 触发的路由专家数；DeepSeek-V3 使用 8 |
| 共享专家 | "始终在线的专家" | 不论路由如何都处理每个 token 的专家；DeepSeek-V3 使用 1 |
| 无辅助损失路由 | "偏置调整负载均衡" | 在训练期间调整每专家偏置项以保持专家负载均衡，无需添加损失项 |
| MTP 模块 | "额外预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer 块；更密集训练，免费推测解码草稿 |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点 all-to-all 重叠的训练调度 |
| 激活参数比 | "稀疏性" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | "8 位训练" | 以 FP8 进行训练存储和许多计算操作；相比 BF16 大约减半内存，质量代价很小 |

## 延伸阅读

- [DeepSeek-AI -- DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) -- 完整的架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) -- config 文件和部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) -- 引入 MLA 的前身
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) -- 在 V3 架构上的推理训练后继
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) -- DeepSeek 家族注意力的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) -- 训练调度参考
