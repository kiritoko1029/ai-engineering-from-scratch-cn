# 毕业项目 07 — 端到端微调流水线（数据到 SFT 到 DPO 到服务）

> 一个用你的数据训练的 80 亿模型，用你自己的偏好进行 DPO 对齐，量化，投机解码，并以可衡量的 $/百万 Token 成本提供服务。2026 年的开源技术栈是 Axolotl v0.8、TRL 0.15、用于迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF、用于服务的 vLLM 0.7 + EAGLE-3。本毕业项目要求你可复现地运行整个流水线——从 YAML 输入到服务端点输出——并在 2026 年 Model Openness Framework 下发布模型卡片。

**类型：** 毕业项目
**语言：** Python（流水线）、YAML（配置）、Bash（脚本）
**前置要求：** 阶段 2（机器学习）、阶段 3（深度学习）、阶段 7（Transformer）、阶段 10（从零构建 LLM）、阶段 11（LLM 工程）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**所需时间：** 35 小时

## 问题所在

2026 年，每个严肃的 AI 团队都随时保持着一条微调流水线。不是因为他们发布前沿的基础模型，而是因为下游适配——领域 SFT、针对标注偏好的 DPO、为投机解码蒸馏的草稿、使用 EAGLE-3 服务——才是可衡量收益所在。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让你快速进行单 GPU 迭代。vLLM 0.7 + EAGLE-3 将解码吞吐量提升 2-3 倍且不损失质量。工具都已就绪；技艺在于 YAML 配置、数据卫生和评估纪律。

你将把一个 80 亿参数的基础模型（Llama 3.3、Qwen3 或 Gemma 3）经过 SFT 然后 DPO 处理任务特定数据，量化后提供服务，并通过 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量收益。你将在 2026 年 Model Openness Framework 下生成模型卡片。核心在于可复现性——一条命令就能端到端重跑整个流水线。

## 概念说明

流水线有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清洗、针对公开基准污染的拆分卫生检查。**SFT**：Axolotl YAML，8xH100 上的 ZeRO-3，余弦调度，打包序列，2-3 个 epoch。**DPO 或 GRPO**：TRL 配置，1 个 epoch，偏好对可以是人工标注或模型评判，beta 调优。**量化**：GPTQ + AWQ + GGUF 提供部署灵活性。**服务**：vLLM 0.7 + EAGLE-3 投机头（或 SGLang + SpecForge），K8s 部署，基于队列等待的 HPA。

消融实验是交付成果：在三个任务特定基准上对比 SFT-only、SFT+DPO 和 SFT+GRPO。服务指标：批量 1/8/32 时的 Token/s、EAGLE-3 接受率、$/百万 Token。安全评估：Llama Guard 4 通过率。模型卡片：偏差评估、可复现性种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量，Presidio 用于 PII
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，ZeRO-3、Flash Attention 3、打包序列
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、GGUF（via llama.cpp）
- 服务：vLLM 0.7 + EAGLE-3 投机解码（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin，基于队列等待指标的 HPA
- 可观测性：W&B 用于训练，Langfuse 用于推理

## 开始构建

1. **数据流水线。** 对原始语料库运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。Presidio 清洗 PII。用明确的种子写入训练/验证拆分。

2. **污染检查。** 对每个验证拆分，对 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集计算 MinHash。拒绝任何重叠。

3. **Axolotl SFT。** YAML 配置 ZeRO-3、FA3、序列打包。8xH100 上运行 2-3 个 epoch。记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT 检查点，对偏好对运行一个 epoch 的 DPO（或使用可验证奖励在数学/代码上的 GRPO）。扫描 beta。

5. **量化。** 生成三种量化：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M（用于 llama.cpp）。记录大小和标称吞吐量。

6. **使用投机解码提供服务。** vLLM 0.7 配置 EAGLE-3 草稿头（通过 Red Hat Speculators 训练）。衡量批量 1/8/32 时的接受率和尾延迟。报告 $/百万 Token 与 Anthropic / OpenAI 在同一评估上的对比。

7. **评估矩阵。** 在基础模型、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **安全评估。** Llama Guard 4 在开发集上的通过率。ShieldGemma-2 输出过滤器。

9. **模型卡片。** MOF 2026 模板：数据、训练、评估、安全、许可、可复现性部分（含 YAML 和提交 SHA）。

## 使用示例

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付成果

`outputs/skill-finetuning-pipeline.md` 描述了交付成果。一条命令运行数据经过 SFT、DPO、量化、服务、评估的全流程，输出模型卡片和服务端点。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 与基础模型的评估差异 | 在目标任务（MMLU-Pro、MT-Bench-v2、任务特定）上的衡量收益 |
| 20 | 流水线可复现性 | 使用相同种子的一条命令端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查通过 |
| 20 | 服务效率 | 批量 1/8/32 的 Token/s、EAGLE-3 接受率、$/百万 Token |
| 15 | 模型卡片 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一任务特定基准上对比 SFT-only、SFT+DPO 和 SFT+GRPO。报告哪种偏好方法获胜以及幅度多大。

2. 将 Llama 3.3 8B 替换为 Qwen3 14B。衡量匹配质量时的 $/百万 Token。

3. 衡量 EAGLE-3 在领域数据与通用 ShareGPT 上的接受率。报告差异及其对延迟预算的影响。

4. 注入 1% 的污染（将 MMLU-Pro 答案泄露到训练数据中）并重跑评估。观察 MMLU-Pro 准确率不切实际地飙升。构建一个能捕获此问题的污染检查 CI 门控。

5. 添加 LoRA SFT 作为全参数微调的替代方案。衡量 10 倍内存节省下的质量差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Axolotl | "SFT 训练器" | 统一的 YAML 驱动训练器，支持 SFT、DPO 和蒸馏 |
| TRL | "偏好调优器" | Hugging Face 库，用于 LLM 的 DPO、GRPO、PPO |
| GRPO | "群组相对策略优化" | DeepSeek R1 的 RL 方案，使用可验证奖励 |
| EAGLE-3 | "投机解码草稿" | 预测 N 个 Token 的草稿头；vLLM 用目标模型验证 |
| MOF | "Model Openness Framework" | 2026 年对模型发布在数据、代码、许可方面的评级标准 |
| 污染检查 | "拆分卫生" | 基于 MinHash 的测试集泄露到训练集的检测 |
| 接受率 | "EAGLE / MTP 指标" | 目标模型接受的草稿 Token 比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — SFT / DPO 训练器参考
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务技术栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选投机解码训练器
- [Model Openness Framework 2026](https://isocpp.org/) — 开放发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评估运行器
