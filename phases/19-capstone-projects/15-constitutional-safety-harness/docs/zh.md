# 毕业项目 15 — 宪法安全框架 + 红队靶场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety 以及 X-Guard 的多语言覆盖，共同定义了 2026 年的安全分类器技术栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们串联成生产级流水线。本毕业项目将这一切整合在一起：围绕目标应用构建分层安全框架，运行 6 种以上攻击族的自主红队智能体，以及一次宪法自我批评流程，产出可度量的无害性增量。

**类型：** 毕业项目
**语言：** Python（安全流水线、红队）、YAML（策略配置）
**前置要求：** Phase 10（从零构建 LLM）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 18（伦理、安全、对齐）
**涉及阶段：** P10 · P11 · P13 · P14 · P18
**所需时间：** 25 小时

## 问题所在

2026 年 LLM 安全的前沿不在于分类器是否有效（它们大致有效），而在于如何在生产应用周围正确组合它们，既不过度拒绝也不留下明显漏洞。Llama Guard 4 处理英语策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 Content Safety 覆盖企业级分类。Anthropic 的 Constitutional Classifiers 则是一种在训练而非服务阶段使用的独立方法。

攻击演进同样重要。PAIR 和 TAP 自动化越狱发现。GCG 执行基于梯度的后缀攻击。多轮和语码切换攻击利用智能体记忆。任何已部署的 LLM 都需要一个红队靶场——garak 和 PyRIT 是标准驱动器——加上有文档记录的缓解措施和 CVSS 评分的发现。

你将加固一个目标应用（8B 指令微调模型或其他毕业项目的 RAG 聊天机器人），对其运行 6 种以上攻击族，并产出前后无害性度量。

## 概念说明

安全流水线分为五层。**输入清洗**：去除零宽字符、解码 base64/rot13、Unicode 归一化。**策略层**：NeMo Guardrails v0.12 护栏（离域、毒性、PII 提取）。**分类器门控**：输入端 Llama Guard 4、非英语端 X-Guard、图像输入端 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：输出端 Llama Guard 4、Presidio PII 清洗、适用场景的引用强制。**HITL 层**：标记为高风险的输出进入 Slack 队列。

红队靶场运行在调度器上。PAIR 和 TAP 自主发现越狱。GCG 执行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（人格代入、记忆利用）。语码切换攻击（混合英语与斯瓦希里语或泰语）。每次运行产出结构化发现文件，包含 CVSS 评分和披露时间线。

宪法自我批评流程是一种训练时干预。取 1k 条有害意图提示，让模型草拟回复，根据书面宪法（不伤害规则）进行批评，并在批评循环上重新训练。在留出评估集上度量前后无害性增量。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动器：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao 等，2023）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪法训练：Anthropic 风格自我批评循环 + 批评上的 SFT
- PII 清洗：Presidio
- 目标：8B 指令微调模型或其他毕业项目的 RAG 聊天机器人

## 开始构建

1. **目标搭建。** 在 vLLM 上部署 8B 指令微调模型（或复用其他毕业项目的 RAG 聊天机器人）。这是待测应用。

2. **安全流水线封装。** 围绕目标接线五层流水线。验证每层可独立观测（Langfuse 中每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在小型标注集上分别运行以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、PAIR 智能体、TAP 智能体、GCG 运行器、多轮攻击器和语码切换攻击器。每个运行在独立队列上。

5. **攻击套件。** 六种攻击族：（1）PAIR 自动化越狱，（2）TAP 攻击树，（3）GCG 梯度后缀，（4）ASCII / base64 / rot13 编码，（5）多轮人格，（6）多语言语码切换。报告每族成功率。

6. **宪法自我批评。** 整理 1k 条有害意图提示。目标为每条草拟回复。批评 LLM 根据书面宪法（"不伤害"、"引用证据"、"拒绝非法请求"）评分。批评反对的提示被重写；目标在批评改进对上微调。在留出评估集上度量前后无害性。

7. **过度拒绝度量。** 在良性提示套件（如 XSTest）上跟踪假阳性率。目标必须在良性问题上保持有用性。

8. **CVSS 评分。** 每次成功越狱按 CVSS 4.0 评分（攻击向量、复杂度、影响）。产出披露时间线和缓解计划。

9. **靶场自动化。** 上述所有流程运行在 cron 上；发现写入队列；过度拒绝回归告警发送到 Slack。

## 使用示例

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付成果

`outputs/skill-safety-harness.md` 是交付物。生产级分层安全流水线加上可复现的红队靶场，包含前后无害性增量。

| 权重 | 标准 | 度量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6 种以上攻击族、2 种以上语言 |
| 20 | 真阳性/假阳性权衡 | 攻击拦截率 vs XSTest 良性通过率 |
| 20 | 自我批评增量 | 留出评估集上的前后无害性 |
| 20 | 文档与披露 | CVSS 评分的发现及时间线 |
| 15 | 自动化与可复现性 | 全部在 cron 上运行并带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，比较有无输出过滤层的攻击成功率。

2. 增加第七种攻击族：通过检索文档的间接提示注入。度量所需的额外防御。

3. 实现"拒绝但有帮助"模式：当护栏拦截时，目标提供更安全的相关回答而非简单拒绝。度量 XSTest 增量。

4. 多语言覆盖差距：找到一种 X-Guard 表现不佳的语言。提出针对该语言的微调数据集。

5. 在 30B 模型上运行宪法自我批评，度量增量是否随规模变化。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | "纵深防御" | 输入、门控、输出、HITL 多重护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考级输入/输出内容分类器 |
| PAIR | "越狱智能体" | Chao 等人的论文，关于 LLM 驱动的越狱发现 |
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪心坐标梯度" | 基于梯度的对抗性后缀攻击 |
| 宪法自我批评 | "Anthropic 风格训练" | 目标草拟 -> 批评评分 -> 重写 -> 重训 |
| XSTest | "良性探针集" | 过度拒绝回归基准 |
| CVSS 4.0 | "严重性评分" | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132 语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
