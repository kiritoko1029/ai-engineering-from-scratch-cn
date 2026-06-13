# 毕业项目 05 — 自主研究智能体（AI-Scientist 级别）

> Sakana 的 AI-Scientist-v2 已发表完整论文。Agent Laboratory 运行了实验。Allen AI 分享了追踪记录。2026 年的形态是：在实验树上进行规划—执行—验证的树搜索、受控的预算成本、沙箱化的代码执行、带视觉反馈的 LaTeX 写作器，以及自动化的 NeurIPS 风格评审集成。本毕业项目要求你构建一个这样的系统，在每篇论文 $30 的预算内端到端运行，并通过 Sakana 记录的沙箱逃逸红队测试。

**类型：** 毕业项目
**语言：** Python（智能体 + 沙箱）、LaTeX（输出）
**前置要求：** 阶段 2（机器学习）、阶段 3（深度学习）、阶段 7（Transformer）、阶段 10（从零构建 LLM）、阶段 14（智能体）、阶段 15（自主系统）、阶段 16（多智能体）、阶段 18（安全）
**涉及阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**所需时间：** 40 小时

## 问题所在

自主研究智能体在 2026 年跨越了一个门槛。Sakana AI 的 AI-Scientist-v2 发表在 Nature 上，生成的论文通过了研讨会同行评审。ShinkaEvolve（ICLR 2026）将其扩展到演化假设。AMD 的 Agent Laboratory 提供了可复现的追踪记录。这些智能体并非魔法——它们是运行在候选实验树上的规划—执行—验证循环，带有成本上限、种子绑定的沙箱和自动评审。技艺在于循环设计、预算控制和安全考量。

你需要通过针对一个种子想法在狭窄领域（例如，1 亿参数 Transformer 上的注意力稀疏性消融）实现一个来学习这个循环。价值不在于首次运行就能发现新东西。价值在于基础设施：树搜索、实验沙箱、写作—评审循环、红队报告。Sakana 团队记录了沙箱逃逸失败；你的智能体必须通过同样的红队测试。

## 概念说明

该智能体是一个最佳优先树搜索。节点是实验规格：（假设、配置、代码、预期结果）。扩展步骤通过小幅修改（替换优化器、调整批量大小、消融一个组件）提出子节点。每个子节点在全新的沙箱中运行，带有硬性资源限制。结果反馈到评分函数中，按（新颖性 x 质量 x 剩余预算）对节点排序。树持续生长直到预算耗尽，然后将最佳分支撰写成文。

写作器是多模态的。它生成 LaTeX 草稿，编译它，渲染图表，然后将渲染后的 PDF 反馈给 Claude Opus 4.7 的视觉模式，对排版、图表可读性和论点—证据一致性进行批评。由五个 LLM 评委组成的评审集成输出 NeurIPS 风格的评分（新颖性、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文会带着批评意见返回给写作器。

安全是核心考量。每个实验在 E2B 或 Daytona 沙箱中运行，无网络出口、有界的墙钟时间和固定的资源限制。智能体的代码生成步骤通过策略层，阻止能逃逸沙箱的系统调用。红队报告复现了 Sakana 记录的攻击面（fork bomb、文件系统逃逸、LLM 编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph，支持检查点和人工审批门控
- 树搜索：自定义最佳优先搜索，遍历实验节点（Sakana v2 的 AB-MCTS 风格）
- 沙箱：每个实验使用 E2B，Docker-in-Docker 作为备选；通过 cgroups 进行资源限制
- 文献：Semantic Scholar Graph API + OpenAlex + 本地 FAISS 摘要缓存
- 写作器：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表批评和排版
- 评审：5 个评委的集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架：PyTorch 2.5 用于物理实验，W&B 用于日志记录
- 可观测性：Langfuse 用于智能体追踪，每篇论文 $30 硬性预算

## 开始构建

1. **种子与领域界定。** 取一个种子想法（例如，"研究 10 亿以下参数 Transformer 中注意力图的稀疏模式"）。定义搜索空间：模型、数据集、计算预算。

2. **文献检索。** 通过 Semantic Scholar + OpenAlex 查询 50 篇最高引用的相关论文；本地缓存摘要；生成 1 页的领域摘要。

3. **树结构搭建。** 用种子假设初始化根节点。实现 `expand(node) -> children`，采用小幅修改建议（每个子节点一个配置更改）。实现 `score(node)`，使用加权的新颖性 x 质量 x 预算项。

4. **沙箱封装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载回宿主机。

5. **规划—执行—验证循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（损失是否下降？消融是否隔离了效果？）。失败节点会在树上记录失败原因。

6. **写作器。** 预算耗尽后，选择最佳分支。用 matplotlib 渲染图表。通过 Claude Opus 4.7 生成 LaTeX 草稿，将分支追踪作为上下文。编译。将编译后的 PDF 反馈给 Opus 4.7 视觉模式进行批评。迭代。

7. **评审集成。** 五个评委按 NeurIPS 风格的评分标准对草稿评分（新颖性、严谨性、清晰度、可复现性、影响力）。如果平均分 < 4.0/5，带着批评意见返回写作器。3 次重写后硬性停止。

8. **红队。** 构建或集成一组针对沙箱的对抗性任务：fork bomb、网络数据外泄尝试、文件系统逃逸、LLM 编写的 shell 元字符。确认全部被拦截。撰写发现报告。

9. **可复现性。** 每篇论文附带其树搜索追踪 JSON、种子、W&B 运行链接、沙箱配置，以及一个端到端复现的 README。

## 使用示例

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付成果

`outputs/skill-ai-scientist.md` 是交付成果。给定一个种子想法、一个领域和 $30 预算，它运行完整流水线，输出可评审的论文和可复现性包。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 论文质量 | 对已发表的研讨会论文进行盲评 |
| 20 | 实验严谨性 | 基线、种子、消融；每个声明都有结果表中的单元格支撑 |
| 20 | 成本与计算纪律 | $30/篇论文的硬性上限，Langfuse 追踪 |
| 20 | 安全性 | 沙箱红队测试通过；网络策略和终止开关已验证 |
| 15 | 可复现性 | 使用相同种子的一键重跑能复现论文 |
| **100** | | |

## 练习

1. 在同一领域对三个不同的种子想法运行流水线。对比树搜索的哪些部分重叠。识别重复浪费的计算。

2. 在实验执行前添加人工审批门控，针对预估超过 $5 的节点。衡量总成本下降多少。

3. 将评审集成替换为单一评委。在已知质量差的论文保留集上衡量误接受率。

4. 引入网络数据外泄红队测试：智能体编写尝试 `curl` 外部地址的代码。确认 `--network=none` 策略将其拦截。记录尝试。

5. 将你的树搜索与扁平随机基线（相同预算，无扩展策略）进行对比。报告新颖性 x 质量的提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 树搜索 | "AB-MCTS 风格扩展" | 最佳优先搜索，遍历实验节点，使用新颖性 x 质量 x 预算评分 |
| 沙箱 | "实验隔离" | 无网络、有界 CPU/内存、固定种子、只读输入的容器 |
| 视觉批评 | "渲染后读取" | 将论文编译为 PDF，将 PDF 反馈给 VLM 进行排版和论点—证据批评 |
| 评审集成 | "自动化同行评审" | 多个 LLM 评委按 NeurIPS 评分标准打分；加权聚合控制流水线 |
| 新颖性评分 | "这是新的吗？" | 惩罚与 50 篇文献缓存接近度的启发式方法 |
| 成本上限 | "$ 预算" | 每篇论文的总支出硬性上限；Langfuse 计数器 + 运行前估算 |
| 红队 | "沙箱逃逸审计" | 如果策略有误就会逃逸沙箱的对抗性任务 |

## 延伸阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 生产级研究智能体参考
- [Sakana AI-Scientist-v1 论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292) — 原始方法论
- [ShinkaEvolve（Sakana ICLR 2026）](https://sakana.ai) — 演化扩展
- [Agent Laboratory（AMD）](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B sandboxes](https://e2b.dev) — 实验隔离参考
- [NeurIPS 评审指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 评审集成编码的评分标准
