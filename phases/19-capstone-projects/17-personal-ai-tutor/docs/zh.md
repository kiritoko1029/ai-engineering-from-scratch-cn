# 毕业项目 17 — 个人 AI 辅导老师（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在 2026 年大规模推出了自适应多模态辅导。共同形态是苏格拉底式策略（绝不直接给出答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音 + 文字 + 拍照数学输入、课程图检索、间隔重复调度，以及针对年龄适配内容的严格安全过滤器。本毕业项目的目标是推出一个学科特定辅导工具（K-12 代数或 Python 入门），对 10 名学习者进行为期两周的效果研究，并通过内容安全审计。

**类型：** 毕业项目
**语言：** Python（后端、学习者模型）、TypeScript（Web 应用）、SQL（课程图，Postgres + Neo4j）
**前置要求：** Phase 5（NLP）、Phase 6（语音）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 14（智能体）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**所需时间：** 30 小时

## 问题所在

自适应辅导曾经是教育科技的研究小众领域。到 2026 年它已成为消费产品。Khanmigo 部署在美国大多数学区。Duolingo Max 达到数千万月活用户。Google 的 LearnLM / Gemini for Education 为 Google Classroom 提供辅导能力。Quizlet Q-Chat 与闪卡并列。Synthesis Tutor 以"面向好奇孩子的辅导"走红。共同元素：多模态输入（打字、语音、拍照方程）、苏格拉底式教学法（先问后解释）、每次交互后更新的学习者模型，以及严格的年龄适配安全。

你将为特定群体构建其中一个。衡量标准是实际效果研究：10 名学习者两周的前后测分数。语音循环必须自然流畅（毕业项目 03 子技术栈）。记忆必须尊重隐私。安全过滤器必须通过面向 K-12 的 COPPA 感知红队测试。

## 概念说明

四个组件。**辅导策略**是一个苏格拉底式循环：当学习者要求答案时，策略提出引导性问题；当他们答对时，进入下一个概念；当他们卡住时，提供脚手架式提示。**学习者模型**是贝叶斯知识追踪（或简单变体），每次交互后更新每个课程节点的掌握概率。**课程图**是带有前置边的 Neo4j 概念图；策略沿图行走以选择下一个概念。**记忆**是情景 + 语义存储（agentmemory 风格），保存过去的交互、错误和偏好。

用户体验是多模态的。文字输入用于打字回答。语音输入通过 LiveKit + Whisper（复用毕业项目 03）。拍照输入用于数学题，通过 dots.ocr 或 PaliGemma 2。语音输出通过 Cartesia Sonic-2。安全使用 Llama Guard 4 加年龄适配过滤器（屏蔽成人内容、暴力、自残）和 COPPA 感知的记忆保留策略。

效果研究是交付成果。10 名学习者，前后测，两周。报告学习增益增量和置信区间。与非自适应基线对比（相同内容线性呈现，无辅导策略）。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12 代数或 Python 入门（二选一深入）
- 辅导策略：LangGraph + Claude Sonnet 4.7（带提示缓存）
- 学习者模型：贝叶斯知识追踪（经典）或 FSRS 间隔调度
- 课程图：Neo4j 概念 + 前置边 + OER 内容
- 记忆：agentmemory 风格持久化向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用毕业项目 03 子技术栈）
- 拍照数学：dots.ocr 或 PaliGemma 2 方程识别
- 安全：Llama Guard 4 + 自定义年龄适配过滤器
- 评估：Bloom 层级题目生成、前后测工具、效果研究工具

## 开始构建

1. **课程图。** 构建包含 50-150 个概念节点的 Neo4j（如 K-12 代数从"数轴"到"求根公式"），带前置边。每个节点附加 OER 内容（Open Textbook、OpenStax）。

2. **学习者模型。** 初始化贝叶斯知识追踪，设置先验：猜测率、失误率、学习率。每次交互后更新每个概念的掌握度。按学习者持久化。

3. **辅导策略。** LangGraph 节点：`read_signal`（学习者答案正确/部分正确/卡住？）、`select_concept`（沿课程图选择最高优先级概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. **记忆。** 每次交互写入情景存储。错误和偏好提升到语义记忆。COPPA 感知的保留策略：1 年后自动删除，家长可访问。

5. **语音通路。** LiveKit Agents worker 附加到辅导策略。ASR 通过 Whisper-v3-turbo。TTS 通过 Cartesia Sonic-2。支持打断（复用毕业项目 03 机制）。

6. **拍照数学通路。** 上传或捕获图像；运行 dots.ocr 或 PaliGemma 2 识别方程；作为结构化输入传递给辅导工具。

7. **安全。** 每个模型输出经过 Llama Guard 4 + 年龄适配过滤器（屏蔽自残、成人内容、暴力）。记忆访问按学习者 ID 限定作用域；家长可访问删除界面。

8. **效果研究。** 10 名学习者，前测（标准化 30 题基线），两周辅导交互（每周 3 次），后测。与 10 名学习者的非自适应基线队列对比相同内容。

9. **每周进度报告。** 为每位学习者自动生成 PDF 摘要，包含已探索主题、掌握度轨迹和推荐的后续步骤。

## 使用示例

```
learner: "我不明白为什么 3x + 6 = 12 意味着 x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "你会先从两边减去什么数字？"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "很好。那 3x / 3 等于多少？"
```

## 交付成果

`outputs/skill-ai-tutor.md` 是交付物。一个学科特定的自适应辅导工具，具备多模态输入、学习者模型、记忆、安全和可度量的效果。

| 权重 | 标准 | 度量方式 |
|:-:|---|---|
| 25 | 学习增益增量 | 10 名学习者两周研究的前后测增量 |
| 20 | 苏格拉底式忠实度 | 对话记录样本的评分量表得分 |
| 20 | 多模态体验 | 语音 + 拍照 + 文字端到端连贯性 |
| 20 | 安全 + 隐私姿态 | Llama Guard 4 通过率 + COPPA 感知保留策略 |
| 15 | 课程广度和图质量 | 概念覆盖 + 前置图一致性 |
| **100** | | |

## 练习

1. 在有无自适应学习者模型（随机概念顺序）的情况下分别运行效果研究。报告增量。预计自适应获胜，但增量大小才是关键数字。

2. 增加多模态探针：同一概念问题以文字、语音和拍照三种模态呈现。度量学习者是否在其偏好的模态上更快收敛。

3. 构建家长仪表盘：已练习主题、掌握度轨迹、即将学习的概念、安全事件（任何护栏触发）。符合 COPPA。

4. 增加语言切换模式：辅导工具接受西班牙语输入并以西班牙语教学。度量 X-Guard 覆盖。

5. 压力测试记忆隐私：验证学习者 A 无法看到学习者 B 的数据，即使通过语音片段重注入攻击。记录尝试的访问并告警。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 苏格拉底式策略 | "提问而非灌输" | 辅导工具提出引导性问题而非直接给出答案 |
| 贝叶斯知识追踪 | "BKT" | 经典的学习者模型方程，用于计算每个概念的掌握概率 |
| FSRS | "自由间隔重复调度器" | 2024 年间隔重复调度器，优于 SM-2 |
| 课程图 | "概念 DAG" | 带前置边的 Neo4j 概念图 |
| 情景记忆 | "逐次交互日志" | 每次交互存储供后续检索 |
| 语义记忆 | "学习模式存储" | 从情景记忆中提炼的错误和偏好 |
| COPPA | "儿童隐私法" | 美国限制从 13 岁以下儿童收集数据的法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 参考级消费 K-12 辅导工具
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考级语言学习辅导工具
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 备选参考
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈
