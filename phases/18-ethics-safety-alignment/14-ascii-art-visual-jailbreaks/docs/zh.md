# ASCII 艺术与视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，"ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs"（ACL 2024，arXiv:2402.11753）。掩盖有害请求中与安全相关的令牌，用相同字母的 ASCII 艺术渲染替换它们，然后发送伪装提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳健地识别 ASCII 艺术令牌。攻击绕过了 PPL（困惑度过滤器）、改述防御和重新分词。相关：ViTC 基准测量非语义视觉提示的识别；StructuralSleight 泛化到不常见的文本编码结构（树、图、嵌套 JSON）作为编码攻击家族。

**类型：** 构建
**语言：** Python（标准库，ArtPrompt 令牌掩盖工具）
**前置要求：** 第 18 阶段 · 12（PAIR），第 18 阶段 · 13（MSJ）
**所需时间：** 约60分钟

## 学习目标

- 描述 ArtPrompt 攻击：词识别步骤、ASCII 艺术替换、最终伪装提示。
- 解释为什么标准防御（PPL、改述、重新分词）在 ArtPrompt 上失败。
- 定义 ViTC 并描述它测量什么。
- 描述 StructuralSleight 作为对任意不常见文本编码结构的泛化。

## 问题所在

通过改述和角色扮演（第 12 课）以及长上下文（第 13 课）的攻击在文本层面模式上运作。ArtPrompt 在识别层面运作：模型不解析被禁止的令牌。它解析一个用字符渲染的图像。安全过滤器看到无害的标点符号。模型看到一个词。

## 概念说明

### ArtPrompt，两个步骤

步骤 1。词识别。给定有害请求，攻击者使用 LLM 识别与安全相关的词（例如，"how to make a bomb"中的"bomb"）。

步骤 2。伪装提示生成。将每个识别出的词替换为其 ASCII 艺术渲染（形成字母形状的 7x5 或 7x7 字符块）。模型接收标点符号和空格的网格，足够强大的模型可以将其识别为词；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 都失败了。在其基准子集上攻击成功率超过 75%。

### 为什么标准防御失败

- **PPL（困惑度过滤器）。** ASCII 艺术有高困惑度——但所有新输入都是如此。阻止 ArtPrompt 的阈值选择也会阻止合法的结构化输入。
- **改述。** 改述提示会破坏 ASCII 艺术。在实践中，改述 LLM 通常保留或重建艺术。
- **重新分词。** 不同地分割令牌不会改变模型的视觉是在识别字母形状。

根本问题是安全过滤器在令牌或语义层面；ArtPrompt 在视觉识别层面运作。

### ViTC 基准

非语义视觉提示的识别。测量模型阅读 ASCII 艺术、wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型阅读视觉文本越好，ArtPrompt 在其上效果越好。这是能力-安全权衡。

### StructuralSleight

泛化 ArtPrompt：不常见文本编码结构（UTES）。树、图、嵌套 JSON、JSON 中的 CSV、diff 风格代码块。如果一个结构在训练安全数据中罕见但可被模型解析，它可以隐藏有害内容。

防御含义：安全必须泛化到模型可以解析的结构化表示。这个集合很大且在增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。带有真实图像的 ArtPrompt 风格攻击比 ASCII 艺术类比更强，因为图像编码器产生更丰富的信号。

### 在第 18 阶段中的位置

第 12-14 课描述三个正交攻击向量：迭代细化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从模型中心攻击转向系统边界攻击（间接提示注入）。第 16 课描述防御工具响应。

## 开始构建

`code/main.py` 构建一个玩具 ArtPrompt。你可以用 ASCII 艺术字形掩盖有害查询中的特定词，验证伪装字符串通过关键词过滤器，并（可选地）使用简单识别器将伪装字符串解码回来。

## 交付产出

本课程生成 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它枚举涵盖的编码攻击家族（ASCII 艺术、base64、leet-speak、UTF-8 同形字、UTES）以及捕获每种的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装字符串通过简单关键词过滤器。报告所需的字符级更改。

2. 实现第二种编码：同一目标词的 base64。比较过滤器绕过率与 ArtPrompt 以及恢复难度。

3. 阅读 Jiang 等人 2024 年第 4.3 节（五模型结果）。提出一个原因说明为什么 Claude 在相同基准上对 ArtPrompt 的抵抗力高于 Gemini。

4. 设计一种预生成防御，检测提示中 ASCII 艺术形状的区域。测量在合法代码、表格和数学符号上的误报率。

5. StructuralSleight 列出了 10 种编码结构。勾画一种处理所有 10 种的通用防御，并估计每个受防御提示的计算成本。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| ArtPrompt | "ASCII 艺术攻击" | 用 ASCII 艺术渲染掩盖安全词的两步越狱 |
| 掩盖 | "隐藏词" | 用模型可读但过滤器不可读的视觉表示替换被禁止的令牌 |
| UTES | "不常见结构" | 不常见文本编码结构——树、图、嵌套 JSON 等，用于偷运内容 |
| ViTC | "视觉文本能力" | 模型阅读非语义视觉编码能力的基准 |
| 困惑度过滤器 | "PPL 防御" | 拒绝高困惑度提示；失败是因为合法结构化输入也得分高 |
| 重新分词 | "分词器偏移防御" | 用不同分词器预处理提示；失败是因为识别是视觉的 |
| 同形字 | "相似字符" | 与拉丁字母看起来相同的 Unicode 字符；绕过子串检查 |

## 延伸阅读

- [Jiang 等 — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li 等 — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 泛化
- [Chao 等 — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补迭代攻击
- [Anil 等 — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补长度攻击
