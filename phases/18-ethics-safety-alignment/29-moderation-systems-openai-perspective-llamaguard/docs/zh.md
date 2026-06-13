# 内容审核系统 —— OpenAI、Perspective、Llama Guard

> 生产内容审核系统将第 12-16 课中定义的安全政策付诸实践。OpenAI Moderation API：`omni-moderation-latest`（2024 年）基于 GPT-4o 构建，一次调用即可分类文本+图像；在多语言测试集上比上一版本提升 42%；响应模式返回 13 个类别布尔值——骚扰、骚扰/威胁、仇恨、仇恨/威胁、违法、违法/暴力、自残、自残/意图、自残/指导、性、性/未成年人、暴力、暴力/血腥；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时使用占位符响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危险类别，代码解释器滥用，8 种语言（v3），多图像（v4）。Perspective API（Google Jigsaw）：在 LLM 作为审核者浪潮之前的毒性评分；主要是单维度毒性，附带严重毒性/侮辱/亵渎变体；内容审核研究的基线。弃用：Azure Content Moderator 于 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI Content Safety 替代。

**类型：** 构建
**语言：** Python（标准库、三层审核框架）
**前置要求：** 第 18 阶段 · 16（Llama Guard / Garak / PyRIT）
**所需时间：** 约60分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别分类法及其与 Llama Guard 3 的 MLCommons 集合的区别。
- 描述三层审核模式（输入、输出、自定义），并指出每层的一个失败模式。
- 描述 Perspective API 作为 LLM 时代之前基线的定位及其在研究中仍被使用的原因。
- 说明 Azure 的弃用时间表。

## 问题所在

第 12-16 课描述了攻击和防御工具。第 29 课涵盖了在用户接触产品的界面上将防御付诸实践的已部署审核系统。三层模式是 2026 年的默认配置。

## 概念说明

### OpenAI Moderation API

`omni-moderation-latest`（2024 年）。基于 GPT-4o 构建。一次调用即可分类文本+图像。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- 骚扰、骚扰/威胁
- 仇恨、仇恨/威胁
- 自残、自残/意图、自残/指导
- 性、性/未成年人
- 暴力、暴力/血腥
- 违法、违法/暴力

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不包括 `sexual/minors`；其余为纯文本。

在 `code/main.py` 的代码框架中，为了教学简洁，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别合并到其顶级父类别中。生产代码应使用完整的 13 类别模式。

在多语言测试集上比上一代审核端点提升 42%。逐类别分数；应用程序设置阈值。

### Llama Guard 3/4

在第 16 课中介绍。14 个 MLCommons 危险类别（组织方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是原生多模态，12B 参数。

OpenAI 和 Llama Guard 的分类法有重叠但有分歧。OpenAI 有"违法"作为宽泛类别；Llama Guard 将"暴力犯罪"和"非暴力犯罪"分开。部署根据其政策分类法适配来选择。

### Perspective API（Google Jigsaw）

在 LLM 作为审核者浪潮之前的毒性评分系统（2020 年之前）。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维度主分数（TOXICITY）附带子维度变体。

作为内容审核研究基线被广泛使用，因为该 API 稳定、有文档、有多年的校准数据。对于现代 LLM 相关用例，Llama Guard 或 OpenAI Moderation 通常是更好的选择。

### 三层模式

1. **输入审核。** 在生成前对用户提示进行分类。如果被标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前对模型输出进行分类。如果被标记则替换为拒绝。延迟：生成后一次分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、白名单、业务策略）。在输入或输出时运行。

三层按设计顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。并行性适用于层内——对同一文本并发运行多个分类器（如 OpenAI Moderation + Llama Guard + Perspective）可以隐藏每个分类器的延迟。作为可选优化，可以在输入审核完成时显示占位符响应（"请稍候，正在检查..."），并延迟 token-1 的流式传输。标记行为可配置：拒绝、消毒、升级到人工审核。

### 失败模式

- **仅输入。** 不捕获输出幻觉（第 12-14 课的编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；向攻击者暴露内部推理。
- **仅自定义。** 跨类别不鲁棒；正则表达式很脆弱。

分层是默认配置。双重保险。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI Content Safety 替代，后者基于 LLM 并与 Azure OpenAI 集成。迁移是 2024-2027 年 Azure 部署的领域级项目。

### 本课在第 18 阶段中的位置

第 16 课涵盖红队背景下的审核工具。第 29 课涵盖已部署的审核。第 30 课以当前的双重用途能力证据收尾。

## 开始构建

`code/main.py` 构建了一个三层审核框架：输入审核器（关键词+类别分数）、输出审核器（对输出使用相同的分类器）、自定义审核器（领域规则）。你可以运行输入并观察哪一层捕获了什么。

## 交付成果

本课生成 `outputs/skill-moderation-stack.md`。给定一个部署，它会推荐审核栈配置：输入端使用哪个分类器，输出端使用哪个，哪些自定义规则，以及边缘案例使用哪个裁判。

## 练习

1. 运行 `code/main.py`。对一个良性、一个边界和一个有害输入运行所有三层。报告每层对每个输入的触发情况。

2. 扩展框架，在特定类别上加入 Perspective API 风格的毒性评分。将其阈值行为与类别分数进行比较。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法干净映射的类别。

4. 为代码助手部署（如 GitHub Copilot）设计审核栈。找出最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 于 2027 年 2 月退役。规划迁移到 Azure AI Content Safety。找出迁移中风险最高的元素。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，部分支持多模态 |
| Perspective API | "Google Jigsaw 毒性" | LLM 时代之前的毒性评分基线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危险分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| 输入审核 | "生成前过滤器" | 在模型调用前对用户提示运行的分类器 |
| 输出审核 | "生成后过滤器" | 在交付前对模型输出运行的分类器 |
| 自定义审核 | "领域规则" | 部署特定规则（正则表达式、白名单、策略） |
| 分层审核 | "三层都用" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) —— omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) —— Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) —— 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) —— Azure 替代方案
