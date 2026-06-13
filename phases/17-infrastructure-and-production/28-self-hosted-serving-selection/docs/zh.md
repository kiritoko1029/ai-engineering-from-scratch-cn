# 自托管推理选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年，四大引擎主导自托管推理领域。根据硬件、规模和生态进行选择。**llama.cpp** 在 CPU 上速度最快 — 支持最广泛的模型，可完全控制量化和线程配置。**Ollama** 是开发笔记本上一条命令即可安装的方案，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化开销），在类生产负载下吞吐量差距达 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式** — 仅修复 Bug，原始吞吐量比 vLLM 慢约 10%，但历史上拥有顶级的可观测性和 HF 生态集成能力。维护状态使其成为长期选择的风险项 — 对于新项目，SGLang 或 vLLM 是更安全的默认选项。**vLLM** 是通用生产环境的默认选择 — v0.15.1（2026 年 2 月）新增 PyTorch 2.10、RTX Blackwell SM120、H200 优化支持。**SGLang** 是多轮智能体 / 前缀密集型场景的专家 — 生产环境中运行超过 400,000 块 GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 只能选 llama.cpp。AMD / 非 NVIDIA → 只能选 vLLM（TRT-LLM 仅限 NVIDIA）。2026 年流水线模式：开发 = Ollama，预发布 = llama.cpp，生产 = vLLM 或 SGLang。全程使用同一套 GGUF/HF 权重。

**类型：** 学习
**语言：** Python（标准库，引擎决策树遍历器）
**前置要求：** 第 17 阶段所有涉及引擎的课程（04、06、07、09、18）
**所需时间：** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用对话 / 智能体 / 长上下文）选择推理引擎。
- 说明 2026 年 TGI 维护模式状态（2025 年 12 月 11 日）以及它为何使新项目倾向于选择 vLLM 或 SGLang。
- 描述开发/预发布/生产流水线，全程使用同一套 GGUF 或 HF 权重。
- 解释为什么"仅 CPU"场景必须选择 llama.cpp，以及"AMD"场景为何排除 TRT-LLM。

## 问题所在

你的团队启动了一个新的自托管 LLM 项目。一位工程师说用 Ollama，另一位说用 vLLM，第三位说"TGI 不是开箱即用吗？"三个人在各自的语境下都是对的，但没有任何一个引擎适用于所有场景。

2026 年，选择决策树至关重要：硬件优先，规模其次，工作负载再次。而 2025 年的一个特定事件 — TGI 于 12 月 11 日进入维护模式 — 改变了新项目的默认选项。

## 概念说明

### 五大引擎

| 引擎 | 最佳适用场景 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘设备 / 最小依赖 / 最广泛的模型支持 | CPU 上速度最快，完全可控 |
| **Ollama** | 开发笔记本、单用户、一条命令安装 | 比 llama.cpp 慢 15-30%；生产吞吐量差距达 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产环境、100+ 用户 | 广泛的生产默认选项；v0.15.1 于 2026 年 2 月发布 |
| **SGLang** | 多轮智能体、前缀密集型工作负载 | 生产环境中运行 400,000+ 块 GPU |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也能运行但速度更慢。在 CPU 上没有其他引擎具有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也能运行。TRT-LLM 仅限 NVIDIA，因此排除。

**NVIDIA Hopper（H100 / H200）** → vLLM 或 SGLang 或 TRT-LLM。三者均为顶级方案。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 吞吐量领先（第 17 阶段 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal 支持）。Ollama 在其基础上封装。

### 规模其次决策

**1 用户 / 本地开发** → Ollama。一条命令，秒级获得首个 Token。

**10-100 用户 / 小团队** → vLLM 单 GPU 方案。

**100-10,000 用户 / 生产环境** → vLLM 生产栈（第 17 阶段 · 18）或 SGLang。

**10,000+ 用户 / 企业级** → vLLM 生产栈 + 分离式架构（第 17 阶段 · 17）+ LMCache（第 17 阶段 · 18）。

### 工作负载再次决策

**通用对话 / 问答** → vLLM 凭借广泛的默认支持胜出。

**多轮智能体交互（工具调用、规划、记忆）** → SGLang 的 RadixAttention（第 17 阶段 · 06）占主导地位。

**重前缀复用的 RAG** → SGLang。

**代码生成** → vLLM 表现良好；SGLang 在缓存方面略胜一筹。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 的维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式 — 此后仅修复 Bug。历史上：顶级可观测性、业界最佳的 HF 生态集成（模型卡片、安全工具），原始吞吐量略低于 vLLM。

对于 2026 年的新项目：默认不再选择 TGI。现有的 TGI 部署可以继续运行，但最终应迁移。SGLang 和 vLLM 是更安全的默认选项。

### 流水线模式

开发（Ollama）→ 预发布（llama.cpp）→ 生产（vLLM）。全程使用同一套 GGUF 或 HF 权重。工程师在笔记本上快速迭代；预发布环境镜像生产环境的量化配置；生产环境是最终的服务目标。

### Ollama 的注意事项

Ollama 非常适合开发。但不适合共享生产环境：Go 的 HTTP 序列化会增加开销，并发管理比 vLLM 更简单，OpenTelemetry 支持也相对滞后。在 Ollama 擅长的场景中使用它 — 单用户、一条命令 — 共享场景则切换到 vLLM。

### 自托管与托管服务是另一个决策

第 17 阶段 · 01（托管超大规模云服务）、· 02（推理平台）涵盖托管服务。本课程假设你已决定自托管。自托管的理由包括：数据驻留要求、自定义微调、大规模场景下的总拥有成本、托管平台上不可用的领域模型。

### 应当记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产部署规模：400,000+ 块 GPU。
- Ollama 与 llama.cpp 的吞吐量差距：慢 15-30%；生产负载下差距达 3 倍。

```figure
data-parallel
```

## 开始使用

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择引擎并解释原因。

## 部署产出

本课程生成 `outputs/skill-engine-picker.md`。给定约束条件，选择引擎并编写迁移计划。

## 练习

1. 使用你的硬件 / 规模 / 工作负载运行 `code/main.py`。输出结果是否符合你的直觉？
2. 你的基础设施是 12 块 H100 和 8 块 MI300X AMD。应选择哪个引擎？为什么 TRT-LLM 不在考虑范围内？
3. 一个团队想在 2026 年使用 TGI，因为"这是我们熟悉的方案"。请论证迁移的理由。
4. 从 Ollama 开发环境切换到 vLLM 生产环境：量化配置、参数设置和可观测性方面有哪些变化？
5. 一个 RAG 产品，P99 前缀长度为 8K，且跨租户复用率高。选择一个引擎，并结合第 17 阶段 · 11 + 18 构建完整方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | "CPU 上的那个" | 最广泛的模型支持，CPU 上速度最快 |
| Ollama | "笔记本上的那个" | 一条命令安装，开发级吞吐量 |
| TGI | "HF 的推理服务" | 自 2025 年 12 月起进入维护模式 |
| vLLM | "默认选项" | 2026 年广泛的生产基线 |
| SGLang | "智能体那个" | 前缀密集型，RadixAttention |
| TRT-LLM | "仅限 NVIDIA" | Blackwell 吞吐量领先者，仅限 NVIDIA |
| GGUF | "llama.cpp 的格式" | 内置 K-quant 变体 |
| 生产栈 | "vLLM K8s" | 第 17 阶段 · 18 参考部署 |
| 流水线模式 | "开发→预发布→生产" | Ollama → llama.cpp → vLLM 使用相同权重 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — 发行说明。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
