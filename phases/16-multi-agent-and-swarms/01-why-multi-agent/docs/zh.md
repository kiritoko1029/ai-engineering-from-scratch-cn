# 为什么需要多智能体？

> 一个智能体撞了墙。聪明的做法不是造一个更大的智能体——而是造更多智能体。

**类型：** 学习
**语言：** TypeScript
**前置课程：** 第 14 阶段（智能体工程）
**时间：** 约 60 分钟

## 学习目标

- 识别单智能体的天花板（上下文溢出、专业能力混合、顺序瓶颈）并解释何时拆分为多智能体是正确的选择
- 比较编排模式（流水线、并行扇出、监督者、层级式）并为给定的任务结构选择合适的模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简洁性之间的权衡

## 问题所在

你在第 14 阶段构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实的代码库：200 个文件、三种语言、依赖基础设施的测试，以及需要先研究外部 API 再编写代码的需求。

智能体卡住了。不是因为 LLM 笨，而是因为任务超出了单个智能体循环所能处理的范围。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用之前读过的内容。它试图同时充当研究员、编码者和审查者，结果三件事都做得很差。

这就是单智能体的天花板。每当任务需要以下条件时，你就会碰到它：

- **超出单个窗口容量的上下文** - 读取 50 个文件会轻松超过 200k token
- **不同阶段需要不同专业能力** - 研究需要的提示词与代码生成不同
- **可以并行执行的工作** - 既然可以同时读取三个文件，为什么还要按顺序读？

## 概念

### 单智能体的天花板

单智能体是一个循环、一个上下文窗口、一个系统提示词。想象一下：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三件事会崩溃：

1. **上下文饱和** - 工具结果不断堆积。到第 30 轮时，智能体已经消耗了 150k token 的文件内容、命令输出和先前的推理。第 5 轮的关键细节丢失了。

2. **角色混乱** - 一个系统提示词说"你是研究员、编码者、审查者和测试者"，会产生一个半研究、半编码、永远完不成审查的智能体。

3. **顺序瓶颈** - 智能体先读文件 A，再读文件 B，再读文件 C。三次串行 LLM 调用。三次串行工具执行。没有并行。

### 多智能体解决方案

拆分工作。给每个智能体一项任务、一个上下文窗口和一个针对该任务调优的系统提示词：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个智能体都有：
- 一个专注的系统提示词（"你是一位代码审查者。你的唯一任务是查找 bug。"）
- 自己的上下文窗口（不会被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 真实系统中的实践

**Claude Code 子智能体** - 当 Claude Code 使用 `Task` 生成子智能体时，它会创建一个具有作用域任务的子智能体。父智能体保持上下文干净。子智能体执行专注的工作并返回摘要。

**Devin** - 运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划智能体将工作分解为步骤。编码智能体编写代码。浏览器智能体研究文档。每个都有独立的上下文。

**多智能体编码团队（SWE-bench）** - SWE-bench 上表现最好的系统使用一个研究员阅读代码库、一个规划者设计修复方案、一个编码者实现它。单智能体系统得分更低。

**ChatGPT Deep Research** - 并行生成多个搜索智能体，每个探索不同角度，然后综合结果。

### 光谱

多智能体不是二元的。它是一个光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单智能体** - 一个循环、一个提示词。适合简单任务。

**子智能体** - 父智能体生成子智能体处理专注的子任务。父智能体维护计划。子智能体汇报结果。这就是 Claude Code 的做法。

**流水线** - 智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适合分阶段工作流：研究 -> 编码 -> 审查 -> 测试。

**团队** - 智能体通过共享消息总线并行运行。每个都有角色。编排器协调。适合同时需要不同技能的场景。

**群体** - 许多相同或近似的智能体共享状态。没有固定的编排器。智能体从队列中获取工作。适合高吞吐量并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个智能体转换数据并向前传递。易于理解。一个阶段的失败会阻塞其余阶段。

#### 模式 2：扇出 / 扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将工作拆分给并行智能体，然后合并结果。适合可分解为独立子任务的任务。

#### 模式 3：编排器-工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个智能的编排器决定做什么，委派给工作者，并综合结果。编排器本身是一个具有生成工作者工具的智能体。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。智能体点对点通信。决策从交互中涌现。更难调试，但可扩展到大量智能体。

### 何时不该使用多智能体

多智能体会增加复杂性。智能体之间的每条消息都是潜在的故障点。调试从"阅读一个对话"变成了"在五个智能体之间追踪消息"。

**保持单智能体的情况：**
- 任务能装进一个上下文窗口（工作数据低于约 100k token）
- 不同阶段不需要不同的系统提示词
- 顺序执行足够快
- 任务足够简单，拆分带来的开销大于收益

**复杂性成本：**
- 每个智能体边界都是有损压缩步骤：智能体 A 的完整上下文被摘要为一条消息发送给智能体 B
- 协调逻辑（谁做什么、何时做、按什么顺序做）本身就是 bug 来源
- 延迟增加：N 个智能体意味着至少 N 次串行 LLM 调用，如果需要来回通信则更多
- 成本倍增：每个智能体独立消耗 token

经验法则：如果一个任务需要少于 20 次工具调用且能装进 100k token，就保持单智能体。

```figure
swarm-messages
```

## 动手构建

### 步骤 1：过载的单智能体

这是一个试图做所有事情的单智能体。它有一个庞大的系统提示词和一个存放研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随每个阶段增长。到审查步骤时，它包含研究笔记、代码和先前的推理。
- 系统提示词是通用的。无法针对每个阶段调优。
- 没有并行执行。

### 步骤 2：专家智能体

现在拆分它。每个智能体负责一项任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有专注的提示词。每个都获得一个干净的上下文窗口，只包含它需要的输入。

### 步骤 3：通过消息协调

用显式消息传递将专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发送给它的消息。没有上下文污染。研究员的 50k token 文档阅读永远不会进入审查者的上下文。

### 步骤 4：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用更多的总 token（三个智能体、三次独立 LLM 调用），但每个智能体的上下文保持干净。每个阶段的质量都因为系统提示词的专业化而提升。

## 应用

本课生成了一个可复用的提示词，用于决定何时采用多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家："测试者"智能体，接收编码者的代码和审查者的审查反馈，然后编写测试
2. 修改流水线，使审查者可以将反馈发回给编码者进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出：并行运行研究员和"需求分析"智能体，然后在传递给编码者之前合并它们的输出

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 群体（Swarm） | "AI 智能体的蜂群思维" | 一组具有共享状态且没有固定领导者的对等智能体。行为从局部交互中涌现。 |
| 编排器（Orchestrator） | "老板智能体" | 一个工具包含生成和管理其他智能体的智能体。它规划和委派，但不一定执行实际工作。 |
| 协调器（Coordinator） | "交通警察" | 一个非智能体组件（通常只是代码，不是 LLM），根据规则在智能体之间路由消息。 |
| 共识（Consensus） | "智能体们达成一致" | 一种协议，多个智能体必须在继续之前达成一致。用于需要解决冲突输出的场景。 |
| 涌现行为（Emergent behavior） | "智能体们自己想出来的" | 从智能体交互中产生的系统级模式，但并非显式编程的。可能有益也可能有害。 |
| 扇出 / 扇入（Fan-out / fan-in） | "智能体的 Map-Reduce" | 将任务拆分给并行智能体（扇出），然后合并它们的结果（扇入）。 |
| 消息传递（Message passing） | "智能体之间互相交谈" | 智能体之间的通信机制：从一个智能体发送到另一个智能体的结构化数据，取代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 微软的多智能体对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何通过 Task 进行委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多智能体框架
