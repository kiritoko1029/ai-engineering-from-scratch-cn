# LangGraph —— 面向 Agent 的状态机

> 手写的 ReAct 循环就是一个 `while True`。用 LangGraph 写出的 ReAct 循环则是一张图，你可以对它做 checkpoint、中断、分支以及时间回溯。Agent 本身没有变，变的是包裹在它周围的运行框架（harness）。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 11 · 09（Function Calling）、Phase 11 · 14（Model Context Protocol）
**所需时间：** 约75分钟

## 问题所在

你交付了一个 function-calling agent。它跑了三轮都没问题，然后出岔子了：模型调用的某个工具返回了 500，用户在任务进行到一半时改了主意，或者 agent 决定在没有人类签字确认的情况下给一笔订单退款。`while True:` 循环没有任何钩子。你没法暂停它，没法回退它，也没法分支出去探究「要是模型当初选了另一个工具会怎样」。一旦你把它推到 demo 之外，这个 agent 就变成了一个黑盒：要么成功，要么失败，没有中间地带。

下一步一旦看清就显而易见了。Agent 本身已经是一个状态机了——系统提示加上消息历史，加上待处理的工具调用，再加上下一个动作。把这个状态机显式化：用节点表示「模型在思考」「工具在运行」「人类在审批」，用边表示它们之间的条件转移。一旦图被显式化，运行框架就能免费获得四样东西：checkpointing（在步骤之间保存状态）、interrupts（为人类暂停）、streaming（流式输出 token 和中间事件），以及 time-travel（回退到先前的状态并尝试不同的分支）。

LangGraph 就是交付这层抽象的库。它不是 LangChain 意义上的那种 agent 框架（「这是一个 AgentExecutor，祝你好运」）。它是一个图运行时（graph runtime），具备一流的状态、一流的持久化和一流的中断能力。Agent 循环是你画出来的东西，而不是你手写出来的东西。

## 概念说明

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 由三样东西构成。

1. **State（状态）。** 一个带类型的字典（TypedDict 或 Pydantic model），它在图中流动。每个节点都接收完整的状态，并返回一份局部更新，LangGraph 会按字段使用一个 *reducer* 来合并这份更新——对于应当累积的列表使用 `operator.add`，默认则为覆盖。
2. **Nodes（节点）。** 形如 `state -> partial_state` 的 Python 函数。每个节点是一个离散的步骤：「调用模型」「运行工具」「做总结」。
3. **Edges（边）。** 节点之间的转移。静态边只指向一个地方。条件边则接收一个路由函数 `state -> next_node_name`，使图能够根据模型输出进行分支。

你需要编译这张图。编译会绑定拓扑结构，挂上一个 checkpointer（可选，但对生产环境至关重要），并返回一个可运行对象（runnable）。你用一个初始状态和一个 `thread_id` 来调用它。执行的每一步都会持久化一个 checkpoint，其键为 `(thread_id, checkpoint_id)`。

### 四大超能力

**Checkpointing（检查点）。** 每次节点转移都会把新状态写入一个存储（测试用内存，生产用 Postgres/Redis/SQLite）。要恢复执行，只需用相同的 `thread_id` 再次调用图。图会从它暂停的地方接着跑。

**Interrupts（中断）。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行就会在该节点运行之前停下。状态被持久化保存。你的 API 给用户回复「等待审批中」。之后对同一个 `thread_id` 发起的请求只要带上 `Command(resume=...)` 就能恢复执行。

**Streaming（流式）。** `graph.stream(state, mode="updates")` 会在状态增量发生时逐个产出（yield）它们。`mode="messages"` 会流式输出模型节点内部的 LLM token。`mode="values"` 则产出完整快照。你来决定在 UI 中呈现哪一种。

**Time-travel（时间回溯）。** `graph.get_state_history(thread_id)` 返回完整的 checkpoint 日志。把任意一个先前的 `checkpoint_id` 传给 `graph.invoke`，你就能从那个点分叉出去。这对调试（「要是模型当初选了 tool B 会怎样？」）以及对重放生产环境轨迹的回归测试都非常有用。

### Reducer 才是关键

每个状态字段都有一个 reducer。大多数默认值都没问题——新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息才会追加而不是替换。并行边通过 reducer 合并它们的更新。如果两个节点都更新 `messages`，而你忘了写 `Annotated[list, add_messages]`，那么后者会悄无声息地胜出，你就丢掉了半轮对话。reducer 是这个库里唯一微妙的东西；把它弄对了，其余部分自然就组合起来了。

### 四个节点搭出 ReAct 图

一个生产级的 ReAct agent 就是四个节点加两条边：

1. `agent` —— 用当前的消息历史调用 LLM。返回 assistant 消息（其中可能包含 tool_calls）。
2. `tools` —— 执行最后一条 assistant 消息中的所有 tool_calls，把工具结果作为 tool 消息追加上去。
3. 从 `agent` 出发的一条条件边：如果最后一条消息含有 tool_calls 就路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的一条静态边。

就这样。你就得到了完整的 ReAct 循环（Thought → Action → Observation → Thought → …），带有 checkpointing、interrupts 和 streaming，大约只需 40 行代码。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让一个节点能够派发并行的子图。例如：agent 决定同时查询三个 retriever。每个 `Send` 都会派生一次目标节点的并行执行；它们的输出通过状态 reducer 合并。这就是 LangGraph 在不动用线程原语的情况下表达 orchestrator-workers 模式的方式。

### 子图（Subgraphs）

一张已编译的图可以作为另一张图中的一个节点。外层图只看到一个节点；内层图有自己的状态和自己的 checkpoint。团队就是这样构建 supervisor-worker agent 的：supervisor 图把用户意图路由到各个领域专属的 worker 子图。

## 开始构建

### 第 1 步：状态与节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 就是那个让消息列表累积而非覆盖的 reducer。忘记它是最常见的 LangGraph bug。

### 第 2 步：带 thread 运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以把这些流式推送到 UI，让用户看到「agent 正在思考……正在调用 search_web……拿到结果……正在作答」。

### 第 3 步：加入 human-in-the-loop 中断

标记一个节点，让执行在它运行之前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、checkpoint 和 thread 在中断期间全都持久化保存。除了执行过程之外，没有任何东西只存在于内存里。

### 第 4 步：用时间回溯来调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

把 `None` 作为输入传入会从给定的 checkpoint 开始重放；传入一个值则会在恢复执行之前，把它作为一份更新追加到该 checkpoint 的状态上。这就是你在不重跑整段对话的情况下复现一次糟糕的 agent 运行的方法。

### 第 5 步：为生产环境替换 checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都是开箱即用的。`MemorySaver` 只用于测试。任何需要跨重启保持的场景都需要一个真正的存储。

## 技能要点

> 你把 agent 构建成图，而不是 `while True` 循环。

在你动手用 LangGraph 之前，先做一个 60 秒的设计：

1. **给节点命名。** 每一个离散的决策或带副作用的动作都是一个节点。「Agent 思考」「工具运行」「审核者审批」「响应流式输出」。如果你列不出这些节点，那这个任务还没有成型为 agent 的形态。
2. **声明状态。** 用一个极简的 TypedDict，给每个列表字段都配上一个 reducer。不要把所有东西都塞进 `messages`；把任务专属的字段（一个正在生效的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画出边。** 默认用静态边，除非下一步取决于模型输出。每条条件边都需要一个带具名分支的路由函数。
4. **一开始就选好 checkpointer。** 测试用 `MemorySaver`，其余一切都用 Postgres/Redis/SQLite。没有它就不要交付——没有 checkpointer 就意味着没有恢复、没有中断、没有时间回溯。
5. **在工具运行之前决定中断，而不是之后。** 审批放在通向带副作用节点的那条入边上，这样你就能在造成伤害之前取消；校验放在从模型出来的那条出边上，这样你就能低成本地拒掉糟糕的调用。
6. **默认开启流式。** UI 用 `mode="updates"`，模型节点内部的 token 级流式用 `mode="messages"`，评估期间的完整快照用 `mode="values"`。

拒绝交付一个没有 checkpointer 的 LangGraph agent。拒绝交付一个在副作用*之后*才中断的 agent。拒绝交付一个 `messages` 字段没有把 `add_messages` 作为其 reducer 的 agent。

## 练习

1. **简单。** 用一个计算器工具和一个网页搜索工具，实现上面那个四节点的 ReAct 图。验证对于一段两轮的对话，`list(app.get_state_history(config))` 至少返回四个 checkpoint。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，它把一个结构化的 `plan: list[str]` 写入状态。让 `agent` 把计划步骤标记为已完成。如果 `plan` 在一次 checkpoint 恢复中丢失（用错了 reducer），就让测试失败。
3. **困难。** 构建一个 supervisor 图，用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图都有自己的状态和 checkpointer。在外层图上加一个 `interrupt_before=["writer"]`，这样人类就可以审批研究简报。确认从先前某个 checkpoint 的时间回溯只会重跑被分叉出来的那条分支。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|-----------------|-----------------------|
| StateGraph | 「那张 LangGraph 图」 | 在编译之前你往上添加节点和边的那个构建器对象。 |
| Reducer | 「字段怎么合并」 | 一个函数 `(old, new) -> merged`，当某个节点为某字段返回更新时应用；默认是覆盖，`add_messages` 则是追加。 |
| Thread | 「一个对话 ID」 | 一个 `thread_id` 字符串，它为一个会话的所有 checkpoint 划定作用域。 |
| Checkpoint | 「一个被暂停的状态」 | 在一次节点转移之后，对完整图状态的一份持久化快照，键为 `(thread_id, checkpoint_id)`。 |
| Interrupt | 「为人类暂停」 | `interrupt_before` / `interrupt_after` 在某个节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | 「从先前某一步分叉」 | `graph.invoke(None, config_with_old_checkpoint_id)` 从那个 checkpoint 起向前重放。 |
| Send | 「并行子图派发」 | 一个构造器，节点可以返回它来派生 N 次目标节点的并行执行。 |
| Subgraph | 「作为节点的一张已编译图」 | 一张被用作另一张图中节点的、已编译的 StateGraph；保留它自己的状态作用域。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— 关于 StateGraph、reducer、checkpointer 和 interrupt 的权威参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) —— 本课所用的心智模型，直接出自源头。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) —— 关于 Postgres/SQLite/Redis 存储、checkpoint 命名空间和 thread ID 的细节。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) —— `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态的模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个 LangGraph agent 都实现的那个模式；读它是为了理解推理轨迹（reasoning trace）背后的原理。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 该优先选择哪些图形态（chain、router、orchestrator-workers、evaluator-optimizer）以及何时选择它们。
- Phase 11 · 09（Function Calling）—— 每个 LangGraph agent 节点都复用的工具调用原语。
- Phase 11 · 14（Model Context Protocol）—— 通过 MCP adapter 接入 LangGraph `ToolNode` 的外部工具发现机制。
- Phase 11 · 17（Agent framework tradeoffs）—— 何时该选 LangGraph 而非 CrewAI、AutoGen 或 Agno。
