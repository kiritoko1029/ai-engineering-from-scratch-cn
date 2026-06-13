# 构建 MCP 服务器 -- Python + TypeScript SDK

> 大多数 MCP 教程只展示了 stdio 的 hello-world 示例。一个真正的服务器需要暴露工具加上资源加上提示、处理能力协商、发出结构化错误，并且在不同 SDK 之间表现一致。本课端到端构建一个笔记服务器：标准库 stdio 传输、JSON-RPC 分发、三个服务器端原语、以及一种纯函数风格，可以在你升级时直接迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建
**语言：** Python（标准库，stdio MCP 服务器）
**前置要求：** 第 13 阶段第 06 课（MCP 基础）
**所需时间：** 约75分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个从 stdin 读取 JSON-RPC 消息并向 stdout 写入响应的分发循环。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化错误响应。
- 将标准库实现迁移到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## 问题所在

在你可以使用远程传输（第 13 阶段第 09 课）或认证层（第 13 阶段第 16 课）之前，你需要一个干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程启动，消息通过 stdin/stdout 以换行分隔的方式流动。

2025-11-25 规范规定 stdio 消息编码为 JSON 对象，以显式的 `\n` 分隔。这里没有 SSE；SSE 是旧的远程模式，将在 2026 年中期移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是全部线路格式。

笔记服务器是一个很好的形式，因为它涵盖了所有三个服务器端原语。工具执行变更操作（`notes_create`）。资源暴露数据（`notes://{id}`）。提示提供模板（`review_note`）。本课的形式可以泛化到任何领域。

## 概念说明

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要在 stdout 上打印任何不是 JSON-RPC 封装的内容。调试日志输出到 stderr。
- 每个请求都必须匹配一个携带相同 `id` 的响应。
- 通知不得被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的内容。客户端依赖能力集来门控功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目有 `name`、`description`、`inputSchema`。`tools/call` 接受 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是类型化的。最常见的：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、参数错误）是 JSON-RPC 错误。工具级错误（有效调用但工具失败）以 `{content: [...], isError: true}` 返回。这让模型可以在其上下文中看到失败。

### 实现资源

资源在设计上是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义方案如 `notes://`。

当你将数据暴露为资源而非工具时：

- 模型不会"调用"它；客户端可以在用户请求时将其注入上下文。
- 订阅让服务器在资源变更时推送更新（第 13 阶段第 10 课）。
- 第 13 阶段第 14 课扩展了 `ui://` 交互式资源。

### 实现提示

提示是带命名参数的模板。宿主将其作为斜杠命令展示。一个 `review_note` 提示可能接受一个 `note_id` 参数，并生成一个多消息提示模板供客户端馈送给其模型。

### stdio 传输细节

- 换行分隔的 JSON。没有长度前缀的帧。
- 不要缓冲。每次写入后执行 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 标注

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` -- 纯读取，可安全重试。
- `destructiveHint: true` -- 不可逆副作用；客户端应确认。
- `idempotentHint: true` -- 相同输入产生相同输出。
- `openWorldHint: true` -- 与外部系统交互。

客户端利用这些来决定 UX（确认对话框、状态指示器）和路由（第 13 阶段第 17 课）。

### 升级路径

`code/main.py` 中的标准库服务器约 180 行。FastMCP（Python）将同样的逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价的形式。升级路径在你准备就绪时可以直接替换；概念（能力、分发、内容块）是相同的。

## 开始构建

`code/main.py` 是一个完整的基于 stdio 的笔记 MCP 服务器，仅使用标准库。它处理 `initialize`、`tools/list`、`tools/call`（三个工具：`notes_list`、`notes_search`、`notes_create`）、`resources/list` 和 `resources/read`（每个笔记），以及一个 `review_note` 提示。你可以通过管道 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要关注的要点：

- 分发器是一个按方法名索引的 `dict[str, Callable]`。
- 每个工具执行器返回内容块列表，而非裸字符串。
- 当执行器抛出异常时设置 `isError: true`。

## 发布成果

本课生成 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能会搭建一个带有正确工具/资源/提示划分和 SDK 升级路径的 MCP 服务器。

## 练习

1. 运行 `code/main.py`，用手工构建的 JSON-RPC 消息驱动它。先执行 `notes_create`，然后用 `resources/read` 检索新笔记。

2. 添加一个 `notes_delete` 工具，标注 `annotations: {destructiveHint: true}`。验证客户端会展示确认对话框（这需要真正的宿主；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，使服务器在笔记被修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应缩减到 80 行以下。线路行为必须完全相同；用相同的 JSON-RPC 测试脚手架验证。

5. 阅读规范中 `server/tools` 部分，找出本课服务器未实现的工具定义中的一个字段。（提示：有几个可选；选一个并添加。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP 服务器 | "暴露工具的东西" | 通过 stdio 或 HTTP 调用 MCP JSON-RPC 的进程 |
| stdio 传输 | "子进程模型" | 服务器由客户端启动；通过 stdin/stdout 通信 |
| 分发器（Dispatcher） | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| 内容块（Content block） | "工具结果块" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级失败" | 信号工具失败；与 JSON-RPC 错误区分 |
| 标注（Annotations） | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议高层框架 |
| 资源 URI（Resource URI） | "可寻址数据" | `file://`、`db://` 或标识资源的自定义方案 |
| 提示模板（Prompt template） | "斜杠命令说明" | 服务器提供的带参数槽的模板，用于宿主 UI |
| 能力声明（Capability declaration） | "功能开关" | 在 `initialize` 中声明的每原语标志 |

## 延伸阅读

- [Model Context Protocol -- Python SDK](https://github.com/modelcontextprotocol/python-sdk) -- 参考 Python 实现
- [Model Context Protocol -- TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) -- 并行 TS 实现
- [FastMCP -- server framework](https://gofastmcp.com/) -- MCP 服务器的装饰器风格 Python API
- [MCP -- Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) -- 使用任一 SDK 的端到端教程
- [MCP -- Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) -- `tools/*` 消息的完整参考
