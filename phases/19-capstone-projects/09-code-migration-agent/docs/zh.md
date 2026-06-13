# 毕业项目 09 — 代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移器设定了 2026 年的标准。Moderne 的 OpenRewrite 大规模执行确定性 AST 重写。Grit 用 codemod 风格的 DSL 解决同一问题。生产模式结合了两者：安全重写的确定性底层加处理模糊情况的智能体层、每分支构建的沙箱、以及在 PR 打开前变绿的测试框架。毕业项目是迁移 50 个真实仓库并发布通过率和失败分类。

**类型：** 毕业项目
**语言：** Python（智能体）、Java / Python（目标）、TypeScript（仪表盘）
**前置要求：** 阶段 5（NLP）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 15（自主系统）、阶段 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**所需时间：** 30 小时

## 问题所在

大规模代码迁移是 2026 年编码智能体最清晰的生产应用之一。真实标准显而易见（迁移后测试套件是否通过？），收益真实（Java 8 机群迁移是人力级别的项目），且基准公开（MigrationBench 50 仓库子集）。Moderne 的 OpenRewrite 处理确定性部分。智能体层处理 OpenRewrite 配方无法处理的一切：模糊重写、构建系统漂移、长尾语法、传递依赖破坏。

你将构建一个智能体，接收 Java 8 仓库（或 Python 2 仓库）并生成 CI 变绿的迁移分支。你将衡量通过率、测试覆盖率保持、每仓库成本，并构建失败分类。与纯确定性基线的并排对比告诉你智能体的价值实际在哪里。

## 概念说明

流水线有两层。**确定性底层**（Java 用 OpenRewrite，Python 用 libcst）安全地执行大部分机械重写：导入、方法签名、空安全编辑、try-with-resources、弃用 API 替换。它快速且产生可审计的 diff。**智能体层**（OpenAI Agents SDK 或 LangGraph，基于 Claude Opus 4.7 和 GPT-5.4-Codex）处理配方无法处理的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试闪烁、自定义注解。

每个仓库获得一个预装目标运行时的 Daytona 沙箱。智能体迭代：运行构建、分类失败、应用修复、重跑。硬性限制：每仓库 30 分钟、每仓库 $8、20 轮智能体。如果所有测试通过且覆盖率差异非负，分支打开 PR。如果不通过，仓库归入某个失败类别并附带证据。

失败分类是交付成果。跨 50 个仓库，什么出了问题？传递依赖？自定义注解？构建工具版本？与迁移无关的测试闪烁？每个类别有一个计数和一个示例 diff。未来的配方作者可以针对前三名。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性底层：OpenRewrite（Java）或 libcst（Python）
- 智能体：OpenAI Agents SDK 或 LangGraph，基于 Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：Daytona devcontainers，每分支，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50 仓库子集（Java 8 到 17）、Google App Engine Py2-to-Py3 仓库
- 测试框架：并行运行器，Jacoco（Java）或 coverage.py（Python）覆盖
- 可观测性：Langfuse + 每仓库追踪包，含每个 diff 块
- 仪表盘：失败分类仪表盘，含每类别计数和示例 diff

## 开始构建

1. **配方遍历。** 先运行 OpenRewrite（Java）或 libcst（Python）配方。捕获 70-80% 的机械迁移。提交为"recipe"提交。

2. **构建试验。** Daytona 沙箱：安装目标运行时，运行构建。如果变绿，跳到测试。如果变红，交给智能体。

3. **智能体循环。** LangGraph 带工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类失败（依赖、语法、测试、构建工具）并应用针对性修复。重跑。

4. **预算上限。** 每仓库 30 分钟墙钟时间、$8 成本、20 轮智能体。任何超限都会停止并以"budget_exhausted"归档当前 diff。

5. **测试 + 覆盖率门控。** 构建变绿后，运行测试套件。与基础仓库对比覆盖率。如果覆盖率下降超过 2%，归入"coverage_regression"。

6. **PR 打开。** 成功后，推送分支，打开 PR，附带 diff 和应用了哪些配方、智能体编写了哪些提交的摘要。

7. **失败分类。** 对每个失败仓库，标记类别：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50 仓库运行。** 在 MigrationBench 子集上执行。报告每类别通过率、每仓库成本、覆盖率保持，以及与纯确定性基线的对比。

## 使用示例

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付成果

`outputs/skill-migration-agent.md` 是交付成果。给定一个仓库，它执行确定性配方然后智能体循环以生成变绿的迁移分支，或按分类归档。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集 pass@1 |
| 20 | 测试覆盖率保持 | 与基础的平均覆盖率差异 |
| 20 | 每迁移仓库成本 | 通过运行的 $/仓库 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理的修复 vs 智能体编写的修复比例 |
| 15 | 失败分析报告 | 分类完整性，含示例 |
| **100** | | |

## 练习

1. 仅用 OpenRewrite（无智能体）运行迁移流水线。与完整流水线对比通过率。识别智能体单独产生差异的场景。

2. 实现"lint-clean"检查：迁移后，运行风格 linter（Java 用 spotless，Python 用 ruff）。如果出现新 lint 错误则 PR 失败。衡量覆盖率保持但风格回退的比率。

3. 添加"最小 diff"优化器：智能体分支通过测试后，用第二次遍历修剪不必要的更改。报告 diff 大小缩减。

4. 扩展到第三种迁移：Node 18 到 Node 22。复用沙箱封装；将配方层替换为自定义 codemod。

5. 衡量首次变绿构建时间（TTFGB）作为 UX 指标。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 确定性底层 | "配方引擎" | OpenRewrite / libcst：带安全保证的声明式 AST 重写 |
| Codemod | "代码修改程序" | 机械地更改源代码的重写规则 |
| 构建漂移 | "工具版本偏差" | Maven / Gradle / uv 主版本间的微妙行为变化 |
| 失败类别 | "分类桶" | 仓库未迁移的标注原因：依赖、语法、测试、构建工具、预算 |
| 覆盖率差异 | "覆盖率保持" | 从基础到迁移分支的测试覆盖率 % 变化 |
| 智能体轮次 | "工具调用轮" | 智能体循环中的一次规划 -> 执行 -> 观察 |
| 预算耗尽 | "触顶" | 仓库在 30 分钟/$8/20 轮限制内未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年权威基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性底层参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 配方编写
- [Grit.io](https://www.grit.io) — 备选 codemod DSL
- [OpenAI 沙箱化迁移手册](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 to Py3 迁移器](https://cloud.google.com/appengine) — 备选迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性底层
- [Daytona sandboxes](https://daytona.io) — 每分支沙箱参考
