# RAG 的分块策略

> 分块配置对检索质量的影响与嵌入模型选择同等重要（Vectara NAACL 2025）。若分块方式有误，再多的重排序也无法挽救局面。

**类型：** 构建
**语言：** Python
**先修要求：** 第5阶段 · 14（信息检索）、第5阶段 · 22（嵌入模型）
**耗时：** 约60分钟

## 问题所在

你将一份50页的合同输入到RAG系统中。用户询问：“终止条款是什么？”检索器却返回了封面页。为何如此？因为该模型是基于512个标记的块进行训练的，而终止条款位于第20页，跨越了分页符，且周围没有能将其与查询关联的关键词。

解决办法并非“购买更优秀的嵌入模型”，而是合理分块。分块大小应设为多少？是否需要重叠？应在何处分割？是否要包含周边上下文？

2026年2月的测试结果令人意外：

- Vectara在2026年的研究显示：递归式512标记分块方法的准确率比语义分块方法高出69%，达到54%。
- 在处理自然语言问题时，使用SPLADE与Mistral-8B组合并未带来任何可测量的性能提升。
- 存在“上下文断崖”现象：当上下文长度超过2,500个标记时，响应质量会急剧下降。

那些看似“显而易见”的方案（语义分块、20%的重叠率、1,000个标记）往往并不正确。本课程将帮助你建立对六种分块策略的直觉，并指导你在何种情况下选择使用哪种策略。

## 概念概述

![在一个段落上展示的六种分块策略](../assets/chunking.svg)

**固定长度分块。** 每 N 个字符或标记进行分割。最简单的基准方法。会在句子中间断开。压缩效果较好，但连贯性较差。

**递归分块。** 使用 LangChain 的 `RecursiveCharacterTextSplitter`。首先尝试按 `\n\n` 分割，然后是 `\n`，接着是 `.`，最后是空格。若失败则自动回退到其他分割方式。这是 2026 年的默认方案。

**语义分块。** 对每个句子进行嵌入处理，计算相邻句子之间的余弦相似度。当相似度低于特定阈值时进行分割。能够保持主题连贯性。但速度较慢；有时会产生仅包含 40 个标记的极小片段，从而影响检索效果。

**按句分块。** 在句子边界处进行分割。每个分块包含一个句子或 N 句组成的窗口。其效果与语义分块类似，可在成本大幅降低的情况下处理多达约 5k 个标记的文本。

**父文档分块。** 同时存储用于检索的小型子分块以及用于提供上下文的较大父分块。先通过子分块进行检索，再返回对应的父分块。即便子分块质量较差，也能依然得到相对合理的父分块结果。

**延迟分块（2024 年）。** 首先在标记层面对整个文档进行嵌入，然后将这些标记嵌入汇总为分块嵌入。能够保留跨分块的上下文信息。适用于长上下文嵌入模型（如 BGE-M3、Jina v3）。但计算成本较高。

**上下文感知检索（Anthropic，2024 年）。** 在每个分块的前面添加由大语言模型生成的关于其在文档中位置的摘要（例如“该分块属于终止条款的第 3.2 节……”）。在 Anthropic 自身的基准测试中，该方法可使检索准确率提升 35% 至 50%。但构建索引的成本较高。

### 能够击败所有默认规则的规则

根据查询类型匹配分块大小：

| 查询类型 | 分块大小 |
|------------|-----------|
| 事实类查询（如“CEO的名字是什么？”） | 256-512 个标记 |
| 分析型/多跳查询 | 512-1024 个标记 |
| 整节内容理解 | 1024-2048 个标记 |

基于 NVIDIA 的 2026 年基准标准。分块大小应足够大以包含答案及相关局部上下文，同时又需足够小，以便检索器的 Top-K 结果能够聚焦于答案而非背景噪声。

## 构建它

### 步骤 1：固定分块与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

根据您的领域调整 `threshold` 值。数值过高会导致数据碎片化；数值过低则会产生一个巨大的数据块。

### 步骤 3：父文档

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：去重父节点。多个子节点可能对应同一个父节点；返回所有子节点会浪费上下文资源。

### 第 4 步：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

对经过上下文标注的文本块进行索引。在查询时，这些额外的周围信号有助于提升检索效果。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

务必进行基准测试。针对您的语料库而言，“最佳”策略可能并不适用于任何博客文章。

## 常见陷阱

- **分块功能仅在事实查询上生效。** 多跳查询的优胜方案则截然不同，应使用按查询类型划分的评估集。
- **无最小长度限制的语义分块。** 生成的40个词元片段会损害检索效果，务必设置 `min_tokens` 参数。
- **过度重叠毫无意义。** 2026年的研究指出，过度重叠往往不会带来任何好处，还会使索引成本翻倍，需通过测量而非猜测来处理。
- **缺乏最小/最大长度约束。** 5个词元或5000个词元的片段都可能破坏检索效果，必须进行限制。
- **跨文档分块不可行。** 绝不允许一个分块跨越两个文档，应先对每个文档单独分块，再合并结果。

## 使用它

2026年技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建，语料未知 | 递归分割，512个标记，无重叠 |
| 事实问答 | 递归分割，256–512个标记 |
| 分析型/多跳查询 | 递归分割，512–1024个标记 + 上层文档信息 |
| 大量交叉引用（合同、论文） | 后期分块或上下文检索 |
| 对话式语料 | 每轮对话作为一个分块 + 发言者元数据 |
| 短文本（推文、评论） | 单个文档对应一个分块 |

初始采用512标记的递归分割方式。在包含50个查询的评估集上测量@5召回率，再据此进行参数调整。

## 发布它

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习题

1. **简单级。** 将一份20页的文档按固定分割（fixed(512, 0)）、递归分割（recursive(512, 0)）以及递归分割（recursive(512, 100)）三种方式拆分。比较不同分割方式的块数及边界质量。
2. **中等级。** 基于5份文档构建包含30个查询的评估集。分别测量递归检索、语义检索以及父文档检索在@5召回率上的表现。哪种方法最优？其结果是否与博客文章内容相符？
3. **高级别。** 实现上下文感知检索功能。衡量相较于基础递归检索，MRR指标的提升幅度。同时报告索引构建成本（即LLM调用次数）与精度提升之间的关系。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Chunk | 文档片段 | 被嵌入、索引并检索的子文档单元。 |
| Overlap | 安全裕度 | 相邻 Chunk 之间共享的令牌数；在 2026 年的基准测试中通常无实际作用。 |
| Semantic chunking | 智能分块 | 在相邻句子的嵌入相似度下降处进行分割。 |
| Parent-document | 两级检索 | 先检索较小的子文档，再返回较大的父文档。 |
| Late chunking | 嵌入后再分块 | 在令牌层面对完整文档进行嵌入，随后将其聚合为 Chunk 向量。 |
| Contextual retrieval | Anthropic 的技巧 | 在索引前将大语言模型生成的摘要附加到每个 Chunk 之前。 |
| Context cliff | 2500令牌壁垒 | 在 RAG 系统中，当上下文令牌数达到约 2.5k 时会出现质量下降现象（2026 年 1 月数据）。 |

## 延伸阅读

- [Yepes 等人 / LangChain — 递归字符分割文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认选择。
- [Vectara (2024, NAACL 2025). 分块配置分析](https://arxiv.org/abs/2410.13070) — 分块策略与嵌入模型选择同样重要。
- [Jina AI — 长上下文嵌入模型中的延迟分块技术 (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 关于延迟分块的论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — 使用大语言模型生成的上下文前缀可使检索效率提升 35% 至 50%。
- [NVIDIA 2026 年分块大小基准测试 — Premai 摘要](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 不同查询类型对应的分块大小标准。
