# LM Studio i18n notes

## Local API

The user's local LM Studio server is available at:

```text
http://127.0.0.1:1234
```

The model list includes `hy-mt2-1.8b`. A smoke request to `/v1/chat/completions` with that model returned a normal OpenAI-compatible response containing:

```text
从零开始构建每一个算法。
```

for the source sentence:

```text
Build every algorithm from scratch.
```

## Repo mapping

The site is static HTML plus generated `site/data.js`.

* `site/build.js` reads `README.md`, `ROADMAP.md`, `glossary/terms.md`, and per-lesson `docs/en.md` metadata.
* `site/lesson.html` fetches lesson Markdown client-side from raw GitHub.
* `scripts/audit_lessons.py` enforces canonical lesson presence and schema around `docs/en.md`.

## Recommended MVP

Use `docs/zh.md` as optional translated content next to each canonical `docs/en.md`. Keep English canonical and let the site fall back to English if Chinese is missing. Add a restartable CLI that uses the local OpenAI-compatible endpoint to generate Chinese files in small batches.

## Risks

* Bulk-translating 503 lessons in one commit conflicts with the repo's one-commit-per-lesson policy.
* Markdown structure can be damaged by translation, especially fenced code blocks, tables, links, and custom `figure` fences.
* The small model may produce inconsistent terminology; generated translations should be reviewable and resumable.
