# Add Chinese internationalization using local LLM translation

## Goal

Add a maintainable Chinese internationalization path for the curriculum site. The site should be able to show Simplified Chinese lesson content when a translated `docs/zh.md` exists, while falling back cleanly to English. The project should also include a stdlib-only translation script that uses the user's local LM Studio OpenAI-compatible endpoint at `http://127.0.0.1:1234` with model `hy-mt2-1.8b`.

## What I Already Know

* The user has LM Studio running locally at `http://127.0.0.1:1234`.
* `/v1/models` lists `hy-mt2-1.8b`.
* A `/v1/chat/completions` smoke test with `hy-mt2-1.8b` translated "Build every algorithm from scratch." to "从零开始构建每一个算法。"
* The repo has 503 lesson docs at `phases/*/*/docs/en.md`.
* The current lesson page loads Markdown from `https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/main/<path>/docs/en.md`.
* `site/build.js` derives catalog summaries and keywords from `docs/en.md`.
* `scripts/audit_lessons.py` requires `docs/en.md`; it does not know about localized docs.
* Project policy says generated site files such as `site/data.js`, `site/sitemap.xml`, and `site/llms.txt` should not be manually committed.
* Project policy says one commit per lesson directory, so this task should not bulk-commit hundreds of translated lesson files.

## Assumptions

* MVP scope is infrastructure plus a small verified sample, not translating all 503 lessons in one task.
* Chinese lesson files live next to the source lesson as `docs/zh.md`, with English remaining canonical at `docs/en.md`.
* A translated lesson should preserve Markdown structure, fenced code blocks, Mermaid/figure blocks, links, tables, frontmatter-like metadata labels, and technical identifiers.
* Quiz translation is out of scope for the first pass because quiz JSON has a stricter data contract and may need a separate validation workflow.

## Requirements

* Add a stdlib-only translation CLI under `scripts/` that:
  * discovers lesson `docs/en.md` files or accepts explicit lesson paths,
  * calls `http://127.0.0.1:1234/v1/chat/completions` by default,
  * uses `hy-mt2-1.8b` by default,
  * writes `docs/zh.md`,
  * supports dry-run, force overwrite, timeout, and retry controls,
  * protects fenced code blocks from translation,
  * validates the translated Markdown enough to catch obvious structural damage.
* Add site support for language selection on lesson pages:
  * `?lang=zh` attempts to load `docs/zh.md`,
  * missing `docs/zh.md` falls back to English with a small status note,
  * the language selector preserves `path`,
  * lesson navigation links preserve the selected language.
* Keep English as default and canonical.
* Add build metadata so `site/data.js` can know whether a lesson has `docs/zh.md` when generated.
* Do not commit generated `site/data.js`, `site/sitemap.xml`, or `site/llms.txt`.

## Acceptance Criteria

* [ ] `python3 scripts/translate_lessons.py --help` works.
* [ ] A dry run can translate a small sample through the local LM Studio endpoint without writing files.
* [ ] A real run can create at least one `docs/zh.md` sample.
* [ ] `node site/build.js` succeeds.
* [ ] `python3 scripts/audit_lessons.py` succeeds or any pre-existing failures are clearly identified.
* [ ] Lesson page URLs with `?lang=zh` render Chinese content when `docs/zh.md` exists and English fallback when it does not.

## Out of Scope

* Translating all 503 lesson docs in this task.
* Translating quiz JSON.
* Translating every static site label across index/catalog/glossary/prereqs pages.
* Introducing non-stdlib Python dependencies or a client-side i18n framework.

## Technical Notes

* Relevant files inspected: `site/lesson.html`, `site/build.js`, `scripts/build_catalog.py`, `scripts/audit_lessons.py`, `.gitignore`.
* `site/lesson.html` has many hardcoded English UI labels. This task focuses on lesson-body language switching and minimal related chrome.
* Since the site currently fetches raw GitHub Markdown, local/production support should try same-origin files first and then fall back to raw GitHub.
* The translation script should be restartable because 503 lessons is too large for a single fragile batch.

## Definition of Done

* Tests and smoke checks run locally.
* Task-specific implementation is committed only after user confirmation of the commit plan.
* Trellis finish workflow can archive the task afterwards.
