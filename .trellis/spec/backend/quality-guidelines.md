# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

<!--
Document your project's quality standards here.

Questions to answer:
- What patterns are forbidden?
- What linting rules do you enforce?
- What are your testing requirements?
- What code review standards apply?
-->

(To be filled by the team)

---

## Forbidden Patterns

<!-- Patterns that should never be used and why -->

(To be filled by the team)

---

## Required Patterns

<!-- Patterns that must always be used -->

(To be filled by the team)

---

## Testing Requirements

<!-- What level of testing is expected -->

(To be filled by the team)

---

## Code Review Checklist

<!-- What reviewers should check -->

(To be filled by the team)

## Scenario: Lesson Chinese Translation Pipeline

### 1. Scope / Trigger
- Trigger: adding or updating Simplified Chinese lesson translations for the static curriculum site.
- English remains canonical in `phases/<phase>/<lesson>/docs/en.md`.
- Chinese translations are optional per lesson and live beside English as `docs/zh.md`.

### 2. Signatures
- Generate one lesson:
  ```bash
  python3 scripts/translate_lessons.py phases/00-setup-and-tooling/02-git-and-collaboration
  ```
- Dry-run without writing:
  ```bash
  python3 scripts/translate_lessons.py <lesson-dir> --dry-run
  ```
- Batch safely:
  ```bash
  python3 scripts/translate_lessons.py --limit 5
  ```

### 3. Contracts
- Default endpoint: `LM_STUDIO_BASE_URL` or `http://127.0.0.1:1234`.
- Default model: `LM_STUDIO_MODEL` or `hy-mt2-1.8b`.
- Source: `docs/en.md`.
- Target: `docs/zh.md`.
- Existing `docs/zh.md` is skipped unless `--force` is passed.
- `site/lesson.html?path=<lesson>&lang=zh` loads Chinese when present and falls back to English with a visible notice when missing.
- `site/build.js` copies lesson Markdown into `site/phases/**/docs/` so Vercel's `outputDirectory: "site"` can serve the docs.
- `site/build.js` and `scripts/build_catalog.py` expose `translations.zh` based on `docs/zh.md` presence.

### 4. Validation & Error Matrix
- Missing lesson `docs/en.md` -> CLI exits non-zero with a path error.
- LM Studio request failure -> CLI exits non-zero after retries.
- Fenced code block languages differ -> CLI exits non-zero.
- Heading levels differ -> CLI exits non-zero.
- Translated Markdown has no H1 or no CJK characters -> CLI exits non-zero.

### 5. Good/Base/Bad Cases
- Good: run `--dry-run` first, review preview, then generate a small batch with `--limit`.
- Base: a lesson without `docs/zh.md` still renders in English.
- Bad: bulk-commit hundreds of generated `docs/zh.md` files in one change; lesson commits must remain reviewable.

### 6. Tests Required
- `python3 scripts/translate_lessons.py --help`
- `python3 -m py_compile scripts/translate_lessons.py scripts/build_catalog.py scripts/audit_lessons.py`
- `python3 scripts/audit_lessons.py`
- `node --check site/build.js`
- Browser smoke: open `site/lesson.html?path=<translated-lesson>&lang=zh` and a missing-translation lesson with `&lang=zh`.

### 7. Wrong vs Correct
#### Wrong
```bash
python3 scripts/translate_lessons.py --force
git add phases
git commit -m "translate everything"
```

#### Correct
```bash
python3 scripts/translate_lessons.py --limit 5 --dry-run
python3 scripts/translate_lessons.py --limit 5
python3 scripts/audit_lessons.py
```
