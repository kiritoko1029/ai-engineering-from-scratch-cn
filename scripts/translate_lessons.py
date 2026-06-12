#!/usr/bin/env python3
"""Translate lesson docs and quizzes with a local OpenAI-compatible LM Studio server.

Default target:
    http://198.18.0.1:11234/v1/chat/completions
    model: hy-mt2-7b@q4_k_m

The script keeps English canonical at docs/en.md and quiz.json, and writes
optional Simplified Chinese copies to docs/zh.md and quiz.zh.json. It is
restartable by default: existing zh.md / quiz.zh.json files are skipped unless
--force is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
DEFAULT_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://198.18.0.1:11234")
DEFAULT_MODEL = os.environ.get("LM_STUDIO_MODEL", "hy-mt2-7b@q4_k_m")

FENCE_RE = re.compile(r"(^```[^\n]*\n.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
HEADING_LINE_RE = re.compile(r"^(#{1,6})([ \t]+)(.+?)([ \t]*)$")
CJK_RE = re.compile(r"[\u3400-\u9fff]")

# Quiz options are frequently pure code/commands (e.g. "import torch; ...").
# These match such option strings so we leave them untranslated.
CODE_OPTION_RE = re.compile(
    r"^\s*(?:"
    r"import\s|from\s|print\(|def\s|class\s|return\s|lambda\s|"
    r"sudo\s|pip[0-9]?\s|python[0-9]?\s|conda\s|uv\s|git\s|docker\s|npm\s|"
    r"cargo\s|node\s|nvidia-smi|export\s|curl\s|wget\s|ssh\s|mkdir\s|"
    r"chmod\s|source\s|bash\s|sh\s|cd\s|ls\s|cat\s|echo\s"
    r")"
)
COMMAND_FLAG_RE = re.compile(r"^\s*[A-Za-z][\w./-]*\s+--?[A-Za-z]")

MARKDOWN_SYSTEM_PROMPT = (
    "You are a precise technical translator. Translate English Markdown "
    "to Simplified Chinese for an AI engineering curriculum. Preserve "
    "Markdown syntax, heading levels, lists, tables, HTML tags, URLs, "
    "file paths, CLI commands, environment variables, JSON keys, product "
    "names, and inline code wrapped in backticks. Do not add commentary. "
    "Return only the translated Markdown."
)

QUIZ_SYSTEM_PROMPT = (
    "You are a precise technical translator for an AI engineering quiz. "
    "Translate the English quiz text to Simplified Chinese, keeping the "
    "meaning exact. Preserve all code, identifiers, function calls, CLI "
    "commands, file paths, environment variables, numbers, math notation, "
    "and symbols exactly as written. If the input is entirely a code "
    "expression or command, return it unchanged. Do not add explanations, "
    "labels, or quotes. Return only the translated text."
)


@dataclass(frozen=True)
class LessonDoc:
    source: Path
    target: Path


@dataclass(frozen=True)
class QuizFile:
    source: Path
    target: Path


class TranslationError(RuntimeError):
    """Raised when translation or validation fails."""


def iter_all_docs() -> Iterable[LessonDoc]:
    for source in sorted(PHASES_DIR.glob("[0-9][0-9]-*/*/docs/en.md")):
        yield LessonDoc(source=source, target=source.with_name("zh.md"))


def resolve_doc(path_arg: str) -> LessonDoc:
    path = Path(path_arg)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()

    if path.is_dir():
        if (path / "docs" / "en.md").is_file():
            source = path / "docs" / "en.md"
        elif (path / "en.md").is_file():
            source = path / "en.md"
        else:
            raise TranslationError(f"no docs/en.md found under {path}")
    elif path.name == "en.md":
        source = path
    else:
        raise TranslationError(
            f"expected a lesson directory or docs/en.md path: {path_arg}"
        )

    return LessonDoc(source=source, target=source.with_name("zh.md"))


def discover_docs(paths: list[str], limit: int | None, force: bool) -> list[LessonDoc]:
    docs = [resolve_doc(p) for p in paths] if paths else list(iter_all_docs())
    if not force:
        docs = [d for d in docs if not d.target.exists()]
    if limit is not None:
        docs = docs[:limit]
    return docs


def iter_all_quizzes() -> Iterable[QuizFile]:
    for source in sorted(PHASES_DIR.glob("[0-9][0-9]-*/*/quiz.json")):
        yield QuizFile(source=source, target=source.with_name("quiz.zh.json"))


def resolve_quiz(path_arg: str) -> QuizFile | None:
    path = Path(path_arg)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()

    if path.is_dir():
        source = path / "quiz.json"
    elif path.name == "quiz.json":
        source = path
    elif path.name == "en.md":  # a docs/en.md path points back at its lesson quiz
        source = path.parent.parent / "quiz.json"
    else:
        return None

    if not source.is_file():
        return None
    return QuizFile(source=source, target=source.with_name("quiz.zh.json"))


def discover_quizzes(
    paths: list[str], limit: int | None, force: bool
) -> list[QuizFile]:
    if paths:
        quizzes = [q for q in (resolve_quiz(p) for p in paths) if q is not None]
    else:
        quizzes = list(iter_all_quizzes())
    if not force:
        quizzes = [q for q in quizzes if not q.target.exists()]
    if limit is not None:
        quizzes = quizzes[:limit]
    return quizzes


def split_fenced_markdown(markdown: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pos = 0
    for match in FENCE_RE.finditer(markdown):
        if match.start() > pos:
            parts.append(("text", markdown[pos : match.start()]))
        parts.append(("code", match.group(0)))
        pos = match.end()
    if pos < len(markdown):
        parts.append(("text", markdown[pos:]))
    return parts


def chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    blocks = re.split(r"(\n{2,})", text)
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_block(block, max_chars))
            continue
        if len(current) + len(block) > max_chars and current:
            chunks.append(current)
            current = block
        else:
            current += block

    if current:
        chunks.append(current)
    return chunks


def split_long_block(block: str, max_chars: int) -> list[str]:
    lines = block.splitlines(keepends=True)
    chunks: list[str] = []
    current = ""
    for line in lines:
        if len(current) + len(line) > max_chars and current:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


def strip_wrapping_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(
        r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", stripped, re.IGNORECASE
    )
    if match:
        return match.group(1).strip() + "\n"
    return text


def preserve_edge_whitespace(source: str, translated: str) -> str:
    if not source.strip() or not translated.strip():
        return source if not source.strip() else translated
    leading = re.match(r"^\s*", source).group(0)
    trailing = re.search(r"\s*$", source).group(0)
    return leading + normalize_translated_markdown_text(translated.strip()) + trailing


def normalize_translated_markdown_text(text: str) -> str:
    text = re.sub(r"^(#{1,6})([^\s#].*)$", r"\1 \2", text, flags=re.MULTILINE)
    text = re.sub(r"^>([^\s].*)$", r"> \1", text, flags=re.MULTILINE)
    return text


# LM Studio runs locally, so never route these requests through a system proxy.
# On Windows urllib otherwise picks up the proxy from the registry (e.g. a
# Clash/VPN HTTP proxy), which 502s on a local or fake-IP address. This is the
# equivalent of curl --noproxy '*'.
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def translate_chunk(
    text: str,
    *,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    system_prompt: str = MARKDOWN_SYSTEM_PROMPT,
) -> str:
    if not text.strip():
        return text

    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with _DIRECT_OPENER.open(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return preserve_edge_whitespace(
                text, strip_wrapping_markdown_fence(str(content))
            )
        except (
            KeyError,
            IndexError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))

    raise TranslationError(
        f"translation request failed after {retries + 1} attempt(s): {last_error}"
    )


def translate_markdown(markdown: str, args: argparse.Namespace) -> str:
    output: list[str] = []
    for kind, value in split_fenced_markdown(markdown):
        if kind == "code":
            output.append(value)
            continue
        output.append(translate_text_segment(value, args))
        if args.sleep:
            time.sleep(args.sleep)
    return "".join(output)


def translate_text_segment(text: str, args: argparse.Namespace) -> str:
    """Translate Markdown text while preserving ATX heading structure exactly."""
    output: list[str] = []
    pending_lines: list[str] = []

    def flush_pending_lines() -> None:
        if not pending_lines:
            return
        block = "".join(pending_lines)
        pending_lines.clear()
        output.extend(
            translate_plain_text_chunk(chunk, args)
            for chunk in chunk_text(block, args.max_chars)
        )

    for line in text.splitlines(keepends=True):
        content, line_ending = split_line_ending(line)
        if HEADING_LINE_RE.match(content):
            flush_pending_lines()
            output.append(translate_heading_line(content, line_ending, args))
        else:
            pending_lines.append(line)

    flush_pending_lines()
    return "".join(output)


def split_line_ending(line: str) -> tuple[str, str]:
    match = re.search(r"(\r?\n)$", line)
    if not match:
        return line, ""
    return line[: match.start()], match.group(1)


def translate_plain_text_chunk(text: str, args: argparse.Namespace) -> str:
    return translate_chunk(
        text,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
    )


def translate_heading_line(
    content: str, line_ending: str, args: argparse.Namespace
) -> str:
    match = HEADING_LINE_RE.match(content)
    if not match:
        return content + line_ending

    hashes, spacing, title, trailing = match.groups()
    translated_title = normalize_translated_heading_title(
        title, translate_plain_text_chunk(title, args)
    )
    return f"{hashes}{spacing}{translated_title}{trailing}{line_ending}"


def normalize_translated_heading_title(source_title: str, translated_title: str) -> str:
    lines = [line.strip() for line in translated_title.splitlines() if line.strip()]
    if not lines:
        return source_title

    title = lines[0]
    title = re.sub(r"^#{1,6}\s+", "", title)
    return title.strip() or source_title


def fence_infos(markdown: str) -> list[str]:
    infos: list[str] = []
    for kind, value in split_fenced_markdown(markdown):
        if kind != "code":
            continue
        first_line = value.splitlines()[0] if value.splitlines() else "```"
        infos.append(first_line.removeprefix("```").strip())
    return infos


def heading_levels(markdown: str) -> list[str]:
    without_code = "".join(
        value for kind, value in split_fenced_markdown(markdown) if kind == "text"
    )
    return [m.group(1) for m in HEADING_RE.finditer(without_code)]


def validate_translation(source: str, translated: str, source_path: Path) -> None:
    source_fences = fence_infos(source)
    translated_fences = fence_infos(translated)
    if source_fences != translated_fences:
        raise TranslationError(
            f"{source_path}: fenced code blocks changed "
            f"(source={source_fences!r}, translated={translated_fences!r})"
        )

    source_headings = heading_levels(source)
    translated_headings = heading_levels(translated)
    if source_headings != translated_headings:
        raise TranslationError(
            f"{source_path}: heading structure changed "
            f"(source={source_headings!r}, translated={translated_headings!r})"
        )

    if not re.search(r"^#\s+\S", translated, re.MULTILINE):
        raise TranslationError(f"{source_path}: translated Markdown is missing an H1")

    if not CJK_RE.search(translated):
        raise TranslationError(
            f"{source_path}: translated Markdown contains no CJK characters"
        )


def write_translation(doc: LessonDoc, args: argparse.Namespace) -> None:
    source = doc.source.read_text(encoding="utf-8")
    translated = translate_markdown(source, args)
    validate_translation(source, translated, doc.source)

    rel_source = doc.source.relative_to(ROOT)
    rel_target = doc.target.relative_to(ROOT)
    if args.dry_run:
        preview = translated[: args.preview_chars].rstrip()
        print(f"[dry-run] {rel_source} -> {rel_target}")
        print(preview)
        if len(translated) > args.preview_chars:
            print("\n... preview truncated ...")
        return

    doc.target.write_text(translated, encoding="utf-8")
    print(f"[written] {rel_target}")


def quiz_questions(data: object) -> list:
    if isinstance(data, dict):
        questions = data.get("questions")
        return questions if isinstance(questions, list) else []
    if isinstance(data, list):
        return data
    return []


def looks_like_code(text: str) -> bool:
    """True when an option string is a bare code expression or command."""
    stripped = text.strip()
    if not stripped or CJK_RE.search(stripped):
        return False
    return bool(CODE_OPTION_RE.match(stripped) or COMMAND_FLAG_RE.match(stripped))


def translate_quiz_string(text: str, args: argparse.Namespace) -> str:
    if not text.strip():
        return text
    translated = translate_chunk(
        text,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        system_prompt=QUIZ_SYSTEM_PROMPT,
    )
    return re.sub(r"\s+", " ", translated).strip()


def translate_quiz_question(question: object, args: argparse.Namespace) -> object:
    """Translate the prose fields of one question, leaving stage/correct intact."""
    if not isinstance(question, dict):
        return question
    translated = dict(question)
    if isinstance(question.get("question"), str):
        translated["question"] = translate_quiz_string(question["question"], args)
    if isinstance(question.get("options"), list):
        translated["options"] = [
            opt
            if not isinstance(opt, str) or looks_like_code(opt)
            else translate_quiz_string(opt, args)
            for opt in question["options"]
        ]
    if isinstance(question.get("explanation"), str):
        translated["explanation"] = translate_quiz_string(question["explanation"], args)
    return translated


def translate_quiz_questions(questions: list, args: argparse.Namespace) -> list:
    translated: list = []
    for question in questions:
        translated.append(translate_quiz_question(question, args))
        if args.sleep:
            time.sleep(args.sleep)
    return translated


def translate_quiz(data: object, args: argparse.Namespace) -> object:
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        translated = dict(data)
        translated["questions"] = translate_quiz_questions(data["questions"], args)
        return translated
    if isinstance(data, list):
        return translate_quiz_questions(data, args)
    raise TranslationError(
        "unsupported quiz schema (expected an object with questions[] or an array)"
    )


def validate_quiz_translation(
    source: object, translated: object, source_path: Path
) -> None:
    source_questions = quiz_questions(source)
    translated_questions = quiz_questions(translated)
    if len(source_questions) != len(translated_questions):
        raise TranslationError(
            f"{source_path}: question count changed "
            f"(source={len(source_questions)}, translated={len(translated_questions)})"
        )
    for idx, (src_q, tr_q) in enumerate(zip(source_questions, translated_questions)):
        if not isinstance(src_q, dict) or not isinstance(tr_q, dict):
            continue
        if src_q.get("stage") != tr_q.get("stage"):
            raise TranslationError(f"{source_path}: question[{idx}] stage changed")
        if src_q.get("correct") != tr_q.get("correct"):
            raise TranslationError(
                f"{source_path}: question[{idx}] correct index changed"
            )
        src_opts = src_q.get("options")
        if isinstance(src_opts, list):
            tr_opts = tr_q.get("options")
            if not isinstance(tr_opts, list) or len(src_opts) != len(tr_opts):
                raise TranslationError(
                    f"{source_path}: question[{idx}] options count changed"
                )
    if not CJK_RE.search(json.dumps(translated, ensure_ascii=False)):
        raise TranslationError(
            f"{source_path}: translated quiz contains no CJK characters"
        )


def write_quiz_translation(quiz: QuizFile, args: argparse.Namespace) -> None:
    raw = quiz.source.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TranslationError(f"{quiz.source}: invalid JSON: {exc}") from exc

    translated = translate_quiz(data, args)
    validate_quiz_translation(data, translated, quiz.source)
    serialized = json.dumps(translated, ensure_ascii=False, indent=2) + "\n"

    rel_source = quiz.source.relative_to(ROOT)
    rel_target = quiz.target.relative_to(ROOT)
    if args.dry_run:
        preview = serialized[: args.preview_chars].rstrip()
        print(f"[dry-run] {rel_source} -> {rel_target}")
        print(preview)
        if len(serialized) > args.preview_chars:
            print("\n... preview truncated ...")
        return

    quiz.target.write_text(serialized, encoding="utf-8")
    print(f"[written] {rel_target}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="lesson directories or docs/en.md files; defaults to every untranslated lesson",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"LM Studio base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("LM_API_TOKEN")
            or os.environ.get("LM_STUDIO_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
        help="bearer token for the LM Studio server (default: $LM_API_TOKEN, $LM_STUDIO_API_KEY, or $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="maximum number of lessons to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing docs/zh.md and quiz.zh.json files",
    )
    parser.add_argument(
        "--skip-quiz",
        action="store_true",
        help="translate docs only; skip quiz.json files",
    )
    parser.add_argument(
        "--quiz-only",
        action="store_true",
        help="translate quiz.json files only; skip docs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="translate and validate without writing files",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="request timeout in seconds"
    )
    parser.add_argument("--retries", type=int, default=1, help="retry count per chunk")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="model temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="maximum completion tokens per chunk",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4500,
        help="maximum source characters per text chunk",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to sleep after each text segment",
    )
    parser.add_argument(
        "--preview-chars", type=int, default=1200, help="dry-run preview length"
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.skip_quiz and args.quiz_only:
        print(
            "error: --skip-quiz and --quiz-only are mutually exclusive", file=sys.stderr
        )
        return 2
    try:
        docs = (
            [] if args.quiz_only else discover_docs(args.paths, args.limit, args.force)
        )
        quizzes = (
            []
            if args.skip_quiz
            else discover_quizzes(args.paths, args.limit, args.force)
        )
        if not docs and not quizzes:
            print("No lesson docs or quizzes to translate.")
            return 0
        for doc in docs:
            write_translation(doc, args)
        for quiz in quizzes:
            write_quiz_translation(quiz, args)
        return 0
    except TranslationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
