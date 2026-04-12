# Implementation Plan: Multi-Format Loaders

**Status**: migrated | **Date**: 2026-04-13

## Summary

Extends the loader infrastructure with 7 new format loaders (TXT, Markdown, DOCX, PPTX, HTML, CSV, Video) and a factory-based extension routing system. Each loader follows the `BaseLoader` contract and integrates with `LoaderFactory`.

## Technical Context

**Language/Version**: Python >= 3.11
**Primary Dependencies**: python-docx, python-pptx, beautifulsoup4, openai-whisper, opencv-python, moviepy
**Storage**: Filesystem (extracted images, frames)
**Testing**: pytest (unit with contract tests)

## Affected Modules

| Module | Impact | Files |
|--------|--------|-------|
| `src/libs/loader/` | 7 new loaders + factory update | 7 implementation files + factory |

## Project Structure

```
src/libs/loader/txt_loader.py       — 28 lines
src/libs/loader/markdown_loader.py  — 30 lines
src/libs/loader/docx_loader.py      — 115 lines
src/libs/loader/pptx_loader.py      — 78 lines
src/libs/loader/html_loader.py      — 59 lines
src/libs/loader/csv_loader.py       — 54 lines
src/libs/loader/video_loader.py     — 191 lines

tests/unit/test_loader_txt_markdown_contract.py  — 102 lines
tests/unit/test_loader_office_contract.py        — 102 lines
tests/unit/test_loader_structured_contract.py    — HTML + CSV tests
tests/unit/test_loader_video_contract.py         — 110 lines
tests/unit/test_pipeline_multiformat.py          — 133 lines
```
