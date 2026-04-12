# Tasks: Multi-Format Loaders

**Status**: migrated — all tasks completed

## Phase 1: Factory + Simple Loaders

- [x] T001 [J1] Implement `LoaderFactory` with extension-based routing in `src/libs/loader/loader_factory.py`
- [x] T002 [J2] Implement `TxtLoader` in `src/libs/loader/txt_loader.py`
- [x] T003 [J2] Implement `MarkdownLoader` in `src/libs/loader/markdown_loader.py`

## Phase 2: Office Document Loaders

- [x] T004 [J3] Implement `DocxLoader` with image extraction in `src/libs/loader/docx_loader.py`
- [x] T005 [J3] Implement `PptxLoader` with speaker notes in `src/libs/loader/pptx_loader.py`

## Phase 3: Web + Tabular Loaders

- [x] T006 [J4] Implement `HtmlLoader` with script/style stripping in `src/libs/loader/html_loader.py`
- [x] T007 [J4] Implement `CsvLoader` with Markdown table conversion in `src/libs/loader/csv_loader.py`

## Phase 4: Video Loader

- [x] T008 [J5] Implement `VideoLoader` with Whisper transcription + frame extraction in `src/libs/loader/video_loader.py`

## Phase 5: Tests

- [x] T009 [J2] Contract tests for TXT + Markdown in `tests/unit/test_loader_txt_markdown_contract.py`
- [x] T010 [J3] Contract tests for DOCX + PPTX in `tests/unit/test_loader_office_contract.py`
- [x] T011 [J4] Contract tests for HTML + CSV in `tests/unit/test_loader_structured_contract.py`
- [x] T012 [J5] Contract tests for Video in `tests/unit/test_loader_video_contract.py`
- [x] T013 Multi-format pipeline tests in `tests/unit/test_pipeline_multiformat.py`

## Gaps

- No OCR support for image-based documents
- No audio-only file support
- Video loader requires heavy deps (Whisper, OpenCV, moviepy)
- No batch loading optimization
