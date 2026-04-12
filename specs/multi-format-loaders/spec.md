# Feature Specification: Multi-Format Loaders

**Status**: migrated
**Created**: 2026-04-13
**Original commits**: J1–J5

## User Scenarios & Testing

### User Story 1 - Load text and markdown files (P1)

As a user, I want to load plain text and Markdown files as documents.

**Acceptance Scenarios**:
1. **Given** a `.txt` file, **When** loaded, **Then** full text content is returned as a `Document`
2. **Given** a `.md` file, **When** loaded, **Then** Markdown content is returned preserving structure

### User Story 2 - Load office documents (P2)

As a user, I want to load DOCX and PPTX files with content and image extraction.

**Acceptance Scenarios**:
1. **Given** a `.docx` file, **When** loaded, **Then** paragraphs are extracted and images saved to disk
2. **Given** a `.pptx` file, **When** loaded, **Then** slide text and speaker notes are extracted

### User Story 3 - Load web and tabular data (P3)

As a user, I want to load HTML and CSV files.

**Acceptance Scenarios**:
1. **Given** an `.html` file, **When** loaded, **Then** scripts/styles are stripped and text content returned
2. **Given** a `.csv` file, **When** loaded, **Then** data is converted to Markdown table format

### User Story 4 - Load and transcribe video files (P3)

As a user, I want to load video files with audio transcription and key frame extraction.

**Acceptance Scenarios**:
1. **Given** a `.mp4` file, **When** loaded, **Then** audio is transcribed via Whisper and key frames extracted
2. **Given** video dependencies unavailable, **When** loaded, **Then** a clear error message is returned

## Requirements

### Functional Requirements

- **FR-001**: System MUST implement `TxtLoader` for plain text files
- **FR-002**: System MUST implement `MarkdownLoader` for Markdown files
- **FR-003**: System MUST implement `DocxLoader` with image extraction
- **FR-004**: System MUST implement `PptxLoader` with speaker notes
- **FR-005**: System MUST implement `HtmlLoader` with script/style stripping
- **FR-006**: System MUST implement `CsvLoader` with Markdown table conversion
- **FR-007**: System MUST implement `VideoLoader` with Whisper transcription + frame extraction
- **FR-008**: All loaders MUST be registered in `LoaderFactory`
- **FR-009**: All loaders MUST inherit from `BaseLoader` and return `Document`
- **FR-010**: Loaders with optional deps MUST provide clear error messages when deps are missing

### Key Entities

- **TxtLoader**: Reads plain text, returns as Document
- **DocxLoader**: Extracts paragraphs + images via python-docx
- **VideoLoader**: Transcribes audio via Whisper, extracts frames via OpenCV

## Success Criteria

- **SC-001**: All 8 formats load without error on valid input
- **SC-002**: Factory routes to correct loader by file extension
- **SC-003**: Missing optional deps produce clear error messages (not crashes)

## Gaps Identified

- No OCR support for image-based documents
- No audio-only file support
- Video loader requires heavy external dependencies (Whisper, OpenCV, moviepy)
- No batch loading optimization
