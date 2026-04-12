# Feature Specification: Core Types + Loaders

**Status**: migrated
**Created**: 2026-04-13
**Original commits**: C1–C3, B8–B10

## User Scenarios & Testing

### User Story 1 - Load PDF documents with image extraction (P1)

As a user, I want to ingest PDF files so that their text and images are available for retrieval.

**Acceptance Scenarios**:
1. **Given** a valid PDF file, **When** loaded, **Then** text content and image references are extracted
2. **Given** a PDF with embedded images, **When** loaded, **Then** images are saved to `data/images/` and referenced via `ImageRef`
3. **Given** a previously processed file, **When** integrity check runs, **Then** the file is skipped (idempotent)

### User Story 2 - Load documents via extension-based routing (P2)

As a user, I want to load any supported file format through a single factory interface.

**Acceptance Scenarios**:
1. **Given** a file path ending in `.pdf`, **When** `LoaderFactory.create_from_path()` is called, **Then** a `PdfLoader` is returned
2. **Given** an unsupported extension, **When** `create_from_path()` is called, **Then** a clear error is raised

### User Story 3 - Chunk documents with image distribution (P3)

As a user, I want loaded documents split into chunks with metadata and image references preserved.

**Acceptance Scenarios**:
1. **Given** a document with images, **When** chunked, **Then** images are distributed to chunks based on text offset overlap
2. **Given** chunked output, **Then** each chunk has `start_offset`, `end_offset`, inherited metadata, and a deterministic ID

## Requirements

### Functional Requirements

- **FR-001**: System MUST define core data types: `Document`, `Chunk`, `ChunkRecord`, `ImageRef`, `ProcessedQuery`, `RetrievalResult`
- **FR-002**: System MUST provide `BaseLoader` abstract class with `load(path) -> Document`
- **FR-003**: System MUST provide `PdfLoader` that extracts text and images from PDFs via PyMuPDF
- **FR-004**: System MUST provide `LoaderFactory` with extension-based routing and runtime registration
- **FR-005**: System MUST provide `FileIntegrityChecker` with SQLite backend for idempotent ingestion
- **FR-006**: System MUST provide `DocumentChunker` that splits documents into chunks via `SplitterFactory`
- **FR-007**: All types MUST support `from_dict()` deserialization for JSON persistence

### Key Entities

- **Document**: Full loaded document with text, metadata, and ID
- **Chunk**: A text segment with offsets, metadata, and source reference
- **ChunkRecord**: A chunk enriched with dense and sparse vectors
- **ImageRef**: Reference to an extracted image file with position metadata

## Success Criteria

- **SC-001**: All supported file types load without error on valid input
- **SC-002**: File integrity checker prevents duplicate processing
- **SC-003**: Chunk IDs are deterministic (based on doc ID + index + text hash)

## Gaps Identified

- No unit tests for `ImageRef`, `ProcessedQuery`, `RetrievalResult` types
- No error handling for corrupted PDFs or out-of-memory on large files
- `BaseLoader._validate_path` has no dedicated tests
