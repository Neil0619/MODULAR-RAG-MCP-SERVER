# Feature Specification: Dashboard

**Status**: migrated
**Created**: 2026-04-13
**Original commits**: G1–G6

## User Scenarios & Testing

### User Story 1 - Overview of system status (P1)

As a user, I want a dashboard overview showing system config, data stats, and recent activity.

**Acceptance Scenarios**:
1. **Given** the dashboard is running, **When** the Overview page loads, **Then** system config, collection stats, and recent traces are displayed

### User Story 2 - Browse and manage documents (P1)

As a user, I want to browse documents, chunks, and images, and manage the ingestion lifecycle.

**Acceptance Scenarios**:
1. **Given** ingested documents, **When** Data Browser loads, **Then** documents, chunks, and images are browsable
2. **Given** a file upload, **When** Ingestion Manager processes it, **Then** progress is tracked in real-time
3. **Given** an ingested document, **When** delete is triggered, **Then** all associated data is removed

### User Story 3 - View and compare retrieval traces (P2)

As a developer, I want to view ingestion and query traces with timing breakdowns and Dense vs Sparse comparison.

**Acceptance Scenarios**:
1. **Given** trace history, **When** Ingestion Traces page loads, **Then** waterfall chart shows stage timing
2. **Given** query traces, **When** Query Traces page loads, **Then** Dense vs Sparse score comparison is displayed

### User Story 4 - Run evaluations from the dashboard (P3)

As a user, I want to run evaluations and view metrics from the dashboard.

**Acceptance Scenarios**:
1. **Given** a test set, **When** evaluation is triggered, **Then** metrics are displayed with history

## Requirements

### Functional Requirements

- **FR-001**: System MUST implement Streamlit multi-page dashboard with navigation
- **FR-002**: System MUST implement Overview page with system config and stats
- **FR-003**: System MUST implement Data Browser page with document/chunk/image viewing
- **FR-004**: System MUST implement Ingestion Manager page with upload, progress, delete
- **FR-005**: System MUST implement Ingestion Traces page with waterfall timing charts
- **FR-006**: System MUST implement Query Traces page with Dense/Sparse comparison
- **FR-007**: System MUST implement Evaluation Panel page with run/history view
- **FR-008**: Each page MUST follow render() + service module pattern
- **FR-009**: System MUST provide `scripts/start_dashboard.py` entry point

### Key Entities

- **ConfigService**: Reads system settings and returns overview data
- **DataService**: Queries ChromaDB and image storage for document data
- **TraceService**: Reads JSONL trace files and formats for display

## Success Criteria

- **SC-001**: Dashboard starts and all 6 pages render without error
- **SC-002**: Ingestion Manager can upload and delete documents
- **SC-003**: Traces display with accurate timing data

## Gaps Identified

- Only smoke test exists — no page-level unit tests
- No authentication/authorization
- No data export capabilities
