# Waqf Record Verifier (DocVerify Chain — POC-C) — End-to-End Documentation

**Project:** AI-assisted verification for Waqf-style records (multi-script OCR, field extraction, validation, human-in-the-loop review, and semantic duplicate detection).
**Source FSD:** `Waqf_POC-C_Waqf_DocVerify_20260706.md`
**Status:** Working demo build — synthetic/sample documents only (see [§10 Compliance & Risk](#10-compliance--risk-dpdp)).

---

## Table of Contents

1. [Objective & Scope](#1-objective--scope)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack](#3-tech-stack)
4. [Data Model](#4-data-model)
5. [Document Processing Pipeline](#5-document-processing-pipeline)
6. [Vector Search / Duplicate Detection](#6-vector-search--duplicate-detection)
7. [Backend API Reference](#7-backend-api-reference)
8. [Frontend Application](#8-frontend-application)
9. [Auth & Roles](#9-auth--roles)
10. [Compliance & Risk (DPDP)](#10-compliance--risk-dpdp)
11. [Configuration Reference](#11-configuration-reference)
12. [Setup & Running Locally](#12-setup--running-locally)
13. [Known Limitations / Roadmap](#13-known-limitations--roadmap)

---

## 1. Objective & Scope

### 1.1 Objective (from FSD)

A working demo of AI-assisted verification for Waqf-style records: multi-script OCR (Urdu, Marathi/Devanagari, and older formats), field extraction, cross-field validation, and a human-in-the-loop review UI with per-document confidence scoring.

### 1.2 In-scope (per FSD, and implemented)

| FSD requirement | Implementation |
|---|---|
| OCR for Urdu (Nastaliq) + Marathi Devanagari | `sarvam_engine.py` (primary), `tesseract_engine.py` (fallback/comparison) |
| Field extraction: property ID, mutawalli name, survey number, registration date, extent | `qwen_mapper.py` (Qwen2.5 via Ollama), backfilled by `gemini_engine.py` |
| Validation rules: mandatory-field completeness, date sanity, cross-document consistency | `validation.py` + `ValidationRuleConfig` (admin-tunable) |
| Review UI: side-by-side scan vs extracted fields, confidence heat, approve/correct/flag | `Review.tsx`, `documents.review` endpoint, `FieldCorrection` audit table |
| Synthetic/public sample documents only | Enforced by process, not code — see [§10](#10-compliance--risk-dpdp) |

### 1.3 Out-of-scope (per FSD)

- Real Waqf board records (blocked pending DPDP terms)
- Modi script OCR (flagged as roadmap)
- Blockchain layer

### 1.4 Beyond-FSD additions (built during implementation)

These extend the original POC-C scope and are documented here since they materially affect the architecture:

- **Vector search / semantic duplicate detection** — a POC layer that embeds each document's extracted fields and flags likely-duplicate property records (see [§6](#6-vector-search--duplicate-detection))
- **Translation pass** — Gemini-based transliteration of extracted fields into English, with a local digit-conversion fallback
- **Background/async OCR processing** — documents are accepted instantly and processed in a background task, rather than blocking the upload request
- **Reupload flow** — a flagged/rejected document can be corrected and reprocessed without creating a new document record
- **Multi-engine OCR fallback chain** with automatic script-mismatch correction (see [§5](#5-document-processing-pipeline))
- **Reports/Analytics module** — throughput, status breakdown, confidence distribution, corrections history
- **Admin console** — user management, OCR settings, validation-rule tuning, CER benchmark viewer

---

## 2. Architecture Overview

### 2.1 High-level system diagram

```
┌─────────────────────┐        HTTPS / REST (JSON)        ┌──────────────────────────┐
│                      │ ─────────────────────────────────▶ │                          │
│   React Frontend     │                                     │   FastAPI Backend        │
│   (waqf-frontend)    │ ◀───────────────────────────────── │   (waqf-backend)         │
│                      │        JSON responses               │                          │
└──────────────────────┘                                     └──────────┬───────────────┘
                                                                          │
                              ┌───────────────────────────────────────────┼───────────────────────────────┐
                              │                                           │                                 │
                    ┌─────────▼─────────┐               ┌─────────────────▼─────────────┐      ┌────────────▼───────────┐
                    │ PostgreSQL /       │               │  Background OCR Task           │      │  S3 (optional) /        │
                    │ SQLite (dev)       │               │  (FastAPI BackgroundTasks)      │      │  local disk storage     │
                    │ - users            │               │                                 │      │  for uploaded scans     │
                    │ - waqf_documents   │               │  ┌──────────────────────────┐  │      └─────────────────────────┘
                    │ - extracted_fields │               │  │ 1. Sarvam Vision 3B      │  │
                    │ - validation_...   │               │  │    (primary OCR)         │  │
                    │ - reviews          │               │  ├──────────────────────────┤  │
                    │ - field_corrections│               │  │ 2. Tesseract             │  │
                    │ - document_        │               │  │    (fallback / compare)  │  │
                    │   embeddings       │               │  ├──────────────────────────┤  │
                    │ - ocr_settings     │               │  │ 3. Gemini Vision         │  │
                    │ - validation_rule_ │               │  │    (last-resort OCR)     │  │
                    │   configs          │               │  ├──────────────────────────┤  │
                    │ - cer_benchmark_   │               │  │ 4. Qwen2.5 (via Ollama)  │  │
                    │   entries          │               │  │    (field extraction)    │  │
                    └────────────────────┘               │  ├──────────────────────────┤  │
                                                            │  │ 5. Gemini (gap-fill +    │  │
                                                            │  │    translation pass)     │  │
                                                            │  ├──────────────────────────┤  │
                                                            │  │ 6. Gemini Embeddings +   │  │
                                                            │  │    Vector Search (POC)   │  │
                                                            │  └──────────────────────────┘  │
                                                            └─────────────────────────────────┘
```

### 2.2 Request flow (document upload → verified record)

1. **Reviewer uploads a scan** via the React frontend (`Upload.tsx`) → `POST /api/v1/documents/upload`.
2. Backend validates file type, stores the raw file (local disk or S3), creates a `WaqfDocument` row with `status = uploaded`, and immediately returns a `201` with the document ID — **it does not wait for OCR**.
3. A **FastAPI `BackgroundTasks`** job (`_run_ocr_pipeline` in `documents.py`) runs asynchronously:
   - Runs the OCR fallback chain (Sarvam → Tesseract → Gemini Vision) to get raw text + detected script.
   - Runs Qwen2.5 (via Ollama) to map raw text into the six structured fields.
   - Backfills any low-confidence field via Gemini Vision field-extraction.
   - Runs script/transliteration guards (catches wrong-script or Latin-transliterated values and clears them for manual entry rather than showing a fluent-but-wrong value).
   - Runs a Gemini translation pass (+ local digit-conversion fallback) to populate English renderings.
   - Persists `ExtractedField` rows, sets `document.status = extracted`.
   - Runs `validation.run_validations()` — cross-field/date/mandatory-field rules — sets `status = validated` (or `flagged` if rules fail).
   - Runs the **vector search pipeline** — embeds the document, finds top-K similar documents, verifies by Property ID, stores results (currently logged only — see [§6](#6-vector-search--duplicate-detection)).
4. The frontend **polls** `GET /documents/{id}` every few seconds until status leaves `processing`.
5. A **reviewer** opens the document in `Review.tsx`: sees the scan side-by-side with extracted fields, confidence highlighting, and validation results. They **approve**, **correct** (which logs a `FieldCorrection` for future training), or **flag** it.
6. If flagged/rejected, the same document can be **reuploaded** (`POST /{id}/reupload`) with a corrected scan, re-running the whole pipeline against the same document record (preserving review history, incrementing `reupload_count`).
7. Once approved, the record is considered verified. Reports (`Reports.tsx`) aggregate throughput, confidence distribution, and correction history across all processed documents.

### 2.3 Why background processing (not synchronous)?

OCR + LLM calls can take anywhere from a few seconds (Sarvam/Gemini) to 30–90+ seconds (Ollama on CPU-only hardware). Returning the upload response immediately keeps the UI responsive; the router endpoint is a plain `def` (not `async def`), so FastAPI/Starlette runs it in a worker thread, and the blocking network calls inside don't block the event loop. This avoids adding a message-queue dependency (Celery/Redis) while still meeting a "process in under ~30 seconds" target for well-behaved documents.

---

## 3. Tech Stack

### 3.1 Backend (`waqf-backend`)

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.115 (Python) |
| ORM / DB toolkit | SQLAlchemy 2.0 |
| Database | PostgreSQL (production) / SQLite (local dev — auto-detected) |
| Migrations | Alembic (present in requirements; also has lightweight in-code column migration helpers in `main.py`) |
| Auth | JWT (`python-jose`), `bcrypt` password hashing |
| File validation | `python-multipart` |
| Primary OCR | **Sarvam Vision 3B** (`sarvam_engine.py`) — via Sarvam AI API |
| Fallback OCR | **Tesseract** (`pytesseract`, `tesseract_engine.py`) — offline, urd/mar/eng language packs |
| Last-resort OCR | **Gemini Vision** (`gemini_engine.py`) — only when both above fail outright |
| Field extraction | **Qwen2.5 7B** via a local **Ollama** server (`qwen_mapper.py`) |
| Field-extraction backfill / translation | **Gemini** (text + vision, `gemini_model` = `gemini-flash-latest`) |
| Embeddings / vector search | **Gemini embeddings** (`gemini-embedding-001`) + **pgvector** (Postgres extension, with a pure-Python cosine-similarity fallback when unavailable) |
| PDF handling | PyMuPDF |
| Object storage | AWS S3 (`boto3`) — optional, falls back to local disk |
| HTTP client | `httpx` |

### 3.2 Frontend (`waqf-frontend`)

| Layer | Technology |
|---|---|
| Framework | React 18 + TypeScript |
| Build tool | Vite |
| Routing | React Router v6 |
| Styling | Tailwind CSS + Radix UI primitives (dialog, dropdown, select, switch, tabs, checkbox) |
| Forms & validation | `react-hook-form` + `zod` |
| Charts | `recharts` (Reports page) |
| HTTP client | `axios` |
| Notifications | `react-hot-toast` |
| Icons | `lucide-react` |

### 3.3 External AI services used

| Service | Purpose | Configured via |
|---|---|---|
| Sarvam AI (Vision 3B) | Primary multi-script OCR | `SARVAM_API_KEY` |
| Google Gemini (`gemini-flash-latest`) | OCR last-resort, field-extraction backfill, translation, embeddings | `GEMINI_API_KEY` |
| Ollama (local, self-hosted) | Field extraction from raw OCR text via Qwen2.5 | `OLLAMA_URL`, `OLLAMA_MODEL` |
| Shasan-SLM (Pod B) | Originally planned extraction-assist API; **superseded by Qwen2.5/Ollama** in this build — code retained (`shasan_stub.py`) for compatibility but not wired into the pipeline | `SHASAN_SLM_API_URL/KEY` (unused currently) |

---

## 4. Data Model

All tables live in a single PostgreSQL (or SQLite for dev) database, defined in `app/models.py`.

### 4.1 Entity-relationship summary

```
User ──< Review
 │
 └──< WaqfDocument ──< ExtractedField
          │      │
          │      ├──< ValidationResult
          │      ├──< Review ──< FieldCorrection
          │      └──1─1 DocumentEmbedding

ValidationRuleConfig   (admin-tunable, standalone)
OcrSettings             (singleton row, admin-tunable)
CerBenchmarkEntry       (OCR engine benchmark data, standalone)
```

### 4.2 Key tables

**`users`**
Role-based accounts (`Role` enum: e.g. `USER`, `SUPERVISOR`) with bcrypt-hashed passwords and active/inactive flag.

**`waqf_documents`** — the core entity, one row per uploaded scan
| Field | Notes |
|---|---|
| `id` | `doc-<hex>` generated ID |
| `status` | `DocumentStatus` enum — `uploaded → processing → extracted → validated → reviewed/approved` or `flagged` at any validation/review step |
| `script_type` | `ScriptType` enum — `urdu_nastaliq`, `marathi_devanagari`, `hindi_devanagari`, `sanskrit_devanagari`, `english_latin` |
| `overall_confidence` | Average confidence across all six extracted fields |
| `extraction_notes` | Human-readable log of what each pipeline stage did/decided — surfaced in the review UI |
| `dpdp_status` | `DpdpStatus` enum (`checking`/`compliant`/`needs_review`) |
| `reupload_count` | Incremented each time the same document is corrected & reprocessed |
| `file_path` / storage key | Where the raw scan lives (local disk or S3) |

**`extracted_fields`** — one row per (`document_id`, `field_name`) pair
| Field | Notes |
|---|---|
| `field_name` | `FieldName` enum: `property_id`, `mutawalli_name`, `survey_number`, `registration_date`, `extent`, `village` |
| `value` | The value as read in its native script |
| `value_en` | English rendering (transliteration/translation/digit-conversion) |
| `confidence` | 0.0–1.0 |
| `source` | `ExtractionSource` enum — which engine produced this specific field's value (`sarvam_vision`, `tesseract`, `gemini_vision`, `qwen_slm`, `reconciled`; `shasan_slm`/`gpt4o_mini` retained only for historical rows from before engine swaps) |

`MANDATORY_FIELDS = [property_id, mutawalli_name, survey_number]` — used by validation rules for completeness checks.

**`validation_results`** — one row per rule evaluated per document, with `ValidationRuleResult` (`pass`/`fail`/`warning`).

**`reviews`** — one row per human review action (`ReviewAction`: `approve`/`correct`/`flag`), linked to the reviewing `User`.

**`field_corrections`** — audit log of every manual correction made during a review (original value, corrected value, field name) — intended to feed future model fine-tuning/training.

**`document_embeddings`** *(added for the vector search POC)*
| Field | Notes |
|---|---|
| `document_id` | PK, FK → `waqf_documents.id`, cascade delete |
| `searchable_text` | Concatenated text built from the document's extracted fields |
| `embedding` | `float[]` (Postgres `ARRAY(Float)`, or JSON on SQLite) — the Gemini embedding vector |
| `created_at` | Timestamp |

**`ocr_settings`** — singleton row for admin-tunable OCR behavior (e.g. `ocr_fallback_threshold`).

**`validation_rule_configs`** — admin-tunable per-rule enable/disable and parameters.

**`cer_benchmark_entries`** — stores Character Error Rate benchmark data per OCR engine per script, surfaced via `/admin/cer-benchmark` (this is the FSD's Week-9 "CER reported per script per engine" deliverable).

---

## 5. Document Processing Pipeline

Implemented in `app/services/ocr/pipeline.py`, orchestrated per-document by `_run_ocr_pipeline()` in `app/routers/documents.py`.

### 5.1 Stage 1 — Script detection (quick pass)

A fast, offline Tesseract pass runs first purely to produce a **script-type hint** (not necessarily used as the final transcription) — this hint, combined with a filename-based hint, is passed into Sarvam Vision's language parameter so its full read is requested in roughly the right script from the start.

### 5.2 Stage 2 — Primary OCR (confidence-driven fallback chain)

1. **Sarvam Vision 3B** is always tried first (materially more accurate than Tesseract or Gemini Vision on Urdu Nastaliq and Marathi Devanagari, per the FSD's Week-9 benchmark).
2. If Sarvam's own confidence is **below `ocr_fallback_threshold`** (default `0.6`, admin-tunable), **Tesseract** is also run and the two results are compared — Sarvam wins ties.
3. **Gemini Vision OCR** is reserved as a genuine last resort — only invoked if **both** Sarvam and Tesseract **fail outright** (not just score low confidence). This is a deliberate cost/rate-limit control: Gemini Vision OCR is expensive and rate-limited, so it's not triggered by low-confidence-but-successful reads from the other two engines.
4. **Script-mismatch correction:** if Sarvam was asked for one script family (e.g. Urdu/Arabic) but its own returned text is unambiguously in another (e.g. Devanagari), the engine is re-run once with the corrected language before accepting the result — catching the specific "Sanskrit document requested/read as Urdu" failure mode.

### 5.3 Stage 3 — Field extraction (Qwen2.5 via Ollama)

The winning engine's raw OCR **text** (not the image) is sent to a locally-running **Qwen2.5** model via Ollama's HTTP API, with a prompt that returns structured JSON for all six fields. This replaced an earlier regex/heuristic parser (`shasan_stub.py`, retained in the tree but unused). Never raises — if Ollama is unreachable or the model isn't pulled, all fields come back with `confidence = 0.0` rather than crashing the pipeline.

### 5.4 Stage 4 — Gemini backfill (conditional)

Only fields Qwen left below `GAP_FILL_THRESHOLD` (0.4) trigger a **Gemini Vision field-extraction** call against the original image — and only if that call returns something *better* than what Qwen already had. This keeps Gemini calls conditional rather than guaranteed on every document (rate-limit control, same rationale as Stage 2's Gemini gate).

### 5.5 Stage 5 — Script/transliteration guards

Two safety nets run after extraction (both idempotent, never raise):
- **Latin-transliteration guard** — if a script-sensitive field (e.g. `mutawalli_name`) came back in Latin script on a non-Latin document, it's treated as a transliteration, not a transcription: the value is moved to `value_en`, the main value is cleared, and confidence is dropped — surfacing it in the review UI as "needs manual entry" rather than showing a fluent-but-wrong value.
- **Wrong-native-script guard** — same treatment if a field is read in the wrong non-Latin script entirely (e.g. Urdu Nastaliq text on a Devanagari-only document).

### 5.6 Stage 6 — Translation pass

For non-English documents: Gemini attempts genuine transliteration/translation of each field into `value_en`. If Gemini is unreachable or not configured, a local, dependency-free **digit-conversion fallback** still populates `value_en` for numeric/date-shaped fields (`property_id`, `survey_number`, `registration_date`, `extent`) — name/place fields genuinely need Gemini and have no local fallback.

### 5.7 Stage 7 — Persistence, validation, vector indexing

- `overall_confidence` = average confidence across all six fields.
- `ExtractedField` rows are written; `document.status → extracted`.
- `validation.run_validations()` runs cross-field/mandatory/date-sanity rules (admin-tunable via `ValidationRuleConfig`) → `document.status → validated` or `flagged`.
- The **vector search pipeline** runs last (see [§6](#6-vector-search--duplicate-detection)) — indexing the document and checking for likely duplicate property records. Currently diagnostic-only (logged, not surfaced in the API/UI).

### 5.8 Engine-fallback summary table

| Situation | Behavior |
|---|---|
| Sarvam confident (≥ threshold) | Sarvam result used directly |
| Sarvam below threshold | Tesseract also run, best-of-two used |
| Sarvam **and** Tesseract both fail outright | Gemini Vision OCR invoked as last resort |
| All three fail | Document proceeds with empty OCR text; all fields end up 0 confidence; note added: *"queued for fully manual entry"* |
| Ollama unreachable | All fields 0 confidence from extraction stage; pipeline continues (translation/vector stages still attempt to run) |
| Gemini unreachable/rate-limited/timeout | Backfill and translation steps are skipped with a logged note; never crashes the pipeline |

---

## 6. Vector Search / Duplicate Detection

**Status:** proof-of-concept, backend-internal only — not part of the original FSD scope, added during implementation as an extension. No frontend surface currently; results are logged, not returned in any API response.

### 6.1 Purpose

Waqf property records frequently get re-registered, renewed, or duplicated across different entries (as seen even in the FSD's own sample documents — multiple register entries referencing overlapping properties/mutawallis). This module flags likely duplicates by comparing documents semantically, not just by exact field match.

### 6.2 How it works (`app/services/vector_search.py`)

1. **Build searchable text** — concatenates the document's extracted field values (mutawalli name, village, survey number, etc.) into a single text blob.
2. **Embed** — sends that text to Gemini's embedding model (`gemini-embedding-001`, 768 dimensions) to get a vector.
3. **Store** — persists the vector in `document_embeddings`.
4. **Search** — finds the top-K (default 5, `VECTOR_SEARCH_TOP_K`) most similar existing documents by cosine similarity.
   - Uses **native pgvector SQL** similarity search when the Postgres `vector` extension is available (checked/enabled automatically at startup via `_init_pgvector()` in `main.py`).
   - Falls back to **pure-Python cosine similarity** (looping over all stored embeddings) when pgvector isn't available — e.g. on SQLite in local dev — so the feature degrades gracefully rather than failing.
5. **Verify** — among the semantically similar candidates, the **Property ID field is the sole authoritative signal** for declaring an actual duplicate/match (`property_match_status`: `new_property`, `existing_property`, etc.) — semantic similarity alone never triggers a match; it only produces candidates for that final ID-based check.

### 6.3 Integration point

Runs as the last step of the same background OCR task (`_run_ocr_pipeline`), for **both** fresh uploads and reuploads (they share the same function). It never raises — a failure here (e.g. Gemini embedding call failing) is caught, logged, and does not affect the document's status.

### 6.4 Example output (from live testing)

```
document doc-730223183613 indexed=True candidates=[...] property_status=new_property matched_document=None
```

Querying similar documents directly returns ranked cosine-similarity scores, e.g.:
```
doc-f48929c66a06  0.8183
doc-ac735ea17610  0.7953
doc-67b45f8ef6b3  0.7873
```

### 6.5 Current limitation

Not yet surfaced to reviewers — a natural next step would be showing a "possible duplicate property" banner on the Review page when `property_match_status` indicates a likely match, so a human reviewer can confirm/reject it as part of their normal workflow.

---

## 7. Backend API Reference

Base path: `/api/v1`. All endpoints except `auth/register` and `auth/login` require a Bearer JWT.

### 7.1 Auth (`/auth`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/register` | Create a new user account |
| POST | `/login` | Authenticate, returns JWT |
| GET | `/me` | Current user profile |
| POST | `/logout` | Logout (client-side token discard) |

### 7.2 Documents (`/documents`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Upload a new scan; returns immediately (`201`), kicks off background OCR |
| POST | `/{document_id}/reupload` | Replace the scan on an existing document and reprocess it |
| GET | `` | List all documents (with filters) |
| GET | `/queue` | Documents pending review |
| GET | `/stats/summary` | Dashboard summary stats |
| GET | `/{document_id}` | Full document detail (fields, validations) — polled by the frontend during processing |
| POST | `/{document_id}/revalidate` | Re-run validation rules without re-running OCR |
| POST | `/{document_id}/review` | Submit a review action (approve/correct/flag); logs `FieldCorrection`s if corrected |
| GET | `/{document_id}/reviews` | Review history for a document |
| GET | `/translate/languages` | Supported languages for the translate-flag-reason feature |
| POST | `/translate` | Translate a flag/rejection reason |
| GET | `/{document_id}/file` | Stream/download the original scan (token-authenticated URL) |

### 7.3 Admin (`/admin`)

| Method | Path | Purpose |
|---|---|---|
| GET / PATCH | `/ocr-settings` | View/update the singleton OCR settings (e.g. fallback threshold) |
| GET | `/users` | List users |
| POST | `/users` | Create a user |
| PATCH | `/users/{user_id}/role` | Change a user's role |
| PATCH | `/users/{user_id}/active` | Activate/deactivate a user |
| GET / PATCH | `/validation-rules` / `/validation-rules/{key}` | View/update validation rule configuration |
| GET | `/cer-benchmark` | Character Error Rate benchmark data per OCR engine/script (FSD Week-9 deliverable) |

### 7.4 Reports (`/reports`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/throughput` | Records processed per unit time — the FSD's "X records/hour" headline metric |
| GET | `/status-breakdown` | Document counts by status |
| GET | `/confidence-distribution` | Histogram of extraction confidence across documents |
| GET | `/corrections` | History of manual field corrections (for future training data) |

---

## 8. Frontend Application

### 8.1 Pages (`src/pages`)

| Page | Purpose |
|---|---|
| `Login.tsx` | Authentication |
| `Dashboard.tsx` | Summary stats, quick status overview |
| `Upload.tsx` | Upload a new scan; polls for processing completion |
| `Review.tsx` | Core reviewer workflow — scan preview, extracted fields with confidence highlighting, validation results, approve/correct/flag actions, review history, reupload |
| `Reports.tsx` | Throughput, status breakdown, confidence distribution, corrections history charts (via `recharts`) |
| `Admin.tsx` | User management, OCR settings, validation-rule configuration, CER benchmark viewer |
| `Settings.tsx` | User-level settings |
| `NotFound.tsx` | 404 |

### 8.2 Structure

- **`src/api/`** — typed API client wrappers (axios) per resource (`documents.ts`, `auth.ts`, etc.)
- **`src/types/domain.ts`** / **`auth.ts`** — TypeScript types mirrored 1:1 with backend Pydantic schemas
- **`src/schemas/`** — `zod` validation schemas for forms
- **`src/hooks/`** — shared React hooks (e.g. polling logic)
- **`src/contexts/`** — auth/session context
- **`src/components/documents/`** — reusable review/upload UI pieces
- **`src/components/ui/`** — Radix-based design system primitives (button, dialog, dropdown, tabs, etc.)
- **`src/routes/`** — React Router route definitions, role-gated where relevant

### 8.3 Key UX behaviors (per FSD demo-gate criteria)

- Upload → status polling with visible processing state (targeting the FSD's "< 30 seconds" live-demo target)
- Confidence "heat" highlighting on extracted fields in `Review.tsx`, so a reviewer can visually scan for low-confidence values needing correction
- Designed for a non-technical reviewer to verify a document in under 60 seconds (FSD Week-11 DoD)

---

## 9. Auth & Roles

- JWT-based auth (`python-jose`), tokens issued on login/register.
- Two access-token lifetimes: a standard one (`access_token_expire_minutes`, default 480 min) and a longer one for regular users (`user_access_token_expire_minutes`, default 7 days).
- `Role` enum distinguishes at least a standard reviewing user role from a supervisor/admin role (seen in JWT payloads as `USER` / `SUPERVISOR` during testing) — supervisors get access to the Admin console and review approval actions.
- File-download URLs (`/documents/{id}/file`) are token-authenticated (short-lived signed token appended as a query param), not just relying on the session cookie/header, so preview links can be shared/opened directly.

---

## 10. Compliance & Risk (DPDP)

Per the FSD, this is a **blocking rule**, not a suggestion:

> No real records enter the pipeline until data-handling terms exist with the buyer. Synthetic-only is not a limitation in the pitch — it becomes the "DPDP-by-design" proof point.

**Implementation note:** the codebase includes a `dpdp.py` service and a `dpdp_status` field on every document (`checking` / `compliant` / `needs_review`), but **this is a status field, not an enforcement mechanism** — the actual constraint (real records must not enter the pipeline pre-DPDP-terms) is a **process/operational control**, not something the code currently blocks at the API level. This should be treated as a live risk item, not a solved one, until/unless enforcement is added at the point of upload.

### Other FSD-flagged risks

- **Nastaliq OCR quality** — materially harder than Devanagari; if CER stays high in practice, the FSD's fallback plan is to narrow the demo to Marathi/Devanagari and state Urdu as a calibrated roadmap item. The CER benchmark endpoint (`/admin/cer-benchmark`) exists specifically to make this judgment call with real numbers.
- **Funding source unverified** (state board vs. central UMEED money) — a go-to-market risk, not a technical one; noted here for completeness since it affects deployment timing.

---

## 11. Configuration Reference

All settings are read via `app/config.py` (Pydantic Settings), sourced from a `.env` file in `waqf-backend/`.

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Enables verbose logging when not `production` |
| `DATABASE_URL` | `postgresql://postgres:...@localhost:5432/waqf_docverify` | DB connection string; SQLite URLs also supported for local dev |
| `JWT_SECRET_KEY` | *(insecure dev default — change in prod)* | JWT signing secret |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `480` | Standard token lifetime |
| `USER_ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` (7 days) | Regular-user token lifetime |
| `SARVAM_API_KEY` | — | Enables Sarvam Vision OCR |
| `SARVAM_API_BASE_URL` | `https://api.sarvam.ai` | Sarvam API base |
| `GEMINI_API_KEY` | — | Enables all Gemini calls: OCR fallback, field-extraction backfill, translation, **and embeddings/vector search** |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model alias used |
| `GEMINI_MAX_RETRIES` | `2` | Retry count on Gemini call failure |
| `GEMINI_MIN_CALL_INTERVAL_SECONDS` | `4.5` | Process-wide throttle to avoid rate-limit issues |
| `SHASAN_SLM_API_URL` / `_API_KEY` | — | Unused currently (superseded by Ollama/Qwen) |
| `OLLAMA_URL` | `http://localhost:11434` | Local Ollama server address |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model used for field extraction — **must be pulled locally** (`ollama pull qwen2.5:7b`) |
| `OLLAMA_TIMEOUT_SECONDS` | `120` (project's `.env` sets `300`) | Max wait for an Ollama call |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — | Optional S3 storage credentials |
| `AWS_S3_BUCKET` / `AWS_S3_REGION` | — / `ap-south-1` | Optional S3 storage target; falls back to local disk if unset |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for vector search embeddings |
| `EMBEDDING_DIMENSIONS` | `768` | Embedding vector size |
| `VECTOR_SEARCH_TOP_K` | `5` | Number of similar-document candidates retrieved |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origin(s) |

---

## 12. Setup & Running Locally

### 12.1 Backend

```bash
cd waqf-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# create .env with at least DATABASE_URL, JWT_SECRET_KEY, and any AI service keys you have
uvicorn app.main:app --reload
```
Backend serves on `http://127.0.0.1:8000`; interactive API docs at `/docs`.

On startup, the app:
- Creates all tables if they don't exist (`Base.metadata.create_all`)
- Attempts to enable the `pgvector` Postgres extension (`_init_pgvector`) — silently falls back to Python cosine similarity if unavailable/on SQLite
- Runs lightweight in-code column migrations for OCR settings

### 12.2 Ollama (required for field extraction)

```bash
ollama pull qwen2.5:7b     # ~4.7GB, matches OLLAMA_MODEL default
ollama serve                # if not already running as a service
```

### 12.3 Frontend

```bash
cd waqf-frontend/waqf
npm install --legacy-peer-deps   # peer-dep conflict between eslint 9 and eslint-plugin-react-hooks — safe to ignore
npm run dev
```
Serves on `http://localhost:5173` by default; must match `CORS_ORIGINS` on the backend.

### 12.4 Minimum viable `.env` for a fully working demo

```env
DATABASE_URL=sqlite:///./waqf_docverify.db   # or a real Postgres URL for pgvector support
JWT_SECRET_KEY=change-me
SARVAM_API_KEY=...
GEMINI_API_KEY=...
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT_SECONDS=300
```

---

## 13. Known Limitations / Roadmap

| Area | Limitation | Notes |
|---|---|---|
| Vector search | Diagnostic-only, not surfaced in UI/API | See [§6.5](#65-current-limitation) |
| DPDP | Status field exists, but no code-level enforcement blocking real records | See [§10](#10-compliance--risk-dpdp) |
| Modi script | Not supported | FSD roadmap item, out of current scope |
| Ollama performance | CPU-only inference of a 7B model can take 30–90+ seconds per document on modest hardware | Consider a smaller model (e.g. `llama3.2`) if latency matters more than extraction quality tuned for Qwen |
| Gemini reliability | Occasional timeouts/503s observed in testing | Already handled gracefully (pipeline degrades, never crashes) — but reduces translation/backfill quality on affected documents |
| Shasan-SLM | Planned Pod-B integration point exists in config but is unused; superseded by Ollama/Qwen | Retained for compatibility only |
| Blockchain layer | Explicitly out of scope | Per FSD |
