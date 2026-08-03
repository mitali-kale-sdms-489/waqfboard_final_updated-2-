# POC-C — Waqf Record Verifier (DocVerify Chain Extension)

**Pod:** C · 2 trainees
**BD alignment:** Pipeline #3 — Waqf / Minority Development digitization. 48% digitized, 2.13 lakh records stuck at verification, national scrutiny, no competitive noise. Strategy per decision principles: **demo, not proposal** — distress buyers respond to working flows.

---

## 1. Objective

A working demo of AI-assisted verification for Waqf-style records: multi-script OCR (Urdu, Marathi/Devanagari, and old formats), field extraction, cross-field validation, and a human-in-the-loop review UI with per-document confidence scoring — extending DocVerify Chain.

## 2. Scope

**In:**
- OCR for Urdu (Nastaliq) + Marathi Devanagari; Sarvam Vision 3B benchmark for scanned documents
- Field extraction: property ID, mutawalli name, survey number, registration date, extent
- Validation rules: mandatory-field completeness, date sanity, cross-document consistency
- Review UI: side-by-side scan vs extracted fields, confidence heat, approve/correct/flag actions; corrections logged for future training
- **Synthetic/public sample documents ONLY** (see risk #1)

**Out:** Real Waqf board records until DPDP handling is defined. Modi script (flag as roadmap item — relevant to old land/Waqf records — but out of 12-week scope). Blockchain layer.

## 3. Stack

Sarvam Vision 3B · Tesseract (urd + mar) / Surya · Shasan-SLM API (Pod B) for extraction assist · FastAPI + simple React UI · PostgreSQL

## 4. Deliverables & Definition of Done

| Wk | Deliverable | DoD |
|---|---|---|
| 1–8 | Shared curriculum + domain prep | 100-document synthetic sample set built (template-generated Waqf-style records in Urdu + Marathi with known ground truth) |
| 9 | Multi-script OCR benchmark | CER reported per script per engine on the sample set; engine selected per script |
| 10 | Extraction + validation | ≥ 90% field-level extraction accuracy on the synthetic set; validation rules firing correctly on 20 seeded-error documents |
| 11 | Review UI | A non-technical user can verify a document in < 60 seconds; corrections persisted |
| 12 | End-to-end demo | 50 documents processed live: scan in → verified record out, with a throughput number ("X records/hour per reviewer vs manual baseline") |

## 5. Demo-gate acceptance criteria (Wk 12)

1. Live: upload a scanned record → extracted fields with confidence in < 30 seconds.
2. Seeded-error catch rate: ≥ 18 of 20 deliberately corrupted documents flagged.
3. The throughput headline is defensible — this single number is the pitch to a department with 2.13 lakh stuck records.

## 6. Skills mastered

Multi-script OCR · document AI · field extraction with LLM assist · validation-rule design · human-in-the-loop UX · confidence scoring.

## 7. Pod-specific risks

- **DPDP (blocking rule):** no real records enter the pipeline until data-handling terms exist with the buyer. Synthetic-only is not a limitation in the pitch — it becomes the "DPDP-by-design" proof point for the Suraksha-Stack annex.
- **Nastaliq OCR quality:** materially harder than Devanagari; if CER stays high, narrow the demo to Marathi/Devanagari records and state Urdu as calibrated-roadmap, honestly.
- **Funding source still unverified** (state board vs central UMEED money): do not let the pod's demo readiness pull you into pitching before that check clears.
