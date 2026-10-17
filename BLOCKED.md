# RAGMedX-MA — Blocked Items (root mirror)

**Last updated:** 2026-10-17 (Phase 6 reconciliation)
**Branch:** `yuvraj/llm`

This root file mirrors `docs/progress/BLOCKED.md` and is kept consistent with it.
See that file for full detail. Statuses: RESOLVED | CARRY-FORWARD | NOT MEASURABLE.

| ID | Item | Status | Note |
|----|------|--------|------|
| BLOCKED-001 | MedRGB dataset absent | NOT MEASURABLE (carry-forward) | Not in repo or MIRAGE; MedRGB metrics omitted / NOT MEASURED. |
| BLOCKED-002 | Eval-index leakage smoke test | RESOLVED (Phase 5) | Canary PASS; first retrieved PMID 17128671 not in eval-question set. |
| BLOCKED-003 | 500-question eval split | RESOLVED (built); full 500×8 sweep CARRY-FORWARD | Split 100/150/250 built, leakage checks pass; full GPU sweep (~40+ h) not run — budget-bounded pilot (16×8) run instead. PubMedQA retrieval-quality NOT MEASURED. |
| BLOCKED-004 | NLI model not present | CARRY-FORWARD | CPU NLI model not downloaded; claim verification default OFF; claim support NOT MEASURED. |
| BLOCKED-005 | Human annotation labels | CARRY-FORWARD | Schema exists, annotation_status pending; abstention/faithfulness metrics NOT MEASURED. |
