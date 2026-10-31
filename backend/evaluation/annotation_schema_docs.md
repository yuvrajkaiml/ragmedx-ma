# Human Annotation Schema — RAGMedX-MA

**Schema version:** 1.0  
**Created:** Phase 2 — Workstream B  
**Evidence tier for any metrics derived from this schema:** HUMAN-VALIDATED (only when annotation_status = "complete")

---

## Purpose

This schema defines the structure for human annotation of RAGMedX-MA pipeline outputs. It is kept **separate from automated/model judgment outputs** (which carry the AUTOMATED MODEL JUDGE or SOFTWARE TEST evidence tier). Human-validated metrics must NEVER be combined in a single number with model-judged metrics.

Until an annotation record has `annotation_status = "complete"`, any metrics computed from it must be reported as **"pending annotation"**.

---

## Schema Fields

Each record is a single JSON object on its own line in a `.jsonl` file.

### `question_id`
**Type:** string  
**Description:** Unique identifier for the question. Must match the ID in the source dataset (e.g. BioASQ ID, MIRAGE ID). Used to join back to the pipeline output.  
**Required:** yes

### `reference_answer`
**Type:** string  
**Description:** The ground-truth reference answer from the dataset (e.g. BioASQ `ideal_answer`). This is the gold standard. Must not be modified by the annotator.  
**Required:** yes

### `model_answer`
**Type:** string  
**Description:** The pipeline's generated answer. Empty string ("") for abstentions. Taken verbatim from `predicted_answer` in the result file.  
**Required:** yes

### `claims`
**Type:** list of objects  
**Description:** The model answer decomposed into individual verifiable claims. Each claim is an independent factual assertion extracted from `model_answer`. For abstentions, this list is empty.  
**Each object has:**
- `claim_id`: string — unique within this record (e.g. "c1", "c2")
- `claim_text`: string — the verbatim or lightly rephrased claim

**Required:** yes (empty list for abstentions)

### `evidence_ids`
**Type:** list of strings  
**Description:** The IDs of the retrieved documents/chunks that the pipeline cited. These should match the `chunk_id` or `document_id` fields in the retrieved evidence. Order is not significant.  
**Required:** yes (empty list for abstentions or plain-LLM-no-rag)

### `support_labels`
**Type:** object (mapping)  
**Description:** A mapping from `claim_id` to a support label indicating whether the retrieved evidence supports, contradicts, or is neutral toward each claim.  
**Valid values per claim:**
- `"supported"` — at least one retrieved document directly supports the claim
- `"partial"` — retrieved documents partially support but do not fully confirm the claim
- `"unsupported"` — no retrieved document supports this claim (but no contradiction either)
- `"contradicted"` — at least one retrieved document explicitly contradicts the claim
- `"unverifiable"` — the claim makes statements that cannot be verified from the retrieved evidence (e.g. references to data not in the index)

**Required:** yes (empty object for abstentions)

### `correctness_label`
**Type:** string  
**Description:** Overall correctness assessment of the model answer relative to the reference answer and medical knowledge.  
**Valid values:**
- `"correct"` — model answer is factually correct and substantially addresses the question
- `"partially_correct"` — model answer contains correct elements but also incorrect or incomplete elements
- `"incorrect"` — model answer is factually wrong or completely misses the question

**Required:** yes; use `"incorrect"` for abstentions (abstaining when an answer exists = incorrect; see `abstention_label` for the abstention-specific judgment)

### `abstention_label`
**Type:** string  
**Description:** Judgment on whether the model's decision to abstain (or not abstain) was appropriate.  
**Valid values:**
- `"should_abstain"` — there was genuinely insufficient evidence; abstaining was the right call
- `"should_answer"` — sufficient evidence existed; the model should have given an answer
- `"acceptable_either"` — the question was borderline; either abstaining or answering was defensible

**Required:** yes

### `annotator_notes`
**Type:** string  
**Description:** Free-text notes from the annotator. Should document any uncertainty, edge cases, or reasons for the assigned labels. Can be empty ("").  
**Required:** no (defaults to "")

### `annotation_status`
**Type:** string  
**Description:** Whether this record has been reviewed and finalised by a human annotator.  
**Valid values:**
- `"pending"` — not yet reviewed; all labels are provisional or blank
- `"complete"` — reviewed and signed off by a qualified annotator

**Important:** Any metric computed using records with `annotation_status = "pending"` MUST be reported as **"pending annotation"** and tagged with evidence tier HUMAN-VALIDATED (PENDING). Do NOT report such metrics as final results.

**Required:** yes

---

## Evidence Tier Rules

| annotation_status | Allowed evidence tier | Metric label |
|---|---|---|
| `"complete"` | HUMAN-VALIDATED | Report numeric value |
| `"pending"` | HUMAN-VALIDATED (PENDING) | Report as "pending annotation" |
| (automated) | AUTOMATED MODEL JUDGE | Report with tier label |
| (software test) | SOFTWARE TEST | Report with tier label |

**CRITICAL:** Never mix HUMAN-VALIDATED and AUTOMATED MODEL JUDGE in the same metric number. They must be reported separately with their tier labels.

---

## Separation from Model Judgment Outputs

Model-judge outputs (e.g. outputs from the `CriticAgent` or any NLI-based verifier) use a different schema and must be stored in separate files. The `CriticAgent`'s `evidence_verification` field is tagged AUTOMATED MODEL JUDGE and may not be treated as equivalent to a human-assigned `support_label`.

---

## Annotation Instructions (brief)

1. Read the `question` from the source dataset (join on `question_id`).
2. Read `reference_answer` and `model_answer`.
3. Decompose `model_answer` into individual claims and populate `claims`.
4. For each claim, decide which of the `evidence_ids` support, contradict, or are neutral; assign `support_labels`.
5. Assign `correctness_label` based on comparison to `reference_answer` and medical knowledge.
6. Assign `abstention_label` based on whether abstention was appropriate.
7. Add any notes to `annotator_notes`.
8. Set `annotation_status` to `"complete"`.
9. Do NOT edit `reference_answer` or `model_answer`.
10. Use the existing `evidence_ids` list; do not add IDs not present in the pipeline output.
