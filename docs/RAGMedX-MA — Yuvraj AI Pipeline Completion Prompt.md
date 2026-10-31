# RAGMedX-MA — Yuvraj AI Pipeline Completion Prompt

## How to use this prompt

Paste this entire document into Antigravity IDE while the repository is open on the `yuvraj/llm` branch. Use **Planning mode**, not Fast mode. Execute **one phase at a time**: begin with Phase 0, stop after the phase, report the evidence, and wait for explicit approval before continuing.

Do not enable any setting that automatically approves plans, artifacts, commits, pushes, deletions, or changes outside the stated scope. This prompt intentionally requires human review at ownership boundaries and before irreversible actions.

---

## Role and objective

You are the implementation agent for **RAGMedX-MA**, a multi-agent retrieval-augmented medical question-answering project. The developer responsible for this work is **Yuvraj Karunakaran**. Work directly in the current repository and continue iteratively until the feasible completion gates below are satisfied.

Yuvraj owns the multi-agent LLM pipeline: the LLM wrapper, prompt management, agents, orchestration, confidence calibration, abstention, traces, and evaluation scaffolding. Sana owns the concrete retrieval engine and knowledge-base implementation. Pranav owns the FastAPI, database, authentication, WebSocket, and master API schemas. Shravani owns the frontend.

Your job is to inspect the existing implementation first, preserve working code and user changes, complete the Yuvraj-owned AI pipeline, integrate safely through existing boundaries, and optimize only after correctness has been demonstrated with executable tests.

The repository is the source of truth when it conflicts with this prompt or an audit. Do not treat a filename, TODO comment, dependency declaration, design document, or prior audit as proof that a feature exists or works. Verify current files, imports, schemas, callers, tests, and runtime behavior before editing.

---

## Non-negotiable operating rules

1. **Inspect before editing.** Begin by reading `git status`, the current diff, branch and recent commits, project instructions, dependency files, test configuration, relevant schemas, current agent modules, the LLM wrapper, retrieval interfaces, query service, and mounted API routes.

2. **Preserve user work.** Never run destructive resets, broad automated rewrites, force checkout, history rewriting, or deletion of untracked files. Before changing anything, record modified and untracked files and inspect their diffs.

3. **No automatic Git actions.** Do not commit, push, merge, rebase, reset, delete, or rewrite history. Prepare a reviewable working tree and report exact files changed. A human must explicitly approve any commit or push.

4. **Respect ownership boundaries.** Modify another teammate’s module only when a verified integration mismatch makes it necessary, keep the change minimal, document it, and stop for approval before making the change. If a task appears to require crossing the boundary, report the conflict rather than proceeding.

5. **Use compatibility adapters.** Do not duplicate or redefine Pranav’s master schemas. If a real mismatch exists, add a narrow adapter and contract tests rather than silently breaking callers.

6. **Never fake completion.** Do not claim that real GPU inference, a downloaded medical model, production FAISS/vector search, MedQA benchmarking, or MIRAGE evaluation works unless it was actually executed and the result was recorded. Mock results must be clearly labeled as mock.

7. **Make mock mode explicit.** A mock response must never be indistinguishable from a real model response. Mock mode must be selected explicitly or surfaced prominently through a typed result, metadata, or error. Silent fallback is prohibited.

8. **No unsupported medical claims.** The generator may only produce claims grounded in supplied evidence. Every citation marker must map to an evidence item. If evidence is empty, weak, contradictory, unsupported, or invalid, return a structured abstention or error instead of inventing facts or citations.

9. **Do not expose private chain-of-thought.** Keep internal model reasoning private. Expose only the final answer, concise verification notes, evidence mappings, confidence components, abstention reason, and trace metadata needed by the UI and evaluation.

10. **Bound every loop.** Retrieval refinement is capped at `MAX_RETRIEVAL_ITERATIONS`, currently expected to be `3`. Critic/generator revision is capped at `MAX_CRITIC_ROUNDS`, currently expected to be `2`. Every retry must have a counter, exit condition, and trace event. Caps must be enforced in code even if configuration is malformed.

11. **Keep tests reproducible.** Unit and integration tests must run without network access, GPU access, a downloaded LLM, or a large external dataset. Use deterministic fake providers and small clearly labeled fixtures. Real-GPU tests must skip cleanly when CUDA is unavailable.

12. **Validate immediately.** Run the narrowest relevant test first, then the broader suite. Never weaken assertions, hide failures, or skip tests merely to make the suite green. Any intentional skip must have a written reason in the progress file.

13. **One phase at a time.** At the end of each phase, stop and provide a concise report containing changed files, commands, results, tests, blockers, and the next proposed phase. Do not continue automatically without explicit approval.

---

## Required progress file

Create or maintain `docs/YUVRAJ_ANTIGRAVITY_PROGRESS.md`.

At the beginning and end of every phase, update it with:

- current date and time;
- current phase and selected acceptance gate;
- files inspected and changed;
- commands executed and their results;
- tests passed, failed, skipped, or blocked;
- mock-versus-real inference status;
- known risks and external blockers;
- the next smallest actionable step.

Do not mark a gate complete unless its corresponding command or test actually passes. In the final report, use exactly one of these statuses: **COMPLETE**, **COMPLETE WITH EXTERNAL BLOCKERS**, or **INCOMPLETE — ACTION REQUIRED**.

---

## Hard scope boundary

### Files and directories you may create, edit, or refactor

You may work in:

- `backend/app/agents/**`;
- `backend/app/llm/**`;
- `backend/prompts/**`;
- `backend/tests/` files covering `agents/` or `llm/`, plus `test_document_loader.py`, `test_medqa_loader.py`, `test_local_llm.py`, and `run_real_gpu_test.py`;
- `backend/evaluation/**` and/or `backend/app/evaluation/**`, after determining which is canonical;
- `backend/data/datasets/medqa/`, only to add a small synthetic or fixture sample for tests;
- `backend/requirements.txt` and root `requirements.txt`, only for the explicitly required testing/dependency corrections and only after inspecting which file is authoritative;
- `docs/YUVRAJ_ANTIGRAVITY_PROGRESS.md`.

You may add exactly one temporary retrieval mock/stub implementing `retrieve_evidence()` with Sana’s exact verified signature. Isolate it clearly so replacing it with Sana’s real function is a one-line dependency change. Do not place it inside either competing retrieval package.

### Files and directories you must not implement or modify without explicit approval

Do not implement, consolidate, delete, or refactor Sana’s retrieval engine:

- `backend/app/rag/chunking/**`;
- `backend/app/rag/embeddings/**`;
- `backend/app/rag/vectorstore/**`;
- `backend/app/rag/retrievers/**`;
- `backend/app/rag/rerankers/**`;
- `backend/app/retrieval/**`, except for a narrowly verified compatibility change approved in advance;
- `backend/app/retrieval/document_loader.py` merely to make a test pass;
- `backend/app/retrieval/` versus `backend/app/rag/` package consolidation.

Do not modify Pranav-owned files without explicit approval:

- `backend/app/api/**`;
- `backend/app/schemas/response.py`;
- `backend/app/schemas/query.py`;
- `backend/app/schemas/document.py`;
- authentication, database, WebSocket, Docker, or unrelated backend infrastructure.

Do not modify frontend code.

If integration requires a change outside the permitted scope, stop and report the exact file, mismatch, proposed minimal change, and owner decision required.

---

## Verified facts and decisions to re-check

The following are working assumptions from the latest audit, not substitutes for inspection:

- The active branch is `yuvraj/llm`.
- Pranav’s backend and Sana’s retrieval skeleton have been merged into this branch.
- The local LLM wrapper targets `Qwen/Qwen3-8B` with optional 4-bit quantization.
- Six agent modules exist: query understanding, iterative retrieval, generator, critic, confidence calibration, and abstention decision.
- The orchestrator and agent service may still be placeholders.
- Async pytest configuration may be missing.
- MedQA data may be absent.
- There may be overlapping `app/retrieval/` and `app/rag/` structures.
- The legacy `/chat` route may be dead code while the mounted v1 route is the production path.

Verify every item against the current repository before acting. Do not delete or migrate anything solely because this list says a structure is obsolete.

### Fixed technical decisions

- Use `Qwen/Qwen3-8B` with optional 4-bit quantization unless the repository proves this is impossible. Do not switch to 14B merely as an optimization.
- Focus first on proving the configured model actually runs or clearly reporting the environment blocker.
- Do not allow silent mock fallback.
- Never use `default_factory=uuid4` or similar automatic IDs for document, chunk, or evidence identifiers. Such IDs must be required and sourced from the producer of the real data. Deterministic hash-based IDs are acceptable only for clearly labeled synthetic fixtures.
- Import Pranav’s schemas directly. Keep one boundary function, `build_medical_response()`, as the only place where pipeline internals are translated into master response schema classes.
- Internal agent state remains pipeline-native. Do not force every internal state object to equal an API response schema.
- LangGraph is optional. Use it only if already compatible with the repository and if it improves correctness. A small explicit typed async state machine is acceptable and preferable to introducing a framework merely for appearance.

---

## Verified schema and behavior contracts

Read the actual current files first. The following design contract must be reconciled against current code, not blindly copied over it.

Expected enums include:

```python
class AgentType(str, Enum):
    QUERY_UNDERSTANDING = "query_understanding"
    ITERATIVE_RETRIEVAL = "iterative_retrieval"
    EVIDENCE_RERANKING = "evidence_reranking"
    GENERATOR = "generator"
    CRITIC = "critic"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    ABSTENTION_DECISION = "abstention_decision"

class SourceType(str, Enum):
    PUBMED = "pubmed"
    PMC_OPEN_ACCESS = "pmc_open_access"
    MEDQA = "medqa"
    BIOASQ = "bioasq"
    UPLOADED_PDF = "uploaded_pdf"
    STATPEARLS = "statpearls"

class ConfidenceCategory(str, Enum):
    HIGH = "high"          # >= 0.80
    MODERATE = "moderate"  # 0.65–0.79
    LOW = "low"             # < 0.65

class CriticVerdict(str, Enum):
    VERIFIED = "verified"
    REVISED = "revised"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
```

Expected evidence and output concepts include:

```python
class EvidenceSnippet(BaseModel):
    chunk_id: UUID
    document_id: UUID
    source_type: SourceType
    title: str
    journal: Optional[str]
    publication_year: Optional[int]
    cited_text: str
    relevance_score: float
    cross_encoder_score: float
    evidence_quality_score: float
    citation_marker: str

class CriticReviewOutput(BaseModel):
    verdict: CriticVerdict
    factual_consistency_score: float = Field(..., ge=0.0, le=1.0)
    medical_correctness_score: float = Field(..., ge=0.0, le=1.0)
    evidence_alignment_score: float = Field(..., ge=0.0, le=1.0)
    hallucination_detected: bool
    unsupported_claims: List[Dict[str, str]]
    critic_feedback: str
    suggested_revisions: List[str]
    review_round: int

class ConfidenceScoreOutput(BaseModel):
    retrieval_confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_quality_score: float = Field(..., ge=0.0, le=1.0)
    answer_consistency_score: float = Field(..., ge=0.0, le=1.0)
    inter_source_agreement: float = Field(..., ge=0.0, le=1.0)
    composite_confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: Optional[float]
    confidence_category: ConfidenceCategory
    passes_threshold: bool
    threshold_used: float = 0.65
```

The response union has three arms:

```python
MedicalResponse = Union[
    VerifiedAnswerResponse,
    AbstentionResponse,
    ErrorResponse,
]
```

The actual current Pydantic definitions are authoritative. Do not create local mirrors if the names or module paths differ.

### Configuration values

Read these from the existing configuration object rather than hard-coding them:

- `CONFIDENCE_THRESHOLD = 0.65`;
- `MAX_RETRIEVAL_ITERATIONS = 3`;
- `MAX_CRITIC_ROUNDS = 2`;
- `RETRIEVAL_WEIGHT = 0.25`;
- `EVIDENCE_QUALITY_WEIGHT = 0.30`;
- `ANSWER_CONSISTENCY_WEIGHT = 0.25`;
- `INTER_SOURCE_AGREEMENT_WEIGHT = 0.20`.

The confidence formula is:

```text
composite = 0.25 * retrieval_confidence
          + 0.30 * evidence_quality_score
          + 0.25 * answer_consistency_score
          + 0.20 * inter_source_agreement
```

Clamp components and composite scores to `[0, 1]`. Apply the configured threshold consistently.

### Exact error and abstention copy

Use these strings exactly where the corresponding condition applies:

| Scenario | Required copy |
|---|---|
| Invalid or empty query | `Please enter a valid medical question.` |
| Retrieval failure | `No relevant medical literature found.` |
| Model failure | `Unable to generate a response. Please try again.` |
| Low confidence | Structured abstention with an explanation in `abstention_reason` |
| Contradictory evidence | `Sources contain conflicting information.` |
| Out-of-domain query | `Outside the scope of medical literature.` |

### Critic routing

The critic has four states:

- `verified` or `revised`: proceed to confidence calibration;
- `needs_revision` with `revision_count < MAX_CRITIC_ROUNDS`: return to the generator with critic feedback;
- `needs_revision` at the cap: proceed to confidence calibration, record the cap-out in trace details, and let the confidence score determine the final response;
- `rejected`: do not silently skip diagnostics. Proceed to confidence calibration or the explicit safety/abstention path according to current schema contracts, record the rejection in trace details, and ensure an unsafe answer cannot be returned as verified.

Do not invent downstream semantics if the current repository defines them differently. Report a contract conflict before changing it.

---

## Public interfaces and retrieval boundary

The intended public pipeline contract is conceptually:

```python
async def run_agent_pipeline(
    query_id,
    original_question,
    processed_query,
    evidence,
    include_trace=False,
):
    # returns (MedicalResponse, Optional[AgentTraceOutput])
    ...
```

Inspect actual types and imports before implementing. Preserve compatibility with Pranav’s caller through an adapter if necessary.

The intended retrieval boundary is conceptually:

```python
async def retrieve_evidence(
    query: str,
    optimized_query: str | None = None,
    query_embedding: list | None = None,
    max_iterations: int = 3,
    top_k_initial: int = 50,
    top_k_final: int = 5,
    enable_contradiction_check: bool = True,
):
    # returns evidence items plus iteration outputs
    ...
```

Use Sana’s actual signature when it differs. For tests and temporary orchestration, use one isolated deterministic stub with synthetic, schema-valid evidence. Do not pretend that the stub is production retrieval.

---

## Autonomous completion workflow

Repeat this workflow within the approved phase only:

1. Read the progress file and establish current state.
2. Run a safe status and diff check.
3. Identify the highest-priority failing gate.
4. Inspect the smallest relevant implementation surface, schemas, tests, and callers.
5. Search for existing types and helpers before creating abstractions.
6. Implement the smallest coherent change.
7. Add or update deterministic tests in the same phase.
8. Run the narrowest test, then the broader relevant suite.
9. Inspect the diff for accidental API changes, schema drift, duplicate code, silent fallback, hard-coded secrets, network calls, unbounded loops, or out-of-scope edits.
10. Update the progress file and stop for review.

Use commands equivalent to these where appropriate, adjusting for the repository’s actual working directory and supported environment:

```bash
python -m compileall -q backend/app
python -m pytest backend/tests --collect-only -q
python -m pytest backend/tests -q
```

---

# Required implementation phases

## Phase 0 — Baseline and trustworthy test recovery

Before implementation changes:

1. Create `docs/YUVRAJ_ANTIGRAVITY_PROGRESS.md`.
2. Capture branch, status, recent commits, modified files, untracked files, and current diff.
3. Inspect project instructions, root and backend requirements, Python version, pytest configuration, schemas, agent files, LLM wrapper, retrieval interfaces, services, and mounted routes.
4. Run compilation and test collection to establish a baseline.
5. Resolve only the blockers required for trustworthy validation:
   - missing imports that prevent collection;
   - compatible async pytest configuration and marker handling;
   - the `DocumentLoader` test boundary, without taking over Sana’s retrieval implementation;
   - a small fixture PDF or explicit canonical-loader test decision if required;
   - a small synthetic MedQA JSONL fixture if real data is absent;
   - root/backend dependency divergence, without blindly copying one file over another.
6. Inspect local changes to `backend/app/schemas/document.py`. Do not commit, discard, or rewrite them. Report whether they appear required and whether Pranav/Sana approval is needed.

Acceptance gate: backend tests collect without import errors or unknown async-marker warnings, or every remaining blocker is explicitly documented with its owner and reason.

Stop for review.

## Phase 1 — Stabilize Yuvraj-owned components

Inspect and improve existing components without rewriting working code unnecessarily:

1. Make the local LLM wrapper expose explicit `mock`, `local`, and/or `auto` runtime modes.
2. Make model-loading failures observable and typed. Allow mock fallback only when configuration explicitly permits it.
3. Log real failures at error level and ensure callers can distinguish mock, local, and failed execution.
4. Normalize prompt loading and versioning under the existing prompt directory.
5. Use strict structured-output models for query understanding, generator output, critic verdicts, confidence output, and abstention decisions.
6. Make JSON parsing robust to fenced JSON, harmless preambles, whitespace, malformed output, and validation errors. On unrecoverable parse failure, return a typed failure rather than guessing.
7. Confirm confidence uses the four configured components and weights.
8. Confirm every state transition is represented in the trace.
9. Ensure all six existing agent unit tests remain meaningful and deterministic.

Acceptance gate: component tests pass in deterministic mock mode for verified, revised, abstained, parser-failure, model-failure, empty-evidence, invalid-citation, and explicit-mock cases. Real-GPU validation may remain environment-blocked, but the status must be recorded honestly.

Stop for review.

## Phase 2 — State contract and temporary retrieval boundary

1. Read the actual current `agents/state.py`, `app/schemas/query.py`, `app/schemas/response.py`, and retrieval interfaces.
2. Define or align one typed internal `AgentState` without duplicating master API schemas.
3. Ensure state can carry query identity, original and processed query, evidence, retrieval iterations, draft answer, critic reviews, confidence, abstention decision, trace, counters, status, and typed error information.
4. Remove automatic IDs for real evidence/document/chunk fields.
5. Create one isolated temporary retrieval stub outside `app/retrieval/` and `app/rag/` only if needed for tests. It must return deterministic schema-valid fixture evidence and match Sana’s verified boundary.
6. Add contract tests for evidence shape, deterministic fixture IDs, empty retrieval, and retrieval failure.

Acceptance gate: state and retrieval contract tests pass, and no third retrieval package or competing schema mirror is introduced.

Stop for review.

## Phase 3 — Orchestration state machine

Implement the missing orchestrator using LangGraph only if compatible and useful; otherwise use an explicit typed async state machine.

The logical flow is:

```text
START
  → query understanding
  → iterative retrieval / evidence handoff
  → evidence validation and boundary reranking
  → generator
  → critic
  → bounded revision loop back to generator when required
  → confidence calibration
  → abstention decision
  → build_medical_response()
  → END
```

Requirements:

1. Accept a valid supplied processed query; invoke query understanding only when necessary.
2. Consume supplied evidence through the canonical evidence type.
3. Use an injected retrieval provider when full-query retrieval is needed.
4. Validate evidence shape, remove duplicates, reject invalid citation metadata, preserve quality scores, and record contradiction/quality flags.
5. Do not pretend to implement Sana’s production cross-encoder or vector search.
6. Generate only evidence-grounded text with citation markers that map to supplied evidence.
7. Critically review factual consistency, medical correctness, evidence alignment, hallucination risk, and unsupported claims.
8. Apply the four-state critic routing rules and enforce the maximum review count.
9. Calculate the four-component confidence score from configuration.
10. Abstain when confidence is below threshold, evidence is unsafe, citations are invalid, or safety gates fail.
11. Append a trace step for every node actually visited, including repeated nodes, with status, timestamps, duration, and useful non-chain-of-thought details.
12. Never return a verified answer containing an unsupported citation or unsafe unsupported claim.

Acceptance gate: deterministic tests prove one verified or revised path and one structured abstention path through the complete pipeline. Additional tests must prove retrieval and critic loops terminate at their caps.

Stop for review.

## Phase 4 — Response builder and public interface

1. Implement `build_medical_response()` as the sole boundary function translating internal state into the real `VerifiedAnswerResponse`, `AbstentionResponse`, or `ErrorResponse` classes.
2. Implement the expected `run_agent_pipeline()` public interface at the repository’s verified package path.
3. Preserve compatibility with the existing caller through a narrow adapter if required.
4. Ensure `include_trace=False` does not expose trace data unnecessarily while still allowing diagnostics when requested.
5. Return typed error responses for invalid input, retrieval failure, model failure, parser failure, and unrecoverable orchestration failure.
6. Add contract tests for all three response arms, exact error copy, citations, confidence, processing time, token accounting where available, and optional trace.

Acceptance gate: the public pipeline is importable, deterministic mock tests return valid Pydantic response models, and no local mirror of Pranav’s schemas exists.

Stop for review.

## Phase 5 — Backend integration through existing boundaries

Inspect the mounted v1 query route, query service, response schemas, database models, authentication, and WebSocket payload types. Do not modify Pranav-owned files without explicit approval.

If the existing service boundary supports it, connect the mock pipeline through the current service layer without breaking authentication, persistence, serialization, or response discrimination. If the route or service requires an out-of-scope change, stop and report the exact mismatch rather than editing automatically.

Add deterministic integration tests for:

- authenticated query submission, when test infrastructure already supports it;
- verified or revised response;
- low-confidence abstention;
- malformed or empty query;
- model and retrieval failure;
- citation and confidence serialization;
- optional trace serialization;
- streaming/WebSocket behavior only if already exposed and testable.

Do not repair dead legacy routes merely because they contain stale code unless they are mounted and reachable in the verified production path.

Acceptance gate: the mounted query API can execute the mock pipeline and return the documented response shape, or the exact external integration blocker is recorded.

Stop for review.

## Phase 6 — Prompt centralization and evaluation scaffold

Only after Phases 0–5 are stable:

1. Reconcile inline agent prompts against the canonical prompt wording in the repository’s backend schema/design document.
2. Centralize prompts under the existing prompt directory and add version metadata without changing semantics unnecessarily.
3. Determine whether `backend/evaluation/` or `backend/app/evaluation/` is canonical. Do not duplicate evaluation systems.
4. Add a small deterministic evaluation scaffold over synthetic fixtures. Clearly label it as a scaffold.
5. Implement only metrics that can be defined and tested honestly, such as accuracy, retrieval precision/recall at k, abstention rate, latency, citation support, and simple evidence-overlap diagnostics.
6. Report unavailable MIRAGE, MedQA, or production retrieval results as “not run” or “blocked,” never as measured results.
7. Add tests for metrics and a mock evaluation command.

Acceptance gate: mock evaluation runs on the small fixture without crashing, reports measured mock results separately from unavailable real benchmarks, and all metric tests pass.

Stop for review.

## Phase 7 — Final validation and documentation

Run the complete feasible validation suite:

```bash
python -m compileall -q backend/app
python -m pytest backend/tests --collect-only -q
python -m pytest backend/tests -q
```

Also run targeted public-interface, orchestration-loop, mock-mode, schema-validation, and integration tests. Inspect the final diff for:

- changes outside approved scope;
- accidental API or schema changes;
- duplicate models or competing implementations;
- silent mock fallback;
- unsupported medical claims;
- invalid citation mapping;
- unbounded loops;
- hard-coded secrets;
- network calls in deterministic tests;
- missing documentation of environment blockers.

Update `docs/YUVRAJ_ANTIGRAVITY_PROGRESS.md` with:

- implementation summary;
- exact files changed;
- exact commands run;
- test counts and pass/fail/skip status;
- mock-versus-real inference status;
- retrieval and data limitations;
- API integration status;
- remaining blockers owned by Sana, Pranav, the environment, or Yuvraj;
- recommended next team action;
- final status: `COMPLETE`, `COMPLETE WITH EXTERNAL BLOCKERS`, or `INCOMPLETE — ACTION REQUIRED`.

Stop and wait for final human review. Do not commit or push.

---

## Minimum required behavioral tests

Ensure tests equivalent to these exist and assert behavior rather than imports:

1. `test_public_run_agent_pipeline_contract`
2. `test_pipeline_verified_answer_with_valid_citations`
3. `test_pipeline_revision_loop_stops_after_success`
4. `test_pipeline_revision_loop_caps_at_two_rounds`
5. `test_pipeline_abstains_when_confidence_is_below_0_65`
6. `test_pipeline_abstains_when_evidence_is_empty_or_unsupported`
7. `test_pipeline_rejects_citation_not_present_in_evidence`
8. `test_pipeline_handles_llm_parse_failure_without_fabrication`
9. `test_pipeline_handles_retrieval_or_model_failure`
10. `test_agent_trace_has_ordered_steps_and_durations`
11. `test_mock_mode_is_explicit_and_does_not_claim_real_inference`
12. `test_retrieval_provider_contract_and_fixture_evidence`
13. `test_query_api_serializes_verified_and_abstention_responses`, when the API boundary is testable without out-of-scope changes
14. `test_no_infinite_retrieval_or_revision_loop`
15. `test_prompt_version_and_structured_output_validation`

Use `unittest.mock.AsyncMock` or deterministic fake providers for model and retrieval tests. Do not require GPU, network access, external datasets, or downloaded models for ordinary test execution.

---

## Required phase-end report

At the end of every phase, report:

```text
PHASE: <number and title>
STATUS: <PASS / PASS WITH BLOCKERS / INCOMPLETE>
FILES INSPECTED: <list>
FILES CHANGED: <list>
COMMANDS RUN: <list>
TEST RESULTS: <pass/fail/skip summary>
MOCK VS REAL STATUS: <explicit status>
OWNERSHIP OR ENVIRONMENT BLOCKERS: <none or precise details>
NEXT PROPOSED STEP: <one smallest actionable step>
```

Do not say “complete” when an acceptance gate is unverified. Begin now with **Phase 0 only**. Inspect the repository, create or update the progress file, establish the baseline, and stop for human review before proceeding.
