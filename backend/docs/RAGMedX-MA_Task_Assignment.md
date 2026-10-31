# RAGMedX-MA — Task Assignment Matrix

**Project:** RAGMedX-MA — A Multi-Agent Iterative Retrieval Framework with Peer-Reviewed Generation for High-Confidence Medical Question Answering  
**Department:** Artificial Intelligence and Machine Learning  
**Academic Year:** 2026–27  
**Team:** Group 09  
**Date:** July 2026

---

## Team Members

| # | Name | Primary Role | Secondary Role |
|---|------|-------------|----------------|
| 1 | **Shravani Waman** | Frontend & User Experience | UI/UX Testing |
| 2 | **Pranav Sukale** | Backend API, Auth & Orchestration | DevOps Lead |
| 3 | **Sana Thorat** | Retrieval Engine & Knowledge Base | Data Pipeline |
| 4 | **Yuvraj K** | Multi-Agent AI Pipeline & LLM | Evaluation Lead |

---

## System Architecture Overview

```
+-------------------------------------------------------------+
|                    RAGMedX-MA SYSTEM                        |
+-------------------------------------------------------------+
|                                                             |
|   +--------------+    +--------------+    +--------------+ |
|   |   Member 1   |    |   Member 2   |    |   Member 1   | |
|   |  SHRAVANI    |<---|   PRANAV     |--->|  SHRAVANI    | |
|   |  (Frontend)  |    |   (Backend)  |    |  (Frontend)  | |
|   +--------------+    +------+-------+    +--------------+ |
|                              |                              |
|                     +--------+--------+                     |
|                     |                 |                     |
|              +------v------+   +------v------+             |
|              |  Member 3   |   |  Member 4  |              |
|              |    SANA     |   |   YUVRAJ   |              |
|              | (Retrieval) |   |   (Agents) |              |
|              +-------------+   +------------+              |
|                                                             |
|   DATA FLOW:  Sana --> Yuvraj --> Pranav --> Shravani      |
|                                                             |
+-------------------------------------------------------------+
```

---

## Member 1: Shravani Waman — Frontend & User Experience

> **Mission:** Build everything the user sees and interacts with. Make the complex multi-agent backend feel simple, trustworthy, and transparent.

### Module Ownership

| # | Module | Description | Priority |
|---|--------|-------------|----------|
| F1 | React Dashboard SPA | Single-page app with React Router, global state (Zustand/Context) | High |
| F2 | Chat Interface | Query input with auto-resize, typing indicators, message bubbles | High |
| F3 | Answer Cards (3 States) | Verified (teal), Revised (amber), Abstention (red) cards | High |
| F4 | Confidence Badge Component | Reusable pill: 'High 87%', 'Moderate 72%', 'Low 58%' | High |
| F5 | Citation Chips | Inline `[1]`, `[2]` markers; click opens Source Card drawer | High |
| F6 | Source Cards | Title, journal, year, DOI, relevance note, external link | High |
| F7 | Agent Trace Timeline | Collapsible vertical stepper: 7 agent phases with status | Medium |
| F8 | Evidence Panel | Desktop: right dock (400px). Tablet: slide-over. Mobile: bottom accordion | High |
| F9 | Session History | Searchable list, topic tags, re-open, export PDF/JSON/Markdown | Medium |
| F10 | Responsive Layout | Desktop 2-col, Tablet 1-col + drawer, Mobile 1-col | High |
| F11 | Accessibility | WCAG AA contrast, keyboard nav, ARIA labels, screen reader support | High |
| F12 | Real-time WebSocket UI | Live status updates: 'Searching PubMed...', 'Reviewing evidence...' | Medium |
| F13 | Empty State / Onboarding | Example queries, dataset badges, one-line disclaimer | Low |
| F14 | Settings Panel | Confidence threshold display toggle, agent verbosity, theme | Low |

### Key Files (You Own)

```
frontend/
├── src/
│   ├── components/
│   │   ├── ConfidenceBadge.tsx
│   │   ├── CitationChip.tsx
│   │   ├── SourceCard.tsx
│   │   ├── AnswerCard.tsx
│   │   ├── AgentTrace.tsx
│   │   ├── EvidencePanel.tsx
│   │   ├── ChatMessage.tsx
│   │   ├── ChatInput.tsx
│   │   └── ...
│   ├── pages/
│   │   ├── AskPage.tsx
│   │   ├── HistoryPage.tsx
│   │   ├── EvidenceLibraryPage.tsx
│   │   └── SettingsPage.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useQuery.ts
│   │   └── useAuth.ts
│   ├── services/
│   │   └── api.ts          # Axios/fetch wrapper for all backend calls
│   ├── types/
│   │   └── index.ts        # TypeScript interfaces matching Pydantic schemas
│   └── App.tsx
```

### API Contracts You Consume (From Pranav)

| Endpoint | Method | What You Send | What You Get |
|----------|--------|---------------|--------------|
| `/api/v1/auth/register` | POST | email, username, password | Token + User |
| `/api/v1/auth/login` | POST | email, password | Token + User |
| `/api/v1/query/ask` | POST | question, conversation_id | MedicalResponse |
| `/api/v1/ws/query-stream` | WS | query_id | Live agent status updates |
| `/api/v1/chat/conversations` | GET | - | ConversationList |
| `/api/v1/chat/conversations/{id}` | GET | - | ConversationDetail |

### Color Tokens (From UI/UX Brief)

| Token | Hex | Usage |
|-------|-----|-------|
| Clinical Navy | `#1e3a5f` | Headers, nav, AI messages |
| Evidence Teal | `#0d9488` | High confidence, verified badges |
| Caution Amber | `#d97706` | Moderate confidence, revised tags |
| Alert Red | `#dc2626` | Low confidence, abstention banners |
| Neutral Ink | `#1f2937` | Body text |
| Neutral Subtle | `#6b7280` | Metadata, timestamps |

### Integration Checkpoints

| Week | Milestone | Action |
|------|-----------|--------|
| Week 2 | Contract Review | Review Pydantic schemas from Pranav; flag any UI concerns |
| Week 4 | Mock Integration #1 | Build against Pranav's mock API server |
| Week 6 | Component Library | ConfidenceBadge, CitationChip, SourceCard ready |
| Week 10 | Full UI Complete | All screens built with mock data |
| Week 12 | Live Integration | Switch from mock to real backend |

### Deliverables

- [ ] Complete React frontend (all screens, all states)
- [ ] Reusable component library (Badge, Chip, Card, Trace)
- [ ] Responsive layouts (Desktop, Tablet, Mobile)
- [ ] Accessibility audit (WCAG AA)
- [ ] Frontend unit tests (Jest + React Testing Library)

---

## Member 2: Pranav Sukale — Backend API, Auth & Orchestration

> **Mission:** Build the HTTP layer that connects everything. You are the conductor — frontend talks to you, you talk to retrieval and agents. Define the contracts everyone follows.

### Module Ownership

| # | Module | Description | Priority |
|---|--------|-------------|----------|
| B1 | FastAPI App Shell | App factory, CORS, middleware, lifespan events, exception handlers | High |
| B2 | Authentication | JWT access/refresh tokens, bcrypt hashing, email verification | High |
| B3 | User Management | Registration, login, logout, profile CRUD, password reset | High |
| B4 | Conversation API | Create, list, get, update, delete, export (PDF/JSON/Markdown) | High |
| B5 | Query Orchestrator | `/query/ask` — the main endpoint that wires all 7 agents together | High |
| B6 | WebSocket Handler | Real-time streaming: agent_status, retrieval_update, final_response | Medium |
| B7 | Admin Endpoints | Metrics, KB stats, user management, feedback, latency breakdown | Medium |
| B8 | Document Upload API | PDF upload (max 50MB), status tracking, deletion | Medium |
| B9 | Pydantic Schemas | ALL request/response models — the single source of truth for all members | High |
| B10 | API Documentation | Auto-generated OpenAPI/Swagger docs | Medium |
| B11 | Docker & DevOps | `docker-compose.yml`, Dockerfile, `.env.example`, service orchestration | Medium |
| B12 | Redis Caching | Response cache, session store | Low |

### Key Files (You Own)

```
backend/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # AppConfig (Pydantic Settings)
│   ├── api/
│   │   ├── deps.py                # Auth dependencies, DB session
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── auth.py        # B2, B3
│   │           ├── users.py       # B3
│   │           ├── chat.py        # B4
│   │           ├── query.py       # B5 (THE ORCHESTRATOR)
│   │           ├── documents.py   # B8
│   │           └── admin.py       # B7
│   ├── schemas/
│   │   ├── auth.py
│   │   ├── user.py
│   │   ├── query.py
│   │   ├── response.py
│   │   ├── conversation.py
│   │   ├── document.py
│   │   ├── admin.py
│   │   └── websocket.py           # B9 — EVERYONE USES THESE
│   ├── core/
│   │   ├── security.py            # JWT, password hashing
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── chat_service.py
│   │   └── query_service.py       # B5 — orchestration logic
│   └── api/v1/websockets.py       # B6
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

### Integration Contracts You Define

**Schema Freeze Deadline: Week 2** — All members must approve these.

```python
# schemas/response.py — THE MASTER CONTRACT
class VerifiedAnswerResponse(BaseModel):
    response_type: Literal['verified_answer'] = 'verified_answer'
    answer: str
    confidence: ConfidenceScoreOutput
    citations: List[CitationOutput]
    agent_trace: Optional[AgentTraceOutput]
    processing_time_ms: int
    total_tokens_used: int

class AbstentionResponse(BaseModel):
    response_type: Literal['abstention'] = 'abstention'
    message: str
    abstention_reason: str
    partial_findings: Optional[List[EvidenceSnippet]]
    suggested_followup: Optional[str]
    confidence: Optional[ConfidenceScoreOutput]
    agent_trace: Optional[AgentTraceOutput]
    processing_time_ms: int

MedicalResponse = Union[VerifiedAnswerResponse, AbstentionResponse, ErrorResponse]
```

### Functions You Call (From Sana & Yuvraj)

| Function | From | Input | Output |
|----------|------|-------|--------|
| `retrieve_evidence(query, max_iter=3)` | Sana | Raw query string | `List[EvidenceSnippet]`, iteration log |
| `run_agent_pipeline(query, evidence)` | Yuvraj | Question + evidence list | `MedicalResponse`, `AgentTraceOutput` |

### Integration Checkpoints

| Week | Milestone | Action |
|------|-----------|--------|
| Week 2 | **SCHEMA FREEZE** | Finalize all Pydantic models; share with team |
| Week 3 | Mock Server | Deploy mock API with sample responses for Shravani |
| Week 4 | Auth Complete | JWT flow working; Shravani can login/register |
| Week 6 | Mock Retrieval | Wire mock `retrieve_evidence()` call |
| Week 8 | Mock Agents | Wire mock `run_agent_pipeline()` call |
| Week 10 | Full Backend | Connect real Sana + Yuvraj modules |
| Week 12 | Live System | End-to-end working with frontend |

### Deliverables

- [ ] Complete FastAPI backend with all endpoints
- [ ] JWT authentication system (register, login, refresh, logout)
- [ ] Query orchestrator that calls retrieval + agents
- [ ] WebSocket streaming endpoint
- [ ] OpenAPI/Swagger documentation
- [ ] Docker + docker-compose setup
- [ ] Backend unit tests (pytest + pytest-asyncio)

---

## Member 3: Sana Thorat — Retrieval Engine & Knowledge Base

> **Mission:** Build the evidence-finding engine. Your job is to take a medical question and return the most relevant, high-quality, non-contradictory evidence from PubMed, PMC, MedQA, and BioASQ.

### Module Ownership

| # | Module | Description | Priority |
|---|--------|-------------|----------|
| R1 | Document Ingestion | PubMed/PMC/MedQA/BioASQ loaders, PDF parsing, metadata extraction | High |
| R2 | Text Processing | Chunking (2000 chars, 400 overlap), cleaning, OCR error handling | High |
| R3 | Embedding Service | BGE-M3 (1024-dim) + PubMedBERT (768-dim), batch inference | High |
| R4 | FAISS Index Manager | IVF index creation, incremental updates, persistence, ID mapping | High |
| R5 | Hybrid Search | BM25 (sparse) + Dense (FAISS) + RRF fusion scoring | High |
| R6 | Cross-Encoder Re-ranker | MedCPT / BGE-Reranker-v2-m3: top-32 to top-5 | High |
| R7 | Contradiction Detection | NLI-based contradiction flagging among retrieved chunks | Medium |
| R8 | Evidence Quality Scorer | GRADE hierarchy (Systematic Review > RCT > Cohort > Case Report) + temporal boost | Medium |
| R9 | Iterative Retrieval Loop | Medical Gap Analysis (MGA), sub-query generation, max 3 iterations | High |
| R10 | Database Models | SQLAlchemy ORM models for documents, chunks, retrieved_evidence | High |
| R11 | Data Pipeline Scripts | Ingestion -> Clean -> Chunk -> Embed -> Index automation | Medium |

### Key Files (You Own)

```
backend/
├── app/
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # R3 — BGE-M3 + PubMedBERT
│   │   ├── vector_store.py      # R4 — FAISS index management
│   │   ├── hybrid_search.py     # R5 — BM25 + Dense + RRF
│   │   ├── reranker.py          # R6 — Cross-encoder scoring
│   │   ├── document_loader.py   # R1 — PubMed/PMC ingestion
│   │   └── interface.py         # PUBLIC API for Pranav
│   ├── db/
│   │   └── models.py            # R10 — ORM models (shared)
│   └── utils/
│       ├── text_processing.py   # R2 — Chunking, cleaning
│       └── citation_parser.py   # Metadata extraction
├── scripts/
│   ├── ingest_pubmed.py
│   ├── ingest_pmc.py
│   ├── build_index.py
│   └── update_index.py
└── data/
    ├── faiss_index/             # FAISS index files
    ├── documents/               # Raw documents
    └── cache/                   # Embedding cache
```

### Public Interface (For Pranav)

```python
# app/retrieval/interface.py — YOUR CONTRACT WITH THE ORCHESTRATOR

from typing import List, Tuple, Optional
from uuid import UUID

async def retrieve_evidence(
    query: str,
    optimized_query: Optional[str] = None,
    query_embedding: Optional[list] = None,
    max_iterations: int = 3,
    top_k_initial: int = 50,
    top_k_final: int = 5,
    enable_contradiction_check: bool = True
) -> Tuple[List[EvidenceSnippet], List[RetrievalIterationOutput]]:
    # THE ONLY FUNCTION PRANAV CALLS
    pass

class EvidenceSnippet(BaseModel):
    chunk_id: UUID
    document_id: UUID
    source_type: SourceType        # pubmed, pmc, medqa, bioasq
    title: str
    journal: Optional[str]
    publication_year: Optional[int]
    cited_text: str                # The actual passage
    relevance_score: float         # Final composite score
    cross_encoder_score: float     # Re-ranker score
    evidence_quality_score: float  # GRADE + temporal
    citation_marker: str           # e.g., '[1]', '[2]'

class RetrievalIterationOutput(BaseModel):
    iteration_number: int
    sub_query: str
    medical_gap_analysis: Optional[str]
    documents_retrieved: int
    documents_filtered: int
    top_evidence: List[EvidenceSnippet]
    duration_ms: int
```

### Internal Data Flow

```
User Query
    |
    v
+------------------+
|  Hybrid Search   |  <- BM25 + FAISS + RRF
|   (top-50)       |
+--------+---------+
         |
         v
+------------------+
| Cross-Encoder    |  <- MedCPT / BGE-Reranker
| Re-ranking       |
|  (top-50->top-5) |
+--------+---------+
         |
         v
+------------------+
| Contradiction    |  <- NLI detection
|    Check         |
+--------+---------+
         |
         v
+------------------+
| Quality Scoring  |  <- GRADE + temporal
|   + Filtering    |
+--------+---------+
         |
         v
+------------------+
|  Gap Analysis    |  <- Need more evidence?
|   (MGA)          |    Yes -> generate sub-query -> loop (max 3)
+------------------+    No  -> return final evidence
```

### Integration Checkpoints

| Week | Milestone | Action |
|------|-----------|--------|
| Week 2 | Schema Review | Review and approve Pydantic schemas |
| Week 4 | Embedding Ready | BGE-M3 + PubMedBERT loading, test embeddings |
| Week 6 | FAISS Index | Index built with sample data (10K docs) |
| Week 7 | Hybrid Search | BM25 + Dense + RRF working end-to-end |
| Week 8 | Mock Interface | `retrieve_evidence()` returns mock data for Pranav |
| Week 9 | Re-ranker | Cross-encoder integrated |
| Week 10 | Full Pipeline | Iterative retrieval + contradiction + quality |
| Week 12 | Integration | Pranav calls your real function |

### Deliverables

- [ ] Document ingestion pipeline (PubMed, PMC, MedQA, BioASQ)
- [ ] Text chunking and cleaning module
- [ ] Dual embedding service (BGE-M3 + PubMedBERT)
- [ ] FAISS index manager (IVF, incremental updates)
- [ ] Hybrid search (BM25 + Dense + RRF)
- [ ] Cross-encoder re-ranker (top-32 -> top-5)
- [ ] Contradiction detection (NLI)
- [ ] Evidence quality scoring (GRADE + temporal)
- [ ] Iterative retrieval with MGA (max 3 iterations)
- [ ] Public `retrieve_evidence()` interface
- [ ] Retrieval unit tests

---

## Member 4: Yuvraj K — Multi-Agent AI Pipeline & LLM

> **Mission:** Build the brain of the system. You own all 7 agents, the LLM wrapper, prompt engineering, and the safety mechanisms that decide whether to answer or abstain.

### Module Ownership

| # | Module | Description | Priority |
|---|--------|-------------|----------|
| A1 | Local LLM Wrapper | Qwen3-14B / Llama3.1 loader, 4-bit quantization (bitsandbytes), token counting | High |
| A2 | Prompt Templates | All 7 agent prompts (Jinja2/f-string), versioned and testable | High |
| A3 | Agent 1: Query Understanding | Specialty detection, abbreviation resolution, UMLS expansion, intent classification | High |
| A4 | Agent 2: Iterative Retrieval | Medical Gap Analysis (MGA), sub-query generation for Sana's loop | High |
| A5 | Agent 4: Generator | Citation-aware answer drafting with `[Doc X]` inline citations | High |
| A6 | Agent 5: Critic | Peer review: factual consistency, hallucination detection, revision requests (max 2 rounds) | High |
| A7 | Agent 6: Confidence Calibration | 4-component composite score with calibrated threshold | High |
| A8 | Agent 7: Abstention Decision | Safety gate: tau=0.65, structured refusal with partial findings | High |
| A9 | Agent Orchestration | LangGraph state machine: conditional edges, retry logic, state management | High |
| A10 | Negative Knowledge Assertion | 'I don't know' instead of hallucination when evidence is insufficient | High |
| A11 | Evaluation Pipeline | MIRAGE benchmark runner, ablation studies, metrics collection | Medium |

### Key Files (You Own)

```
backend/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAgent class with common logic
│   │   ├── query_understanding.py   # A3 — Agent 1
│   │   ├── iterative_retrieval.py   # A4 — Agent 2 (MGA generator)
│   │   ├── evidence_reranking.py    # (Agent 3 — mostly Sana's, you coordinate)
│   │   ├── generator.py             # A5 — Agent 4
│   │   ├── critic.py                # A6 — Agent 5
│   │   ├── confidence_calibration.py # A7 — Agent 6
│   │   ├── abstention_decision.py   # A8 — Agent 7
│   │   └── orchestrator.py          # A9 — LangGraph wiring
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── local_llm.py             # A1 — Model loader, generate()
│   │   ├── prompt_templates.py      # A2 — All prompts
│   │   └── tokenizer.py             # Token counting
│   └── utils/
│       └── confidence_utils.py      # Score computation helpers
├── evaluation/
│   ├── mirage_runner.py
│   ├── ablation_studies.py
│   └── metrics.py
└── prompts/                         # Versioned prompt backups
    ├── v1/
    │   ├── query_understanding.txt
    │   ├── generator.txt
    │   ├── critic.txt
    │   ├── confidence.txt
    │   └── abstention.txt
```

### Public Interface (For Pranav)

```python
# app/agents/interface.py — YOUR CONTRACT WITH THE ORCHESTRATOR

from typing import Tuple, Optional
from uuid import UUID

async def run_agent_pipeline(
    query_id: UUID,
    original_question: str,
    processed_query: QueryUnderstandingOutput,
    evidence: List[EvidenceSnippet],
    include_trace: bool = False
) -> Tuple[MedicalResponse, Optional[AgentTraceOutput]]:
    # THE ONLY FUNCTION PRANAV CALLS
    pass

class AgentTraceStep(BaseModel):
    agent_type: AgentType
    status: Literal['running', 'completed', 'failed']
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    details: Optional[Dict[str, Any]]

class AgentTraceOutput(BaseModel):
    steps: List[AgentTraceStep]
    total_duration_ms: int
```

### Agent Workflow (LangGraph)

```
+---------------------------------------------------------------+
|                  LANGGRAPH STATE MACHINE                       |
+---------------------------------------------------------------+
|                                                                |
|   +-------------+                                             |
|   |   START     |                                             |
|   +------+------+                                             |
|          |                                                     |
|          v                                                     |
|   +-------------+     +-----------------+                     |
|   |  Agent 1    |---->| QueryUnderstandingOutput              |
|   |  (QUA)      |     +-----------------+                     |
|   +------+------+                                             |
|          |                                                     |
|          v                                                     |
|   +-------------+     +-----------------+                     |
|   |  Agent 2    |---->| Call Sana's     |                     |
|   |  (IRA)      |     | retrieve_evidence()                   |
|   +------+------+     +-----------------+                     |
|          |         ^                                           |
|          |         | (loop max 3)                               |
|          |    +----+----+                                      |
|          +--->|  MGA    |--Need more evidence?---+            |
|               | Check   |        Yes             |            |
|               +----+----+                        |            |
|                    | No                          |            |
|                    v                             |            |
|   +-------------+     +-----------------+        |            |
|   |  Agent 3    |---->| Evidence list   |        |            |
|   |  (ERA)      |     | (from Sana)     |        |            |
|   +------+------+     +-----------------+        |            |
|          |                                       |            |
|          v                                       |            |
|   +-------------+                                |            |
|   |  Agent 4    |                                |            |
|   | (Generator) |--> Draft answer with [Doc X]   |            |
|   +------+------+      citations                 |            |
|          |                                       |            |
|          v                                       |            |
|   +-------------+     +-----------------+        |            |
|   |  Agent 5    |---->| CriticVerdict   |        |            |
|   |  (Critic)   |     | verified /      |        |            |
|   +------+------+     | needs_revision  |        |            |
|          |            +-----------------+        |            |
|          |         ^                              |            |
|          |         | (loop max 2)                  |            |
|          |    +----+----+                         |            |
|          +--->| Needs   |--Yes--> Generator       |            |
|               |Revision?|         revises         |            |
|               +----+----+                         |            |
|                    | No                           |            |
|                    v                              |            |
|   +-------------+                                |            |
|   |  Agent 6    |                                |            |
|   | (Confidence)|--> Composite score (0.0-1.0)   |            |
|   +------+------+                                |            |
|          |                                       |            |
|          v                                       |            |
|   +-------------+     +-----------------+        |            |
|   |  Agent 7    |---->|  tau = 0.65     |        |            |
|   | (Abstention)|     |  Pass -> Answer |        |            |
|   +------+------+     |  Fail -> Abstain|        |            |
|          |            +-----------------+        |            |
|          v                                       |            |
|   +-------------+                                |            |
|   |    END      |                                |            |
|   +-------------+                                |            |
|                                                                |
+---------------------------------------------------------------+
```

### Confidence Score Formula (You Implement)

```python
composite_confidence = (
    0.25 * retrieval_confidence +       # From Sana: precision@5, similarity
    0.30 * evidence_quality_score +      # GRADE levels, temporal, impact factor
    0.25 * answer_consistency_score +    # Claims supported by citations (NLI)
    0.20 * inter_source_agreement        # Sources agreeing / total sources
)

calibrated_confidence = temperature_scale(composite_confidence)
# Target ECE < 0.15
```

### Integration Checkpoints

| Week | Milestone | Action |
|------|-----------|--------|
| Week 2 | Schema Review | Review and approve Pydantic schemas |
| Week 4 | LLM Setup | Qwen3/Llama3.1 loading locally, test generation |
| Week 5 | Prompt V1 | All 7 prompt templates drafted |
| Week 6 | Agent 1+4 | Query Understanding + Generator working |
| Week 7 | Agent 5 | Critic agent with JSON output |
| Week 8 | Mock Interface | `run_agent_pipeline()` returns mock data for Pranav |
| Week 9 | Agent 6+7 | Confidence + Abstention working |
| Week 10 | LangGraph | Full state machine wired |
| Week 12 | Integration | Pranav calls your real function |
| Week 14 | Evaluation | MIRAGE benchmark + ablation studies |

### Deliverables

- [ ] Local LLM wrapper with 4-bit quantization
- [ ] All 7 agent implementations
- [ ] LangGraph orchestration with state management
- [ ] Prompt template library (versioned)
- [ ] Confidence calibration with 4-component scoring
- [ ] Abstention decision module (tau=0.65)
- [ ] Critic agent with hallucination detection
- [ ] Agent trace generation
- [ ] MIRAGE benchmark evaluation script
- [ ] Ablation study framework
- [ ] Public `run_agent_pipeline()` interface
- [ ] Agent unit tests

---

## Shared Responsibilities (All Members)

### Week-by-Week Collaboration

| Week | Activity | Who Leads | Who Participates |
|------|----------|-----------|------------------|
| 1 | Project kickoff, repo setup | Pranav | All |
| 2 | **Schema Freeze Meeting** | Pranav | All |
| 3 | Docker environment | Pranav | All |
| 4 | Mock API server | Pranav | Shravani |
| 5 | Prompt review session | Yuvraj | All |
| 6 | Retrieval demo | Sana | Yuvraj, Pranav |
| 7 | Agent pipeline demo | Yuvraj | Pranav |
| 8 | Frontend component review | Shravani | All |
| 10 | **Full Backend Integration** | Pranav | Sana, Yuvraj |
| 12 | **End-to-End Integration** | Pranav | All |
| 13 | Bug bash & fix | All | All |
| 14 | MIRAGE evaluation | Yuvraj | All |
| 15 | Performance optimization | All | All |
| 16 | Documentation & paper draft | All | All |
| 17-20 | Paper writing, presentation | All | All |

### Joint Deliverables

| # | Deliverable | Owner | Contributors |
|---|-------------|-------|-------------|
| J1 | `docker-compose.yml` (full stack) | Pranav | All review |
| J2 | `README.md` | Pranav | All review |
| J3 | API Documentation (Swagger) | Pranav | All review |
| J4 | Database Migrations (Alembic) | Sana | Pranav |
| J5 | `.env.example` | Pranav | All |
| J6 | Integration Tests | All | All |
| J7 | MIRAGE Benchmark Report | Yuvraj | All |
| J8 | IEEE Research Paper | All | Divide by section |
| J9 | Final Presentation | All | Divide by slide |
| J10 | Technical Documentation | All | Divide by module |

---

## Integration Contract Summary

### Data Flow

```
Shravani (Frontend) 
    | HTTP/WebSocket
    v
Pranav (Backend Orchestrator)
    | function call
    +------------------+
    v                  v
Sana (Retrieval)   Yuvraj (Agents)
    |                  |
    +-------><---------+
         (Agent 2 calls
          retrieve_evidence)
```

### Critical Interfaces

| Caller | Callee | Function | Data Passed | Data Returned |
|--------|--------|----------|-------------|---------------|
| Pranav | Sana | `retrieve_evidence()` | query string, max_iter=3 | `List[EvidenceSnippet]`, iteration log |
| Pranav | Yuvraj | `run_agent_pipeline()` | question, evidence, query_id | `MedicalResponse`, `AgentTraceOutput` |
| Shravani | Pranav | `POST /api/v1/query/ask` | question JSON | `MedicalResponse` JSON |
| Shravani | Pranav | `WS /api/v1/ws/query-stream` | query_id | `WSMessage` stream |

### Schema Ownership

| Schema File | Owner | Reviewers |
|-------------|-------|-----------|
| `schemas/auth.py` | Pranav | All |
| `schemas/user.py` | Pranav | All |
| `schemas/query.py` | Pranav | Sana, Yuvraj |
| `schemas/response.py` | Pranav | All |
| `schemas/conversation.py` | Pranav | Shravani |
| `schemas/websocket.py` | Pranav | Shravani |
| `schemas/document.py` | Pranav | Sana |
| `schemas/admin.py` | Pranav | All |

---

## Quick Reference: Who to Ask

| Question About... | Ask... |
|-------------------|--------|
"Why is this API returning 422?" | Pranav |
"Can we change the response shape?" | Pranav (lead schema owner) |
"The retrieval is returning wrong evidence" | Sana |
"How do I call the retrieval from my agent?" | Sana (then Pranav for wiring) |
"The answer has hallucinations" | Yuvraj |
"How does the confidence score work?" | Yuvraj |
"The confidence badge looks wrong" | Shravani |
"Can we add a new screen?" | Shravani |
"Docker build is failing" | Pranav |
"How do we evaluate on MIRAGE?" | Yuvraj |
"I need more training data" | Sana |
"The WebSocket is disconnecting" | Pranav |

---

*Document Version: 1.0*  
*Last Updated: July 2026*  
*Next Review: Week 2 (Schema Freeze)*