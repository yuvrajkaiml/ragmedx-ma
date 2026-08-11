RAGMedX-MA

*A Multi-Agent Retrieval-Augmented Framework with Evidence Verification  
and Confidence-Based Abstention for Reliable Medical Question Answering*

**Product Requirements Document (PRD)  
**Version 2.0 (Research Edition) • July 2026  
Department of Artificial Intelligence and Machine Learning  
Academic Year 2026–27

# 1\. Executive Summary

RAGMedX-MA is an intelligent medical question-answering system that addresses the critical failure modes of standard Retrieval-Augmented Generation (RAG) in high-stakes healthcare applications. Unlike conventional medical chatbots that rely on a single retrieval pass and direct generation, RAGMedX-MA implements a multi-agent collaborative architecture featuring iterative evidence retrieval, peer-reviewed answer generation, cross-encoder evidence re-ranking, and a knowledge-driven abstention mechanism calibrated to clinical safety protocols.

The system is designed not merely as an engineering project but as a research contribution targeting top-tier venues (ACL, IEEE JBHI, MICCAI). It directly responds to recent large-scale evaluations—specifically the MIRAGE benchmark—which have demonstrated that vanilla RAG can degrade performance on complex medical queries due to the "lost-in-the-middle" effect and insufficient evidence verification.

# 2\. Problem Statement & Research Gap

## 2.1 The Problem

Medical professionals and patients increasingly rely on AI systems for clinical information. However, current Large Language Models (LLMs) suffer from:

-   Hallucination: Generating plausible-sounding but factually incorrect medical information.
-   Outdated Knowledge: Training corpora lack the latest clinical guidelines and research.
-   Lack of Transparency: No supporting evidence or confidence scores are provided.
-   Overconfidence: Models guess rather than abstain when evidence is insufficient.

## 2.2 The Research Gap

Recent frontier research (2025–2026) has moved far beyond simple semantic similarity search:

-   Iterative RAG (e.g., i-MedRAG, MedTrust-RAG): Systems that allow the LLM to ask follow-up queries iteratively.
-   Confidence & Abstention Mechanisms (e.g., I-CALM, MedConf): Teaching LLMs to measure their own uncertainty.
-   Multi-Agent Peer Review: Using multiple LLM agents collaboratively for reasoning and verification.

The Gap: Most existing medical RAG systems use a static, single-pass retrieval pipeline. They lack dynamic evidence refinement, multi-agent verification, and calibrated abstention—creating safety risks in clinical deployment.

# 3\. Objectives & Research Contributions

## 3.1 Primary Objectives

1\. Develop a multi-agent iterative retrieval framework that dynamically refines medical evidence.

2\. Implement a Critic Agent for peer-reviewed generation that verifies factual consistency.

3\. Design a Confidence Calibration Agent that computes a composite reliability score.

4\. Build a Knowledge-Driven Abstention Mechanism returning "Insufficient Evidence" when confidence is low.

5\. Validate the system against the MIRAGE benchmark suite and established medical QA datasets.

## 3.2 Novel Research Contributions

Contribution 1: First integration of iterative retrieval-verification with multi-agent peer review in an open-source medical RAG system.

Contribution 2: A composite confidence calibration method combining retrieval precision, evidence re-ranking scores, and cross-source agreement.

Contribution 3: Empirical demonstration that the proposed architecture outperforms vanilla RAG and Chain-of-Thought baselines on MIRAGE benchmark tasks.

# 4\. System Architecture

## 4.1 High-Level Architecture

User Query  
│  
▼  
\[Phase 1: Query Understanding Agent\] → Intent, entities, specialty, abbreviations  
│  
▼  
\[Phase 2: Iterative Retrieval Agent\] → Hybrid retrieval, MGA, max 3 iterations  
│  
▼  
\[Phase 3: Evidence Re-Ranking Agent\] → Cross-encoder, dedup, contradiction, GRADE  
│  
▼  
\[Phase 4: Generator Agent (LLM)\] → Citation-aware reasoning, draft answer  
│  
▼  
\[Phase 5: Critic Agent (Peer Review)\] → Factual consistency, hallucination detection  
│  
▼  
\[Phase 6: Confidence Calibration\] → 4-component composite score  
│  
▼  
\[Phase 7: Abstention Decision\] → Threshold τ=0.65 → Final Answer or Abstain  
│  
▼  
\[Phase 8: Response Delivery\] → Answer + citations + confidence + evidence

## 4.2 Multi-Agent Workflow Detail

**Agent 1: Query Understanding Agent (QUA)**

Pre-process and optimize the user query. Uses medical synonym expansion, UMLS concept mapping, and abbreviation resolution.

**Agent 2: Iterative Retrieval Agent (IRA)**

Dynamically retrieve evidence using hybrid BM25 + Dense (BGE-M3/PubMedBERT) with RRF. Generates Medical Gap Analysis (MGA) and refines queries up to 3 iterations.

**Agent 3: Evidence Re-Ranking Agent (ERA)**

Cross-encoder re-ranking (MedCPT), contradiction detection via NLI, and GRADE hierarchy weighting (Systematic reviews > RCTs > Cohort > Case reports).

**Agent 4: Generator Agent (GA)**

Produces citation-aware draft answers strictly grounded in verified evidence. Format: statement \[Doc X\] statement \[Doc Y\].

**Agent 5: Critic Agent (CA)**

Independent peer review checking factual consistency, medical correctness, evidence alignment, and hallucination. Max 2 review rounds.

**Agent 6: Confidence Calibration Agent (CCA)**

Computes composite confidence: w₁·Retrieval + w₂·Evidence\_Quality + w₃·Answer\_Consistency + w₄·Inter\_Source\_Agreement (initial weights: 0.25, 0.30, 0.25, 0.20).

**Agent 7: Abstention Decision Module (ADM)**

Clinical safety gate. If Confidence ≥ τ (0.65) → Final Answer. Otherwise → structured abstention message recommending professional consultation.

# 5\. Technology Stack

Component

Technology

Justification

Frontend

React.js

Interactive UI with real-time confidence visualization

Backend

FastAPI

High-performance async API for multi-agent orchestration

Language

Python 3.12

Ecosystem for ML/LLM frameworks

Multi-Agent Framework

LangGraph (LangChain)

Cyclic workflows, state management, conditional edges

Embedding Models

BGE-M3, PubMedBERT

General + domain-specific biomedical encoding

Vector DB

FAISS

Efficient approximate nearest neighbor search

Re-ranker

MedCPT / BGE-Reranker-v2-m3

Precise query-document relevance scoring

Local LLM

Qwen 3 (8B/14B) or Llama 3.1 (8B/70B)

Open-source, privacy-preserving local deployment

Retrieval Strategy

Hybrid (BM25 + Dense) + RRF

Proven optimal for medical RAG

Evaluation

MIRAGE + MedRGB

Standardized medical RAG evaluation

# 6\. Functional Requirements

## 6.1 User Module

ID

Requirement

Priority

FR-U1

User registration/login with secure authentication (JWT)

High

FR-U2

Medical question submission with specialty auto-detection

High

FR-U3

Real-time chat interface with typing indicators

Medium

FR-U4

Display of final answer with inline citations and confidence score

High

FR-U5

Visualization of evidence sources (clickable references)

High

FR-U6

Chat history with search, export (PDF), and delete functionality

Medium

FR-U7

Bookmark/save important responses with annotations

Medium

FR-U8

Profile management and settings

Low

## 6.2 AI Module (Multi-Agent System)

ID

Requirement

Priority

FR-A1

Query understanding with medical entity extraction

High

FR-A2

Iterative retrieval with max 3 rounds and MGA

High

FR-A3

Hybrid retrieval (BM25 + Dense) with RRF merging

High

FR-A4

Cross-encoder re-ranking (top-32 → top-5)

High

FR-A5

Evidence quality scoring (GRADE + temporal)

High

FR-A6

Citation-aware answer generation

High

FR-A7

Critic Agent peer review with hallucination detection

High

FR-A8

Confidence calibration (4-component score)

High

FR-A9

Abstention decision (τ=0.65)

High

FR-A10

Negative Knowledge Assertion for insufficient evidence

High

FR-A11

Contradiction detection among sources (NLI)

Medium

FR-A12

Support for uploaded PDF/document indexing

Medium

## 6.3 Admin Module

ID

Requirement

Priority

FR-AD1

Upload and index medical literature

High

FR-AD2

Document chunking, embedding, FAISS storage

High

FR-AD3

Knowledge base monitoring and metrics

Medium

FR-AD4

User management and permission control

Medium

FR-AD5

System performance dashboard

Medium

# 7\. Non-Functional Requirements

## 7.1 Performance

-   Response Time: < 5s (single-pass); < 15s (iterative, max 3 rounds)
-   Throughput: 50 concurrent users
-   Retrieval Latency: < 500ms initial; < 2s for cross-encoder re-ranking

## 7.2 Reliability

-   Retrieval Precision@5 ≥ 0.75 on MIRAGE
-   Answer Accuracy ≥ 70% on MedQA-US
-   Abstention Calibration (ECE) < 0.15

## 7.3 Security

-   JWT-based secure authentication
-   Local LLM deployment (no external API calls for medical queries)
-   AES-256 at rest, TLS 1.3 in transit

# 8\. Data Requirements

## 8.1 Datasets

Dataset

Size

Purpose

PubMed Abstracts

~35M articles

Primary retrieval corpus

PMC Open Access

~5M full-text

Extended evidence

StatPearls

~10K articles

Point-of-care summaries

MedQA-US

1,273 questions

USMLE-style evaluation

MedMCQA

4,183 questions

Indian medical exams

MMLU-Med

1,089 questions

Multitask medical knowledge

PubMedQA\*

618 questions

Literature-based yes/no

BioASQ-Y/N

17 questions

Biomedical semantic QA

## 8.2 Data Processing Pipeline

1\. Ingestion: Download/process raw documents.

2\. Cleaning: Remove headers, footers, references; handle OCR errors.

3\. Chunking: 2,000 chars/chunk with 400-char overlap.

4\. Embedding: BGE-M3 (1024-dim) + PubMedBERT (768-dim).

5\. Indexing: FAISS with IVF index + metadata (PMID, title, year, study type).

6\. Metadata Extraction: Publication year, study design, journal impact factor.

# 9\. Validation Strategy & Benchmarking

## 9.2 Metrics

Metric

Description

Target

Accuracy

% correct answers on MCQA

≥ 70% on MIRAGE

Retrieval Precision@5

Top-5 chunks with ground-truth

≥ 0.75

Retrieval Recall@5

Ground-truth found in top-5

≥ 0.65

Abstention Rate

Queries triggering abstention

10–20%

Abstention Calibration (ECE)

Accuracy of abstention decisions

< 0.15

Factual Consistency

Claims supported by citations (NLI)

≥ 0.90

Hallucination Rate

Unsupported claims

< 5%

Latency

End-to-end response time

< 15s / < 5s

## 9.3 Baseline Comparisons

-   Vanilla RAG: Single-pass retrieval + direct generation
-   Chain-of-Thought (CoT): LLM reasoning without retrieval
-   MedRAG: State-of-the-art medical RAG toolkit
-   MedTrust-RAG: Iterative retrieval with verification (2025 SOTA)

## 9.4 Ablation Studies

-   w/o Iterative Retrieval
-   w/o Critic Agent
-   w/o Cross-Encoder Re-ranking
-   w/o Confidence Calibration
-   w/o Evidence Quality Scoring

# 10\. User Interface Requirements

## 10.1 Dashboard

Clean clinical-themed interface (blues/whites) with quick-access cards: "Ask a Question", "Upload Literature", "Chat History", "Saved References".

## 10.2 Chat Interface

-   Question input with medical specialty auto-suggestion
-   Real-time processing visualization showing active agent
-   Structured answer with expandable citation blocks
-   Confidence score as colored progress bar (green ≥80%, yellow 65–79%, red <65%)
-   Evidence cards showing source title, year, study type, and relevant passage
-   Clear abstention messages with professional consultation suggestion

# 11\. Error Handling & Edge Cases

Scenario

System Behavior

Invalid/empty query

"Please enter a valid medical question."

Retrieval failure

Abstention: "No relevant medical literature found."

Model failure

"Unable to generate a response. Please try again."

Low confidence (< τ)

Abstention with explanation

Contradictory evidence

UI flag: "Sources contain conflicting information."

Out-of-domain query

Abstention: "Outside the scope of medical literature."

# 12\. Implementation Roadmap

## Phase 1: Foundation (Weeks 1–4)

-   FastAPI + React scaffold
-   Basic RAG pipeline
-   FAISS + BGE-M3 integration
-   PubMed/PMC dataset loading

## Phase 2: Core Agents (Weeks 5–8)

-   Query Understanding Agent
-   Iterative Retrieval Agent with MGA
-   Evidence Re-Ranking Agent
-   Generator Agent with citations

## Phase 3: Safety & Verification (Weeks 9–12)

-   Critic Agent with hallucination detection
-   Confidence Calibration Agent
-   Abstention Decision Module
-   Contradiction detection (NLI)

## Phase 4: Evaluation (Weeks 13–16)

-   MIRAGE benchmark evaluations
-   Ablation studies
-   Threshold tuning
-   Latency optimization

## Phase 5: Documentation & Publication (Weeks 17–20)

-   Technical documentation
-   IEEE/ACL research paper
-   Project presentation
-   Final testing

# 13\. Risks & Mitigation

Risk

Impact

Mitigation

High compute cost for multi-agent loops

High

Use smaller models (7B) for verifier/critic; implement caching

Latency exceeding 15s

Medium

Parallelize agents; limit max iterations to 3

Poor abstention calibration

High

Iterative threshold tuning; temperature scaling

Limited ground-truth for retrieval

Medium

Focus on PubMedQA/BioASQ; manual annotation subset

Local LLM below API performance

Medium

Use Qwen3-14B / Llama-3.1-70B; 4-bit/8-bit quantization

# 14\. Future Scope

1\. EHR Integration for personalized clinical decision support

2\. Multilingual support (Hindi, Marathi, regional languages)

3\. Medical image understanding (radiology/pathology)

4\. Voice-enabled consultation (speech-to-text)

5\. Knowledge graph enhancement (UMLS, SNOMED CT)

6\. Personalized medicine based on demographics and history

# 15\. Expected Deliverables

1\. Functional RAGMedX-MA system with all 7 agent phases

2\. Production-ready React frontend

3\. Modular FastAPI backend with documentation

4\. FAISS-based biomedical retrieval engine

5\. MIRAGE benchmark evaluation report with ablation studies

6\. Technical documentation (architecture, API, deployment)

7\. Major project academic report

8\. IEEE research paper (JBHI or ACL submission)

9\. Final defense presentation

# 16\. References

\[1\] Xiong, G., et al. (2024). "Benchmarking Retrieval-Augmented Generation for Medicine." arXiv:2402.13178.

\[2\] Jiang, F., et al. (2025). "Beyond Retrieval: Ensembling Cross-Encoders and GPT Rerankers." arXiv:2507.05577.

\[3\] MedTrust-RAG Team (2025). "MedTrust-RAG: Evidence Verification and Trust Alignment." arXiv:2510.14400.

\[4\] I-CALM Team (2026). "I-CALM: Incentivizing Confidence-Aware Abstention." arXiv:2604.03904.

\[5\] ET-RAG Team (2026). "A Multi-Agent RAG Framework with Evidence-Temporal Scoring." bioRxiv.

\[6\] MedRGB Team (2024). "Comprehensive Evaluation of RAG for Medical QA." arXiv:2411.09213.

\[7\] Li, W., et al. (2025). "Refine Medical Diagnosis Using Generation-Augmented Retrieval." IEEE JBHI, 29(4).

\[8\] Zhao, H., et al. (2025). "Nursing Retrieval-Augmented Generation." J. Biomedical Informatics, 168.