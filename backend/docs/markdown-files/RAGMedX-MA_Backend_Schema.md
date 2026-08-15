RAGMedX-MA BACKEND SCHEMA

*Multi-Agent Retrieval-Augmented Framework  
Version 2.0 | July 2026*

**Technical Specification Document  
**Database Schema, API Contracts, Prompts, and Architecture

# TABLE OF CONTENTS

1.  1\. Database Schema (PostgreSQL + pgvector)
2.  2\. Pydantic API Schemas (FastAPI)
3.  3\. Configuration & Prompt Templates
4.  4\. API Endpoint Specification
5.  5\. Project Directory Structure
6.  6\. Environment Variables Template
7.  7\. Requirements.txt

# 1\. Database Schema (PostgreSQL + pgvector)

\-- ============================================================  
\-- RAGMedX-MA Database Schema  
\-- PostgreSQL 15+ with pgvector extension  
\-- ============================================================  
  
\-- Enable pgvector extension  
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  
CREATE EXTENSION IF NOT EXISTS "vector";  
  
\-- ============================================================  
\-- 1. USER MANAGEMENT TABLES  
\-- ============================================================  
CREATE TABLE users (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
email VARCHAR(255) UNIQUE NOT NULL,  
username VARCHAR(100) UNIQUE NOT NULL,  
full\_name VARCHAR(255),  
password\_hash VARCHAR(255) NOT NULL,  
role VARCHAR(50) NOT NULL DEFAULT 'healthcare\_professional',  
specialty VARCHAR(100),  
institution VARCHAR(255),  
is\_active BOOLEAN DEFAULT TRUE,  
is\_verified BOOLEAN DEFAULT FALSE,  
email\_verified\_at TIMESTAMP,  
last\_login\_at TIMESTAMP,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
updated\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE user\_sessions (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
user\_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  
access\_token TEXT NOT NULL,  
refresh\_token TEXT NOT NULL,  
device\_info JSONB,  
expires\_at TIMESTAMP NOT NULL,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
revoked\_at TIMESTAMP  
);  
  
\-- ============================================================  
\-- 2. CHAT / CONVERSATION TABLES  
\-- ============================================================  
CREATE TABLE conversations (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
user\_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  
title VARCHAR(255),  
specialty\_tag VARCHAR(100),  
status VARCHAR(50) DEFAULT 'active',  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
updated\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE messages (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
conversation\_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,  
role VARCHAR(50) NOT NULL,  
content TEXT NOT NULL,  
message\_type VARCHAR(50) DEFAULT 'text',  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
\-- ============================================================  
\-- 3. QUERY & RETRIEVAL TABLES  
\-- ============================================================  
CREATE TABLE queries (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
conversation\_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,  
user\_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  
original\_query TEXT NOT NULL,  
processed\_query TEXT,  
detected\_intent VARCHAR(100),  
detected\_specialty VARCHAR(100),  
expanded\_terms JSONB,  
resolved\_abbreviations JSONB,  
query\_embedding VECTOR(1024),  
processing\_status VARCHAR(50) DEFAULT 'pending',  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
completed\_at TIMESTAMP  
);  
  
CREATE TABLE retrieval\_iterations (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
query\_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,  
iteration\_number INTEGER NOT NULL,  
sub\_query TEXT NOT NULL,  
medical\_gap\_analysis TEXT,  
retrieval\_strategy VARCHAR(50),  
top\_k\_initial INTEGER DEFAULT 50,  
documents\_retrieved INTEGER,  
documents\_filtered INTEGER,  
started\_at TIMESTAMP,  
completed\_at TIMESTAMP,  
UNIQUE(query\_id, iteration\_number)  
);  
  
\-- ============================================================  
\-- 4. MEDICAL EVIDENCE / DOCUMENT TABLES  
\-- ============================================================  
CREATE TABLE medical\_documents (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
source\_type VARCHAR(50) NOT NULL,  
external\_id VARCHAR(100),  
doi VARCHAR(255),  
title TEXT NOT NULL,  
abstract TEXT,  
full\_text TEXT,  
authors JSONB,  
journal VARCHAR(255),  
publication\_year INTEGER,  
publication\_date DATE,  
study\_type VARCHAR(100),  
grade\_level VARCHAR(50),  
impact\_factor DECIMAL(5,2),  
chunk\_count INTEGER DEFAULT 0,  
is\_indexed BOOLEAN DEFAULT FALSE,  
indexed\_at TIMESTAMP,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
updated\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE document\_chunks (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
document\_id UUID NOT NULL REFERENCES medical\_documents(id) ON DELETE CASCADE,  
chunk\_text TEXT NOT NULL,  
chunk\_index INTEGER NOT NULL,  
char\_start INTEGER,  
char\_end INTEGER,  
embedding\_bge\_m3 VECTOR(1024),  
embedding\_pubmedbert VECTOR(768),  
faiss\_index\_id BIGINT,  
section\_type VARCHAR(100),  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE retrieved\_evidence (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
query\_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,  
iteration\_id UUID REFERENCES retrieval\_iterations(id) ON DELETE CASCADE,  
chunk\_id UUID NOT NULL REFERENCES document\_chunks(id) ON DELETE CASCADE,  
initial\_similarity\_score DECIMAL(5,4),  
bm25\_score DECIMAL(10,6),  
rrf\_score DECIMAL(10,6),  
cross\_encoder\_score DECIMAL(5,4),  
final\_rank INTEGER,  
is\_selected BOOLEAN DEFAULT FALSE,  
evidence\_quality\_score DECIMAL(5,4),  
temporal\_score DECIMAL(5,4),  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
\-- ============================================================  
\-- 5. MULTI-AGENT WORKFLOW TABLES  
\-- ============================================================  
CREATE TABLE agent\_runs (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
query\_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,  
agent\_type VARCHAR(50) NOT NULL,  
input\_payload JSONB NOT NULL,  
output\_payload JSONB,  
status VARCHAR(50) DEFAULT 'running',  
started\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
completed\_at TIMESTAMP,  
duration\_ms INTEGER,  
error\_message TEXT,  
retry\_count INTEGER DEFAULT 0  
);  
  
CREATE TABLE generator\_outputs (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
query\_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,  
agent\_run\_id UUID NOT NULL REFERENCES agent\_runs(id) ON DELETE CASCADE,  
draft\_answer TEXT NOT NULL,  
reasoning\_chain TEXT,  
cited\_chunks JSONB,  
model\_used VARCHAR(100),  
temperature DECIMAL(3,2),  
tokens\_used INTEGER,  
version\_number INTEGER DEFAULT 1,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE critic\_reviews (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
query\_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,  
generator\_output\_id UUID NOT NULL REFERENCES generator\_outputs(id) ON DELETE CASCADE,  
agent\_run\_id UUID NOT NULL REFERENCES agent\_runs(id) ON DELETE CASCADE,  
verdict VARCHAR(50) NOT NULL,  
factual\_consistency\_score DECIMAL(5,4),  
medical\_correctness\_score DECIMAL(5,4),  
evidence\_alignment\_score DECIMAL(5,4),  
hallucination\_detected BOOLEAN DEFAULT FALSE,  
unsupported\_claims JSONB,  
critic\_feedback TEXT,  
suggested\_revisions JSONB,  
review\_round INTEGER DEFAULT 1,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE confidence\_scores (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
query\_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,  
retrieval\_confidence DECIMAL(5,4) NOT NULL,  
evidence\_quality\_score DECIMAL(5,4) NOT NULL,  
answer\_consistency\_score DECIMAL(5,4) NOT NULL,  
inter\_source\_agreement DECIMAL(5,4) NOT NULL,  
composite\_confidence DECIMAL(5,4) NOT NULL,  
calibrated\_confidence DECIMAL(5,4),  
confidence\_category VARCHAR(50),  
threshold\_used DECIMAL(3,2) DEFAULT 0.65,  
passes\_threshold BOOLEAN,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
\-- ============================================================  
\-- 6. FINAL RESPONSE & ANSWER TABLES  
\-- ============================================================  
CREATE TABLE final\_responses (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
query\_id UUID NOT NULL UNIQUE REFERENCES queries(id) ON DELETE CASCADE,  
conversation\_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,  
response\_type VARCHAR(50) NOT NULL,  
final\_answer TEXT,  
confidence\_score\_id UUID REFERENCES confidence\_scores(id),  
abstention\_reason TEXT,  
partial\_findings JSONB,  
suggested\_followup TEXT,  
error\_type VARCHAR(100),  
error\_details TEXT,  
total\_processing\_time\_ms INTEGER,  
total\_tokens\_used INTEGER,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE response\_citations (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
response\_id UUID NOT NULL REFERENCES final\_responses(id) ON DELETE CASCADE,  
chunk\_id UUID NOT NULL REFERENCES document\_chunks(id) ON DELETE CASCADE,  
citation\_marker VARCHAR(20) NOT NULL,  
cited\_text TEXT,  
source\_title TEXT,  
source\_journal VARCHAR(255),  
source\_year INTEGER,  
source\_doi VARCHAR(255),  
source\_url TEXT,  
relevance\_note TEXT,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
\-- ============================================================  
\-- 7. USER INTERACTION TABLES  
\-- ============================================================  
CREATE TABLE saved\_responses (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
user\_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  
response\_id UUID NOT NULL REFERENCES final\_responses(id) ON DELETE CASCADE,  
user\_note TEXT,  
tags JSONB,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
UNIQUE(user\_id, response\_id)  
);  
  
CREATE TABLE user\_feedback (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
user\_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  
response\_id UUID NOT NULL REFERENCES final\_responses(id) ON DELETE CASCADE,  
rating INTEGER CHECK (rating >= 1 AND rating <= 5),  
is\_helpful BOOLEAN,  
feedback\_text TEXT,  
issue\_type VARCHAR(100),  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
\-- ============================================================  
\-- 8. ADMIN / SYSTEM TABLES  
\-- ============================================================  
CREATE TABLE uploaded\_documents (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
uploaded\_by UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  
original\_filename VARCHAR(255) NOT NULL,  
file\_path TEXT NOT NULL,  
file\_size\_bytes BIGINT,  
file\_hash VARCHAR(64),  
mime\_type VARCHAR(100),  
processing\_status VARCHAR(50) DEFAULT 'pending',  
document\_id UUID REFERENCES medical\_documents(id),  
upload\_notes TEXT,  
created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,  
processed\_at TIMESTAMP  
);  
  
CREATE TABLE system\_metrics (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
metric\_type VARCHAR(100) NOT NULL,  
metric\_value DECIMAL(10,4),  
metric\_unit VARCHAR(50),  
query\_id UUID REFERENCES queries(id),  
agent\_type VARCHAR(50),  
recorded\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
CREATE TABLE knowledge\_base\_stats (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
source\_type VARCHAR(50) NOT NULL,  
document\_count BIGINT DEFAULT 0,  
chunk\_count BIGINT DEFAULT 0,  
index\_size\_mb DECIMAL(10,2),  
last\_updated TIMESTAMP,  
avg\_chunk\_quality DECIMAL(5,4),  
recorded\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP  
);  
  
\-- ============================================================  
\-- INDEXES  
\-- ============================================================  
CREATE INDEX idx\_users\_email ON users(email);  
CREATE INDEX idx\_users\_role ON users(role);  
CREATE INDEX idx\_conversations\_user ON conversations(user\_id);  
CREATE INDEX idx\_messages\_conversation ON messages(conversation\_id);  
CREATE INDEX idx\_queries\_user ON queries(user\_id);  
CREATE INDEX idx\_queries\_conversation ON queries(conversation\_id);  
CREATE INDEX idx\_queries\_status ON queries(processing\_status);  
CREATE INDEX idx\_retrieval\_iterations\_query ON retrieval\_iterations(query\_id);  
CREATE INDEX idx\_medical\_documents\_source ON medical\_documents(source\_type, external\_id);  
CREATE INDEX idx\_medical\_documents\_year ON medical\_documents(publication\_year);  
CREATE INDEX idx\_document\_chunks\_document ON document\_chunks(document\_id);  
CREATE INDEX idx\_retrieved\_evidence\_query ON retrieved\_evidence(query\_id);  
CREATE INDEX idx\_agent\_runs\_query ON agent\_runs(query\_id);  
CREATE INDEX idx\_agent\_runs\_type ON agent\_runs(agent\_type);  
CREATE INDEX idx\_final\_responses\_query ON final\_responses(query\_id);  
CREATE INDEX idx\_final\_responses\_type ON final\_responses(response\_type);  
CREATE INDEX idx\_system\_metrics\_type ON system\_metrics(metric\_type);  
CREATE INDEX idx\_system\_metrics\_time ON system\_metrics(recorded\_at);  
  
\-- Vector similarity indexes (using pgvector)  
CREATE INDEX idx\_chunks\_embedding\_bge ON document\_chunks USING ivfflat (embedding\_bge\_m3 vector\_cosine\_ops) WITH (lists = 100);  
CREATE INDEX idx\_chunks\_embedding\_pubmedbert ON document\_chunks USING ivfflat (embedding\_pubmedbert vector\_cosine\_ops) WITH (lists = 100);  
CREATE INDEX idx\_query\_embedding ON queries USING ivfflat (query\_embedding vector\_cosine\_ops) WITH (lists = 100);

# 2\. Pydantic API Schemas (FastAPI)

from datetime import datetime  
from typing import Optional, List, Dict, Any, Literal, Union  
from uuid import UUID  
from pydantic import BaseModel, Field, EmailStr, ConfigDict  
from enum import Enum  
  
\# ============================================================  
\# ENUMS  
\# ============================================================  
class UserRole(str, Enum):  
HEALTHCARE\_PROFESSIONAL = "healthcare\_professional"  
MEDICAL\_STUDENT = "medical\_student"  
RESEARCHER = "researcher"  
SYSTEM\_ADMIN = "system\_admin"  
  
class QueryStatus(str, Enum):  
PENDING = "pending"  
PROCESSING = "processing"  
COMPLETED = "completed"  
FAILED = "failed"  
ABSTAINED = "abstained"  
  
class AgentType(str, Enum):  
QUERY\_UNDERSTANDING = "query\_understanding"  
ITERATIVE\_RETRIEVAL = "iterative\_retrieval"  
EVIDENCE\_RERANKING = "evidence\_reranking"  
GENERATOR = "generator"  
CRITIC = "critic"  
CONFIDENCE\_CALIBRATION = "confidence\_calibration"  
ABSTENTION\_DECISION = "abstention\_decision"  
  
class ResponseType(str, Enum):  
VERIFIED\_ANSWER = "verified\_answer"  
ABSTENTION = "abstention"  
ERROR = "error"  
  
class ConfidenceCategory(str, Enum):  
HIGH = "high" # >= 0.80  
MODERATE = "moderate" # 0.65 - 0.79  
LOW = "low" # < 0.65  
  
class CriticVerdict(str, Enum):  
VERIFIED = "verified"  
REVISED = "revised"  
REJECTED = "rejected"  
NEEDS\_REVISION = "needs\_revision"  
  
class StudyType(str, Enum):  
SYSTEMATIC\_REVIEW = "systematic\_review"  
RCT = "rct"  
COHORT = "cohort"  
CASE\_CONTROL = "case\_control"  
CASE\_REPORT = "case\_report"  
GUIDELINE = "guideline"  
REVIEW = "review"  
META\_ANALYSIS = "meta\_analysis"  
  
class SourceType(str, Enum):  
PUBMED = "pubmed"  
PMC\_OPEN\_ACCESS = "pmc\_open\_access"  
MEDQA = "medqa"  
BIOASQ = "bioasq"  
UPLOADED\_PDF = "uploaded\_pdf"  
STATPEARLS = "statpearls"  
  
class WSMessageType(str, Enum):  
AGENT\_STATUS = "agent\_status"  
RETRIEVAL\_UPDATE = "retrieval\_update"  
GENERATION\_UPDATE = "generation\_update"  
CRITIC\_UPDATE = "critic\_update"  
CONFIDENCE\_UPDATE = "confidence\_update"  
FINAL\_RESPONSE = "final\_response"  
ERROR = "error"  
  
\# ============================================================  
\# AUTH SCHEMAS  
\# ============================================================  
class UserRegisterRequest(BaseModel):  
email: EmailStr  
username: str = Field(..., min\_length=3, max\_length=100)  
password: str = Field(..., min\_length=8)  
full\_name: Optional\[str\] = None  
role: UserRole = UserRole.HEALTHCARE\_PROFESSIONAL  
specialty: Optional\[str\] = None  
institution: Optional\[str\] = None  
  
class UserLoginRequest(BaseModel):  
email: EmailStr  
password: str  
  
class TokenResponse(BaseModel):  
access\_token: str  
refresh\_token: str  
token\_type: str = "bearer"  
expires\_in: int  
user: "UserResponse"  
  
class UserResponse(BaseModel):  
model\_config = ConfigDict(from\_attributes=True)  
id: UUID  
email: EmailStr  
username: str  
full\_name: Optional\[str\]  
role: UserRole  
specialty: Optional\[str\]  
institution: Optional\[str\]  
is\_active: bool  
is\_verified: bool  
created\_at: datetime  
  
class UserProfileUpdate(BaseModel):  
full\_name: Optional\[str\] = None  
specialty: Optional\[str\] = None  
institution: Optional\[str\] = None  
  
\# ============================================================  
\# QUERY & AGENT SCHEMAS  
\# ============================================================  
class MedicalQuestionRequest(BaseModel):  
question: str = Field(..., min\_length=10, max\_length=5000, description="The medical question to be answered")  
conversation\_id: Optional\[UUID\] = None  
specialty\_hint: Optional\[str\] = None  
include\_agent\_trace: bool = False  
  
class QueryUnderstandingOutput(BaseModel):  
original\_query: str  
processed\_query: str  
detected\_intent: str  
detected\_specialty: Optional\[str\]  
expanded\_terms: Dict\[str, List\[str\]\]  
resolved\_abbreviations: Dict\[str, str\]  
optimized\_search\_query: str  
  
class EvidenceSnippet(BaseModel):  
chunk\_id: UUID  
document\_id: UUID  
source\_type: SourceType  
title: str  
journal: Optional\[str\]  
publication\_year: Optional\[int\]  
cited\_text: str  
relevance\_score: float  
cross\_encoder\_score: float  
evidence\_quality\_score: float  
citation\_marker: str  
  
class RetrievalIterationOutput(BaseModel):  
iteration\_number: int  
sub\_query: str  
medical\_gap\_analysis: Optional\[str\]  
documents\_retrieved: int  
documents\_filtered: int  
top\_evidence: List\[EvidenceSnippet\]  
duration\_ms: int  
  
class CriticReviewOutput(BaseModel):  
verdict: CriticVerdict  
factual\_consistency\_score: float = Field(..., ge=0.0, le=1.0)  
medical\_correctness\_score: float = Field(..., ge=0.0, le=1.0)  
evidence\_alignment\_score: float = Field(..., ge=0.0, le=1.0)  
hallucination\_detected: bool  
unsupported\_claims: List\[Dict\[str, str\]\]  
critic\_feedback: str  
suggested\_revisions: List\[str\]  
review\_round: int  
  
class ConfidenceScoreOutput(BaseModel):  
retrieval\_confidence: float = Field(..., ge=0.0, le=1.0)  
evidence\_quality\_score: float = Field(..., ge=0.0, le=1.0)  
answer\_consistency\_score: float = Field(..., ge=0.0, le=1.0)  
inter\_source\_agreement: float = Field(..., ge=0.0, le=1.0)  
composite\_confidence: float = Field(..., ge=0.0, le=1.0)  
calibrated\_confidence: Optional\[float\]  
confidence\_category: ConfidenceCategory  
passes\_threshold: bool  
threshold\_used: float = 0.65  
  
class AgentTraceStep(BaseModel):  
agent\_type: AgentType  
status: Literal\["running", "completed", "failed"\]  
started\_at: datetime  
completed\_at: Optional\[datetime\]  
duration\_ms: Optional\[int\]  
details: Optional\[Dict\[str, Any\]\]  
  
class AgentTraceOutput(BaseModel):  
steps: List\[AgentTraceStep\]  
total\_duration\_ms: int  
  
\# ============================================================  
\# RESPONSE SCHEMAS  
\# ============================================================  
class CitationOutput(BaseModel):  
citation\_marker: str  
cited\_text: str  
source\_title: str  
source\_journal: Optional\[str\]  
source\_year: Optional\[int\]  
source\_doi: Optional\[str\]  
source\_url: Optional\[str\]  
relevance\_note: Optional\[str\]  
  
class VerifiedAnswerResponse(BaseModel):  
response\_type: Literal\["verified\_answer"\] = "verified\_answer"  
answer: str  
confidence: ConfidenceScoreOutput  
citations: List\[CitationOutput\]  
agent\_trace: Optional\[AgentTraceOutput\]  
processing\_time\_ms: int  
total\_tokens\_used: int  
  
class AbstentionResponse(BaseModel):  
response\_type: Literal\["abstention"\] = "abstention"  
message: str = "Insufficient reliable evidence available to provide a confident medical answer."  
abstention\_reason: str  
partial\_findings: Optional\[List\[EvidenceSnippet\]\]  
suggested\_followup: Optional\[str\]  
confidence: Optional\[ConfidenceScoreOutput\]  
agent\_trace: Optional\[AgentTraceOutput\]  
processing\_time\_ms: int  
  
class ErrorResponse(BaseModel):  
response\_type: Literal\["error"\] = "error"  
error\_type: str  
error\_message: str  
suggestion: Optional\[str\]  
  
MedicalResponse = Union\[VerifiedAnswerResponse, AbstentionResponse, ErrorResponse\]  
  
\# ============================================================  
\# CONVERSATION SCHEMAS  
\# ============================================================  
class ConversationCreate(BaseModel):  
title: Optional\[str\] = None  
  
class ConversationResponse(BaseModel):  
model\_config = ConfigDict(from\_attributes=True)  
id: UUID  
title: str  
specialty\_tag: Optional\[str\]  
status: str  
created\_at: datetime  
updated\_at: datetime  
message\_count: int  
  
class MessageResponse(BaseModel):  
model\_config = ConfigDict(from\_attributes=True)  
id: UUID  
role: str  
content: str  
message\_type: str  
created\_at: datetime  
  
class ConversationDetailResponse(ConversationResponse):  
messages: List\[MessageResponse\]  
  
class ConversationListResponse(BaseModel):  
conversations: List\[ConversationResponse\]  
total: int  
page: int  
page\_size: int  
  
\# ============================================================  
\# DOCUMENT UPLOAD SCHEMAS  
\# ============================================================  
class DocumentUploadRequest(BaseModel):  
upload\_notes: Optional\[str\] = None  
  
class DocumentUploadResponse(BaseModel):  
id: UUID  
original\_filename: str  
file\_size\_bytes: int  
processing\_status: str  
created\_at: datetime  
  
class DocumentProcessingStatus(BaseModel):  
id: UUID  
original\_filename: str  
processing\_status: str  
document\_id: Optional\[UUID\]  
progress\_percent: Optional\[int\]  
error\_message: Optional\[str\]  
created\_at: datetime  
processed\_at: Optional\[datetime\]  
  
\# ============================================================  
\# ADMIN / SYSTEM SCHEMAS  
\# ============================================================  
class SystemMetricsResponse(BaseModel):  
total\_queries: int  
total\_abstentions: int  
avg\_response\_time\_ms: float  
avg\_confidence\_score: float  
retrieval\_precision\_at\_5: float  
hallucination\_rate: float  
abstention\_rate: float  
active\_users: int  
  
class KnowledgeBaseStatsResponse(BaseModel):  
source\_type: SourceType  
document\_count: int  
chunk\_count: int  
index\_size\_mb: float  
last\_updated: Optional\[datetime\]  
avg\_chunk\_quality: Optional\[float\]  
  
class UserManagementResponse(BaseModel):  
users: List\[UserResponse\]  
total: int  
page: int  
page\_size: int  
  
class FeedbackSubmitRequest(BaseModel):  
response\_id: UUID  
rating: Optional\[int\] = Field(None, ge=1, le=5)  
is\_helpful: Optional\[bool\] = None  
feedback\_text: Optional\[str\] = None  
issue\_type: Optional\[str\] = None  
  
\# ============================================================  
\# WEBSOCKET SCHEMAS  
\# ============================================================  
class WSAgentStatusPayload(BaseModel):  
agent\_type: AgentType  
status: str  
message: str  
progress\_percent: Optional\[int\]  
  
class WSRetrievalPayload(BaseModel):  
iteration: int  
sub\_query: str  
documents\_found: int  
current\_status: str  
  
class WSFinalResponsePayload(BaseModel):  
response: MedicalResponse  
  
class WSMessage(BaseModel):  
type: WSMessageType  
query\_id: UUID  
payload: Dict\[str, Any\]  
timestamp: datetime = Field(default\_factory=datetime.utcnow)

# 3\. Configuration & Prompt Templates

from pydantic\_settings import BaseSettings  
from typing import List  
  
\# ============================================================  
\# APPLICATION CONFIGURATION  
\# ============================================================  
class AppConfig(BaseSettings):  
\# Application  
APP\_NAME: str = "RAGMedX-MA"  
APP\_VERSION: str = "2.0.0"  
DEBUG: bool = False  
  
\# Security  
SECRET\_KEY: str  
JWT\_ALGORITHM: str = "HS256"  
ACCESS\_TOKEN\_EXPIRE\_MINUTES: int = 30  
REFRESH\_TOKEN\_EXPIRE\_DAYS: int = 7  
  
\# Database  
DATABASE\_URL: str = "postgresql+asyncpg://user:pass@localhost/ragmedx"  
  
\# Vector DB (FAISS)  
FAISS\_INDEX\_PATH: str = "./data/faiss\_index"  
FAISS\_INDEX\_TYPE: str = "IVF\_FLAT"  
FAISS\_NLIST: int = 100  
FAISS\_NPROBE: int = 10  
  
\# Embedding Models  
BGE\_M3\_MODEL\_PATH: str = "BAAI/bge-m3"  
PUBMEDBERT\_MODEL\_PATH: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"  
EMBEDDING\_BATCH\_SIZE: int = 32  
  
\# Re-ranker  
RERANKER\_MODEL\_PATH: str = "BAAI/bge-reranker-v2-m3"  
MEDCPT\_MODEL\_PATH: str = "ncbi/MedCPT-CrossEncoder"  
RERANKER\_TOP\_K: int = 32  
RERANKER\_FINAL\_K: int = 5  
  
\# LLM Configuration  
LLM\_MODEL\_PATH: str = "Qwen/Qwen3-14B"  
LLM\_MAX\_TOKENS: int = 2048  
LLM\_TEMPERATURE: float = 0.3  
LLM\_TOP\_P: float = 0.9  
LLM\_QUANTIZATION: str = "4bit"  
  
\# Multi-Agent Settings  
MAX\_RETRIEVAL\_ITERATIONS: int = 3  
MAX\_CRITIC\_ROUNDS: int = 2  
CONFIDENCE\_THRESHOLD: float = 0.65  
  
\# Confidence Weights  
RETRIEVAL\_WEIGHT: float = 0.25  
EVIDENCE\_QUALITY\_WEIGHT: float = 0.30  
ANSWER\_CONSISTENCY\_WEIGHT: float = 0.25  
INTER\_SOURCE\_AGREEMENT\_WEIGHT: float = 0.20  
  
\# Chunking  
CHUNK\_SIZE: int = 2000  
CHUNK\_OVERLAP: int = 400  
  
\# Performance  
MAX\_CONCURRENT\_USERS: int = 50  
REQUEST\_TIMEOUT\_SECONDS: int = 60  
CACHE\_TTL\_SECONDS: int = 3600  
  
\# Data Sources  
ENABLED\_SOURCES: List\[str\] = \["pubmed", "pmc\_open\_access", "medqa", "bioasq"\]  
  
\# Logging  
LOG\_LEVEL: str = "INFO"  
LOG\_FORMAT: str = "json"  
  
class Config:  
env\_file = ".env"  
case\_sensitive = True  
  
\# ============================================================  
\# AGENT PROMPT TEMPLATES  
\# ============================================================  
QUERY\_UNDERSTANDING\_PROMPT = """  
You are a medical query understanding agent. Your task is to analyze and optimize the user's medical question for evidence retrieval.  
  
INPUT: {user\_query}  
  
TASKS:  
1\. Detect the medical specialty (e.g., Cardiology, Neurology, Oncology)  
2\. Identify and expand medical abbreviations  
3\. Expand medical terms with synonyms (using UMLS concept mapping)  
4\. Detect query intent: diagnostic | treatment | drug\_interaction | prognosis | epidemiology  
5\. Reformulate ambiguous queries into precise search queries  
6\. Generate an optimized search query for retrieval  
  
OUTPUT FORMAT (JSON):  
{  
"detected\_specialty": "string",  
"detected\_intent": "string",  
"resolved\_abbreviations": {"abbr": "full\_form"},  
"expanded\_terms": {"term": \["synonym1", "synonym2"\]},  
"optimized\_search\_query": "string"  
}  
"""  
  
GENERATOR\_PROMPT = """  
You are a medical evidence synthesis agent. Generate a comprehensive, evidence-based answer to the medical question using ONLY the provided retrieved evidence.  
  
QUESTION: {question}  
RETRIEVED EVIDENCE: {evidence\_context}  
  
RULES:  
1\. Ground EVERY claim in the retrieved evidence  
2\. Use inline citations in format \[Doc X\] where X is the document number  
3\. If evidence is conflicting, present both sides and note the conflict  
4\. Do NOT include information not present in the retrieved evidence  
5\. Use clinical, precise language appropriate for healthcare professionals  
6\. Structure the answer clearly with headings if needed  
  
OUTPUT FORMAT:  
\- Direct answer to the question  
\- Supporting evidence with inline citations \[Doc X\]  
\- Any noted limitations or conflicts  
"""  
  
CRITIC\_PROMPT = """  
You are an independent medical peer-review agent. Review the following draft answer against the retrieved evidence for factual accuracy and alignment.  
  
QUESTION: {question}  
DRAFT ANSWER: {draft\_answer}  
RETRIEVED EVIDENCE: {evidence\_context}  
  
REVIEW CRITERIA:  
1\. Factual Consistency: Are all claims supported by the evidence?  
2\. Medical Correctness: Is the medical information accurate?  
3\. Evidence Alignment: Does the answer reflect what the evidence says?  
4\. Hallucination Detection: Are there any unsupported claims?  
5\. Logical Soundness: Is the reasoning valid?  
  
OUTPUT FORMAT (JSON):  
{  
"verdict": "verified|revised|rejected|needs\_revision",  
"factual\_consistency\_score": 0.0-1.0,  
"medical\_correctness\_score": 0.0-1.0,  
"evidence\_alignment\_score": 0.0-1.0,  
"hallucination\_detected": true/false,  
"unsupported\_claims": \[{"claim\_text": "...", "reason": "..."}\],  
"critic\_feedback": "detailed feedback",  
"suggested\_revisions": \["revision suggestion 1", "revision suggestion 2"\]  
}  
"""  
  
CONFIDENCE\_PROMPT = """  
You are a confidence calibration agent. Evaluate the reliability of the medical answer based on the following components:  
  
RETRIEVAL METRICS:  
\- Precision@5: {precision\_at\_5}  
\- Recall@5: {recall\_at\_5}  
\- Average similarity score: {avg\_similarity}  
  
EVIDENCE QUALITY:  
\- Study types present: {study\_types}  
\- GRADE levels: {grade\_levels}  
\- Publication years range: {year\_range}  
\- Contradictions detected: {has\_contradictions}  
  
ANSWER CONSISTENCY:  
\- Claims supported by citations: {supported\_claims}/{total\_claims}  
\- Critic verdict: {critic\_verdict}  
  
INTER-SOURCE AGREEMENT:  
\- Sources agreeing: {agreeing\_sources}/{total\_sources}  
\- Conflict severity: {conflict\_severity}  
  
Compute a composite confidence score (0.0-1.0) and categorize as:  
\- HIGH (>=0.80): Strong evidence, consistent sources  
\- MODERATE (0.65-0.79): Adequate evidence, minor inconsistencies  
\- LOW (<0.65): Weak evidence, significant gaps or conflicts  
  
OUTPUT FORMAT (JSON):  
{  
"retrieval\_confidence": 0.0-1.0,  
"evidence\_quality\_score": 0.0-1.0,  
"answer\_consistency\_score": 0.0-1.0,  
"inter\_source\_agreement": 0.0-1.0,  
"composite\_confidence": 0.0-1.0,  
"confidence\_category": "high|moderate|low",  
"reasoning": "brief explanation"  
}  
"""  
  
ABSTENTION\_PROMPT = """  
You are a clinical safety gate. Based on the confidence calibration results, decide whether to provide the answer or abstain.  
  
CONFIDENCE SCORE: {confidence\_score}  
THRESHOLD: {threshold}  
CONFIDENCE CATEGORY: {confidence\_category}  
EVIDENCE SUMMARY: {evidence\_summary}  
  
DECISION RULES:  
\- If confidence >= threshold AND no critical contradictions: ALLOW answer  
\- If confidence < threshold: ABSTAIN with explanation  
\- If critical contradictions exist: ABSTAIN with conflict note  
\- If no relevant evidence found: ABSTAIN with "no evidence" message  
  
OUTPUT FORMAT (JSON):  
{  
"should\_abstain": true/false,  
"abstention\_reason": "reason if abstaining",  
"partial\_findings": "any partial evidence to share",  
"suggested\_followup": "suggested reformulated question if applicable"  
}  
"""

# 4\. API Endpoint Specification

from fastapi import APIRouter, Depends, HTTPException, WebSocket, UploadFile, File, Query as QueryParam  
from fastapi.security import OAuth2PasswordBearer  
from typing import List, Optional  
from uuid import UUID  
  
\# Router definitions  
auth\_router = APIRouter(prefix="/api/v1/auth", tags=\["Authentication"\])  
user\_router = APIRouter(prefix="/api/v1/users", tags=\["Users"\])  
chat\_router = APIRouter(prefix="/api/v1/chat", tags=\["Chat / Conversations"\])  
query\_router = APIRouter(prefix="/api/v1/query", tags=\["Medical Queries"\])  
document\_router = APIRouter(prefix="/api/v1/documents", tags=\["Documents"\])  
admin\_router = APIRouter(prefix="/api/v1/admin", tags=\["Admin"\])  
websocket\_router = APIRouter(prefix="/api/v1/ws", tags=\["WebSocket"\])  
  
\# ============================================================  
\# AUTH ENDPOINTS  
\# ============================================================  
@auth\_router.post("/register", response\_model=TokenResponse)  
async def register\_user(request: UserRegisterRequest):  
"""Register a new user account with email verification."""  
pass  
  
@auth\_router.post("/login", response\_model=TokenResponse)  
async def login\_user(request: UserLoginRequest):  
"""Authenticate user and return JWT access & refresh tokens."""  
pass  
  
@auth\_router.post("/refresh", response\_model=TokenResponse)  
async def refresh\_access\_token(refresh\_token: str):  
"""Refresh access token using a valid refresh token."""  
pass  
  
@auth\_router.post("/logout")  
async def logout\_user(current\_user: UserResponse = Depends(get\_current\_user)):  
"""Revoke current user session and invalidate tokens."""  
pass  
  
@auth\_router.get("/verify-email/{token}")  
async def verify\_email(token: str):  
"""Verify user email address using verification token."""  
pass  
  
@auth\_router.post("/forgot-password")  
async def forgot\_password(email: EmailStr):  
"""Request password reset email."""  
pass  
  
@auth\_router.post("/reset-password")  
async def reset\_password(token: str, new\_password: str):  
"""Reset password using reset token."""  
pass  
  
\# ============================================================  
\# USER ENDPOINTS  
\# ============================================================  
@user\_router.get("/me", response\_model=UserResponse)  
async def get\_current\_user\_profile(current\_user: UserResponse = Depends(get\_current\_user)):  
"""Get current authenticated user's profile."""  
pass  
  
@user\_router.patch("/me", response\_model=UserResponse)  
async def update\_profile(update: UserProfileUpdate, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Update current user's profile information."""  
pass  
  
@user\_router.get("/me/saved-responses")  
async def get\_saved\_responses(page: int = QueryParam(1, ge=1), page\_size: int = QueryParam(20, ge=1, le=100), current\_user: UserResponse = Depends(get\_current\_user)):  
"""Get paginated list of user's saved/bookmarked responses."""  
pass  
  
@user\_router.post("/me/saved-responses/{response\_id}")  
async def save\_response(response\_id: UUID, note: Optional\[str\] = None, tags: Optional\[List\[str\]\] = None, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Save/bookmark a response with optional note and tags."""  
pass  
  
@user\_router.delete("/me/saved-responses/{response\_id}")  
async def unsave\_response(response\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Remove a saved response from bookmarks."""  
pass  
  
\# ============================================================  
\# CHAT / CONVERSATION ENDPOINTS  
\# ============================================================  
@chat\_router.post("/conversations", response\_model=ConversationResponse)  
async def create\_conversation(request: ConversationCreate, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Create a new conversation session."""  
pass  
  
@chat\_router.get("/conversations", response\_model=ConversationListResponse)  
async def list\_conversations(page: int = QueryParam(1, ge=1), page\_size: int = QueryParam(20, ge=1, le=100), search: Optional\[str\] = None, current\_user: UserResponse = Depends(get\_current\_user)):  
"""List user's conversations with pagination and optional search."""  
pass  
  
@chat\_router.get("/conversations/{conversation\_id}", response\_model=ConversationDetailResponse)  
async def get\_conversation(conversation\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Get conversation details with all messages."""  
pass  
  
@chat\_router.patch("/conversations/{conversation\_id}")  
async def update\_conversation(conversation\_id: UUID, title: Optional\[str\] = None, status: Optional\[str\] = None, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Update conversation title or status."""  
pass  
  
@chat\_router.delete("/conversations/{conversation\_id}")  
async def delete\_conversation(conversation\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Soft-delete a conversation."""  
pass  
  
@chat\_router.post("/conversations/{conversation\_id}/export")  
async def export\_conversation(conversation\_id: UUID, format: str = QueryParam("pdf", regex="^(pdf|json|markdown)$"), current\_user: UserResponse = Depends(get\_current\_user)):  
"""Export conversation as PDF, JSON, or Markdown."""  
pass  
  
\# ============================================================  
\# QUERY / MEDICAL QUESTION ENDPOINTS  
\# ============================================================  
@query\_router.post("/ask", response\_model=MedicalResponse)  
async def ask\_medical\_question(request: MedicalQuestionRequest, current\_user: UserResponse = Depends(get\_current\_user)):  
"""  
Submit a medical question and receive an evidence-based answer.  
The system executes the full multi-agent pipeline:  
1\. Query Understanding Agent  
2\. Iterative Retrieval Agent (up to 3 iterations)  
3\. Evidence Re-Ranking Agent  
4\. Generator Agent  
5\. Critic Agent  
6\. Confidence Calibration Agent  
7\. Abstention Decision  
"""  
pass  
  
@query\_router.get("/{query\_id}/status")  
async def get\_query\_status(query\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Get real-time processing status of a query."""  
pass  
  
@query\_router.get("/{query\_id}/agent-trace")  
async def get\_agent\_trace(query\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Get detailed agent reasoning trace for transparency."""  
pass  
  
@query\_router.get("/{query\_id}/evidence")  
async def get\_retrieved\_evidence(query\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Get all retrieved evidence with relevance scores."""  
pass  
  
@query\_router.get("/{query\_id}/confidence")  
async def get\_confidence\_details(query\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Get detailed confidence score breakdown."""  
pass  
  
@query\_router.post("/{query\_id}/feedback")  
async def submit\_feedback(query\_id: UUID, feedback: FeedbackSubmitRequest, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Submit user feedback on a response for system improvement."""  
pass  
  
\# ============================================================  
\# DOCUMENT UPLOAD ENDPOINTS  
\# ============================================================  
@document\_router.post("/upload")  
async def upload\_document(file: UploadFile = File(...), notes: Optional\[str\] = None, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Upload a medical document (PDF) for indexing. Max 50MB."""  
pass  
  
@document\_router.get("/uploads")  
async def list\_uploads(page: int = QueryParam(1, ge=1), page\_size: int = QueryParam(20, ge=1, le=100), current\_user: UserResponse = Depends(get\_current\_user)):  
"""List user's uploaded documents with processing status."""  
pass  
  
@document\_router.get("/uploads/{upload\_id}/status", response\_model=DocumentProcessingStatus)  
async def get\_upload\_status(upload\_id: UUID):  
"""Get document processing status and progress."""  
pass  
  
@document\_router.delete("/uploads/{upload\_id}")  
async def delete\_upload(upload\_id: UUID, current\_user: UserResponse = Depends(get\_current\_user)):  
"""Delete an uploaded document and its indexed chunks."""  
pass  
  
\# ============================================================  
\# WEBSOCKET ENDPOINT  
\# ============================================================  
@websocket\_router.websocket("/query-stream")  
async def query\_stream\_websocket(websocket: WebSocket):  
"""  
WebSocket endpoint for real-time query processing updates.  
Message Types: agent\_status, retrieval\_update, generation\_update,  
critic\_update, confidence\_update, final\_response, error  
"""  
pass  
  
\# ============================================================  
\# ADMIN ENDPOINTS  
\# ============================================================  
@admin\_router.get("/metrics", response\_model=SystemMetricsResponse)  
async def get\_system\_metrics(current\_user: UserResponse = Depends(require\_admin)):  
"""Get system-wide performance and quality metrics."""  
pass  
  
@admin\_router.get("/metrics/timeseries")  
async def get\_metrics\_timeseries(metric\_type: str, start\_date: datetime, end\_date: datetime, granularity: str = QueryParam("day", regex="^(hour|day|week|month)$"), current\_user: UserResponse = Depends(require\_admin)):  
"""Get time-series metrics for trend analysis."""  
pass  
  
@admin\_router.get("/knowledge-base/stats", response\_model=List\[KnowledgeBaseStatsResponse\])  
async def get\_knowledge\_base\_stats(current\_user: UserResponse = Depends(require\_admin)):  
"""Get knowledge base statistics by source type."""  
pass  
  
@admin\_router.post("/knowledge-base/reindex")  
async def trigger\_reindex(source\_type: Optional\[str\] = None, current\_user: UserResponse = Depends(require\_admin)):  
"""Trigger knowledge base reindexing."""  
pass  
  
@admin\_router.get("/users", response\_model=UserManagementResponse)  
async def list\_all\_users(page: int = QueryParam(1, ge=1), page\_size: int = QueryParam(50, ge=1, le=200), role: Optional\[str\] = None, current\_user: UserResponse = Depends(require\_admin)):  
"""List all users with pagination and role filtering (admin only)."""  
pass  
  
@admin\_router.patch("/users/{user\_id}/role")  
async def update\_user\_role(user\_id: UUID, new\_role: UserRole, current\_user: UserResponse = Depends(require\_admin)):  
"""Update user role (admin only)."""  
pass  
  
@admin\_router.get("/queries")  
async def list\_all\_queries(page: int = QueryParam(1, ge=1), page\_size: int = QueryParam(50, ge=1, le=200), status: Optional\[str\] = None, start\_date: Optional\[datetime\] = None, end\_date: Optional\[datetime\] = None, current\_user: UserResponse = Depends(require\_admin)):  
"""List all queries with filtering (admin only)."""  
pass  
  
@admin\_router.get("/feedback")  
async def list\_all\_feedback(page: int = QueryParam(1, ge=1), page\_size: int = QueryParam(50, ge=1, le=200), issue\_type: Optional\[str\] = None, current\_user: UserResponse = Depends(require\_admin)):  
"""List all user feedback for quality monitoring (admin only)."""  
pass  
  
@admin\_router.get("/feedback/summary")  
async def get\_feedback\_summary(current\_user: UserResponse = Depends(require\_admin)):  
"""Get aggregated feedback statistics and trends."""  
pass  
  
@admin\_router.get("/performance/latency")  
async def get\_latency\_breakdown(current\_user: UserResponse = Depends(require\_admin)):  
"""Get per-agent latency breakdown for optimization."""  
pass

# 5\. Project Directory Structure

ragmedx-backend/  
│  
├── app/  
│ ├── \_\_init\_\_.py  
│ ├── main.py # FastAPI application entry point  
│ ├── config.py # AppConfig (Pydantic Settings)  
│ │  
│ ├── api/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── deps.py # Dependencies (auth, DB, etc.)  
│ │ └── v1/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── endpoints/  
│ │ │ ├── \_\_init\_\_.py  
│ │ │ ├── auth.py # Authentication endpoints  
│ │ │ ├── users.py # User management endpoints  
│ │ │ ├── chat.py # Conversation endpoints  
│ │ │ ├── query.py # Medical query endpoints  
│ │ │ ├── documents.py # Document upload endpoints  
│ │ │ └── admin.py # Admin endpoints  
│ │ └── websockets.py # WebSocket handlers  
│ │  
│ ├── core/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── security.py # JWT, password hashing  
│ │ ├── exceptions.py # Custom exceptions  
│ │ └── logging.py # Structured logging  
│ │  
│ ├── db/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── base.py # SQLAlchemy base & session  
│ │ ├── models.py # All ORM models  
│ │ └── migrations/ # Alembic migrations  
│ │  
│ ├── schemas/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── auth.py # Auth request/response schemas  
│ │ ├── user.py # User schemas  
│ │ ├── query.py # Query & agent schemas  
│ │ ├── response.py # Final response schemas  
│ │ ├── conversation.py # Conversation schemas  
│ │ ├── document.py # Document upload schemas  
│ │ ├── admin.py # Admin schemas  
│ │ └── websocket.py # WebSocket message schemas  
│ │  
│ ├── services/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── auth\_service.py # Authentication logic  
│ │ ├── user\_service.py # User CRUD operations  
│ │ ├── chat\_service.py # Conversation management  
│ │ ├── query\_service.py # Query orchestration  
│ │ ├── document\_service.py # Document upload & processing  
│ │ └── admin\_service.py # Admin operations  
│ │  
│ ├── agents/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── base.py # Base agent class  
│ │ ├── query\_understanding.py # Agent 1  
│ │ ├── iterative\_retrieval.py # Agent 2  
│ │ ├── evidence\_reranking.py # Agent 3  
│ │ ├── generator.py # Agent 4  
│ │ ├── critic.py # Agent 5  
│ │ ├── confidence\_calibration.py # Agent 6  
│ │ └── abstention\_decision.py # Agent 7  
│ │  
│ ├── retrieval/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── embeddings.py # Embedding generation (BGE-M3, PubMedBERT)  
│ │ ├── vector\_store.py # FAISS index management  
│ │ ├── hybrid\_search.py # BM25 + Dense + RRF  
│ │ ├── reranker.py # Cross-encoder re-ranking (MedCPT)  
│ │ └── document\_loader.py # PubMed/PMC data ingestion  
│ │  
│ ├── llm/  
│ │ ├── \_\_init\_\_.py  
│ │ ├── local\_llm.py # Local LLM wrapper (Qwen3/Llama3.1)  
│ │ ├── prompt\_templates.py # All prompt templates  
│ │ └── tokenizer.py # Token counting & management  
│ │  
│ └── utils/  
│ ├── \_\_init\_\_.py  
│ ├── text\_processing.py # Text chunking, cleaning  
│ ├── citation\_parser.py # Citation extraction  
│ ├── confidence\_utils.py # Confidence score computation  
│ └── validators.py # Input validation helpers  
│  
├── data/  
│ ├── faiss\_index/ # FAISS index files  
│ ├── documents/ # Uploaded documents  
│ └── cache/ # Response cache  
│  
├── tests/  
│ ├── \_\_init\_\_.py  
│ ├── conftest.py # Pytest fixtures  
│ ├── test\_auth.py  
│ ├── test\_query.py  
│ ├── test\_agents.py  
│ └── test\_retrieval.py  
│  
├── alembic.ini # Alembic config  
├── requirements.txt # Python dependencies  
├── Dockerfile # Container definition  
├── docker-compose.yml # Multi-service orchestration  
└── .env.example # Environment variables template

# 6\. Environment Variables Template

\# ============================================================  
\# RAGMedX-MA Environment Configuration  
\# Copy this file to .env and fill in your values  
\# ============================================================  
  
\# — Application —  
APP\_NAME=RAGMedX-MA  
APP\_VERSION=2.0.0  
DEBUG=false  
  
\# — Security —  
SECRET\_KEY=your-super-secret-key-change-this-in-production  
JWT\_ALGORITHM=HS256  
ACCESS\_TOKEN\_EXPIRE\_MINUTES=30  
REFRESH\_TOKEN\_EXPIRE\_DAYS=7  
  
\# — Database —  
DATABASE\_URL=postgresql+asyncpg://ragmedx\_user:ragmedx\_pass@db:5432/ragmedx  
  
\# — Vector DB (FAISS) —  
FAISS\_INDEX\_PATH=./data/faiss\_index  
FAISS\_INDEX\_TYPE=IVF\_FLAT  
FAISS\_NLIST=100  
FAISS\_NPROBE=10  
  
\# — Embedding Models —  
BGE\_M3\_MODEL\_PATH=BAAI/bge-m3  
PUBMEDBERT\_MODEL\_PATH=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract  
EMBEDDING\_BATCH\_SIZE=32  
  
\# — Re-ranker —  
RERANKER\_MODEL\_PATH=BAAI/bge-reranker-v2-m3  
MEDCPT\_MODEL\_PATH=ncbi/MedCPT-CrossEncoder  
RERANKER\_TOP\_K=32  
RERANKER\_FINAL\_K=5  
  
\# — LLM Configuration —  
LLM\_MODEL\_PATH=Qwen/Qwen3-14B  
LLM\_MAX\_TOKENS=2048  
LLM\_TEMPERATURE=0.3  
LLM\_TOP\_P=0.9  
LLM\_QUANTIZATION=4bit  
  
\# — Multi-Agent Settings —  
MAX\_RETRIEVAL\_ITERATIONS=3  
MAX\_CRITIC\_ROUNDS=2  
CONFIDENCE\_THRESHOLD=0.65  
  
\# — Confidence Weights —  
RETRIEVAL\_WEIGHT=0.25  
EVIDENCE\_QUALITY\_WEIGHT=0.30  
ANSWER\_CONSISTENCY\_WEIGHT=0.25  
INTER\_SOURCE\_AGREEMENT\_WEIGHT=0.20  
  
\# — Chunking —  
CHUNK\_SIZE=2000  
CHUNK\_OVERLAP=400  
  
\# — Performance —  
MAX\_CONCURRENT\_USERS=50  
REQUEST\_TIMEOUT\_SECONDS=60  
CACHE\_TTL\_SECONDS=3600  
  
\# — Data Sources —  
ENABLED\_SOURCES=pubmed,pmc\_open\_access,medqa,bioasq  
  
\# — Logging —  
LOG\_LEVEL=INFO  
LOG\_FORMAT=json

# 7\. Requirements.txt

\# ============================================================  
\# RAGMedX-MA Python Dependencies  
\# ============================================================  
  
\# — Web Framework —  
fastapi==0.111.0  
uvicorn\[standard\]==0.30.0  
python-multipart==0.0.9  
websockets==12.0  
  
\# — Authentication & Security —  
python-jose\[cryptography\]==3.3.0  
passlib\[bcrypt\]==1.7.4  
python-dotenv==1.0.1  
  
\# — Database —  
sqlalchemy\[asyncio\]==2.0.31  
asyncpg==0.29.0  
alembic==1.13.2  
pgvector==0.2.5  
  
\# — Data Validation —  
pydantic==2.8.2  
pydantic-settings==2.3.4  
email-validator==2.2.0  
  
\# — ML / AI —  
torch==2.3.1  
transformers==4.42.4  
sentence-transformers==3.0.1  
huggingface-hub==0.23.4  
accelerate==0.32.1  
bitsandbytes==0.43.1  
  
\# — Vector Search —  
faiss-cpu==1.8.0  
\# faiss-gpu==1.8.0 # Use for GPU deployment  
  
\# — RAG Framework —  
langchain==0.2.11  
langchain-community==0.2.10  
langchain-huggingface==0.0.3  
langgraph==0.1.19  
  
\# — Document Processing —  
pypdf==4.3.1  
python-docx==1.1.2  
beautifulsoup4==4.12.3  
lxml==5.2.2  
  
\# — Text Processing —  
nltk==3.8.1  
spacy==3.7.5  
scispacy==0.5.4  
  
\# — Utilities —  
numpy==1.26.4  
pandas==2.2.2  
scikit-learn==1.5.1  
requests==2.32.3  
aiohttp==3.9.5  
httpx==0.27.0  
  
\# — Testing —  
pytest==8.3.2  
pytest-asyncio==0.23.8  
pytest-cov==5.0.0  
  
\# — Monitoring & Logging —  
structlog==24.4.0  
prometheus-client==0.20.0  
  
\# — Caching —  
redis==5.0.7

# Document Information

Document Owner: RAGMedX Development Team

Reviewers: Lead Backend Engineer, System Architect

Approval Status: Pending Review

Generated: July 2026