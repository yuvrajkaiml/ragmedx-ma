/**
 * RAGMedX-MA — Query API Service
 *
 * Wires POST /api/v1/query/ask (202 Accepted → returns query_id) and
 * WS  /api/v1/ws/query-stream?query_id=&token= (streams events until final_response).
 *
 * The backend response shape differs from the frontend AnswerData type, so an
 * adapter (adaptBackendResponse) converts between them.
 */

import { AnswerData } from '../types/answer';
import { AgentTraceStep } from '../types/agent';
import { PassageItem, EvidenceSourceItem } from '../types/evidence';
import { getAccessToken } from './authService';

const BASE = '/api/v1';

// ── Backend shapes ────────────────────────────────────────────────────────────

// Fix #4: status typed as the actual enum values, not a loose string
export type QueryExecutionStatus = 'queued' | 'running' | 'completed' | 'failed';

interface QueryAcceptedResponse {
  query_id: string;
  status: QueryExecutionStatus;
}

interface BackendConfidence {
  score: number;           // 0.0–1.0
  level: string;           // high | moderate | low
  retrieval_score?: number;
  evidence_quality_score?: number;
  answer_consistency_score?: number;
  inter_source_agreement?: number;
}

interface BackendCitation {
  document_id: string;
  citation_marker: string;
  title: string;
  journal?: string;
  publication_year?: number;
  doi?: string;
  url?: string;
  cited_text: string;
  relevance_score: number;
}

interface BackendEvidenceSnippet {
  chunk_id: string;
  document_id: string;
  source_type: string;
  title: string;
  cited_text: string;
  relevance_score: number;
}

interface BackendTraceStep {
  agent_type: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  details?: Record<string, unknown>;
}

interface BackendAgentTrace {
  steps: BackendTraceStep[];
  total_duration_ms: number;
}

interface BackendVerifiedAnswer {
  response_type: 'verified_answer';
  answer: string;
  confidence: BackendConfidence;
  citations: BackendCitation[];
  processing_time_ms: number;
  total_tokens_used: number;
  agent_trace?: BackendAgentTrace;
}

interface BackendAbstention {
  response_type: 'abstention';
  message: string;
  abstention_reason: string;
  partial_findings?: BackendEvidenceSnippet[];
  suggested_followup?: string;
  confidence?: BackendConfidence;
  // Fix #5: include agent_trace field that backend actually sends
  agent_trace?: BackendAgentTrace;
  processing_time_ms: number;
}

interface BackendError {
  response_type: 'error';
  message: string;
  error_code?: string;
}

type BackendMedicalResponse = BackendVerifiedAnswer | BackendAbstention | BackendError;

interface BackendQueryExecution {
  query_id: string;
  status: QueryExecutionStatus;
  created_at: string;
  completed_at?: string;
  response?: BackendMedicalResponse;
  error_code?: string;
}

// ── WebSocket event shapes (backend) ─────────────────────────────────────────

// Fix #1: event_type values match the backend enum's lowercase string values exactly
type WsEventType =
  | 'agent_status'
  | 'retrieval_update'
  | 'final_response'
  | 'error'
  | 'ping';

interface WsEvent {
  query_id: string;
  event_type: WsEventType;
  timestamp: string;
  data: Record<string, unknown>;
}

// ── Adapter: backend → frontend types ────────────────────────────────────────

function confidenceLevel(level: string): 'High' | 'Moderate' | 'Insufficient' {
  if (level === 'high') return 'High';
  if (level === 'moderate') return 'Moderate';
  return 'Insufficient';
}

function adaptBackendResponse(
  query: string,
  r: BackendMedicalResponse,
): AnswerData {
  const now = new Date().toLocaleTimeString();

  if (r.response_type === 'verified_answer') {
    const sentences = r.answer.split(/(?<=\.)\s+/);
    const summary = sentences.slice(0, 2).join(' ') || r.answer;
    const body = sentences.length > 2 ? sentences.slice(2).join(' ') : undefined;
    return {
      id: `ans-${Date.now()}`,
      query,
      state: 'verified',
      confidence: {
        score: Math.round(r.confidence.score * 100),
        level: confidenceLevel(r.confidence.level),
      },
      summary,
      body,
      createdAt: now,
      citations: r.citations.map((c) => c.citation_marker),
      citationObjects: r.citations.map((c, i) => ({
        id: c.document_id,
        markerIndex: i + 1,
        sourceId: c.document_id,
        claimSnippet: c.cited_text.slice(0, 120),
        relevanceScore: c.relevance_score,
      })),
      sourceIds: r.citations.map((c) => c.document_id),
    };
  }

  if (r.response_type === 'abstention') {
    return {
      id: `ans-${Date.now()}`,
      query,
      state: 'abstained',
      confidence: r.confidence
        ? { score: Math.round(r.confidence.score * 100), level: confidenceLevel(r.confidence.level) }
        : { score: 0, level: 'Insufficient' },
      summary: r.message,
      abstentionReason: r.abstention_reason,
      partialFindings: r.partial_findings?.map((f) => f.cited_text) ?? [],
      recommendedNextActions: r.suggested_followup ? [r.suggested_followup] : [],
      createdAt: now,
      citations: [],
      sourceIds: [],
    };
  }

  // error
  return {
    id: `ans-${Date.now()}`,
    query,
    state: 'abstained',
    confidence: { score: 0, level: 'Insufficient' },
    summary: r.message,
    abstentionReason: (r as BackendError).error_code ?? 'error',
    createdAt: now,
    citations: [],
    sourceIds: [],
  };
}

function adaptTraceSteps(steps: BackendTraceStep[]): AgentTraceStep[] {
  const phaseMap: Record<string, AgentTraceStep['phase']> = {
    query_understanding: 'Query Understanding',
    iterative_retrieval: 'Iterative Retrieval',
    evidence_reranking: 'Re-ranking',
    generator: 'Generator',
    critic: 'Critic',
    confidence_calibration: 'Confidence Calibration',
    abstention_decision: 'Abstention Gatekeeper',
    langgraph: 'Query Understanding',
  };
  return steps.map((s, i) => ({
    id: `step-${i}`,
    phase: phaseMap[s.agent_type] ?? 'Query Understanding',
    action: (s.details?.action as string) ?? s.status,
    status:
      s.status === 'completed' ? 'completed'
      : s.status === 'failed' ? 'failed'
      : 'in_progress',
    timestamp: s.started_at,
    details: s.duration_ms != null ? `${s.duration_ms}ms` : undefined,
  }));
}

function adaptEvidence(
  citations: BackendCitation[],
): { passages: PassageItem[]; sources: EvidenceSourceItem[] } {
  const sources: EvidenceSourceItem[] = citations.map((c) => ({
    id: c.document_id,
    title: c.title,
    journal: c.journal,
    year: c.publication_year,
    doi: c.doi,
    url: c.url,
    snippet: c.cited_text.slice(0, 200),
    relevanceScore: c.relevance_score,
  }));
  const passages: PassageItem[] = citations.map((c, i) => ({
    id: `passage-${i}`,
    markerIndex: i + 1,
    sourceId: c.document_id,
    sourceTitle: c.title,
    journal: c.journal,
    year: c.publication_year,
    snippet: c.cited_text,
    relevanceScore: c.relevance_score,
  }));
  return { passages, sources };
}

// ── Public result type ────────────────────────────────────────────────────────

export interface QueryResult {
  answerData: AnswerData;
  traceSteps: AgentTraceStep[];
  evidence: { passages: PassageItem[]; sources: EvidenceSourceItem[] };
}

// ── Fix #2: derive WebSocket URL from window.location instead of hardcoding ──

function buildWsUrl(path: string): string {
  // In dev the Vite proxy does not handle WebSocket upgrades for /api, so we
  // connect directly to the backend host.  In production, if the reverse proxy
  // handles WS, swap to window.location.host instead.
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const backendHost =
    import.meta.env.VITE_BACKEND_WS_HOST ??
    (import.meta.env.DEV ? 'localhost:8000' : window.location.host);
  return `${protocol}//${backendHost}${path}`;
}

// ── Poll fallback (used if WS connection fails) ───────────────────────────────

async function pollUntilComplete(
  queryId: string,
  token: string,
  maxWaitMs = 120_000,
): Promise<BackendQueryExecution> {
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    const res = await fetch(`${BASE}/query/${queryId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) continue;
    const exec: BackendQueryExecution = await res.json();
    if (exec.status === 'completed' || exec.status === 'failed') return exec;
  }
  throw new Error('Query timed out after 120s');
}

// ── Main entry point ──────────────────────────────────────────────────────────

export async function askQuestion(
  question: string,
  documentId?: string,
  onTraceStep?: (step: AgentTraceStep) => void,
  onStatusMessage?: (msg: string) => void,
): Promise<QueryResult> {
  const token = getAccessToken();
  if (!token) throw new Error('Not authenticated');

  // 1. POST /api/v1/query/ask → 202 with query_id
  const startRes = await fetch(`${BASE}/query/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      question,
      include_agent_trace: true,
      ...(documentId ? { document_id: documentId } : {}),
    }),
  });
  if (!startRes.ok) {
    const err = await startRes.json();
    throw new Error(err.message || err.detail || 'Failed to start query');
  }
  const accepted: QueryAcceptedResponse = await startRes.json();
  const queryId = accepted.query_id;

  onStatusMessage?.('Query started — waiting for results…');

  // 2. Subscribe to WebSocket stream
  // Fix #3 note: token is passed as a query param because the WS handshake
  // cannot carry custom headers in the browser.  This is the standard pattern
  // for browser WebSocket auth; the backend should treat it as short-lived and
  // the connection should be over TLS (wss://) in production to protect it.
  let execution: BackendQueryExecution | null = null;
  try {
    execution = await new Promise<BackendQueryExecution>((resolve, reject) => {
      // Fix #2: URL built dynamically, not hardcoded to localhost:8000
      const wsPath = `${BASE}/ws/query-stream?query_id=${queryId}&token=${token}`;
      const ws = new WebSocket(buildWsUrl(wsPath));
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('ws_timeout'));
      }, 120_000);

      ws.onmessage = (evt) => {
        try {
          const event: WsEvent = JSON.parse(evt.data);

          // Fix #1: compare against lowercase values that the backend actually sends
          if (event.event_type === 'agent_status' && onTraceStep) {
            const s = event.data as unknown as { agent?: string; status?: string; duration_ms?: number };
            const step = adaptTraceSteps([{
              agent_type: s.agent ?? '',
              status: s.status ?? '',
              started_at: event.timestamp,
              duration_ms: s.duration_ms,
            }]);
            onTraceStep(step[0]);
          }

          if (event.event_type === 'retrieval_update') {
            onStatusMessage?.('Retrieving evidence…');
          }

          if (event.event_type === 'final_response') {
            clearTimeout(timeout);
            ws.close();
            const resp = event.data.response as BackendMedicalResponse;
            resolve({
              query_id: queryId,
              status: 'completed',
              created_at: event.timestamp,
              response: resp,
            });
          }

          if (event.event_type === 'error') {
            clearTimeout(timeout);
            ws.close();
            reject(new Error((event.data.message as string) ?? 'Pipeline failed'));
          }
        } catch {
          // ignore malformed frames
        }
      };
      ws.onerror = () => reject(new Error('ws_error'));
      ws.onclose = (e) => {
        if (!e.wasClean) reject(new Error('ws_closed'));
      };
    });
  } catch (wsErr: unknown) {
    const msg = wsErr instanceof Error ? wsErr.message : '';
    if (msg === 'ws_timeout' || msg === 'ws_error' || msg === 'ws_closed') {
      onStatusMessage?.('Streaming unavailable — polling for result…');
      execution = await pollUntilComplete(queryId, token);
    } else {
      throw wsErr;
    }
  }

  if (!execution?.response) {
    throw new Error('No response received from pipeline');
  }

  // 3. Adapt to frontend types
  const answerData = adaptBackendResponse(question, execution.response);

  // Fix #5: extract agent_trace from both verified_answer AND abstention
  let traceSteps: AgentTraceStep[] = [];
  const resp = execution.response;
  if (
    (resp.response_type === 'verified_answer' || resp.response_type === 'abstention') &&
    resp.agent_trace?.steps
  ) {
    traceSteps = adaptTraceSteps(resp.agent_trace.steps);
  }

  const evidence =
    resp.response_type === 'verified_answer'
      ? adaptEvidence(resp.citations)
      : { passages: [], sources: [] };

  return { answerData, traceSteps, evidence };
}
