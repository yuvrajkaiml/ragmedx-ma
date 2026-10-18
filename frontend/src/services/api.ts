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

// Bounded timeout for individual HTTP fetches so a hung TCP connection cannot
// exceed the intended budget (browsers do not time out fetch by default).
const FETCH_TIMEOUT_MS = 30_000;

/** Parse a Response body as JSON, tolerating non-JSON/empty bodies. */
async function safeJson<T>(res: Response): Promise<T | null> {
  try {
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

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

// Adapt the partial findings carried by an abstention so the Evidence panel
// reflects what the backend actually retrieved (instead of forcing 0/0).
// BackendEvidenceSnippet only has chunk_id/document_id/source_type/title/
// cited_text/relevance_score, so journal/year/doi/url/pmid are left undefined
// — we do not fabricate metadata the snippet does not carry.
function adaptPartialEvidence(
  findings: BackendEvidenceSnippet[],
): { passages: PassageItem[]; sources: EvidenceSourceItem[] } {
  const sources: EvidenceSourceItem[] = findings.map((f) => ({
    id: f.document_id,
    title: f.title,
    snippet: f.cited_text.slice(0, 200),
    relevanceScore: f.relevance_score,
  }));
  const passages: PassageItem[] = findings.map((f, i) => ({
    id: `passage-${i}`,
    markerIndex: i + 1,
    sourceId: f.document_id,
    sourceTitle: f.title,
    snippet: f.cited_text,
    relevanceScore: f.relevance_score,
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
  // One rule for dev and prod. In dev the Vite `/api` proxy (ws: true) upgrades
  // WebSocket connections to the backend, so window.location.host works; in prod
  // the reverse proxy upgrades /api on the same origin. VITE_BACKEND_WS_HOST is
  // an optional public override for topologies where the proxy does not upgrade.
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const backendHost = import.meta.env.VITE_BACKEND_WS_HOST ?? window.location.host;
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
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${BASE}/query/${queryId}`, {
        headers: { Authorization: `Bearer ${token}` },
        signal: controller.signal,
      });
    } catch {
      // Network error or aborted poll — retry until the deadline.
      continue;
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) continue;
    const exec = await safeJson<BackendQueryExecution>(res);
    if (!exec) continue;
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
  // The 30s FETCH_TIMEOUT_MS abort below is for the quick 202 accept only; the
  // pipeline runs server-side and is observed via WS/poll (120s budget), so
  // this never aborts the long-running query.
  const controller = new AbortController();
  const startTimer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  let startRes: Response;
  try {
    startRes = await fetch(`${BASE}/query/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({
        question,
        include_agent_trace: true,
        ...(documentId ? { document_id: documentId } : {}),
      }),
      signal: controller.signal,
    });
  } catch (e: unknown) {
    const aborted = e instanceof DOMException && e.name === 'AbortError';
    throw new Error(aborted ? 'Request timed out — please try again' : 'Network error — unable to reach the server');
  } finally {
    clearTimeout(startTimer);
  }
  if (!startRes.ok) {
    // A 401 here means the token is missing/expired/stale (e.g. the backend
    // restarted and rotated its signing key). Surface an actionable sign-in
    // message rather than a raw auth error or a misleading network card.
    if (startRes.status === 401) {
      throw new Error('Your session expired — please sign in again to run a query.');
    }
    const err = await safeJson<{ message?: string; detail?: string }>(startRes);
    throw new Error(err?.message || err?.detail || `Request failed (${startRes.status})`);
  }
  const accepted = await safeJson<QueryAcceptedResponse>(startRes);
  if (!accepted?.query_id) {
    throw new Error(`Request failed (${startRes.status})`);
  }
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
      : resp.response_type === 'abstention'
        ? adaptPartialEvidence(resp.partial_findings ?? [])
        : { passages: [], sources: [] };

  return { answerData, traceSteps, evidence };
}
