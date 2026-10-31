/**
 * RAGMedX-MA — WebSocket Event Contracts
 *
 * These types mirror the backend WebSocketEventType enum exactly (lowercase
 * string values).  They are the canonical reference for event_type comparisons
 * across the codebase — do not use the old uppercase SCREAMING_SNAKE names.
 *
 * Actual WebSocket connection logic lives in api.ts (askQuestion).
 */

import { AgentTraceStep } from '../types/agent';

// Fix #6: event type values now match the backend enum lowercase serialisation.
// Old stubs used AGENT_START / RETRIEVAL_PROGRESS / GENERATION_DRAFT etc. —
// none of those exist on the backend.  The real set is below.
export type WebSocketEventType =
  | 'agent_status'
  | 'retrieval_update'
  | 'final_response'
  | 'error'
  | 'ping';

export interface WebSocketEvent {
  query_id: string;
  event_type: WebSocketEventType;
  timestamp: string;
  data: Record<string, unknown>;
}

// ── Per-event data shapes (match backend WsEvent.data payloads) ──────────────

export interface AgentStatusData {
  agent: string;
  status: string;
  duration_ms?: number;
}

export interface RetrievalUpdateData {
  message?: string;
}

/** data.response is a discriminated union: verified_answer | abstention | error */
export interface FinalResponseData {
  response: {
    response_type: 'verified_answer' | 'abstention' | 'error';
    [key: string]: unknown;
  };
}

export interface WsErrorData {
  message: string;
  error_code?: string;
}

// ── Stream connection options (for callers that manage their own WS) ─────────

export interface StreamConnectionOptions {
  wsUrl: string;
  onAgentStatus?: (data: AgentStatusData, step: AgentTraceStep) => void;
  onRetrievalUpdate?: (data: RetrievalUpdateData) => void;
  onFinalResponse?: (data: FinalResponseData) => void;
  onError?: (data: WsErrorData) => void;
  onClose?: () => void;
}
