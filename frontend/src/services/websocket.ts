import { AgentTraceStep } from '../types/agent';

/**
 * WebSocket Stream Event Contracts for RAGMedX-MA Multi-Agent Pipeline
 * Target WebSocket Endpoint: WS /api/v1/ws/query-stream
 * 
 * Note: These are typed contracts for live agent trace streaming.
 * No actual WebSocket connection or mock socket is instantiated.
 */

export type WebSocketEventType = 
  | 'AGENT_START'
  | 'RETRIEVAL_PROGRESS'
  | 'GENERATION_DRAFT'
  | 'CRITIC_VERDICT'
  | 'REVISION_UPDATE'
  | 'PIPELINE_COMPLETE'
  | 'ERROR';

export interface WebSocketStreamMessage {
  type: WebSocketEventType;
  queryId: string;
  timestamp: string;
  step?: AgentTraceStep;
  message?: string;
}

export interface StreamConnectionOptions {
  url: string;
  onMessage: (event: WebSocketStreamMessage) => void;
  onError: (error: Event) => void;
  onClose: () => void;
}

// Service stub contract for WebSocket stream initialization
export class QueryStreamService {
  constructor(_options: StreamConnectionOptions) {
    // Stub: No active WebSocket connection initiated in frontend-only phase
  }

  public disconnect(): void {
    // Stub method for disconnecting
  }
}
