export type AgentPhase =
  | 'Query Understanding'
  | 'Iterative Retrieval'
  | 'Re-ranking'
  | 'Generator'
  | 'Critic'
  | 'Confidence Calibration'
  | 'Abstention Gatekeeper';

export type PipelineStage = 'Searching' | 'Re-ranking' | 'Drafting' | 'Reviewing';

export type AgentRole = 'Retrieval' | 'Generator' | 'Critic' | 'Revision' | AgentPhase;

export type AgentStatus = 'pending' | 'in_progress' | 'completed' | 'failed';

export interface AgentTraceStep {
  id: string;
  phase: AgentPhase;
  agent?: AgentRole;
  action: string;
  status: AgentStatus;
  timestamp?: string;
  details?: string;
  subQueries?: string[];
  queryOrPayload?: string;
}

