export type AnswerState = 'verified' | 'revised' | 'abstained';

export type ConfidenceLevel = 'High' | 'Moderate' | 'Insufficient';

export interface ConfidenceScore {
  score: number; // 0 - 100 percentage
  level: ConfidenceLevel;
}

export interface AnswerData {
  id: string;
  query: string;
  state: AnswerState;
  confidence: ConfidenceScore;
  summary: string;
  body?: string;
  revisionNote?: string;
  originalAnswer?: string;
  abstentionReason?: string;
  partialFindings?: string[];
  recommendedNextActions?: string[];
  createdAt: string;
  citations: string[]; // citation IDs
  citationObjects?: import('./citation').CitationItem[];
  sourceIds: string[]; // source IDs
}

