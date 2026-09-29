import { AnswerState, ConfidenceLevel, AnswerData } from './answer';
import { CitationItem } from './citation';
import { AgentTraceStep } from './agent';
import { PassageItem, EvidenceSourceItem } from './evidence';

export type HistoryFilterType = 'All' | 'Verified' | 'Revised' | 'Abstained';

export interface HistorySessionItem {
  id: string;
  question: string;
  answerSummary: string;
  state: AnswerState;
  confidenceScore: number;
  confidenceLevel: ConfidenceLevel;
  date: string;
  sourceCount: number;
  fullAnswerData?: AnswerData;
  citations?: CitationItem[];
  agentTraceSteps?: AgentTraceStep[];
  passages?: PassageItem[];
  sources?: EvidenceSourceItem[];
}

