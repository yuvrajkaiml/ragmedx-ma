import React, { createContext, useContext, useState, ReactNode } from 'react';
import { AnswerData } from '../types/answer';
import { CitationItem } from '../types/citation';
import { AgentTraceStep, PipelineStage } from '../types/agent';
import { PassageItem, EvidenceSourceItem } from '../types/evidence';
import { HistorySessionItem } from '../types/history';
import { askQuestion } from '../services/api';

interface QAContextState {
  activeQuestion: string | null;
  answerData: AnswerData | null;
  citations: CitationItem[];
  agentTraceSteps: AgentTraceStep[];
  activeStage: PipelineStage;
  subQueries: string[];
  retrievedEvidence: { passages: PassageItem[]; sources: EvidenceSourceItem[] };
  historySessions: HistorySessionItem[];
  selectedCitationIndex: number | null;
  isEvidencePanelOpen: boolean;
  isLoading: boolean;
  loadingMessage: string;
  error: string | null;

  submitQuestion: (question: string) => void;
  clearQuestion: () => void;
  setAnswerData: (data: AnswerData | null) => void;
  setCitations: (citations: CitationItem[]) => void;
  setAgentTraceSteps: (steps: AgentTraceStep[]) => void;
  setHistorySessions: React.Dispatch<React.SetStateAction<HistorySessionItem[]>>;
  setSelectedCitationIndex: (index: number | null) => void;
  setIsEvidencePanelOpen: (isOpen: boolean) => void;
  reopenSession: (session: HistorySessionItem) => void;
}

const QAContext = createContext<QAContextState | undefined>(undefined);

export const QAProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null);
  const [answerData, setAnswerData] = useState<AnswerData | null>(null);
  const [citations, setCitations] = useState<CitationItem[]>([]);
  const [agentTraceSteps, setAgentTraceSteps] = useState<AgentTraceStep[]>([]);
  const [activeStage] = useState<PipelineStage>('Searching');
  const [subQueries] = useState<string[]>([]);
  const [retrievedEvidence, setRetrievedEvidence] = useState<{
    passages: PassageItem[];
    sources: EvidenceSourceItem[];
  }>({ passages: [], sources: [] });
  const [historySessions, setHistorySessions] = useState<HistorySessionItem[]>([]);
  const [selectedCitationIndex, setSelectedCitationIndex] = useState<number | null>(null);
  const [isEvidencePanelOpen, setIsEvidencePanelOpen] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('Processing…');
  const [error, setError] = useState<string | null>(null);

  const submitQuestion = (question: string) => {
    setActiveQuestion(question);
    setAnswerData(null);
    setCitations([]);
    setAgentTraceSteps([]);
    setRetrievedEvidence({ passages: [], sources: [] });
    setSelectedCitationIndex(null);
    setError(null);
    setIsLoading(true);
    setLoadingMessage('Starting pipeline…');

    askQuestion(
      question,
      // onTraceStep — append each step as it streams in
      (step: AgentTraceStep) => {
        setAgentTraceSteps((prev) => [...prev, step]);
        setLoadingMessage(`Running ${step.phase}…`);
      },
      // onStatusMessage
      (msg: string) => setLoadingMessage(msg),
    )
      .then(({ answerData: result, traceSteps, evidence }) => {
        setAnswerData(result);
        setAgentTraceSteps(traceSteps);
        setRetrievedEvidence(evidence);
        setCitations(result.citationObjects ?? []);
        // Save to history
        const session: HistorySessionItem = {
          id: result.id,
          question,
          answerSummary: result.summary?.slice(0, 120) ?? '',
          state: result.state,
          confidenceScore: result.confidence.score,
          confidenceLevel: result.confidence.level,
          date: new Date().toLocaleString(),
          sourceCount: evidence.sources.length,
          citations: result.citationObjects ?? [],
          agentTraceSteps: traceSteps,
          passages: evidence.passages,
          sources: evidence.sources,
          fullAnswerData: result,
        };
        setHistorySessions((prev) => [session, ...prev]);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'An unexpected error occurred';
        setError(msg);
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  const clearQuestion = () => {
    setActiveQuestion(null);
    setAnswerData(null);
    setCitations([]);
    setAgentTraceSteps([]);
    setRetrievedEvidence({ passages: [], sources: [] });
    setSelectedCitationIndex(null);
    setError(null);
    setIsLoading(false);
  };

  const reopenSession = (session: HistorySessionItem) => {
    setActiveQuestion(session.question);
    setAnswerData(session.fullAnswerData ?? null);
    setCitations(session.citations ?? []);
    setAgentTraceSteps(session.agentTraceSteps ?? []);
    setRetrievedEvidence({
      passages: session.passages ?? [],
      sources: session.sources ?? [],
    });
    setSelectedCitationIndex(null);
    setIsEvidencePanelOpen(true);
    setError(null);
  };

  return (
    <QAContext.Provider
      value={{
        activeQuestion,
        answerData,
        citations,
        agentTraceSteps,
        activeStage,
        subQueries,
        retrievedEvidence,
        historySessions,
        selectedCitationIndex,
        isEvidencePanelOpen,
        isLoading,
        loadingMessage,
        error,
        submitQuestion,
        clearQuestion,
        setAnswerData,
        setCitations,
        setAgentTraceSteps,
        setHistorySessions,
        setSelectedCitationIndex,
        setIsEvidencePanelOpen,
        reopenSession,
      }}
    >
      {children}
    </QAContext.Provider>
  );
};

export const useQA = (): QAContextState => {
  const ctx = useContext(QAContext);
  if (!ctx) throw new Error('useQA must be used within a QAProvider');
  return ctx;
};
