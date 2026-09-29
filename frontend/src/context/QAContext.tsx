import React, { createContext, useContext, useState, ReactNode } from 'react';
import { AnswerData } from '../types/answer';
import { CitationItem } from '../types/citation';
import { AgentTraceStep, PipelineStage } from '../types/agent';
import { PassageItem, EvidenceSourceItem } from '../types/evidence';
import { HistorySessionItem } from '../types/history';
import {
  MOCK_MODE_ENABLED,
  MOCK_VERIFIED_ANSWER,
  MOCK_AGENT_TRACE_STEPS,
  MOCK_PASSAGES,
  MOCK_SOURCES,
  MOCK_HISTORY_SESSIONS,
} from '../mock/mockData';

interface QAContextState {
  activeQuestion: string | null;
  answerData: AnswerData | null;
  citations: CitationItem[];
  agentTraceSteps: AgentTraceStep[];
  activeStage: PipelineStage;
  subQueries: string[];
  retrievedEvidence: {
    passages: PassageItem[];
    sources: EvidenceSourceItem[];
  };
  historySessions: HistorySessionItem[];
  selectedCitationIndex: number | null;
  isEvidencePanelOpen: boolean;

  // Action Handlers
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

  // Initialize history sessions (loads mock history if MOCK_MODE_ENABLED is true)
  const [historySessions, setHistorySessions] = useState<HistorySessionItem[]>(
    MOCK_MODE_ENABLED ? MOCK_HISTORY_SESSIONS : []
  );

  const [selectedCitationIndex, setSelectedCitationIndex] = useState<number | null>(null);
  const [isEvidencePanelOpen, setIsEvidencePanelOpen] = useState<boolean>(true);

  const submitQuestion = (question: string) => {
    setActiveQuestion(question);

    if (MOCK_MODE_ENABLED) {
      // In Mock Testing Mode: Populate mock verified answer fixtures
      setAnswerData({ ...MOCK_VERIFIED_ANSWER, query: question });
      setCitations(MOCK_VERIFIED_ANSWER.citationObjects || []);
      setAgentTraceSteps(MOCK_AGENT_TRACE_STEPS);
      setRetrievedEvidence({
        passages: MOCK_PASSAGES,
        sources: MOCK_SOURCES,
      });
    } else {
      setAnswerData(null);
      setCitations([]);
      setAgentTraceSteps([]);
      setRetrievedEvidence({ passages: [], sources: [] });
    }
    setSelectedCitationIndex(null);
  };

  const clearQuestion = () => {
    setActiveQuestion(null);
    setAnswerData(null);
    setCitations([]);
    setAgentTraceSteps([]);
    setRetrievedEvidence({ passages: [], sources: [] });
    setSelectedCitationIndex(null);
  };

  const reopenSession = (session: HistorySessionItem) => {
    setActiveQuestion(session.question);
    if (session.fullAnswerData) {
      setAnswerData(session.fullAnswerData);
    } else {
      setAnswerData(null);
    }
    setCitations(session.citations || []);
    setAgentTraceSteps(session.agentTraceSteps || []);
    setRetrievedEvidence({
      passages: session.passages || [],
      sources: session.sources || [],
    });
    setSelectedCitationIndex(null);
    setIsEvidencePanelOpen(true);
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
  const context = useContext(QAContext);
  if (!context) {
    throw new Error('useQA must be used within a QAProvider');
  }
  return context;
};
