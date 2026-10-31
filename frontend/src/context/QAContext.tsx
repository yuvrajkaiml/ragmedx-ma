import React, { createContext, useContext, useState, useRef, ReactNode } from 'react';
import { AnswerData } from '../types/answer';
import { CitationItem } from '../types/citation';
import { AgentPhase, AgentTraceStep, PipelineStage } from '../types/agent';
import { PassageItem, EvidenceSourceItem } from '../types/evidence';
import { HistorySessionItem } from '../types/history';
import { askQuestion } from '../services/api';
import { getAccessToken } from '../services/authService';
import {
  uploadDocument,
  pollUntilIndexed,
  deleteDocument,
  DocumentStatus,
} from '../services/documentService';

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

  // Document upload state
  activeDocumentId: string | null;
  activeDocumentName: string | null;
  uploadStatus: DocumentStatus | null; // null = no doc, 'queued'/'processing'/'indexed'/'failed'
  uploadChunkCount: number | null;

  submitQuestion: (question: string) => void;
  clearQuestion: () => void;
  uploadAndAttachDocument: (file: File) => Promise<void>;
  detachDocument: () => void;
  setAnswerData: (data: AnswerData | null) => void;
  setCitations: (citations: CitationItem[]) => void;
  setAgentTraceSteps: (steps: AgentTraceStep[]) => void;
  setHistorySessions: React.Dispatch<React.SetStateAction<HistorySessionItem[]>>;
  setSelectedCitationIndex: (index: number | null) => void;
  setIsEvidencePanelOpen: (isOpen: boolean) => void;
  reopenSession: (session: HistorySessionItem) => void;
}

// Collapse the backend's agent phases (as adapted by api.ts adaptTraceSteps)
// into the four UI PipelineStages shown by the RetrievalProgress stepper. This
// reflects only events the backend actually emits — it never fabricates
// progress. The map is exhaustive over AgentPhase so tsc flags a missing key.
const PHASE_TO_STAGE: Record<AgentPhase, PipelineStage> = {
  'Query Understanding': 'Searching',
  'Iterative Retrieval': 'Searching',
  'Re-ranking': 'Re-ranking',
  'Generator': 'Drafting',
  'Critic': 'Reviewing',
  'Confidence Calibration': 'Reviewing',
  'Abstention Gatekeeper': 'Reviewing',
};

const QAContext = createContext<QAContextState | undefined>(undefined);

export const QAProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null);
  const [answerData, setAnswerData] = useState<AnswerData | null>(null);
  const [citations, setCitations] = useState<CitationItem[]>([]);
  const [agentTraceSteps, setAgentTraceSteps] = useState<AgentTraceStep[]>([]);
  const [activeStage, setActiveStage] = useState<PipelineStage>('Searching');
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

  // Document upload state
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const [activeDocumentName, setActiveDocumentName] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<DocumentStatus | null>(null);
  const [uploadChunkCount, setUploadChunkCount] = useState<number | null>(null);

  // Synchronous in-flight guard. A useRef mutates immediately, so a second
  // same-tick submitQuestion call sees it set — unlike isLoading (useState),
  // whose update is batched and would still read false. This closes the
  // double-submit / duplicate-pipeline-run gap at the context layer.
  const inFlightRef = useRef(false);

  const uploadAndAttachDocument = async (file: File): Promise<void> => {
    setUploadStatus('queued');
    setActiveDocumentName(file.name);
    setActiveDocumentId(null);
    setUploadChunkCount(null);
    try {
      const uploaded = await uploadDocument(file);
      setActiveDocumentId(uploaded.document_id);
      setUploadStatus('processing');
      const result = await pollUntilIndexed(
        uploaded.document_id,
        (status, chunkCount) => {
          setUploadStatus(status);
          setUploadChunkCount(chunkCount);
        },
      );
      setUploadStatus(result.status);
      setUploadChunkCount(result.chunk_count);
      if (result.status === 'failed') {
        setActiveDocumentId(null);
        setError(`Document indexing failed: ${result.error_message ?? 'unknown error'}`);
      }
    } catch (err: unknown) {
      setUploadStatus('failed');
      setActiveDocumentId(null);
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  const detachDocument = () => {
    if (activeDocumentId) {
      deleteDocument(activeDocumentId).catch(() => {/* best effort */});
    }
    setActiveDocumentId(null);
    setActiveDocumentName(null);
    setUploadStatus(null);
    setUploadChunkCount(null);
  };

  const submitQuestion = (question: string) => {
    // Synchronous guard against duplicate/rapid submits (same-tick safe).
    if (inFlightRef.current) return;

    // Lightweight client-side lower-bound mirroring the backend minimum (5
    // chars). The backend 422 remains the authority; this just avoids an
    // obvious round-trip failure and gives an instant, readable message.
    if (question.trim().length < 5) {
      setError('Please enter a question of at least 5 characters.');
      return;
    }

    // Live queries require a real account. Guest mode (and a logged-out
    // session) carries no access token, so POST /query/ask would be rejected
    // with 401. Detect that up front and show an actionable sign-in message
    // instead of letting it fall through to a misleading network error card.
    if (!getAccessToken()) {
      setActiveQuestion(question);
      setError('Guest mode can\u2019t run live queries — please sign in or create an account to ask a question.');
      return;
    }

    inFlightRef.current = true;
    setActiveQuestion(question);
    setAnswerData(null);
    setCitations([]);
    setAgentTraceSteps([]);
    setRetrievedEvidence({ passages: [], sources: [] });
    setSelectedCitationIndex(null);
    setActiveStage('Searching');
    setError(null);
    setIsLoading(true);
    setLoadingMessage('Starting pipeline…');

    askQuestion(
      question,
      activeDocumentId ?? undefined,
      // onTraceStep — append each step as it streams in
      (step: AgentTraceStep) => {
        setAgentTraceSteps((prev) => [...prev, step]);
        setActiveStage(PHASE_TO_STAGE[step.phase]);
        setLoadingMessage(`Running ${step.phase}…`);
      },
      // onStatusMessage
      (msg: string) => setLoadingMessage(msg),
    )
      .then(({ answerData: result, traceSteps, evidence }) => {
        setAnswerData(result);
        setAgentTraceSteps(traceSteps);
        // Settle the stepper on completion so all stages read as done. On
        // error the last reached stage is left as-is; the ErrorCard (FIX A)
        // communicates the failure.
        setActiveStage('Reviewing');
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
        inFlightRef.current = false;
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
    setActiveStage('Searching');
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
    setActiveStage('Searching');
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
        activeDocumentId,
        activeDocumentName,
        uploadStatus,
        uploadChunkCount,
        submitQuestion,
        clearQuestion,
        uploadAndAttachDocument,
        detachDocument,
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
