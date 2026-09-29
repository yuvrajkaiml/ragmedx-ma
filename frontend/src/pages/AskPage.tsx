import React, { useState } from 'react';
import { QuestionInput } from '../components/ask/QuestionInput';
import { AnswerContainer } from '../components/ask/AnswerContainer';
import { EvidencePanel } from '../components/evidence/EvidencePanel';
import { ExportControls } from '../components/history/ExportControls';
import { useQA } from '../context/QAContext';

export const AskPage: React.FC = () => {
  const [draftInput, setDraftInput] = useState<string>('');
  const [replayCount, setReplayCount] = useState<number>(0);

  const {
    activeQuestion,
    answerData,
    agentTraceSteps,
    activeStage,
    subQueries,
    retrievedEvidence,
    selectedCitationIndex,
    isEvidencePanelOpen,
    isLoading,
    error,
    submitQuestion,
    clearQuestion,
    setSelectedCitationIndex,
    setIsEvidencePanelOpen,
  } = useQA();

  const handleSelectCitation = (index: number) => {
    setSelectedCitationIndex(index);
    setIsEvidencePanelOpen(true);
  };

  const handleSelectExampleQuestion = (questionText: string) => {
    setDraftInput(questionText);
  };

  const handleSubmit = (questionText: string) => {
    setDraftInput('');
    submitQuestion(questionText);
  };

  return (
    <div className="space-y-6 pb-4">
      {/* Header Bar with Export Controls & Replay Run Button */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-[var(--surface-2)] text-[var(--accent)] flex items-center justify-center border border-[var(--border)] shrink-0 animate-brand-pulse">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-[var(--text)] tracking-tight">Ask</h2>
            <p className="text-xs text-[var(--text-muted)]">
              Enter a clinical question to retrieve evidence and generate a multi-agent answer.
            </p>
          </div>
        </div>

        {activeQuestion && (
          <div className="flex items-center space-x-2 sm:space-x-3">
            <ExportControls
              question={activeQuestion}
              answerData={answerData}
              sources={retrievedEvidence.sources}
            />
            <button
              type="button"
              onClick={() => setReplayCount((prev) => prev + 1)}
              className="btn-high-energy text-xs font-semibold px-3 py-1.5 rounded-xl border border-[rgba(20,184,166,0.35)] bg-[rgba(20,184,166,0.15)] hover:bg-[rgba(20,184,166,0.25)] text-[var(--mint)] flex items-center space-x-1.5 focus:outline-none cursor-pointer"
            >
              <svg className="w-3.5 h-3.5 text-[var(--accent-bright)] animate-spin-slow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>Replay run</span>
            </button>
            <button
              type="button"
              onClick={() => setIsEvidencePanelOpen(!isEvidencePanelOpen)}
              className="btn-high-energy text-xs font-semibold px-3 py-1.5 rounded-xl border border-[var(--border)] bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text)] flex items-center space-x-1.5 focus:outline-none cursor-pointer"
            >
              <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <span>{isEvidencePanelOpen ? 'Hide Evidence Panel' : 'Show Evidence Panel'}</span>
            </button>
          </div>
        )}
      </div>

      {/* Main Grid: Content Area & Evidence Panel */}
      <div className={`grid grid-cols-1 ${activeQuestion && isEvidencePanelOpen ? 'lg:grid-cols-3' : ''} gap-6`}>
        <div className={activeQuestion && isEvidencePanelOpen ? 'lg:col-span-2' : ''}>
          <AnswerContainer
            key={replayCount}
            submittedQuestion={activeQuestion}
            answerData={answerData}
            agentTraceSteps={agentTraceSteps}
            activeStage={activeStage}
            subQueries={subQueries}
            errorType={error ? 'retrieval_failure' : null}
            onClearQuestion={isLoading ? undefined : clearQuestion}
            onSelectCitation={handleSelectCitation}
            onSelectExampleQuestion={handleSelectExampleQuestion}
          />
        </div>

        {activeQuestion && isEvidencePanelOpen && (
          <div className="lg:col-span-1">
            <EvidencePanel
              passages={retrievedEvidence.passages}
              sources={retrievedEvidence.sources}
              selectedCitationIndex={selectedCitationIndex}
              isOpen={isEvidencePanelOpen}
              onClose={() => setIsEvidencePanelOpen(false)}
              onSelectCitation={handleSelectCitation}
            />
          </div>
        )}
      </div>

      {/* Bottom Input Field */}
      <div className="pt-2">
        <QuestionInput
          value={draftInput}
          onChange={setDraftInput}
          onSubmit={handleSubmit}
        />
      </div>
    </div>
  );
};
