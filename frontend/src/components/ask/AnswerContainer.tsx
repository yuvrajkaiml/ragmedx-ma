import React from 'react';
import { Disclaimer } from '../common/Disclaimer';
import { AnswerCard } from './AnswerCard';
import { RetrievalProgress } from '../agent/RetrievalProgress';
import { AgentTrace } from '../agent/AgentTrace';
import { ExampleChips } from './ExampleChips';
import { ErrorCard, ErrorType } from '../common/ErrorCard';
import { AnswerData } from '../../types/answer';
import { AgentTraceStep, PipelineStage } from '../../types/agent';

interface AnswerContainerProps {
  submittedQuestion?: string | null;
  answerData?: AnswerData | null;
  agentTraceSteps?: AgentTraceStep[];
  activeStage?: PipelineStage;
  subQueries?: string[];
  errorType?: ErrorType | null;
  onClearQuestion?: () => void;
  onSelectCitation?: (index: number) => void;
  onSelectExampleQuestion?: (question: string) => void;
}

export const AnswerContainer: React.FC<AnswerContainerProps> = ({
  submittedQuestion,
  answerData,
  agentTraceSteps = [],
  activeStage = 'Searching',
  subQueries = [],
  errorType = null,
  onClearQuestion,
  onSelectCitation,
  onSelectExampleQuestion,
}) => {
  const [typedQuestion, setTypedQuestion] = React.useState<string>('');
  const [isTyping, setIsTyping] = React.useState<boolean>(true);

  React.useEffect(() => {
    if (!submittedQuestion) return;
    setTypedQuestion('');
    setIsTyping(true);
    let index = 0;
    const interval = setInterval(() => {
      index++;
      setTypedQuestion(submittedQuestion.slice(0, index));
      if (index >= submittedQuestion.length) {
        setIsTyping(false);
        clearInterval(interval);
      }
    }, 18);
    return () => clearInterval(interval);
  }, [submittedQuestion]);

  if (submittedQuestion) {
    const isComplete = Boolean(answerData);

    return (
      <div className="space-y-6 relative perspective-[1000px]">
        {/* Card 1: User Submitted Question Box with Ripple Effect Icon & Typewriter Text */}
        <div
          className="bg-[var(--surface)] p-5 rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] flex items-start space-x-4 relative z-10 animate-card-3d"
          style={{ animationDelay: '0ms' }}
        >
          <div className="w-10 h-10 rounded-xl bg-[var(--surface-2)] text-[var(--accent-bright)] flex items-center justify-center shrink-0 border border-[var(--border)] animate-icon-ripple mt-0.5">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <div className="space-y-1 flex-1 min-w-0">
            <div className="text-xs font-semibold text-[var(--mint)] tracking-wide uppercase">
              Submitted Question
            </div>
            <p className="text-[var(--text)] font-semibold text-base leading-snug">
              {typedQuestion}
              {isTyping && <span className="inline-block w-2 h-4 bg-[var(--accent)] ml-1 translate-y-0.5 animate-caret-blink" />}
            </p>
          </div>
        </div>

        {/* Optional Error State Card */}
        {errorType && <ErrorCard type={errorType} onRetry={onClearQuestion} />}

        {/* Card 2: Retrieval & Pipeline Progress Indicator */}
        <div className="animate-card-3d" style={{ animationDelay: '200ms' }}>
          <RetrievalProgress
            activeStage={activeStage}
            subQueries={subQueries}
            isComplete={isComplete}
          />
        </div>

        {/* Card 3: 7-Phase Agent Reasoning Trace */}
        <div className="animate-card-3d" style={{ animationDelay: '400ms' }}>
          <AgentTrace steps={agentTraceSteps} defaultExpanded={false} />
        </div>

        {/* Card 4: Answer Output or Integration Waiting Box */}
        <div className="animate-card-3d" style={{ animationDelay: '600ms' }}>
          {answerData ? (
            <AnswerCard data={answerData} onSelectCitation={onSelectCitation} />
          ) : (
            <div className="bg-[var(--surface)] p-8 rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] text-center space-y-3 relative z-10">
              <div className="w-12 h-12 rounded-full bg-[var(--surface-2)] text-[var(--accent)] flex items-center justify-center mx-auto border border-[var(--border)] animate-brand-pulse">
                <svg className="w-6 h-6 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 01-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <h3 className="text-base font-semibold text-[var(--text)]">Waiting for backend response</h3>
              <p className="text-xs text-[var(--text-muted)] max-w-md mx-auto">
                The frontend UI is ready to receive real evidence retrieval, multi-agent critique, and verified answers from the backend REST/WebSocket service.
              </p>
            </div>
          )}
        </div>

        {onClearQuestion && (
          <div className="flex justify-end pt-2">
            <button
              onClick={onClearQuestion}
              className="text-xs font-medium text-[var(--mint)] hover:text-[var(--accent-bright)] underline focus:outline-none cursor-pointer"
            >
              Ask another question
            </button>
          </div>
        )}
      </div>
    );
  }

  // Default Empty State with Example Questions
  const WELCOME_TEXT = 'Welcome to RAGMedX-MA. Enter a clinical question to retrieve evidence and generate a multi-agent answer.';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 perspective-[1000px]">
      {/* Welcome Card & Example Questions */}
      <div className="lg:col-span-2 space-y-6">
        {/* Welcome Card Entrance (.1s delay) */}
        <div
          className="bg-[var(--surface)] p-6 rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] flex items-start space-x-4 animate-card-entrance-spec relative overflow-hidden"
          style={{ animationDelay: '100ms' }}
        >
          <div className="w-10 h-10 rounded-full bg-[var(--surface-2)] text-[var(--accent)] flex items-center justify-center shrink-0 border border-[var(--border)] animate-spec-heartbeat">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-[var(--text)] leading-relaxed">
              {WELCOME_TEXT}
            </p>
          </div>
        </div>

        {/* Sample Questions Block Entrance (.3s delay) */}
        <div
          className="animate-card-entrance-spec"
          style={{ animationDelay: '300ms' }}
        >
          {onSelectExampleQuestion && (
            <ExampleChips onSelectExample={onSelectExampleQuestion} />
          )}
        </div>
      </div>

      {/* Card Entrance 3 (.5s delay) - How it works sidebar */}
      <div
        className="bg-[var(--surface)] p-6 rounded-2xl border border-[var(--border)] space-y-4 shadow-[var(--card-shadow)] animate-card-entrance-spec"
        style={{ animationDelay: '500ms' }}
      >
        <div className="flex items-center space-x-2 text-[var(--text)] font-semibold text-sm">
          <svg className="w-4 h-4 text-amber-400 animate-icon-float" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span>How it works</span>
        </div>

        {/* 6 Steps sliding from right (120ms apart) + popping badge */}
        <ol className="space-y-2.5 text-xs text-[var(--text-muted)]">
          {[
            'You submit a clinical question.',
            'Retrieval fetches relevant evidence passages.',
            'A generator drafts an answer with citations.',
            'A critic checks evidentiary support.',
            'Revision fixes unsupported claims.',
            'A verified, revised, or abstained answer is returned.',
          ].map((stepText, stepIdx) => (
            <li
              key={stepIdx}
              className="flex items-start space-x-2 animate-row-slide p-1 rounded-lg transition-all"
              style={{ animationDelay: `${stepIdx * 120}ms` }}
            >
              <span className="w-4 h-4 rounded-full bg-[var(--surface-2)] text-[var(--mint)] border border-[var(--border)] font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5 animate-badge-pop shadow-xs">
                {stepIdx + 1}
              </span>
              <span>{stepText}</span>
            </li>
          ))}
        </ol>

        {/* Breathing Border Glow Yellow Disclaimer */}
        <div className="animate-yellow-breathing rounded-xl border border-[var(--warn-border)] p-3 bg-[var(--warn-bg)]">
          <Disclaimer variant="callout" />
        </div>
      </div>
    </div>
  );
};
