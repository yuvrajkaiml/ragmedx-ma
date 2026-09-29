import React from 'react';
import { AgentTraceStep, AgentStatus } from '../../types/agent';

interface AgentStepCardProps {
  step: AgentTraceStep;
  stepNumber: number;
  isExpanded?: boolean;
  onToggle?: () => void;
}

const StatusIndicator: React.FC<{ status: AgentStatus }> = ({ status }) => {
  switch (status) {
    case 'completed':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[rgba(52,211,153,0.12)] text-[var(--success)] border border-[rgba(52,211,153,0.3)]">
          <svg className="w-3 h-3 mr-1 text-[var(--success)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
          </svg>
          Completed
        </span>
      );
    case 'in_progress':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--warn-bg)] text-[var(--warn-text)] border border-[var(--warn-border)]">
          <svg className="w-3 h-3 mr-1 text-[var(--warn-text)] animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          In Progress
        </span>
      );
    case 'failed':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[#2a0d12] text-[#f87171] border border-[#7a1c28]">
          <svg className="w-3 h-3 mr-1 text-[#f87171]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
          Failed
        </span>
      );
    case 'pending':
    default:
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--surface)] text-[var(--text-faint)] border border-[var(--border)]">
          <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-[var(--text-faint)]" />
          Pending
        </span>
      );
  }
};

export const AgentStepCard: React.FC<AgentStepCardProps> = ({
  step,
  stepNumber,
  isExpanded = false,
  onToggle,
}) => {
  return (
    <div className="bg-[var(--surface-2)] rounded-xl border border-[var(--border)] shadow-xs overflow-hidden transition-all duration-150">
      {/* Step Card Header / Toggle Button */}
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isExpanded}
        aria-label={`Phase ${stepNumber}: ${step.phase} - ${step.status}`}
        className="w-full p-4 flex items-center justify-between text-left hover:bg-[var(--surface)] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] select-none cursor-pointer"
      >
        <div className="flex items-center space-x-3 min-w-0">
          <span className="w-6 h-6 rounded-full bg-[var(--surface)] text-[var(--mint)] font-bold text-xs flex items-center justify-center shrink-0 border border-[var(--border)]">
            {stepNumber}
          </span>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-[var(--text)] truncate">{step.phase}</div>
            <div className="text-xs text-[var(--text-muted)] truncate">{step.action}</div>
          </div>
        </div>

        <div className="flex items-center space-x-3 shrink-0 ml-2">
          <StatusIndicator status={step.status} />
          {step.timestamp && <span className="text-[11px] text-[var(--text-faint)] hidden sm:inline">{step.timestamp}</span>}
          <svg
            className={`w-4 h-4 text-[var(--text-faint)] transform transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Collapsible Details Body */}
      {isExpanded && (
        <div className="p-4 pt-0 border-t border-[var(--border)] space-y-3 bg-[var(--surface)]/60 text-xs">
          {step.details && (
            <div className="space-y-1 pt-3">
              <span className="font-semibold text-[var(--text)]">Execution Log & Reasoning:</span>
              <p className="text-[var(--text-muted)] leading-relaxed whitespace-pre-line">{step.details}</p>
            </div>
          )}

          {step.subQueries && step.subQueries.length > 0 && (
            <div className="space-y-1.5 pt-2">
              <span className="font-semibold text-[var(--text)]">Sub-Queries Executed:</span>
              <ul className="space-y-1 pl-4 list-disc text-[var(--text-muted)]">
                {step.subQueries.map((sq, i) => (
                  <li key={i}>{sq}</li>
                ))}
              </ul>
            </div>
          )}

          {step.queryOrPayload && (
            <div className="space-y-1 pt-2">
              <span className="font-semibold text-[var(--text)]">Payload / Query Data:</span>
              <pre className="p-2.5 bg-[#050b0e] text-[var(--text)] border border-[var(--border)] rounded-lg text-[11px] overflow-x-auto font-mono">
                {step.queryOrPayload}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
