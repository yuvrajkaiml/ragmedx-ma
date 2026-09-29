import React from 'react';
import { PipelineStage } from '../../types/agent';

interface RetrievalProgressProps {
  activeStage?: PipelineStage;
  subQueries?: string[];
  isComplete?: boolean;
  className?: string;
}

const STAGES: PipelineStage[] = ['Searching', 'Re-ranking', 'Drafting', 'Reviewing'];

export const RetrievalProgress: React.FC<RetrievalProgressProps> = ({
  activeStage = 'Searching',
  subQueries = [],
  isComplete = false,
  className = '',
}) => {
  const activeIndex = STAGES.indexOf(activeStage);

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={`Retrieval pipeline progress: ${isComplete ? 'Complete' : activeStage}`}
      className={`bg-[var(--surface)] rounded-2xl border border-[var(--border)] p-5 space-y-4 shadow-[var(--card-shadow)] ${className}`}
    >
      {/* Pipeline Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="w-6 h-6 rounded-lg bg-[var(--surface-2)] text-[var(--accent-bright)] border border-[var(--border)] flex items-center justify-center shrink-0">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            </svg>
          </span>
          <span className="text-xs font-bold text-[var(--text)] tracking-wide uppercase">
            Multi-Agent Pipeline Progress
          </span>
        </div>

        {/* Live Indicator Pill */}
        <div className="flex items-center space-x-2 bg-[var(--surface-2)] border border-[var(--border)] px-3 py-1 rounded-full text-xs font-semibold text-[var(--accent-bright)] shadow-xs">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--success)] animate-dot-ping shadow-xs" />
          <span>{isComplete ? 'Pipeline Complete' : 'Running…'}</span>
        </div>
      </div>

      {/* Stepper Flow Grid with Data Flow Chevron Arrows */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
        {STAGES.map((stage, idx) => {
          const isDone = isComplete || idx < activeIndex;
          const isCurrent = !isComplete && idx === activeIndex;

          return (
            <div key={stage} className="relative flex items-center">
              <div
                className={`w-full p-3.5 rounded-xl border transition-all duration-300 flex flex-col justify-between relative overflow-hidden ${
                  isDone
                    ? 'bg-[var(--accent-deep)]/30 border-[var(--accent)] text-[var(--mint)] shadow-xs animate-stage-pop'
                    : isCurrent
                    ? 'bg-[var(--surface-2)] border-[var(--accent-bright)] ring-2 ring-[var(--accent-bright)]/40 shadow-[0_0_16px_rgba(45,212,191,0.3)] transform -translate-y-1.5 scale-[1.04] z-10'
                    : 'bg-[var(--surface-2)] border-[var(--border)] text-[var(--text-faint)]'
                }`}
              >
                {/* Rotating Conic Gradient Overlay for Running Stage */}
                {isCurrent && (
                  <div className="absolute inset-0 pointer-events-none opacity-30 animate-pipeline-conic bg-[conic-gradient(from_0deg,transparent_70%,#14b8a6)]" />
                )}

                <div className="flex items-center justify-between mb-1.5 relative z-10">
                  <span className="text-[11px] font-semibold tracking-wide">Stage {idx + 1}</span>
                  {isDone ? (
                    <svg className="w-4 h-4 text-[var(--mint)] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path className="animate-checkmark-draw-spec" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : isCurrent ? (
                    <svg className="w-4 h-4 text-[var(--accent-bright)] animate-spin shrink-0" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                  ) : (
                    <span className="w-2 h-2 rounded-full bg-[var(--border-strong)] shrink-0" />
                  )}
                </div>
                <div className={`text-xs font-bold relative z-10 ${isDone ? 'text-[var(--text-muted)]' : isCurrent ? 'text-[var(--text)]' : 'text-[var(--text-faint)]'}`}>
                  {stage}
                </div>
              </div>

              {/* Data Flow Chevron Indicator between steps */}
              {idx < STAGES.length - 1 && (
                <div className="hidden sm:flex absolute -right-2.5 top-1/2 -translate-y-1/2 z-20 w-5 h-5 rounded-full bg-[var(--surface-2)] border border-[var(--border)] shadow-xs items-center justify-center text-[var(--accent-bright)]">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Sub-Queries Section */}
      {subQueries.length > 0 && (
        <div className="pt-3 border-t border-[var(--border)] space-y-2">
          <div className="text-xs font-semibold text-[var(--text-muted)] flex items-center space-x-1.5">
            <svg className="w-3.5 h-3.5 text-[var(--accent)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span>Active Sub-Queries ({subQueries.length})</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {subQueries.map((sq, i) => (
              <span
                key={i}
                className="px-2.5 py-1 rounded-lg text-xs font-medium bg-[var(--surface-2)] text-[var(--text)] border border-[var(--border)]"
              >
                {sq}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
