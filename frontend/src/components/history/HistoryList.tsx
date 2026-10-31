import React from 'react';
import { HistorySessionItem } from '../../types/history';
import { StatusBadge, ConfidenceBadge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';

interface HistoryListProps {
  sessions?: HistorySessionItem[];
  onSelectSession?: (session: HistorySessionItem) => void;
}

export const HistoryList: React.FC<HistoryListProps> = ({ sessions = [], onSelectSession }) => {
  if (sessions.length === 0) {
    return (
      <EmptyState
        icon={
          <svg className="w-8 h-8 text-[var(--text-faint)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        }
        title="No conversation history yet"
        description="Submitted clinical questions and multi-agent adjudicated answers will appear here when connected to the backend service."
      />
    );
  }

  return (
    <div className="space-y-3">
      {sessions.map((session) => (
        <div
          key={session.id}
          onClick={() => onSelectSession?.(session)}
          className="bg-[var(--surface-2)] p-5 rounded-2xl border border-[var(--border)] shadow-xs interactive-card cursor-pointer group select-none hover:border-[var(--border-strong)]"
        >
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <StatusBadge state={session.state} />
            <ConfidenceBadge level={session.confidenceLevel} score={session.confidenceScore} />
            <span className="text-xs text-[var(--text-faint)] flex items-center ml-2">
              <svg className="w-3.5 h-3.5 mr-1 text-[var(--text-faint)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {session.date}
            </span>
            <span className="text-xs text-[var(--text-faint)] font-medium ml-auto">
              {session.sourceCount} {session.sourceCount === 1 ? 'source' : 'sources'}
            </span>
          </div>

          <div className="flex items-start justify-between">
            <div>
              <h4 className="text-sm font-semibold text-[var(--text)] group-hover:text-[var(--mint)] transition-colors">
                {session.question}
              </h4>
              <p className="text-xs text-[var(--text-muted)] mt-1 line-clamp-2">
                {session.answerSummary}
              </p>
            </div>
            <svg
              className="w-5 h-5 text-[var(--text-faint)] group-hover:text-[var(--accent-bright)] transition-colors shrink-0 ml-4 mt-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </div>
      ))}
    </div>
  );
};
