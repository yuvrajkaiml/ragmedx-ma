import React, { useEffect } from 'react';
import { HistorySessionItem } from '../../types/history';
import { StatusBadge, ConfidenceBadge } from '../common/Badge';

interface SessionDetailModalProps {
  session: HistorySessionItem | null;
  isOpen: boolean;
  onClose: () => void;
  onReopen?: (session: HistorySessionItem) => void;
}

export const SessionDetailModal: React.FC<SessionDetailModalProps> = ({
  session,
  isOpen,
  onClose,
  onReopen,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !session) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="session-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs animate-page-entrance"
    >
      <div className="bg-[var(--surface)] rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] max-w-2xl w-full p-6 space-y-6 max-h-[90vh] overflow-y-auto">
        {/* Modal Header */}
        <div className="flex items-start justify-between pb-4 border-b border-[var(--border)]">
          <div className="space-y-1 pr-4">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <StatusBadge state={session.state} />
              <ConfidenceBadge level={session.confidenceLevel} score={session.confidenceScore} />
              <span className="text-xs text-[var(--text-faint)]">{session.date}</span>
            </div>
            <h3 id="session-modal-title" className="text-lg font-bold text-[var(--text)] leading-snug">
              {session.question}
            </h3>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close modal"
            className="p-1 rounded-lg text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] focus:outline-none cursor-pointer"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Answer Summary */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-[var(--mint)] tracking-wide uppercase">
            Answer Summary
          </div>
          <p className="text-sm text-[var(--text)] leading-relaxed bg-[var(--surface-2)] p-4 rounded-xl border border-[var(--border)]">
            {session.answerSummary}
          </p>
        </div>

        {/* Trace Overview Summary */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-[var(--text)] flex items-center space-x-1.5">
            <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 00-2 2h10a2 2 0 00-2-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
            </svg>
            <span>7-Phase Multi-Agent Trace Summary</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            {[
              'Query Understanding',
              'Iterative Retrieval',
              'Re-ranking',
              'Generator',
              'Critic',
              'Confidence Calibration',
              'Abstention Gatekeeper',
            ].map((phase, idx) => (
              <div key={idx} className="p-2 rounded-lg bg-[var(--surface-2)] border border-[var(--border)] flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
                <span className="text-[11px] font-medium text-[var(--text-muted)] truncate">{phase}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sources Summary */}
        <div className="space-y-1 text-xs text-[var(--text-faint)] border-t border-[var(--border)] pt-4 flex items-center justify-between">
          <span className="font-medium text-[var(--text-muted)]">Cited Literature: {session.sourceCount} sources</span>
          <span>Session ID: {session.id}</span>
        </div>

        {/* Modal Actions */}
        <div className="pt-3 border-t border-[var(--border)] flex items-center justify-end space-x-3">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-[var(--text-muted)] hover:text-[var(--text)] bg-[var(--surface-2)] hover:bg-[var(--surface)] border border-[var(--border)] transition-all focus:outline-none cursor-pointer"
          >
            Close
          </button>
          {onReopen && (
            <button
              type="button"
              onClick={() => {
                onReopen(session);
                onClose();
              }}
              className="px-4 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-[#0f766e] to-[#14b8a6] hover:brightness-110 text-[#041214] shadow-[0_0_16px_rgba(20,184,166,0.35)] transition-all focus:outline-none cursor-pointer"
            >
              Reopen in Ask View
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
