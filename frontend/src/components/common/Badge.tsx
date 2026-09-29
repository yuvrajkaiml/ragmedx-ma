import React from 'react';
import { AnswerState, ConfidenceLevel } from '../../types/answer';

interface StatusBadgeProps {
  state: AnswerState;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ state }) => {
  switch (state) {
    case 'verified':
      return (
        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-[rgba(52,211,153,0.12)] text-[var(--success)] border border-[rgba(52,211,153,0.3)] shadow-xs animate-verified-glow-spec">
          <svg className="w-3.5 h-3.5 mr-1.5 text-[var(--success)] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
          </svg>
          Verified
        </span>
      );
    case 'revised':
      return (
        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-[var(--warn-bg)] text-[var(--warn-text)] border border-[var(--warn-border)] shadow-xs">
          <span className="w-2 h-2 mr-1.5 rounded-full bg-[var(--warn-text)] animate-pulse"></span>
          Revised
        </span>
      );
    case 'abstained':
      return (
        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-[#2a0d12] text-[#f87171] border border-[#7a1c28] shadow-xs">
          <span className="w-2 h-2 mr-1.5 rounded-full bg-[#f87171]"></span>
          Abstained
        </span>
      );
  }
};

interface ConfidenceBadgeProps {
  level: ConfidenceLevel;
  score?: number;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ level, score }) => {
  const scoreText = score !== undefined ? ` ${score}%` : '';

  switch (level) {
    case 'High':
      return (
        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-[rgba(45,212,191,0.12)] text-[var(--mint)] border border-[rgba(45,212,191,0.3)] shadow-xs">
          <span className="w-2 h-2 mr-1.5 rounded-full bg-[var(--mint)] animate-pulse"></span>
          High{scoreText}
        </span>
      );
    case 'Moderate':
      return (
        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-[var(--warn-bg)] text-[var(--warn-text)] border border-[var(--warn-border)] shadow-xs">
          <span className="w-2 h-2 mr-1.5 rounded-full bg-[var(--warn-text)]"></span>
          Moderate{scoreText}
        </span>
      );
    case 'Insufficient':
      return (
        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-[#2a0d12] text-[#f87171] border border-[#7a1c28] shadow-xs">
          <span className="w-2 h-2 mr-1.5 rounded-full bg-[#f87171]"></span>
          Insufficient{scoreText}
        </span>
      );
  }
};

interface TagBadgeProps {
  label: string;
  active?: boolean;
  onClick?: () => void;
}

export const TagBadge: React.FC<TagBadgeProps> = ({ label, active = false, onClick }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium transition-all duration-150 ease-in-out active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
        active
          ? 'bg-[var(--accent)] text-[#041214] font-bold shadow-xs'
          : 'bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--surface)] hover:text-[var(--text)] border border-[var(--border)]'
      }`}
    >
      {label}
    </button>
  );
};
