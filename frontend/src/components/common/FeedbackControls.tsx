import React, { useState } from 'react';

interface FeedbackControlsProps {
  onFeedback?: (type: 'up' | 'down' | 'flag') => void;
  className?: string;
}

export const FeedbackControls: React.FC<FeedbackControlsProps> = ({
  onFeedback,
  className = '',
}) => {
  const [feedbackState, setFeedbackState] = useState<'up' | 'down' | 'flag' | null>(null);

  const handleToggle = (type: 'up' | 'down' | 'flag') => {
    const nextState = feedbackState === type ? null : type;
    setFeedbackState(nextState);
    if (onFeedback && nextState) {
      onFeedback(nextState);
    }
  };

  return (
    <div className={`flex items-center space-x-1.5 ${className}`}>
      {/* Thumbs Up Button */}
      <button
        type="button"
        onClick={() => handleToggle('up')}
        aria-pressed={feedbackState === 'up'}
        aria-label="Helpful answer (Thumbs up)"
        title="Helpful answer"
        className={`p-1.5 rounded-lg text-xs font-medium transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
          feedbackState === 'up'
            ? 'bg-[rgba(52,211,153,0.15)] text-[var(--success)] border border-[rgba(52,211,153,0.35)] font-semibold'
            : 'bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] border border-[var(--border)]'
        }`}
      >
        <svg className="w-4 h-4" fill={feedbackState === 'up' ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2" />
        </svg>
      </button>

      {/* Thumbs Down Button */}
      <button
        type="button"
        onClick={() => handleToggle('down')}
        aria-pressed={feedbackState === 'down'}
        aria-label="Unhelpful answer (Thumbs down)"
        title="Unhelpful answer"
        className={`p-1.5 rounded-lg text-xs font-medium transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
          feedbackState === 'down'
            ? 'bg-[#2a0d12] text-[#f87171] border border-[#7a1c28] font-semibold'
            : 'bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] border border-[var(--border)]'
        }`}
      >
        <svg className="w-4 h-4" fill={feedbackState === 'down' ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018c.163 0 .326.02.485.06L17 4m-7 10v5a2 2 0 002 2h.095c.5 0 .905-.405.905-.905 0-.714.211-1.412.608-2.006L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2" />
        </svg>
      </button>

      {/* Flag Answer Button */}
      <button
        type="button"
        onClick={() => handleToggle('flag')}
        aria-pressed={feedbackState === 'flag'}
        aria-label="Flag answer for clinical review"
        title="Flag answer for review"
        className={`p-1.5 rounded-lg text-xs font-medium transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
          feedbackState === 'flag'
            ? 'bg-[var(--warn-bg)] text-[var(--warn-text)] border border-[var(--warn-border)] font-semibold'
            : 'bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] border border-[var(--border)]'
        }`}
      >
        <svg className="w-4 h-4" fill={feedbackState === 'flag' ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V9" />
        </svg>
      </button>
    </div>
  );
};
