import React from 'react';

export type ErrorType =
  | 'invalid_query'
  | 'retrieval_failure'
  | 'model_failure'
  | 'out_of_domain'
  | 'contradictory_evidence';

interface ErrorCardProps {
  type: ErrorType;
  title?: string;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

const ERROR_CONFIGS: Record<
  ErrorType,
  { defaultTitle: string; defaultMessage: string; badgeLabel: string; bgClass: string; textClass: string; borderClass: string }
> = {
  invalid_query: {
    defaultTitle: 'Invalid Clinical Query',
    defaultMessage: 'Please enter a clear, specific medical or clinical question.',
    badgeLabel: 'Validation Error',
    bgClass: 'bg-[var(--warn-bg)]',
    textClass: 'text-[var(--warn-text)]',
    borderClass: 'border-[var(--warn-border)]',
  },
  retrieval_failure: {
    defaultTitle: 'Retrieval Failure',
    defaultMessage: 'No relevant literature or evidence passages could be retrieved for this query.',
    badgeLabel: 'Retrieval Warning',
    bgClass: 'bg-[var(--warn-bg)]',
    textClass: 'text-[var(--warn-text)]',
    borderClass: 'border-[var(--warn-border)]',
  },
  model_failure: {
    defaultTitle: 'Model Generation Error',
    defaultMessage: 'The multi-agent pipeline encountered an unexpected model processing error.',
    badgeLabel: 'Pipeline Exception',
    bgClass: 'bg-[#2a0d12]',
    textClass: 'text-[#f87171]',
    borderClass: 'border-[#7a1c28]',
  },
  out_of_domain: {
    defaultTitle: 'Out of Medical Domain',
    defaultMessage: 'This system is strictly designed for medical evidence retrieval and clinical QA.',
    badgeLabel: 'Scope Boundary',
    bgClass: 'bg-[var(--surface-2)]',
    textClass: 'text-[var(--text)]',
    borderClass: 'border-[var(--border)]',
  },
  contradictory_evidence: {
    defaultTitle: 'Contradictory Evidence Detected',
    defaultMessage: 'Retrieved clinical sources contain directly opposing findings or conflicting recommendations.',
    badgeLabel: 'Contradiction Alert',
    bgClass: 'bg-[#2a0d12]',
    textClass: 'text-[#f87171]',
    borderClass: 'border-[#7a1c28]',
  },
};

export const ErrorCard: React.FC<ErrorCardProps> = ({
  type,
  title,
  message,
  onRetry,
  className = '',
}) => {
  const config = ERROR_CONFIGS[type];

  return (
    <div
      role="alert"
      aria-label={`${config.badgeLabel}: ${title || config.defaultTitle}`}
      className={`rounded-2xl border p-5 space-y-4 shadow-xs ${config.bgClass} ${config.borderClass} ${className}`}
    >
      <div className="flex items-start space-x-3">
        <div className="w-9 h-9 rounded-xl bg-[var(--surface-2)] text-[var(--accent)] flex items-center justify-center shrink-0 border border-[var(--border)] mt-0.5">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>

        <div className="space-y-1 flex-1">
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)]">
              {config.badgeLabel}
            </span>
          </div>
          <h4 className={`text-sm font-bold ${config.textClass} leading-snug`}>
            {title || config.defaultTitle}
          </h4>
          <p className={`text-xs ${config.textClass} opacity-90 leading-relaxed`}>
            {message || config.defaultMessage}
          </p>
        </div>
      </div>

      {onRetry && (
        <div className="pt-2 flex justify-end">
          <button
            type="button"
            onClick={onRetry}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text)] border border-[var(--border)] shadow-xs transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
};
