import React from 'react';

interface DisclaimerProps {
  variant?: 'footer' | 'callout';
}

export const Disclaimer: React.FC<DisclaimerProps> = ({ variant = 'footer' }) => {
  if (variant === 'callout') {
    return (
      <div className="p-4 bg-[var(--warn-bg)] border border-[var(--warn-border)] rounded-lg text-[var(--warn-text)] text-xs flex items-start space-x-2">
        <svg className="w-4 h-4 text-[var(--warn-text)] mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <span>
          Answers are grounded in retrieved medical literature for informational purposes. Always consult a licensed clinician for medical advice.
        </span>
      </div>
    );
  }

  return (
    <footer className="py-3 px-4 text-center text-xs text-[var(--text-faint)] border-t border-[var(--border)]">
      For informational purposes; not a substitute for professional medical advice.
    </footer>
  );
};
