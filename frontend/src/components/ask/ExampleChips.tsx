import React from 'react';

interface ExampleChipsProps {
  onSelectExample: (question: string) => void;
  className?: string;
}

const SAMPLE_QUESTIONS: string[] = [
  'What is the recommended first-line therapy for heart failure with reduced ejection fraction?',
  'How does SGLT2 inhibitor therapy impact renal outcomes in type 2 diabetes?',
  'What are the primary contraindications for dual antiplatelet therapy following PCI?',
  'What evidence supports early statin therapy in acute coronary syndromes?',
];

export const ExampleChips: React.FC<ExampleChipsProps> = ({
  onSelectExample,
  className = '',
}) => {
  return (
    <div className={`space-y-2.5 ${className}`}>
      <div className="flex items-center space-x-1.5 text-xs font-semibold text-[var(--text-muted)]">
        <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        <span>Sample Clinical Questions</span>
      </div>

      <div className="flex flex-wrap gap-2">
        {SAMPLE_QUESTIONS.map((q, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectExample(q)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelectExample(q);
              }
            }}
            className="text-left px-3.5 py-2.5 rounded-xl text-xs font-medium bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text)] hover:text-[var(--mint)] border border-[var(--border)] hover:border-[var(--accent)] chip-hover-spec animate-chip-slide focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer select-none"
            style={{ animationDelay: `${idx * 80}ms` }}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
};
