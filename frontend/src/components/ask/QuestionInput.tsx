import React, { useState, useEffect } from 'react';

interface QuestionInputProps {
  onSubmit: (question: string) => void;
  value?: string;
  onChange?: (value: string) => void;
  isLoading?: boolean;
}

const PLACEHOLDER_QUESTIONS = [
  'What is the recommended first-line therapy for heart failure with reduced ejection fraction?',
  'How does SGLT2 inhibitor therapy impact renal outcomes in type 2 diabetes?',
  'What are the primary contraindications for dual antiplatelet therapy following PCI?',
  'What evidence supports early statin therapy in acute coronary syndromes?',
];

export const QuestionInput: React.FC<QuestionInputProps> = ({
  onSubmit,
  value: externalValue,
  onChange: externalOnChange,
  isLoading = false,
}) => {
  const [question, setQuestion] = useState(externalValue || '');
  const [isFocused, setIsFocused] = useState(false);
  const [placeholderText, setPlaceholderText] = useState('Ask a clinical question...');
  const [ripple, setRipple] = useState(false);

  useEffect(() => {
    if (externalValue !== undefined) {
      setQuestion(externalValue);
    }
  }, [externalValue]);

  // Typewriter placeholder cycling through 4 sample questions when not focused
  useEffect(() => {
    if (isFocused || question.trim()) return;

    let qIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let timer: ReturnType<typeof setTimeout>;

    const typeCycle = () => {
      const currentFullText = PLACEHOLDER_QUESTIONS[qIndex];

      if (!isDeleting) {
        charIndex++;
        setPlaceholderText(currentFullText.slice(0, charIndex));

        if (charIndex === currentFullText.length) {
          timer = setTimeout(() => {
            isDeleting = true;
            typeCycle();
          }, 2400);
          return;
        }
      } else {
        charIndex -= 2;
        setPlaceholderText(currentFullText.slice(0, Math.max(0, charIndex)));

        if (charIndex <= 0) {
          isDeleting = false;
          qIndex = (qIndex + 1) % PLACEHOLDER_QUESTIONS.length;
        }
      }

      timer = setTimeout(typeCycle, isDeleting ? 25 : 40);
    };

    timer = setTimeout(typeCycle, 500);

    return () => clearTimeout(timer);
  }, [isFocused, question]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setQuestion(val);
    if (externalOnChange) {
      externalOnChange(val);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && !isLoading) {
      setRipple(true);
      setTimeout(() => setRipple(false), 600);
      onSubmit(question.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      {/* Animated Gradient Border Conic Container & Soft Pulsing Outer Glow */}
      <div className="input-conic-wrapper transition-all duration-300">
        <div className="flex items-center bg-[var(--surface-2)] rounded-[0.9rem] p-2 pl-4 transition-all duration-200">
          <textarea
            rows={1}
            value={question}
            onChange={handleChange}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onKeyDown={handleKeyDown}
            placeholder={isFocused ? 'Ask a clinical question...' : placeholderText}
            disabled={isLoading}
            className="w-full bg-transparent resize-none outline-none text-[var(--text)] placeholder-[var(--text-faint)] text-sm py-1.5 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!question.trim() || isLoading}
            className={`relative overflow-hidden w-11 h-11 rounded-xl bg-gradient-to-r from-[#0f766e] to-[#14b8a6] text-[#041214] font-bold shadow-[0_0_16px_rgba(20,184,166,0.35)] hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center shrink-0 transition-all duration-200 ml-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer group ${
              question.trim() ? 'animate-spec-heartbeat' : ''
            }`}
            aria-label="Submit Question"
          >
            {/* Click Ripple Effect */}
            {ripple && (
              <span className="absolute w-12 h-12 rounded-full bg-[var(--accent)]/30 animate-ping" />
            )}
            {isLoading ? (
              <svg className="w-5 h-5 animate-spin text-[#041214]" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-5 h-5 transform rotate-90 transition-transform duration-200 group-hover:rotate-[105deg] group-hover:scale-110 text-[#041214]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </form>
  );
};
