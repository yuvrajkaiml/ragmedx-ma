import React, { useState, useEffect, useRef } from 'react';
import { useQA } from '../../context/QAContext';

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
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    activeDocumentName,
    uploadStatus,
    uploadChunkCount,
    uploadAndAttachDocument,
    detachDocument,
  } = useQA();

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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadAndAttachDocument(file);
      // Reset input so same file can be re-uploaded if needed
      e.target.value = '';
    }
  };

  // Status pill config — keys must cover the full DocumentStatus union
  // ('uploaded' | 'queued' | 'processing' | 'indexed' | 'failed' | 'deleted')
  // so indexing by uploadStatus type-checks. 'uploaded' is the pre-queue state
  // right after POST /documents/upload, styled as an early/muted indicator.
  const uploadPillConfig = {
    uploaded:   { label: 'Uploaded…',   color: 'text-[var(--text-muted)]',  bg: 'bg-[var(--surface-2)]',           dot: 'bg-gray-400'    },
    queued:     { label: 'Queued…',     color: 'text-[var(--text-muted)]',  bg: 'bg-[var(--surface-2)]',           dot: 'bg-gray-400'    },
    processing: { label: 'Indexing…',   color: 'text-[var(--accent)]',       bg: 'bg-[rgba(20,184,166,0.12)]',      dot: 'bg-[var(--accent)] animate-pulse' },
    indexed:    { label: uploadChunkCount ? `${uploadChunkCount} chunks` : 'Indexed', color: 'text-[#4ade80]', bg: 'bg-[rgba(74,222,128,0.1)]', dot: 'bg-[#4ade80]' },
    failed:     { label: 'Failed',      color: 'text-[#f87171]',             bg: 'bg-[rgba(248,113,113,0.1)]',      dot: 'bg-[#f87171]'   },
    deleted:    { label: 'Removed',     color: 'text-[var(--text-muted)]',   bg: 'bg-[var(--surface-2)]',           dot: 'bg-gray-400'    },
  };

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Document status pill — shown above the input when a doc is attached */}
      {uploadStatus && activeDocumentName && (() => {
        const cfg = uploadPillConfig[uploadStatus];
        return (
          <div className={`flex items-center gap-1.5 mb-1.5 px-2.5 py-1 rounded-lg text-xs font-medium w-fit ${cfg.bg} ${cfg.color} border border-[var(--border)]`}>
            <span className={`w-2 h-2 rounded-full shrink-0 ${cfg.dot}`} />
            <span className="truncate max-w-[200px]">{activeDocumentName}</span>
            <span className="opacity-70">— {cfg.label}</span>
            {/* Remove button */}
            {(uploadStatus === 'indexed' || uploadStatus === 'failed') && (
              <button
                type="button"
                onClick={detachDocument}
                className="ml-1 opacity-60 hover:opacity-100 transition-opacity"
                aria-label="Remove document"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        );
      })()}

      {/* Animated Gradient Border Conic Container & Soft Pulsing Outer Glow */}
      <div className="input-conic-wrapper transition-all duration-300">
        <div className="flex items-center bg-[var(--surface-2)] rounded-[0.9rem] p-2 pl-3 transition-all duration-200">

          {/* Paperclip / Upload Button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadStatus === 'queued' || uploadStatus === 'processing'}
            title={uploadStatus === 'indexed' ? `Attached: ${activeDocumentName}` : 'Attach a PDF to ask questions about it'}
            className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center mr-1 transition-all duration-200 focus:outline-none
              ${uploadStatus === 'indexed'
                ? 'text-[#4ade80] bg-[rgba(74,222,128,0.1)] border border-[rgba(74,222,128,0.3)]'
                : uploadStatus === 'queued' || uploadStatus === 'processing'
                ? 'text-[var(--accent)] bg-[rgba(20,184,166,0.1)] border border-[rgba(20,184,166,0.2)] cursor-wait'
                : 'text-[var(--text-faint)] hover:text-[var(--accent)] hover:bg-[rgba(20,184,166,0.08)] border border-transparent cursor-pointer'
              }`}
            aria-label="Attach PDF"
          >
            {uploadStatus === 'queued' || uploadStatus === 'processing' ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
            )}
          </button>
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
