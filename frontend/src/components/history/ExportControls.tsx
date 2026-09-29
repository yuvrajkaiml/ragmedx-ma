import React from 'react';
import { AnswerData } from '../../types/answer';
import { EvidenceSourceItem } from '../../types/evidence';

interface ExportControlsProps {
  question?: string | null;
  answerData?: AnswerData | null;
  sources?: EvidenceSourceItem[];
  className?: string;
}

export const ExportControls: React.FC<ExportControlsProps> = ({
  question,
  answerData,
  sources = [],
  className = '',
}) => {
  const exportJSON = () => {
    const payload = {
      question: question || answerData?.query || 'Clinical Query',
      answer: answerData || null,
      sources,
      exportedAt: new Date().toISOString(),
    };
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(payload, null, 2)
    )}`;
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', jsonString);
    downloadAnchor.setAttribute('download', `ragmedx-export-${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const exportMarkdown = () => {
    const q = question || answerData?.query || 'Clinical Query';
    const state = answerData?.state?.toUpperCase() || 'UNANSWERED';
    const confidence = answerData?.confidence ? `${answerData.confidence.level} (${answerData.confidence.score}%)` : 'N/A';
    const summary = answerData?.summary || 'No summary available.';
    const body = answerData?.body || '';

    const mdContent = `# RAGMedX-MA Clinical Export

## Clinical Question
${q}

## Status & Confidence
- **State**: ${state}
- **Confidence**: ${confidence}

## Answer Summary
${summary}

${body ? `### Detailed Findings\n${body}\n` : ''}
## Sources (${sources.length})
${
  sources.length > 0
    ? sources.map((s, idx) => `${idx + 1}. **${s.title}** ${s.journal ? `(${s.journal}, ${s.year || ''})` : ''}`).join('\n')
    : 'No external sources recorded.'
}

---
*Exported from RAGMedX-MA on ${new Date().toLocaleDateString()}*
`;

    const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `ragmedx-export-${Date.now()}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportPDF = () => {
    window.print();
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <span className="text-xs font-semibold text-[var(--text-muted)] mr-1">Export:</span>

      <button
        type="button"
        onClick={exportJSON}
        className="px-2.5 py-1 rounded-lg text-xs font-medium bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text)] border border-[var(--border)] transition-colors focus:outline-none cursor-pointer"
        title="Export as JSON"
      >
        JSON
      </button>

      <button
        type="button"
        onClick={exportMarkdown}
        className="px-2.5 py-1 rounded-lg text-xs font-medium bg-[var(--surface-2)] hover:bg-[var(--surface)] text-[var(--text)] border border-[var(--border)] transition-colors focus:outline-none cursor-pointer"
        title="Export as Markdown"
      >
        Markdown
      </button>

      <button
        type="button"
        onClick={exportPDF}
        className="px-2.5 py-1 rounded-lg text-xs font-medium bg-[rgba(20,184,166,0.15)] hover:bg-[rgba(20,184,166,0.25)] text-[var(--mint)] border border-[rgba(20,184,166,0.35)] transition-colors focus:outline-none flex items-center space-x-1 cursor-pointer"
        title="Print / Save as PDF"
      >
        <svg className="w-3 h-3 text-[var(--accent-bright)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
        </svg>
        <span>PDF</span>
      </button>
    </div>
  );
};
