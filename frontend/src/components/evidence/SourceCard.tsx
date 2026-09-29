import React from 'react';
import { EvidenceSourceItem } from '../../types/evidence';

interface SourceCardProps {
  source: EvidenceSourceItem;
  isSelected?: boolean;
  onSelect?: () => void;
}

export const SourceCard: React.FC<SourceCardProps> = ({
  source,
  isSelected = false,
  onSelect,
}) => {
  return (
    <div
      onClick={onSelect}
      className={`bg-[var(--surface-2)] p-5 rounded-2xl border transition-all duration-200 shadow-xs flex flex-col justify-between space-y-4 cursor-pointer ${
        isSelected
          ? 'border-[var(--accent-bright)] ring-2 ring-[var(--accent-bright)]/30 shadow-md'
          : 'border-[var(--border)] hover:border-[var(--border-strong)] interactive-card'
      }`}
    >
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-[var(--surface)] text-[var(--accent)] border border-[var(--border)] flex items-center justify-center shrink-0">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            {source.relevanceScore !== undefined && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-[rgba(45,212,191,0.12)] text-[var(--mint)] border border-[rgba(45,212,191,0.3)]">
                {Math.round(source.relevanceScore * 100)}% relevant
              </span>
            )}
          </div>

          {source.tag && (
            <span className="px-2.5 py-0.5 rounded-md text-xs font-medium bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)]">
              {source.tag}
            </span>
          )}
        </div>

        <h4 className="text-sm font-semibold text-[var(--text)] leading-snug mb-1.5 hover:text-[var(--mint)] transition-colors">
          {source.title}
        </h4>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[var(--text-faint)] mb-2">
          {source.journal && <span className="font-medium text-[var(--text-muted)]">{source.journal}</span>}
          {source.year && (
            <span className="flex items-center">
              <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              {source.year}
            </span>
          )}
          {source.doi && <span className="font-mono text-[11px] text-[var(--text-faint)]">DOI: {source.doi}</span>}
          {source.pmid && <span className="font-mono text-[11px] text-[var(--text-faint)]">PMID: {source.pmid}</span>}
        </div>

        {source.snippet && (
          <p className="text-xs text-[var(--text-muted)] leading-relaxed line-clamp-3">
            {source.snippet}
          </p>
        )}
      </div>

      <div className="pt-3 border-t border-[var(--border)] flex items-center justify-between text-xs text-[var(--text-faint)]">
        {source.url ? (
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="inline-flex items-center text-[var(--mint)] font-medium hover:text-[var(--accent-bright)] transition-colors"
          >
            <svg className="w-3.5 h-3.5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            View source link
          </a>
        ) : (
          <span className="inline-flex items-center text-[var(--text-faint)]">
            <svg className="w-3.5 h-3.5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101" />
            </svg>
            Internal Reference
          </span>
        )}

        {source.citationCount !== undefined && (
          <span className="text-[var(--text-faint)] font-medium">Cited {source.citationCount}x</span>
        )}
      </div>
    </div>
  );
};
