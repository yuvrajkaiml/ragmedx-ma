import React, { useState } from 'react';
import { PassageItem, EvidenceSourceItem } from '../../types/evidence';
import { PassageCard } from './PassageCard';
import { SourceCard } from './SourceCard';
import { EmptyState } from '../common/EmptyState';

interface EvidencePanelProps {
  passages?: PassageItem[];
  sources?: EvidenceSourceItem[];
  selectedCitationIndex?: number | null;
  isOpen?: boolean;
  onClose?: () => void;
  onSelectCitation?: (index: number) => void;
  className?: string;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  passages = [],
  sources = [],
  selectedCitationIndex = null,
  isOpen = true,
  onClose,
  onSelectCitation,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'passages' | 'sources'>('passages');
  const [selectedPassageId, setSelectedPassageId] = useState<string | null>(null);

  if (!isOpen) return null;

  return (
    <aside
      aria-label="Evidence & Literature Panel"
      className={`bg-[var(--surface)] border border-[var(--border)] shadow-[var(--card-shadow)] rounded-2xl flex flex-col h-full overflow-hidden ${className}`}
    >
      {/* Panel Header */}
      <div className="p-4 border-b border-[var(--border)] flex items-center justify-between bg-[var(--surface-2)]">
        <div className="flex items-center space-x-2">
          <div className="w-7 h-7 rounded-lg bg-[var(--surface)] text-[var(--accent-bright)] flex items-center justify-center border border-[var(--border)] shrink-0">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-bold text-[var(--text)] leading-tight">Evidence Library</h3>
            <p className="text-[11px] text-[var(--text-muted)]">Supporting literature & passages</p>
          </div>
        </div>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Evidence Panel"
            className="p-1 rounded-lg text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface)] focus:outline-none cursor-pointer"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[var(--border)] bg-[var(--surface)] px-3 pt-2 text-xs font-semibold select-none">
        <button
          type="button"
          onClick={() => setActiveTab('passages')}
          className={`pb-2.5 px-3 border-b-2 transition-colors cursor-pointer ${
            activeTab === 'passages'
              ? 'border-[var(--accent)] text-[var(--accent-bright)] font-bold'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          }`}
        >
          Passages ({passages.length})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('sources')}
          className={`pb-2.5 px-3 border-b-2 transition-colors cursor-pointer ${
            activeTab === 'sources'
              ? 'border-[var(--accent)] text-[var(--accent-bright)] font-bold'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
          }`}
        >
          Sources ({sources.length})
        </button>
      </div>

      {/* Panel Content Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {activeTab === 'passages' ? (
          passages.length === 0 ? (
            <EmptyState
              icon={
                <svg className="w-6 h-6 text-[var(--text-faint)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              }
              title="No retrieved passages"
              description="Retrieved evidence passages will appear here when connected to the backend vector store."
            />
          ) : (
            passages.map((p) => {
              const isSelected =
                (selectedCitationIndex !== null && p.markerIndex === selectedCitationIndex) ||
                p.id === selectedPassageId;

              return (
                <PassageCard
                  key={p.id}
                  passage={p}
                  isSelected={isSelected}
                  onSelect={() => {
                    setSelectedPassageId(p.id);
                    if (p.markerIndex && onSelectCitation) {
                      onSelectCitation(p.markerIndex);
                    }
                  }}
                />
              );
            })
          )
        ) : sources.length === 0 ? (
          <EmptyState
            icon={
              <svg className="w-6 h-6 text-[var(--text-faint)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            }
            title="No evidence sources"
            description="PubMed and clinical sources cited by agents will be listed here."
          />
        ) : (
          sources.map((s) => <SourceCard key={s.id} source={s} />)
        )}
      </div>
    </aside>
  );
};
