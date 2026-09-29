import React, { useState } from 'react';
import { EvidenceSearch } from '../components/evidence/EvidenceSearch';
import { EvidenceFilters } from '../components/evidence/EvidenceFilters';
import { SourceCard } from '../components/evidence/SourceCard';
import { DatasetCard } from '../components/evidence/DatasetCard';
import { EmptyState } from '../components/common/EmptyState';
import { EvidenceSourceItem, DatasetItem } from '../types/evidence';
import { MOCK_MODE_ENABLED, MOCK_SOURCES, MOCK_DATASETS } from '../mock/mockData';

const DEFAULT_DATASETS: DatasetItem[] = [
  {
    id: 'dataset-pubmed',
    name: 'PubMed',
    description: 'MEDLINE abstracts and biomedical literature index from NLM.',
    coverage: 'Biomedical & Life Sciences Abstracts',
  },
  {
    id: 'dataset-pmc',
    name: 'PMC',
    description: 'PubMed Central full-text open-access archive.',
    coverage: 'Full-text Peer-reviewed Clinical Papers',
  },
  {
    id: 'dataset-medqa',
    name: 'MedQA',
    description: 'USMLE-style multi-choice clinical reasoning dataset.',
    coverage: 'Clinical Diagnosis & Medical Board Benchmarks',
  },
  {
    id: 'dataset-bioasq',
    name: 'BioASQ',
    description: 'Large-scale biomedical semantic indexing and question answering benchmark.',
    coverage: 'Biomedical Question Answering & Fact Retrieval',
  },
];

export const EvidencePage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<string>('All tags');

  const datasets = MOCK_MODE_ENABLED ? MOCK_DATASETS : DEFAULT_DATASETS;
  const sources: EvidenceSourceItem[] = MOCK_MODE_ENABLED ? MOCK_SOURCES : [];

  const filteredSources = sources.filter((source) => {
    const matchesQuery =
      source.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      source.journal?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      source.snippet?.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesTag = selectedTag === 'All tags' || source.tag === selectedTag;

    return matchesQuery && matchesTag;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-xl bg-[var(--surface-2)] text-[var(--accent)] flex items-center justify-center border border-[var(--border)] shrink-0">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <div>
          <h2 className="text-2xl font-bold text-[var(--text)] tracking-tight">Evidence Library</h2>
          <p className="text-xs text-[var(--text-muted)]">
            Knowledge corpora and cited literature references.
          </p>
        </div>
      </div>

      {/* Dataset Transparency Section */}
      <div className="space-y-3">
        <h3 className="text-sm font-bold text-[var(--text)] tracking-tight flex items-center space-x-2">
          <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
          </svg>
          <span>Integrated Knowledge Datasets</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {datasets.map((dataset) => (
            <DatasetCard key={dataset.id} dataset={dataset} />
          ))}
        </div>
      </div>

      {/* Search & Filter Section */}
      <div className="space-y-4">
        <h3 className="text-sm font-bold text-[var(--text)] tracking-tight">
          Cited Literature Sources
        </h3>

        {/* Search Input */}
        <EvidenceSearch query={searchQuery} onQueryChange={setSearchQuery} />

        {/* Tag Filters */}
        <EvidenceFilters selectedTag={selectedTag} onSelectTag={setSelectedTag} />

        {/* Grid or Empty State */}
        {filteredSources.length === 0 ? (
          <EmptyState
            icon={
              <svg className="w-8 h-8 text-[var(--text-faint)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 01-2-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            }
            title="No sources in evidence library"
            description="Retrieved medical literature, PubMed articles, and clinical guidelines will appear here when connected to the backend knowledge base."
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {filteredSources.map((source) => (
              <SourceCard key={source.id} source={source} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
