import React from 'react';
import { DatasetItem } from '../../types/evidence';

interface DatasetCardProps {
  dataset: DatasetItem;
  className?: string;
}

export const DatasetCard: React.FC<DatasetCardProps> = ({ dataset, className = '' }) => {
  return (
    <div
      className={`bg-[var(--surface-2)] p-5 rounded-2xl border border-[var(--border)] shadow-xs flex flex-col justify-between space-y-4 ${className}`}
    >
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-[var(--surface)] text-[var(--accent)] border border-[var(--border)] flex items-center justify-center font-bold text-xs shrink-0">
              {dataset.name.slice(0, 3).toUpperCase()}
            </div>
            <h4 className="text-base font-bold text-[var(--text)]">{dataset.name}</h4>
          </div>

          <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)]">
            Knowledge Corpus
          </span>
        </div>

        <p className="text-xs text-[var(--text-muted)] leading-relaxed">{dataset.description}</p>

        {dataset.coverage && (
          <div className="text-xs text-[var(--text-muted)] pt-1">
            <strong className="font-semibold text-[var(--text)]">Scope:</strong> {dataset.coverage}
          </div>
        )}
      </div>

      {(dataset.documentCount !== undefined || dataset.lastUpdated || dataset.url) && (
        <div className="pt-3 border-t border-[var(--border)] flex items-center justify-between text-xs text-[var(--text-faint)]">
          <div className="flex items-center space-x-3">
            {dataset.documentCount !== undefined && (
              <span>{dataset.documentCount} records</span>
            )}
            {dataset.lastUpdated && <span>Updated {dataset.lastUpdated}</span>}
          </div>

          {dataset.url && (
            <a
              href={dataset.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--mint)] font-medium hover:text-[var(--accent-bright)] hover:underline flex items-center"
            >
              <span>Info</span>
              <svg className="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          )}
        </div>
      )}
    </div>
  );
};
