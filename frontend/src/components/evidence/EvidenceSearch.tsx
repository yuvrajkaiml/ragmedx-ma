import React from 'react';

interface EvidenceSearchProps {
  query: string;
  onQueryChange: (query: string) => void;
}

export const EvidenceSearch: React.FC<EvidenceSearchProps> = ({ query, onQueryChange }) => {
  return (
    <div className="relative w-full mb-4">
      <svg
        className="w-4 h-4 text-[var(--text-faint)] absolute left-3.5 top-1/2 transform -translate-y-1/2 transition-colors duration-150"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      <input
        type="text"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder="Search sources, journals, or tags..."
        className="w-full bg-[var(--surface-2)] border border-[var(--border)] rounded-xl pl-10 pr-4 py-2.5 text-sm text-[var(--text)] placeholder-[var(--text-faint)] transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/50 focus:border-[var(--accent-bright)] shadow-xs hover:border-[var(--border-strong)]"
      />
    </div>
  );
};
