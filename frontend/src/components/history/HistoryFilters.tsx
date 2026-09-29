import React from 'react';
import { HistoryFilterType } from '../../types/history';

interface HistoryFiltersProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  activeFilter: HistoryFilterType;
  onFilterChange: (filter: HistoryFilterType) => void;
}

export const HistoryFilters: React.FC<HistoryFiltersProps> = ({
  searchQuery,
  onSearchChange,
  activeFilter,
  onFilterChange,
}) => {
  const filterOptions: HistoryFilterType[] = ['All', 'Verified', 'Revised', 'Abstained'];

  return (
    <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between mb-6">
      {/* Search Input */}
      <div className="relative flex-1">
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
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search past questions..."
          className="w-full bg-[var(--surface-2)] border border-[var(--border)] rounded-xl pl-10 pr-4 py-2 text-sm text-[var(--text)] placeholder-[var(--text-faint)] transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/50 focus:border-[var(--accent-bright)] shadow-xs hover:border-[var(--border-strong)]"
        />
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center space-x-1 bg-[var(--surface-2)] p-1 rounded-xl border border-[var(--border)] shrink-0 select-none">
        {filterOptions.map((filter) => (
          <button
            key={filter}
            type="button"
            onClick={() => onFilterChange(filter)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ease-in-out active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
              activeFilter === filter
                ? 'bg-[var(--accent)] text-[#041214] shadow-xs font-bold'
                : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--surface)]'
            }`}
          >
            {filter}
          </button>
        ))}
      </div>
    </div>
  );
};
