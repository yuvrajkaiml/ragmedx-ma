import React from 'react';

interface CitationChipProps {
  index: number;
  title?: string;
  onClick?: (index: number) => void;
  isActive?: boolean;
  className?: string;
}

export const CitationChip: React.FC<CitationChipProps> = ({
  index,
  title,
  onClick,
  isActive = false,
  className = '',
}) => {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    if (onClick) {
      onClick(index);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && onClick) {
      e.preventDefault();
      onClick(index);
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      title={title || `Citation [${index}]`}
      aria-label={`Citation reference [${index}]${title ? `: ${title}` : ''}`}
      className={`citation-chip-animated inline-flex items-center space-x-1 px-2 py-1 mx-0.5 rounded-lg text-xs font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer select-none transition-all duration-150 ${
        isActive
          ? 'bg-[var(--accent)] text-[#041214] font-bold border border-[var(--accent-bright)] ring-2 ring-[var(--accent-bright)]/40 shadow-xs scale-105'
          : 'bg-[rgba(20,184,166,0.15)] text-[var(--mint)] border border-[rgba(20,184,166,0.35)] hover:bg-[rgba(20,184,166,0.25)] hover:border-[var(--accent-bright)]'
      } ${className}`}
    >
      <span className="font-bold">[{index}]</span>
      {title && <span className="truncate max-w-[120px] font-normal opacity-90">{title}</span>}
    </button>
  );
};
