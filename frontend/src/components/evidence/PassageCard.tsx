import React from 'react';
import { PassageItem, GradeRating } from '../../types/evidence';

interface PassageCardProps {
  passage: PassageItem;
  isSelected?: boolean;
  onSelect?: () => void;
  className?: string;
}

const GradeBadge: React.FC<{ rating: GradeRating }> = ({ rating }) => {
  switch (rating) {
    case 'HIGH':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-[rgba(52,211,153,0.12)] text-[var(--success)] border border-[rgba(52,211,153,0.3)]">
          GRADE: High Quality
        </span>
      );
    case 'MODERATE':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-[rgba(45,212,191,0.12)] text-[var(--mint)] border border-[rgba(45,212,191,0.3)]">
          GRADE: Moderate
        </span>
      );
    case 'LOW':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-[var(--warn-bg)] text-[var(--warn-text)] border border-[var(--warn-border)]">
          GRADE: Low Evidence
        </span>
      );
    case 'VERY_LOW':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-[#2a0d12] text-[#f87171] border border-[#7a1c28]">
          GRADE: Very Low
        </span>
      );
  }
};

export const PassageCard: React.FC<PassageCardProps> = ({
  passage,
  isSelected = false,
  onSelect,
  className = '',
}) => {
  return (
    <div
      onClick={onSelect}
      className={`p-4 rounded-xl border transition-all duration-200 text-xs space-y-2.5 cursor-pointer ${
        isSelected
          ? 'bg-[var(--surface-2)] border-[var(--accent-bright)] ring-2 ring-[var(--accent-bright)]/40 shadow-xs'
          : 'bg-[var(--surface-2)] border-[var(--border)] hover:border-[var(--border-strong)]'
      } ${className}`}
    >
      {/* Passage Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {passage.markerIndex !== undefined && (
            <span className="px-1.5 py-0.5 rounded bg-[var(--accent)] text-[#041214] font-bold text-[10px]">
              [{passage.markerIndex}]
            </span>
          )}
          {passage.sourceTitle && (
            <span className="font-semibold text-[var(--text)] truncate max-w-[200px]">
              {passage.sourceTitle}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-1.5">
          {passage.gradeRating && <GradeBadge rating={passage.gradeRating} />}
          {passage.relevanceScore !== undefined && (
            <span className="text-[10px] font-medium text-[var(--text-faint)]">
              {Math.round(passage.relevanceScore * 100)}% match
            </span>
          )}
        </div>
      </div>

      {/* Contradiction Warning Indicator */}
      {passage.isContradictory && (
        <div className="p-2 bg-[#2a0d12] border border-[#7a1c28] rounded-lg text-[#f87171] space-y-0.5">
          <div className="flex items-center space-x-1 font-semibold text-[11px] text-[#fca5a5]">
            <svg className="w-3.5 h-3.5 text-[#f87171] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>Contradictory Evidence Notice</span>
          </div>
          {passage.contradictionNote && (
            <p className="text-[11px] text-[#f87171]/90 leading-relaxed pl-4">{passage.contradictionNote}</p>
          )}
        </div>
      )}

      {/* Snippet Text */}
      <p className="text-[var(--text)] leading-relaxed italic bg-[var(--surface)] p-2.5 rounded-lg border border-[var(--border)]">
        "{passage.snippet}"
      </p>

      {/* Source Citation Info Footer */}
      {(passage.journal || passage.year) && (
        <div className="text-[11px] text-[var(--text-faint)] flex items-center space-x-2 pt-1">
          {passage.journal && <span>{passage.journal}</span>}
          {passage.year && <span>({passage.year})</span>}
        </div>
      )}
    </div>
  );
};
