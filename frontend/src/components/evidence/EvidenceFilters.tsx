import React from 'react';
import { TagBadge } from '../common/Badge';
import { EvidenceTag } from '../../types/evidence';

interface EvidenceFiltersProps {
  selectedTag: string;
  onSelectTag: (tag: string) => void;
}

const ALL_TAGS: (EvidenceTag | 'All tags')[] = [
  'All tags',
  'Cardiology',
  'Cohort study',
  'Combination therapy',
  'Early intervention',
  'Guideline',
  'Lipids',
  'Long-term',
  'Meta-analysis',
  'RCT',
  'Registry',
  'Risk reduction',
  'Safety',
];

export const EvidenceFilters: React.FC<EvidenceFiltersProps> = ({ selectedTag, onSelectTag }) => {
  return (
    <div className="flex flex-wrap gap-2 mb-6">
      {ALL_TAGS.map((tag) => (
        <TagBadge
          key={tag}
          label={tag}
          active={selectedTag === tag}
          onClick={() => onSelectTag(tag)}
        />
      ))}
    </div>
  );
};
