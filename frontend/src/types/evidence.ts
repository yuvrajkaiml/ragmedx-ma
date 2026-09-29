export type EvidenceTag = 
  | 'Cardiology'
  | 'Cohort study'
  | 'Combination therapy'
  | 'Early intervention'
  | 'Guideline'
  | 'Lipids'
  | 'Long-term'
  | 'Meta-analysis'
  | 'RCT'
  | 'Registry'
  | 'Risk reduction'
  | 'Safety';

export type GradeRating = 'HIGH' | 'MODERATE' | 'LOW' | 'VERY_LOW';

export interface EvidenceSourceItem {
  id: string;
  title: string;
  journal?: string;
  year?: number;
  pmid?: string;
  doi?: string;
  snippet?: string;
  relevanceScore?: number;
  citationCount?: number;
  tag?: EvidenceTag;
  url?: string;
}

export interface PassageItem {
  id: string;
  markerIndex?: number; // e.g. 1 for [1]
  sourceId?: string;
  sourceTitle?: string;
  journal?: string;
  year?: number;
  snippet: string;
  gradeRating?: GradeRating;
  relevanceScore?: number;
  isContradictory?: boolean;
  contradictionNote?: string;
}

export interface DatasetItem {
  id: string;
  name: 'PubMed' | 'PMC' | 'MedQA' | 'BioASQ' | string;
  description: string;
  coverage?: string;
  documentCount?: number | string;
  lastUpdated?: string;
  url?: string;
}

