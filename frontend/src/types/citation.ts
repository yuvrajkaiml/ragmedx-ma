export interface CitationItem {
  id: string;
  markerIndex: number; // e.g. 1 for [1]
  sourceId: string;
  claimSnippet: string;
  relevanceScore: number;
}
