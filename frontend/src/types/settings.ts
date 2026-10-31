export type ModelTier = 'Fast' | 'Balanced' | 'Thorough';

export interface UserSettingsState {
  highConfidenceThreshold: number; // percentage, e.g. 80
  moderateConfidenceThreshold: number; // percentage, e.g. 50
  defaultModelTier: ModelTier;
  showAgentTraceByDefault: boolean;
  autoExpandEvidencePanel: boolean;
  highContrastMode: boolean;
}
