import { useState, useCallback } from 'react';
import { UserSettingsState } from '../types/settings';

const DEFAULT_SETTINGS: UserSettingsState = {
  highConfidenceThreshold: 80,
  moderateConfidenceThreshold: 50,
  defaultModelTier: 'Balanced',
  showAgentTraceByDefault: false,
  autoExpandEvidencePanel: false,
  highContrastMode: false,
};

export function useSettings() {
  const [settings, setSettings] = useState<UserSettingsState>(DEFAULT_SETTINGS);
  const [isSaved, setIsSaved] = useState<boolean>(false);

  const updateSetting = useCallback(<K extends keyof UserSettingsState>(key: K, value: UserSettingsState[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setIsSaved(false);
  }, []);

  const saveSettings = useCallback(() => {
    // In local React state mode, confirm save status
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 2500);
  }, []);

  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
    setIsSaved(false);
  }, []);

  return {
    settings,
    updateSetting,
    saveSettings,
    resetSettings,
    isSaved,
  };
}
