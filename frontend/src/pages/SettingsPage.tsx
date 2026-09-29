import React from 'react';
import { SettingsSection } from '../components/settings/SettingsSection';
import { useSettings } from '../hooks/useSettings';

export const SettingsPage: React.FC = () => {
  const { settings, updateSetting, saveSettings, resetSettings, isSaved } = useSettings();

  return (
    <div>
      <SettingsSection
        settings={settings}
        onUpdate={updateSetting}
        onSave={saveSettings}
        onReset={resetSettings}
        isSaved={isSaved}
      />
    </div>
  );
};
