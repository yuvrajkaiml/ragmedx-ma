import React from 'react';
import { UserSettingsState, ModelTier } from '../../types/settings';
import { Button } from '../common/Button';

interface SettingsSectionProps {
  settings: UserSettingsState;
  onUpdate: <K extends keyof UserSettingsState>(key: K, value: UserSettingsState[K]) => void;
  onSave: () => void;
  onReset: () => void;
  isSaved?: boolean;
}

export const SettingsSection: React.FC<SettingsSectionProps> = ({
  settings,
  onUpdate,
  onSave,
  onReset,
  isSaved = false,
}) => {
  const modelTiers: { tier: ModelTier; label: string; desc: string }[] = [
    { tier: 'Fast', label: 'Fast', desc: 'Quick retrieval, lighter critique' },
    { tier: 'Balanced', label: 'Balanced', desc: 'Default multi-agent pipeline' },
    { tier: 'Thorough', label: 'Thorough', desc: 'Deeper retrieval & extended revision' },
  ];

  return (
    <div className="space-y-6">
      {/* Header Actions Bar */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-[var(--text)] tracking-tight">Settings</h2>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            Configure confidence thresholds and display preferences.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <Button variant="outline" size="md" onClick={onReset}>
            <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Reset
          </Button>
          <Button variant="primary" size="md" onClick={onSave}>
            <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
            </svg>
            {isSaved ? 'Saved!' : 'Save'}
          </Button>
        </div>
      </div>

      {/* Card 1: Confidence Thresholds */}
      <div className="bg-[var(--surface)] p-6 rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] space-y-6">
        <div>
          <h3 className="text-base font-semibold text-[var(--text)]">Confidence thresholds</h3>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            Scores at or above these values count as high / moderate.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* High threshold */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <label className="font-medium text-[var(--text)]">High threshold</label>
              <span className="font-semibold text-[var(--mint)]">{settings.highConfidenceThreshold}%</span>
            </div>
            <input
              type="range"
              min="50"
              max="95"
              step="5"
              value={settings.highConfidenceThreshold}
              onChange={(e) => onUpdate('highConfidenceThreshold', Number(e.target.value))}
              className="w-full accent-teal-400 cursor-pointer transition-all duration-150"
            />
          </div>

          {/* Moderate threshold */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <label className="font-medium text-[var(--text)]">Moderate threshold</label>
              <span className="font-semibold text-[var(--warn-text)]">{settings.moderateConfidenceThreshold}%</span>
            </div>
            <input
              type="range"
              min="30"
              max="80"
              step="5"
              value={settings.moderateConfidenceThreshold}
              onChange={(e) => onUpdate('moderateConfidenceThreshold', Number(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer transition-all duration-150"
            />
          </div>
        </div>

        {/* Live Preview Bar */}
        <div className="pt-4 border-t border-[var(--border)]">
          <div className="text-xs font-medium text-[var(--text-faint)] mb-2">Live Badge Preview</div>
          <div className="flex flex-wrap gap-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-[rgba(45,212,191,0.12)] text-[var(--mint)] border border-[rgba(45,212,191,0.3)]">
              <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-[var(--mint)]"></span>
              High 90%
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-[var(--warn-bg)] text-[var(--warn-text)] border border-[var(--warn-border)]">
              <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-[var(--warn-text)]"></span>
              Moderate 65%
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-[#2a0d12] text-[#f87171] border border-[#7a1c28]">
              <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-[#f87171]"></span>
              Insufficient 30%
            </span>
          </div>
        </div>
      </div>

      {/* Card 2: Default Model Tier */}
      <div className="bg-[var(--surface)] p-6 rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] space-y-4">
        <div>
          <h3 className="text-base font-semibold text-[var(--text)]">Default model tier</h3>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            Pipeline depth used for new questions.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {modelTiers.map(({ tier, label, desc }) => {
            const isSelected = settings.defaultModelTier === tier;
            return (
              <div
                key={tier}
                onClick={() => onUpdate('defaultModelTier', tier)}
                className={`p-4 rounded-xl border-2 transition-all duration-200 ease-in-out cursor-pointer active:scale-[0.99] select-none ${
                  isSelected
                    ? 'border-[var(--accent-bright)] bg-[var(--surface-2)] shadow-xs'
                    : 'border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--border-strong)] hover:shadow-xs'
                }`}
              >
                <div className="font-semibold text-sm text-[var(--text)] mb-1">{label}</div>
                <div className="text-xs text-[var(--text-muted)] leading-relaxed">{desc}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Card 3: Display Preferences */}
      <div className="bg-[var(--surface)] p-6 rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] space-y-5">
        <div>
          <h3 className="text-base font-semibold text-[var(--text)]">Display preferences</h3>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            How answers and supporting detail are shown.
          </p>
        </div>

        <div className="space-y-4 divide-y divide-[var(--border)]">
          {/* Toggle 1 */}
          <div className="flex items-center justify-between pt-3">
            <div>
              <div className="text-sm font-medium text-[var(--text)]">Show agent trace by default</div>
              <div className="text-xs text-[var(--text-faint)]">Expand the agent reasoning timeline when an answer loads.</div>
            </div>
            <button
              type="button"
              onClick={() => onUpdate('showAgentTraceByDefault', !settings.showAgentTraceByDefault)}
              className={`w-11 h-6 rounded-full transition-colors duration-200 ease-in-out p-0.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
                settings.showAgentTraceByDefault ? 'bg-[var(--accent)]' : 'bg-[var(--surface-2)] border border-[var(--border)]'
              }`}
            >
              <div
                className={`w-5 h-5 rounded-full bg-[var(--text)] transition-transform duration-200 ease-in-out transform ${
                  settings.showAgentTraceByDefault ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>

          {/* Toggle 2 */}
          <div className="flex items-center justify-between pt-3">
            <div>
              <div className="text-sm font-medium text-[var(--text)]">Auto-expand evidence panel</div>
              <div className="text-xs text-[var(--text-faint)]">Open cited literature panel automatically.</div>
            </div>
            <button
              type="button"
              onClick={() => onUpdate('autoExpandEvidencePanel', !settings.autoExpandEvidencePanel)}
              className={`w-11 h-6 rounded-full transition-colors duration-200 ease-in-out p-0.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
                settings.autoExpandEvidencePanel ? 'bg-[var(--accent)]' : 'bg-[var(--surface-2)] border border-[var(--border)]'
              }`}
            >
              <div
                className={`w-5 h-5 rounded-full bg-[var(--text)] transition-transform duration-200 ease-in-out transform ${
                  settings.autoExpandEvidencePanel ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>

          {/* Toggle 3 */}
          <div className="flex items-center justify-between pt-3">
            <div>
              <div className="text-sm font-medium text-[var(--text)]">High contrast text mode</div>
              <div className="text-xs text-[var(--text-faint)]">Increase visual contrast for clinical review.</div>
            </div>
            <button
              type="button"
              onClick={() => onUpdate('highContrastMode', !settings.highContrastMode)}
              className={`w-11 h-6 rounded-full transition-colors duration-200 ease-in-out p-0.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer ${
                settings.highContrastMode ? 'bg-[var(--accent)]' : 'bg-[var(--surface-2)] border border-[var(--border)]'
              }`}
            >
              <div
                className={`w-5 h-5 rounded-full bg-[var(--text)] transition-transform duration-200 ease-in-out transform ${
                  settings.highContrastMode ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
