import React from 'react';

interface AbstainBannerProps {
  reason?: string;
  partialFindings?: string[];
  recommendedNextActions?: string[];
  className?: string;
}

export const AbstainBanner: React.FC<AbstainBannerProps> = ({
  reason,
  partialFindings = [],
  recommendedNextActions = [],
  className = '',
}) => {
  return (
    <div
      role="alert"
      aria-label="Abstention Notice"
      className={`bg-[#2a0d12] border border-[#7a1c28] rounded-2xl p-5 space-y-4 shadow-xs ${className}`}
    >
      {/* Header Banner */}
      <div className="flex items-start space-x-3">
        <div className="w-9 h-9 rounded-xl bg-[#3f141c] text-[#f87171] flex items-center justify-center shrink-0 border border-[#7a1c28] mt-0.5">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <div className="space-y-1">
          <h4 className="text-sm font-bold text-[#fca5a5] leading-snug">
            Abstained: Insufficient Grounded Evidence
          </h4>
          <p className="text-xs text-[#f87171]/90 leading-relaxed">
            {reason || 'The multi-agent gatekeeper determined that available clinical evidence was insufficient or contradictory to safely form a grounded answer.'}
          </p>
        </div>
      </div>

      {/* Partial Findings Section */}
      {partialFindings.length > 0 && (
        <div className="pt-3 border-t border-[#7a1c28] space-y-2">
          <div className="flex items-center space-x-1.5 text-xs font-semibold text-[#fca5a5]">
            <svg className="w-4 h-4 text-[#f87171]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>Partial Observations & Literature Findings</span>
          </div>
          <ul className="space-y-1 pl-5 list-disc text-xs text-[#f87171]/90 leading-relaxed">
            {partialFindings.map((finding, idx) => (
              <li key={idx}>{finding}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommended Next Actions */}
      {recommendedNextActions.length > 0 && (
        <div className="pt-3 border-t border-[#7a1c28] space-y-2">
          <div className="flex items-center space-x-1.5 text-xs font-semibold text-[#fca5a5]">
            <svg className="w-4 h-4 text-[#f87171]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>Suggested Next Actions</span>
          </div>
          <ul className="space-y-1 pl-5 list-disc text-xs text-[#f87171]/90 leading-relaxed">
            {recommendedNextActions.map((action, idx) => (
              <li key={idx}>{action}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
