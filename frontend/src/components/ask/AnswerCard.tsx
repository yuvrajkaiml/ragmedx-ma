import React, { useState } from 'react';
import { AnswerData } from '../../types/answer';
import { StatusBadge, ConfidenceBadge } from '../common/Badge';
import { CitationChip } from './CitationChip';
import { AbstainBanner } from './AbstainBanner';
import { FeedbackControls } from '../common/FeedbackControls';
import { BookmarkButton } from '../common/BookmarkButton';

interface AnswerCardProps {
  data: AnswerData;
  onSelectCitation?: (citationIndex: number) => void;
  className?: string;
}

export const AnswerCard: React.FC<AnswerCardProps> = ({
  data,
  onSelectCitation,
  className = '',
}) => {
  const [showOriginal, setShowOriginal] = useState(false);
  const [activeCitationIndex, setActiveCitationIndex] = useState<number | null>(null);
  const [countScore, setCountScore] = useState<number>(0);
  const [typedSummary, setTypedSummary] = useState<string>('');
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  const targetScore = data.confidence.score ?? 92;

  // 1. Ease-out cubic count-up counter (1.6s = 1600ms) & typewriter summary reveal
  React.useEffect(() => {
    const startTime = performance.now();
    const duration = 1600;

    const animateCount = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      // Ease-out cubic formula: 1 - pow(1 - x, 3)
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const currentVal = Math.floor(easeProgress * targetScore);

      setCountScore(currentVal);

      if (progress < 1) {
        requestAnimationFrame(animateCount);
      } else {
        setCountScore(targetScore);
      }
    };

    const animReq = requestAnimationFrame(animateCount);

    // Typewriter summary
    let textIndex = 0;
    const summaryText = data.summary;
    const textInterval = setInterval(() => {
      textIndex += 3;
      if (textIndex >= summaryText.length) {
        setTypedSummary(summaryText);
        clearInterval(textInterval);
      } else {
        setTypedSummary(summaryText.slice(0, textIndex));
      }
    }, 20);

    return () => {
      cancelAnimationFrame(animReq);
      clearInterval(textInterval);
    };
  }, [data.summary, targetScore]);

  // 2. High-energy 140-particle Confetti burst when AnswerCard mounts
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.parentElement?.clientWidth || 700;
    canvas.height = 350;

    const colors = ['#22d3ee', '#14b8a6', '#34d399', '#0f766e', '#f59e0b', '#10b981', '#67e8f9'];
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      color: string;
      size: number;
      rotation: number;
      vRot: number;
      opacity: number;
    }> = [];

    // 140 particles as required by visual spec
    for (let i = 0; i < 140; i++) {
      particles.push({
        x: canvas.width / 2 + (Math.random() - 0.5) * 100,
        y: canvas.height / 4,
        vx: (Math.random() - 0.5) * 18,
        vy: (Math.random() - 0.85) * 16,
        color: colors[Math.floor(Math.random() * colors.length)],
        size: Math.random() * 7 + 4,
        rotation: Math.random() * Math.PI * 2,
        vRot: (Math.random() - 0.5) * 0.25,
        opacity: 1,
      });
    }

    let animId: number;
    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let alive = false;

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.38; // gravity
        p.rotation += p.vRot;
        p.opacity -= 0.012;

        if (p.opacity > 0) {
          alive = true;
          ctx.save();
          ctx.translate(p.x, p.y);
          ctx.rotate(p.rotation);
          ctx.globalAlpha = Math.max(0, p.opacity);
          ctx.fillStyle = p.color;
          ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.7);
          ctx.restore();
        }
      });

      if (alive) {
        animId = requestAnimationFrame(render);
      }
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, []);

  const handleCitationClick = (index: number) => {
    setActiveCitationIndex(index === activeCitationIndex ? null : index);
    if (onSelectCitation) {
      onSelectCitation(index);
    }
  };

  const isVerified = data.state === 'verified';
  const isRevised = data.state === 'revised';
  const isAbstained = data.state === 'abstained';

  // SVG confidence ring (r=45, dasharray=283)
  const strokeDashoffset = 283 - (283 * countScore) / 100;

  return (
    <article
      role="region"
      aria-label={`Clinical Answer (${data.state})`}
      className={`bg-[var(--surface)] rounded-2xl border border-[var(--border)] shadow-[var(--card-shadow)] p-6 space-y-6 transition-all duration-300 relative overflow-hidden ${className}`}
    >
      {/* Canvas Overlay for 140-Particle Confetti Burst */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none z-30"
      />

      {/* Card Header Bar with Spec SVG Confidence Ring (r=45, dasharray 283, #22d3ee to #0f766e) */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[var(--border)]">
        <div className="flex items-center space-x-3">
          <StatusBadge state={data.state} />
          <ConfidenceBadge level={data.confidence.level} score={countScore} />
        </div>

        <div className="flex items-center space-x-4">
          {/* Animated SVG Circular Confidence Gauge Ring (r=45, dasharray 283) */}
          <div className="relative w-18 h-18 flex items-center justify-center shrink-0">
            <svg className="w-18 h-18 transform -rotate-90" viewBox="0 0 110 110">
              <circle
                cx="55"
                cy="55"
                r="45"
                fill="transparent"
                stroke="var(--border)"
                strokeWidth="8"
              />
              <circle
                cx="55"
                cy="55"
                r="45"
                fill="transparent"
                stroke="url(#specConfidenceGradient)"
                strokeWidth="8"
                strokeDasharray="283"
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                className="transition-all duration-300 ease-out"
                style={{ filter: 'drop-shadow(0px 0px 8px rgba(34, 211, 238, 0.7))' }}
              />
              <defs>
                <linearGradient id="specConfidenceGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#22d3ee" />
                  <stop offset="100%" stopColor="#0f766e" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute inset-0 flex items-center justify-center font-bold text-xs text-[var(--text)] tracking-tighter">
              {countScore}%
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <BookmarkButton />
            <div className="text-xs text-[var(--text-faint)] flex items-center space-x-1 font-medium border-l border-[var(--border)] pl-3">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{data.createdAt || 'Just now'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Submitted Query Display */}
      <div className="space-y-1">
        <div className="text-xs font-semibold text-[var(--mint)] tracking-wide uppercase">
          Clinical Question
        </div>
        <h3 className="text-base font-semibold text-[var(--text)] leading-snug">
          {data.query}
        </h3>
      </div>

      {/* VERIFIED STATE */}
      {isVerified && (
        <div className="space-y-4">
          <div className="prose prose-invert max-w-none text-sm leading-relaxed text-[var(--text)] space-y-3">
            <p className="font-medium text-[var(--text)]">{typedSummary}</p>
            {data.body && <p className="text-[var(--text-muted)] whitespace-pre-line">{data.body}</p>}
          </div>

          {/* Citations Bar */}
          {data.citationObjects && data.citationObjects.length > 0 && (
            <div className="pt-4 border-t border-[var(--border)] space-y-2">
              <div className="text-xs font-semibold text-[var(--text-muted)] flex items-center space-x-1.5">
                <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101" />
                </svg>
                <span>Citations ({data.citationObjects.length})</span>
              </div>
              <div className="flex flex-wrap gap-1.5 items-center">
                {data.citationObjects.map((cit) => (
                  <CitationChip
                    key={cit.id}
                    index={cit.markerIndex}
                    title={cit.claimSnippet}
                    isActive={activeCitationIndex === cit.markerIndex}
                    onClick={handleCitationClick}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* REVISED STATE */}
      {isRevised && (
        <div className="space-y-4">
          {/* Critic Revision Notice Banner */}
          <div className="bg-[var(--warn-bg)] border border-[var(--warn-border)] rounded-xl p-4 space-y-2">
            <div className="flex items-center space-x-2 text-xs font-bold text-[var(--warn-text)]">
              <svg className="w-4 h-4 text-[var(--warn-text)] shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <span>Critic revised this answer for evidentiary precision</span>
            </div>
            {data.revisionNote && (
              <p className="text-xs text-[var(--warn-text)]/90 leading-relaxed pl-6">
                <strong className="font-semibold">Revision note:</strong> {data.revisionNote}
              </p>
            )}
          </div>

          {/* Revised Answer Content */}
          <div className="prose prose-invert max-w-none text-sm leading-relaxed text-[var(--text)] space-y-3">
            <p className="font-medium text-[var(--text)]">{typedSummary}</p>
            {data.body && <p className="text-[var(--text-muted)] whitespace-pre-line">{data.body}</p>}
          </div>

          {/* Optional Toggle for Original Draft View */}
          {data.originalAnswer && (
            <div className="pt-2">
              <button
                type="button"
                onClick={() => setShowOriginal(!showOriginal)}
                className="text-xs font-medium text-[var(--mint)] hover:text-[var(--accent-bright)] underline focus:outline-none flex items-center space-x-1 cursor-pointer"
              >
                <span>{showOriginal ? 'Hide original draft' : 'View original generator draft'}</span>
                <svg
                  className={`w-3.5 h-3.5 transform transition-transform duration-150 ${showOriginal ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showOriginal && (
                <div className="mt-3 p-4 bg-[var(--surface-2)] border border-[var(--border)] rounded-xl text-xs text-[var(--text-muted)] space-y-1">
                  <div className="font-semibold text-[var(--text)]">Original Unrevised Draft:</div>
                  <p className="whitespace-pre-line leading-relaxed">{data.originalAnswer}</p>
                </div>
              )}
            </div>
          )}

          {/* Citations Bar */}
          {data.citationObjects && data.citationObjects.length > 0 && (
            <div className="pt-4 border-t border-[var(--border)] space-y-2">
              <div className="text-xs font-semibold text-[var(--text-muted)] flex items-center space-x-1.5">
                <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101" />
                </svg>
                <span>Citations ({data.citationObjects.length})</span>
              </div>
              <div className="flex flex-wrap gap-1.5 items-center">
                {data.citationObjects.map((cit) => (
                  <CitationChip
                    key={cit.id}
                    index={cit.markerIndex}
                    title={cit.claimSnippet}
                    isActive={activeCitationIndex === cit.markerIndex}
                    onClick={handleCitationClick}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ABSTAINED STATE */}
      {isAbstained && (
        <AbstainBanner
          reason={data.abstentionReason}
          partialFindings={data.partialFindings}
          recommendedNextActions={data.recommendedNextActions}
        />
      )}

      {/* Card Footer Actions Bar */}
      <div className="pt-4 border-t border-[var(--border)] flex items-center justify-between text-xs">
        <div className="text-[var(--text-faint)] font-medium">Was this response helpful?</div>
        <FeedbackControls />
      </div>
    </article>
  );
};
