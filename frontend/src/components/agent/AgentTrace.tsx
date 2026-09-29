import React, { useState } from 'react';
import { AgentTraceStep, AgentPhase } from '../../types/agent';
import { AgentStepCard } from './AgentStepCard';

interface AgentTraceProps {
  steps?: AgentTraceStep[];
  className?: string;
  defaultExpanded?: boolean;
}

const DEFAULT_7_PHASES: AgentPhase[] = [
  'Query Understanding',
  'Iterative Retrieval',
  'Re-ranking',
  'Generator',
  'Critic',
  'Confidence Calibration',
  'Abstention Gatekeeper',
];

export const AgentTrace: React.FC<AgentTraceProps> = ({
  steps = [],
  className = '',
  defaultExpanded = false,
}) => {
  const [expandedStepIds, setExpandedStepIds] = useState<Record<string, boolean>>({});
  const [isTraceCollapsed, setIsTraceCollapsed] = useState<boolean>(!defaultExpanded);
  const [activeNodeIndex, setActiveNodeIndex] = useState<number>(0);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // 750ms step progression timer for orb and particle bursts
  React.useEffect(() => {
    setActiveNodeIndex(0);
    const interval = setInterval(() => {
      setActiveNodeIndex((prev) => {
        if (prev < 6) return prev + 1;
        clearInterval(interval);
        return 6;
      });
    }, 750);
    return () => clearInterval(interval);
  }, []);

  // Canvas Particle Burst when activeNodeIndex updates
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.parentElement?.clientWidth || 700;
    canvas.height = 100;

    const nodeX = (canvas.width * (7 + (activeNodeIndex / 6) * 86)) / 100;
    const nodeY = 34;

    const colors = ['#34d399', '#14b8a6', '#06b6d4', '#a7f3d0', '#5eead4'];
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      color: string;
      size: number;
      opacity: number;
    }> = [];

    for (let i = 0; i < 26; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 4.5 + 1.5;
      particles.push({
        x: nodeX,
        y: nodeY,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        color: colors[Math.floor(Math.random() * colors.length)],
        size: Math.random() * 4 + 2,
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
        p.opacity -= 0.03;

        if (p.opacity > 0) {
          alive = true;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fillStyle = p.color;
          ctx.globalAlpha = Math.max(0, p.opacity);
          ctx.fill();
        }
      });

      if (alive) {
        animId = requestAnimationFrame(render);
      }
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [activeNodeIndex]);

  // Map provided steps to the 7 defined phases
  const mappedSteps: AgentTraceStep[] = DEFAULT_7_PHASES.map((phase, idx) => {
    const existing = steps.find((s) => s.phase === phase);
    if (existing) return existing;

    return {
      id: `phase-${idx + 1}`,
      phase,
      action: `Execute ${phase.toLowerCase()} pipeline phase`,
      status: 'pending',
    };
  });

  const toggleStep = (id: string) => {
    setExpandedStepIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const expandAll = () => {
    const allExpanded: Record<string, boolean> = {};
    mappedSteps.forEach((s) => {
      allExpanded[s.id] = true;
    });
    setExpandedStepIds(allExpanded);
  };

  const collapseAll = () => {
    setExpandedStepIds({});
  };

  const orbLeftPct = 7 + (activeNodeIndex / 6) * 86;
  const railFillPct = (activeNodeIndex / 6) * 100;

  return (
    <section
      aria-label="Agent Reasoning Trace (7 Phases)"
      className={`bg-[var(--surface)] rounded-2xl border border-[var(--border)] p-5 space-y-5 shadow-[var(--card-shadow)] ${className}`}
    >
      {/* Trace Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[var(--border)]">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-xl bg-[var(--surface-2)] text-[var(--accent-bright)] flex items-center justify-center border border-[var(--border)] shrink-0">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 00-2 2h10a2 2 0 00-2-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-bold text-[var(--text)] leading-tight">
              Agent Execution Trace
            </h3>
            <p className="text-[11px] text-[var(--text-muted)]">7-phase multi-agent reasoning timeline</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={expandAll}
            className="text-xs font-medium text-[var(--mint)] hover:text-[var(--accent-bright)] focus:outline-none cursor-pointer"
          >
            Expand all
          </button>
          <span className="text-[var(--border-strong)]">|</span>
          <button
            type="button"
            onClick={collapseAll}
            className="text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text)] focus:outline-none cursor-pointer"
          >
            Collapse all
          </button>
          <span className="text-[var(--border-strong)]">|</span>
          <button
            type="button"
            onClick={() => setIsTraceCollapsed(!isTraceCollapsed)}
            className="text-xs font-semibold text-[var(--text)] bg-[var(--surface-2)] hover:bg-[var(--surface)] px-2.5 py-1 rounded-lg border border-[var(--border)] focus:outline-none cursor-pointer"
          >
            {isTraceCollapsed ? 'Show Timeline' : 'Hide Timeline'}
          </button>
        </div>
      </div>

      {/* Visual 7-Phase Timeline Graph */}
      <div className="hidden lg:block relative py-6 px-4 bg-[var(--surface-2)] rounded-xl border border-[var(--border)] overflow-hidden">
        {/* Canvas Overlay for 26 Mint/Teal Particle Bursts */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none z-20"
        />

        {/* Timeline Rail */}
        <div className="absolute top-[34px] left-[7%] right-[7%] h-1.5 bg-[var(--border)] rounded-full z-0 overflow-hidden">
          {/* Teal to Cyan Gradient Fill Rail */}
          <div
            className="h-full bg-gradient-to-r from-[var(--accent-deep)] via-[var(--accent)] to-[var(--mint)] transition-all duration-500 ease-out"
            style={{ width: `${railFillPct}%` }}
          />
        </div>

        {/* Glowing Orb: Bright glowing mint/cyan dot (#5eead4), z-10 (below node badges at z-30), fades out over 300ms on completion */}
        <div
          className={`absolute top-[31px] w-3.5 h-3.5 rounded-full bg-[#5eead4] z-10 transition-all duration-300 ease-out pointer-events-none -translate-x-1/2 ${
            activeNodeIndex >= 6 ? 'opacity-0 scale-50' : 'opacity-100'
          }`}
          style={{
            left: `${orbLeftPct}%`,
            boxShadow: '0 0 14px 4px #5eead4',
          }}
        />

        <div className="grid grid-cols-7 gap-2 relative z-30 text-center">
          {mappedSteps.map((step, idx) => {
            const isPassed = idx < activeNodeIndex;
            const isCurrent = idx === activeNodeIndex;

            return (
              <div
                key={step.id}
                onClick={() => toggleStep(step.id)}
                className="flex flex-col items-center space-y-1.5 cursor-pointer group select-none transition-all duration-300 relative z-30"
                style={{
                  transform: isPassed || isCurrent ? 'scale(1)' : 'scale(0.7)',
                  transitionTimingFunction: 'cubic-bezier(0.2, 1.8, 0.4, 1)',
                }}
              >
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs transition-all duration-300 relative z-30 ${
                    isCurrent
                      ? 'bg-[var(--surface-2)] text-[var(--mint)] border-2 border-[var(--accent-bright)] ring-4 ring-[var(--accent-bright)]/30 animate-spec-heartbeat'
                      : isPassed
                      ? 'bg-[var(--accent-deep)] text-[var(--mint)] border border-[var(--accent)] shadow-sm'
                      : 'bg-[var(--surface)] text-[var(--text-faint)] border border-[var(--border)] group-hover:border-[var(--accent)]'
                  }`}
                >
                  {idx + 1}
                </div>
                <span className={`text-[10px] leading-tight line-clamp-2 transition-colors ${
                  isCurrent
                    ? 'text-[var(--text)] font-bold'
                    : isPassed
                    ? 'text-[var(--text-muted)] font-medium group-hover:text-[var(--text)]'
                    : 'text-[var(--text-faint)]'
                }`}>
                  {step.phase}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 7 Phase Stepper List Cards */}
      {!isTraceCollapsed && (
        <div className="space-y-2.5 pt-1">
          {mappedSteps.map((step, idx) => (
            <AgentStepCard
              key={step.id}
              step={step}
              stepNumber={idx + 1}
              isExpanded={!!expandedStepIds[step.id]}
              onToggle={() => toggleStep(step.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
};
