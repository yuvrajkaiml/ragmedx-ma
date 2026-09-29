/**
 * Centralized Animation Timing and Intensity Constants
 * Tweak these parameters to adjust speed, duration, particle counts, and delays.
 */
export const ANIMATION_CONFIG = {
  // Page Background Aurora Blobs
  aurora: {
    durationBlob1: '14s',
    durationBlob2: '18s',
    durationBlob3: '22s',
    blurRadius: '70px',
    opacity: 0.55,
    sizeVmax: '40vmax',
  },

  // Sidebar Heartbeat
  heartbeat: {
    duration: '1.4s',
    shadowGlow: 'rgba(20, 184, 166, 0.6)',
  },

  // Card Entrances (Landing & Results)
  cardEntrance: {
    duration: '0.8s',
    easing: 'cubic-bezier(0.2, 1.2, 0.3, 1)',
    staggerDelays: [100, 300, 500],
  },

  // Typewriter Effects
  typewriter: {
    charIntervalMs: 22,
    placeholderCycleIntervalMs: 3500,
  },

  // Sample Question Chips
  sampleChips: {
    staggerMs: 80,
    hoverScale: 1.04,
    hoverOffsetY: '-4px',
  },

  // "How it works" Rows & Badges
  howItWorks: {
    rowStaggerMs: 120,
    mintGlowLoopDuration: '3s',
  },

  // Pipeline Stages (4)
  pipelineStage: {
    rotateOverlayDuration: '1s',
    checkmarkDrawDuration: '0.5s',
  },

  // Agent Trace (7 Nodes)
  agentTrace: {
    stepTimeMs: 750,
    particleCountPerNode: 26,
    orbShadow: '0 0 14px 4px #14b8a6',
    nodeEasing: 'cubic-bezier(0.2, 1.8, 0.4, 1)',
  },

  // Results & Confidence Ring
  resultCard: {
    ringRadius: 45,
    ringDashArray: 283,
    ringDurationMs: 1600,
    confettiParticleCount: 140,
    verifiedGlowDuration: '2.4s',
  },
};
