import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

/* ──────────────────────────── helpers ──────────────────────────── */



function validateEmail(v: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
}

type Strength = 'weak' | 'fair' | 'strong';
function passwordStrength(pw: string): { label: Strength; pct: number } {
  if (pw.length < 8) return { label: 'weak', pct: 20 };
  let score = 0;
  if (/[a-z]/.test(pw)) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^a-zA-Z0-9]/.test(pw)) score++;
  if (pw.length >= 12) score++;
  if (score <= 2) return { label: 'weak', pct: 30 };
  if (score <= 3) return { label: 'fair', pct: 60 };
  return { label: 'strong', pct: 100 };
}

const strengthColor: Record<Strength, string> = {
  weak: '#f87171',
  fair: '#fbbf24',
  strong: '#34d399',
};

/* ──────────────────────────── confetti ─────────────────────────── */

function launchConfetti(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  canvas.width = canvas.parentElement?.clientWidth || 500;
  canvas.height = canvas.parentElement?.clientHeight || 600;

  const colors = ['#14b8a6', '#5eead4', '#22d3ee', '#34d399', '#fbbf24', '#a78bfa'];
  interface P { x: number; y: number; vx: number; vy: number; w: number; h: number; rot: number; rv: number; color: string; opacity: number; }
  const particles: P[] = [];

  for (let i = 0; i < 80; i++) {
    particles.push({
      x: canvas.width / 2 + (Math.random() - 0.5) * 100,
      y: canvas.height / 2,
      vx: (Math.random() - 0.5) * 14,
      vy: Math.random() * -12 - 4,
      w: Math.random() * 8 + 4,
      h: Math.random() * 5 + 2,
      rot: Math.random() * 360,
      rv: (Math.random() - 0.5) * 15,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: 1,
    });
  }

  let frame: number;
  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let alive = false;
    particles.forEach((p) => {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.35;
      p.rot += p.rv;
      p.opacity -= 0.012;
      if (p.opacity <= 0) return;
      alive = true;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate((p.rot * Math.PI) / 180);
      ctx.globalAlpha = Math.max(0, p.opacity);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    });
    if (alive) frame = requestAnimationFrame(draw);
  };
  draw();

  return () => cancelAnimationFrame(frame);
}

/* ──────────────────────────── ECG line ─────────────────────────── */

function EcgLine() {
  const ref = useRef<SVGSVGElement>(null);
  // Static SVG path animated with CSS
  return (
    <svg
      ref={ref}
      className="absolute bottom-12 left-0 w-full h-16 opacity-25 pointer-events-none"
      viewBox="0 0 600 60"
      fill="none"
      preserveAspectRatio="none"
    >
      <path
        d="M0 30 L60 30 L80 30 L90 10 L100 50 L110 5 L120 55 L130 30 L150 30 L200 30 L260 30 L280 30 L290 10 L300 50 L310 5 L320 55 L330 30 L350 30 L400 30 L460 30 L480 30 L490 10 L500 50 L510 5 L520 55 L530 30 L550 30 L600 30"
        stroke="var(--accent)"
        strokeWidth="2"
        strokeLinecap="round"
        className="animate-ecg-draw"
      />
    </svg>
  );
}

/* ─────────────────────── forgot password modal ────────────────── */

function ForgotPasswordModal({ onClose }: { onClose: () => void }) {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateEmail(email)) {
      setError('Enter a valid email address');
      return;
    }
    setError('');
    setSent(true);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={onClose}>
      <div
        className="bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-6 max-w-sm w-full shadow-2xl animate-card-entrance-spec"
        onClick={(e) => e.stopPropagation()}
      >
        {sent ? (
          <div className="text-center space-y-4 py-4">
            <div className="w-14 h-14 mx-auto rounded-full bg-[var(--accent-deep)] flex items-center justify-center">
              <svg className="w-7 h-7 text-[var(--mint)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path className="animate-checkmark-draw-spec" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="text-lg font-bold text-[var(--text)]">Check your inbox</h3>
            <p className="text-sm text-[var(--text-muted)]">
              If an account exists for <strong className="text-[var(--text)]">{email}</strong>, we've sent a password reset link.
            </p>
            <button onClick={onClose} className="mt-2 text-sm font-medium text-[var(--mint)] hover:text-[var(--accent-bright)] cursor-pointer">
              Back to login
            </button>
          </div>
        ) : (
          <>
            <h3 className="text-lg font-bold text-[var(--text)] mb-1">Reset your password</h3>
            <p className="text-sm text-[var(--text-muted)] mb-5">Enter your email and we'll send you a reset link.</p>
            <form onSubmit={handleSubmit} noValidate>
              <label htmlFor="forgot-email" className="block text-xs font-semibold text-[var(--text-muted)] mb-1.5">Email</label>
              <input
                id="forgot-email"
                type="email"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setError(''); }}
                className="w-full px-4 py-2.5 rounded-xl bg-[var(--surface-2)] border border-[var(--border)] text-[var(--text)] text-sm outline-none focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent)]/30 transition-all"
                placeholder="you@example.com"
                autoFocus
              />
              {error && <p className="text-xs text-[#f87171] mt-1.5" role="alert">{error}</p>}
              <button
                type="submit"
                className="w-full mt-4 py-2.5 rounded-xl bg-gradient-to-r from-[#0f766e] to-[#14b8a6] text-[#041214] font-bold text-sm shadow-[0_0_16px_rgba(20,184,166,0.35)] hover:brightness-110 transition-all cursor-pointer"
              >
                Send reset link
              </button>
            </form>
            <button onClick={onClose} className="mt-3 w-full text-center text-sm text-[var(--text-muted)] hover:text-[var(--text)] cursor-pointer">
              Cancel
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/* ─────────────────────── form field component ─────────────────── */

interface FieldProps {
  id: string;
  label: string;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  error?: string;
  autoFocus?: boolean;
  placeholder?: string;
  showToggle?: boolean;
}

function Field({ id, label, type = 'text', value, onChange, error, autoFocus, placeholder, showToggle }: FieldProps) {
  const [visible, setVisible] = useState(false);
  const resolvedType = showToggle ? (visible ? 'text' : 'password') : type;

  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-xs font-semibold text-[var(--text-muted)]">{label}</label>
      <div className="relative">
        <input
          id={id}
          type={resolvedType}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoFocus={autoFocus}
          placeholder={placeholder}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-err` : undefined}
          className={`w-full px-4 py-2.5 rounded-xl bg-[var(--surface-2)] border text-[var(--text)] text-sm outline-none transition-all duration-200 focus:ring-2 focus:ring-[var(--accent)]/30 ${error ? 'border-[#f87171] focus:border-[#f87171]' : 'border-[var(--border)] focus:border-[var(--accent)]'}`}
        />
        {showToggle && (
          <button
            type="button"
            tabIndex={-1}
            onClick={() => setVisible(!visible)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)] hover:text-[var(--text)] cursor-pointer"
            aria-label={visible ? 'Hide password' : 'Show password'}
          >
            {visible ? (
              <svg className="w-4.5 h-4.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L6.59 6.59m7.532 7.532l3.29 3.29M3 3l18 18" /></svg>
            ) : (
              <svg className="w-4.5 h-4.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
            )}
          </button>
        )}
      </div>
      {error && <p id={`${id}-err`} className="text-xs text-[#f87171]" role="alert" aria-live="polite">{error}</p>}
    </div>
  );
}

/* ═══════════════════════════ MAIN PAGE ═══════════════════════════ */

type Tab = 'login' | 'signup';

export const AuthPage: React.FC = () => {
  const [tab, setTab] = useState<Tab>('login');
  const navigate = useNavigate();
  const location = useLocation();
  const { signIn, signUp, continueAsGuest, user } = useAuth();
  const confettiRef = useRef<HTMLCanvasElement>(null);

  // Redirect if already authenticated
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/ask';
  useEffect(() => {
    if (user) navigate(from, { replace: true });
  }, [user, navigate, from]);

  // Sync URL to tab
  useEffect(() => {
    if (location.pathname === '/signup') setTab('signup');
    else setTab('login');
  }, [location.pathname]);

  const switchTab = (t: Tab) => {
    setTab(t);
    navigate(t === 'login' ? '/login' : '/signup', { replace: true, state: location.state });
  };

  /* ── login state ── */
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPass, setLoginPass] = useState('');
  const [remember, setRemember] = useState(false);
  const [loginErrors, setLoginErrors] = useState<Record<string, string>>({});
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginGlobalError, setLoginGlobalError] = useState('');
  const [shake, setShake] = useState(false);

  /* ── signup state ── */
  const [signupName, setSignupName] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPass, setSignupPass] = useState('');
  const [signupConfirm, setSignupConfirm] = useState('');
  const [disclaimer, setDisclaimer] = useState(false);
  const [signupErrors, setSignupErrors] = useState<Record<string, string>>({});
  const [signupLoading, setSignupLoading] = useState(false);
  const [signupGlobalError, setSignupGlobalError] = useState('');
  const [signupShake, setSignupShake] = useState(false);
  const [signupSuccess, setSignupSuccess] = useState(false);

  /* ── forgot password ── */
  const [showForgot, setShowForgot] = useState(false);

  /* ── validation ── */
  const validateLogin = (): boolean => {
    const errs: Record<string, string> = {};
    if (!loginEmail.trim()) errs.email = 'Email is required';
    else if (!validateEmail(loginEmail)) errs.email = 'Enter a valid email address';
    if (!loginPass) errs.password = 'Password is required';
    else if (loginPass.length < 8) errs.password = 'Password must be at least 8 characters';
    setLoginErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const validateSignup = (): boolean => {
    const errs: Record<string, string> = {};
    if (!signupName.trim()) errs.name = 'Full name is required';
    if (!signupEmail.trim()) errs.email = 'Email is required';
    else if (!validateEmail(signupEmail)) errs.email = 'Enter a valid email address';
    if (!signupPass) errs.password = 'Password is required';
    else if (signupPass.length < 8) errs.password = 'Password must be at least 8 characters';
    if (signupPass !== signupConfirm) errs.confirm = 'Passwords do not match';
    if (!disclaimer) errs.disclaimer = 'You must acknowledge this to continue';
    setSignupErrors(errs);
    return Object.keys(errs).length === 0;
  };

  /* ── submit handlers ── */
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginGlobalError('');
    if (!validateLogin()) {
      setShake(true);
      setTimeout(() => setShake(false), 500);
      return;
    }
    setLoginLoading(true);
    try {
      await signIn(loginEmail, loginPass);
      // navigate happens via useEffect when user is set
    } catch (err: unknown) {
      setLoginGlobalError(err instanceof Error ? err.message : 'Something went wrong');
      setShake(true);
      setTimeout(() => setShake(false), 500);
    } finally {
      setLoginLoading(false);
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setSignupGlobalError('');
    if (!validateSignup()) {
      setSignupShake(true);
      setTimeout(() => setSignupShake(false), 500);
      return;
    }
    setSignupLoading(true);
    try {
      await signUp(signupName, signupEmail, signupPass);
      setSignupSuccess(true);
      if (confettiRef.current) launchConfetti(confettiRef.current);
      // Give confetti 1.2s, then navigate
      setTimeout(() => navigate(from, { replace: true }), 1200);
    } catch (err: unknown) {
      setSignupGlobalError(err instanceof Error ? err.message : 'Something went wrong');
      setSignupShake(true);
      setTimeout(() => setSignupShake(false), 500);
    } finally {
      setSignupLoading(false);
    }
  };

  const handleGuest = useCallback(() => {
    continueAsGuest();
    navigate(from, { replace: true });
  }, [continueAsGuest, navigate, from]);

  /* ── blur validators ── */
  const blurLogin = (field: string) => {
    const errs = { ...loginErrors };
    if (field === 'email') {
      if (!loginEmail.trim()) errs.email = 'Email is required';
      else if (!validateEmail(loginEmail)) errs.email = 'Enter a valid email address';
      else delete errs.email;
    }
    if (field === 'password') {
      if (!loginPass) errs.password = 'Password is required';
      else if (loginPass.length < 8) errs.password = 'Password must be at least 8 characters';
      else delete errs.password;
    }
    setLoginErrors(errs);
  };

  const blurSignup = (field: string) => {
    const errs = { ...signupErrors };
    if (field === 'name') { if (!signupName.trim()) errs.name = 'Full name is required'; else delete errs.name; }
    if (field === 'email') { if (!signupEmail.trim()) errs.email = 'Email is required'; else if (!validateEmail(signupEmail)) errs.email = 'Enter a valid email address'; else delete errs.email; }
    if (field === 'password') { if (!signupPass) errs.password = 'Password is required'; else if (signupPass.length < 8) errs.password = 'Password must be at least 8 characters'; else delete errs.password; }
    if (field === 'confirm') { if (signupPass !== signupConfirm) errs.confirm = 'Passwords do not match'; else delete errs.confirm; }
    setSignupErrors(errs);
  };

  const strength = passwordStrength(signupPass);

  /* ══════════════════════════ RENDER ══════════════════════════ */

  return (
    <div className="flex min-h-screen bg-[var(--bg)] text-[var(--text)] relative overflow-hidden">

      {/* ──── LEFT: Brand panel (hidden on mobile) ──── */}
      <div className="hidden lg:flex lg:w-[48%] flex-col justify-between p-10 relative overflow-hidden">
        {/* Aurora blobs */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
          <div className="absolute top-[-15%] left-[-10%] w-[50vmax] h-[50vmax] rounded-full bg-[#0f766e] blur-[100px] opacity-30 animate-aurora-spec-1" />
          <div className="absolute top-[30%] right-[-20%] w-[45vmax] h-[45vmax] rounded-full bg-[#0e7490] blur-[100px] opacity-25 animate-aurora-spec-2" />
          <div className="absolute bottom-[-10%] left-[10%] w-[40vmax] h-[40vmax] rounded-full bg-[#115e59] blur-[100px] opacity-30 animate-aurora-spec-3" />
        </div>

        {/* ECG line */}
        <EcgLine />

        {/* Content */}
        <div className="relative z-10 space-y-8">
          {/* Logo */}
          <div className="flex items-center space-x-3 animate-card-entrance-spec">
            <div className="w-12 h-12 rounded-full bg-[var(--accent)] flex items-center justify-center text-[#050b0e] shadow-lg animate-spec-heartbeat">
              <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[var(--text)] tracking-tight">RAGMedX-MA</h1>
              <p className="text-xs text-[var(--text-muted)] font-medium">Medical Evidence QA</p>
            </div>
          </div>

          {/* Headline */}
          <div className="animate-card-entrance-spec" style={{ animationDelay: '200ms' }}>
            <h2 className="text-3xl md:text-4xl font-bold leading-tight tracking-tight">
              Evidence-based answers<br />
              for <span className="text-transparent bg-clip-text bg-gradient-to-r from-[var(--accent)] to-[var(--cyan)]">clinical questions</span>
            </h2>
          </div>

          {/* Bullets */}
          <ul className="space-y-4 animate-card-entrance-spec" style={{ animationDelay: '400ms' }}>
            {[
              { icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z', text: 'Multi-agent verification pipeline' },
              { icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z', text: 'Every claim backed by cited sources' },
              { icon: 'M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636', text: 'Abstains when evidence is insufficient' },
            ].map((b, i) => (
              <li key={i} className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-lg bg-[var(--accent-deep)]/40 border border-[var(--accent)]/30 flex items-center justify-center shrink-0">
                  <svg className="w-4 h-4 text-[var(--mint)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={b.icon} />
                  </svg>
                </div>
                <span className="text-sm text-[var(--text-muted)] font-medium">{b.text}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Bottom tagline */}
        <p className="relative z-10 text-xs text-[var(--text-faint)] mt-8 animate-card-entrance-spec" style={{ animationDelay: '600ms' }}>
          Smarter Evidence. Better Decisions.
        </p>
      </div>

      {/* ──── RIGHT: Form panel ──── */}
      <div className="flex-1 flex flex-col items-center justify-center px-4 py-8 sm:px-8 relative z-10">
        {/* Confetti canvas */}
        <canvas ref={confettiRef} className="absolute inset-0 pointer-events-none z-50" />

        {/* Mobile compact brand header */}
        <div className="lg:hidden flex items-center space-x-3 mb-8 animate-card-entrance-spec">
          <div className="w-10 h-10 rounded-full bg-[var(--accent)] flex items-center justify-center text-[#050b0e] animate-spec-heartbeat">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold text-[var(--text)] tracking-tight">RAGMedX-MA</h1>
            <p className="text-xs text-[var(--text-muted)]">Medical Evidence QA</p>
          </div>
        </div>

        {/* Form card */}
        <div className="w-full max-w-md">
          <div className="auth-form-card bg-[var(--surface)] rounded-2xl border border-[var(--border)] p-6 sm:p-8 shadow-[var(--card-shadow)] animate-card-entrance-spec relative overflow-hidden">

            {/* Tab toggle */}
            <div className="flex bg-[var(--surface-2)] rounded-xl p-1 mb-7 relative">
              {/* Sliding indicator */}
              <div
                className="absolute top-1 bottom-1 w-[calc(50%-4px)] rounded-lg bg-gradient-to-r from-[#0f766e] to-[#14b8a6] shadow-[0_0_12px_rgba(20,184,166,0.3)] transition-transform duration-300 ease-out"
                style={{ transform: tab === 'login' ? 'translateX(0)' : 'translateX(calc(100% + 8px))' }}
              />
              <button
                type="button"
                onClick={() => switchTab('login')}
                className={`relative z-10 flex-1 py-2 text-sm font-bold rounded-lg transition-colors duration-200 cursor-pointer ${tab === 'login' ? 'text-[#041214]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
              >
                Sign in
              </button>
              <button
                type="button"
                onClick={() => switchTab('signup')}
                className={`relative z-10 flex-1 py-2 text-sm font-bold rounded-lg transition-colors duration-200 cursor-pointer ${tab === 'signup' ? 'text-[#041214]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}`}
              >
                Create account
              </button>
            </div>

            {/* ── LOGIN FORM ── */}
            {tab === 'login' && (
              <form
                onSubmit={handleLogin}
                noValidate
                className={`space-y-5 animate-auth-fade-in ${shake ? 'animate-auth-shake' : ''}`}
              >
                {loginGlobalError && (
                  <div className="p-3 rounded-xl bg-[#2a0d12] border border-[#7a1c28] text-sm text-[#f87171] font-medium" role="alert" aria-live="assertive">
                    {loginGlobalError}
                  </div>
                )}

                <div onBlur={() => blurLogin('email')}>
                  <Field
                    id="login-email"
                    label="Email"
                    type="email"
                    value={loginEmail}
                    onChange={(v) => { setLoginEmail(v); setLoginGlobalError(''); }}
                    error={loginErrors.email}
                    autoFocus
                    placeholder="you@example.com"
                  />
                </div>

                <div onBlur={() => blurLogin('password')}>
                  <Field
                    id="login-password"
                    label="Password"
                    type="password"
                    value={loginPass}
                    onChange={(v) => { setLoginPass(v); setLoginGlobalError(''); }}
                    error={loginErrors.password}
                    showToggle
                    placeholder="••••••••"
                  />
                </div>

                <div className="flex items-center justify-between text-xs">
                  <label className="flex items-center space-x-2 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={remember}
                      onChange={(e) => setRemember(e.target.checked)}
                      className="w-4 h-4 rounded border-[var(--border)] bg-[var(--surface-2)] accent-[var(--accent)] cursor-pointer"
                    />
                    <span className="text-[var(--text-muted)] font-medium">Remember me</span>
                  </label>
                  <button
                    type="button"
                    onClick={() => setShowForgot(true)}
                    className="text-[var(--mint)] hover:text-[var(--accent-bright)] font-medium cursor-pointer"
                  >
                    Forgot password?
                  </button>
                </div>

                <button
                  type="submit"
                  disabled={loginLoading}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-[#0f766e] to-[#14b8a6] text-[#041214] font-bold text-sm shadow-[0_0_16px_rgba(20,184,166,0.35)] hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 cursor-pointer btn-spec flex items-center justify-center space-x-2"
                >
                  {loginLoading ? (
                    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                  ) : (
                    <span>Sign in</span>
                  )}
                </button>
              </form>
            )}

            {/* ── SIGNUP FORM ── */}
            {tab === 'signup' && (
              <form
                onSubmit={handleSignup}
                noValidate
                className={`space-y-4 animate-auth-fade-in ${signupShake ? 'animate-auth-shake' : ''}`}
              >
                {signupGlobalError && (
                  <div className="p-3 rounded-xl bg-[#2a0d12] border border-[#7a1c28] text-sm text-[#f87171] font-medium" role="alert" aria-live="assertive">
                    {signupGlobalError}
                  </div>
                )}

                {signupSuccess && (
                  <div className="p-3 rounded-xl bg-[var(--accent-deep)]/30 border border-[var(--accent)] text-sm text-[var(--mint)] font-medium flex items-center space-x-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path className="animate-checkmark-draw-spec" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                    </svg>
                    <span>Account created! Redirecting…</span>
                  </div>
                )}

                <div onBlur={() => blurSignup('name')}>
                  <Field
                    id="signup-name"
                    label="Full name"
                    value={signupName}
                    onChange={(v) => { setSignupName(v); setSignupGlobalError(''); }}
                    error={signupErrors.name}
                    autoFocus
                    placeholder="Dr. Jane Smith"
                  />
                </div>

                <div onBlur={() => blurSignup('email')}>
                  <Field
                    id="signup-email"
                    label="Email"
                    type="email"
                    value={signupEmail}
                    onChange={(v) => { setSignupEmail(v); setSignupGlobalError(''); }}
                    error={signupErrors.email}
                    placeholder="you@example.com"
                  />
                </div>

                <div onBlur={() => blurSignup('password')}>
                  <Field
                    id="signup-password"
                    label="Password"
                    type="password"
                    value={signupPass}
                    onChange={(v) => { setSignupPass(v); setSignupGlobalError(''); }}
                    error={signupErrors.password}
                    showToggle
                    placeholder="Minimum 8 characters"
                  />
                  {/* Password strength meter */}
                  {signupPass.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <div className="h-1.5 w-full bg-[var(--surface-2)] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500 ease-out"
                          style={{ width: `${strength.pct}%`, backgroundColor: strengthColor[strength.label] }}
                        />
                      </div>
                      <p className="text-[11px] font-medium capitalize" style={{ color: strengthColor[strength.label] }}>
                        {strength.label}
                      </p>
                    </div>
                  )}
                </div>

                <div onBlur={() => blurSignup('confirm')}>
                  <Field
                    id="signup-confirm"
                    label="Confirm password"
                    type="password"
                    value={signupConfirm}
                    onChange={(v) => { setSignupConfirm(v); setSignupGlobalError(''); }}
                    error={signupErrors.confirm}
                    showToggle
                    placeholder="Re-enter password"
                  />
                </div>

                {/* Disclaimer checkbox */}
                <div className="space-y-1">
                  <label className="flex items-start space-x-2 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={disclaimer}
                      onChange={(e) => { setDisclaimer(e.target.checked); setSignupErrors((p) => { const n = { ...p }; delete n.disclaimer; return n; }); }}
                      className="w-4 h-4 mt-0.5 rounded border-[var(--border)] bg-[var(--surface-2)] accent-[var(--accent)] cursor-pointer shrink-0"
                      aria-describedby={signupErrors.disclaimer ? 'disclaimer-err' : undefined}
                    />
                    <span className="text-xs text-[var(--text-muted)] leading-relaxed">
                      I understand that answers are informational and <strong className="text-[var(--text)]">not medical advice</strong>.
                    </span>
                  </label>
                  {signupErrors.disclaimer && <p id="disclaimer-err" className="text-xs text-[#f87171] ml-6" role="alert">{signupErrors.disclaimer}</p>}
                </div>

                <button
                  type="submit"
                  disabled={signupLoading || signupSuccess}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-[#0f766e] to-[#14b8a6] text-[#041214] font-bold text-sm shadow-[0_0_16px_rgba(20,184,166,0.35)] hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 cursor-pointer btn-spec flex items-center justify-center space-x-2"
                >
                  {signupLoading ? (
                    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                  ) : signupSuccess ? (
                    <svg className="w-5 h-5 text-[#041214]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path className="animate-checkmark-draw-spec" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <span>Create account</span>
                  )}
                </button>
              </form>
            )}

            {/* Guest link */}
            <div className="mt-6 text-center">
              <button
                type="button"
                onClick={handleGuest}
                className="text-sm font-medium text-[var(--text-faint)] hover:text-[var(--mint)] transition-colors cursor-pointer"
              >
                Continue as guest →
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Forgot password modal */}
      {showForgot && <ForgotPasswordModal onClose={() => setShowForgot(false)} />}
    </div>
  );
};
