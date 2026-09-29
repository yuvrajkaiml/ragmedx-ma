/**
 * ============================================================================
 *  RAGMedX-MA  —  Authentication Service  (MOCK / DEMO-ONLY)
 * ============================================================================
 *
 *  ⚠️  DEMO-ONLY: This module simulates authentication entirely on the client
 *  using localStorage.  Passwords are hashed with SHA-256 via SubtleCrypto,
 *  but this is NOT a substitute for a real server-side auth system.
 *
 *  ── REPLACING WITH REAL API CALLS ──────────────────────────────────────────
 *
 *  When connecting to a real backend, replace the body of these functions:
 *
 *  1. signIn(email, password)
 *     → POST /api/auth/login  { email, password }
 *     ← { user: { id, name, email }, token }
 *     Store the token (httpOnly cookie or secure header).
 *
 *  2. signUp(name, email, password)
 *     → POST /api/auth/register  { name, email, password }
 *     ← { user: { id, name, email }, token }
 *
 *  3. signOut()
 *     → POST /api/auth/logout  (with token)
 *     ← 204 No Content
 *     Clear stored token / cookie.
 *
 *  4. getCurrentUser()
 *     → GET /api/auth/me  (with token)
 *     ← { user: { id, name, email } } | null
 *     Called on app boot to restore the session.
 *
 *  5. continueAsGuest()
 *     → No backend call needed (or POST /api/auth/guest if you track guests).
 *     Returns a synthetic guest user object.
 *
 *  The AuthUser interface below is the shape every consumer expects.
 * ============================================================================
 */

// ── Public types ────────────────────────────────────────────────────────────

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  isGuest: boolean;
}

// ── Internal storage helpers (demo-only) ─────────────────────────────────────

interface StoredUser {
  id: string;
  name: string;
  email: string;
  passwordHash: string;
}

const USERS_KEY = 'ragmedx_users';
const SESSION_KEY = 'ragmedx_session';

function getStoredUsers(): StoredUser[] {
  try {
    return JSON.parse(localStorage.getItem(USERS_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveStoredUsers(users: StoredUser[]): void {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

function saveSession(user: AuthUser): void {
  localStorage.setItem(SESSION_KEY, JSON.stringify(user));
}

function clearSession(): void {
  localStorage.removeItem(SESSION_KEY);
}

function loadSession(): AuthUser | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

// ── SHA-256 hashing (SubtleCrypto) ──────────────────────────────────────────
// ⚠️  Demo-only.  Real auth should hash passwords SERVER-SIDE with bcrypt/argon2.

async function sha256(text: string): Promise<string> {
  const msgBuffer = new TextEncoder().encode(text);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

// ── Simulated network delay ─────────────────────────────────────────────────

function delay(ms = 600): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Unique ID generator ─────────────────────────────────────────────────────

function uid(): string {
  return `u_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

// ── Public API ──────────────────────────────────────────────────────────────

export async function signIn(
  email: string,
  password: string,
): Promise<AuthUser> {
  await delay();
  const users = getStoredUsers();
  const hash = await sha256(password);
  const found = users.find(
    (u) => u.email.toLowerCase() === email.toLowerCase() && u.passwordHash === hash,
  );

  if (!found) {
    throw new Error('Incorrect email or password');
  }

  const user: AuthUser = {
    id: found.id,
    name: found.name,
    email: found.email,
    isGuest: false,
  };
  saveSession(user);
  return user;
}

export async function signUp(
  name: string,
  email: string,
  password: string,
): Promise<AuthUser> {
  await delay();
  const users = getStoredUsers();
  const exists = users.some(
    (u) => u.email.toLowerCase() === email.toLowerCase(),
  );

  if (exists) {
    throw new Error('An account with this email already exists');
  }

  const hash = await sha256(password);
  const newUser: StoredUser = {
    id: uid(),
    name,
    email: email.toLowerCase(),
    passwordHash: hash,
  };
  users.push(newUser);
  saveStoredUsers(users);

  const user: AuthUser = {
    id: newUser.id,
    name: newUser.name,
    email: newUser.email,
    isGuest: false,
  };
  saveSession(user);
  return user;
}

export async function signOut(): Promise<void> {
  await delay(300);
  clearSession();
}

export function getCurrentUser(): AuthUser | null {
  return loadSession();
}

export function continueAsGuest(): AuthUser {
  const guest: AuthUser = {
    id: 'guest',
    name: 'Guest',
    email: '',
    isGuest: true,
  };
  saveSession(guest);
  return guest;
}
