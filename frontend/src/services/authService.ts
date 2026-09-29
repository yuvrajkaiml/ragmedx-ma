/**
 * RAGMedX-MA — Authentication Service (Real backend integration)
 *
 * Connects to the FastAPI backend at /api/v1/auth/*.
 * JWT access + refresh tokens are stored in localStorage.
 * Every outbound API request should include:
 *   Authorization: Bearer <access_token>
 *
 * Token storage keys:
 *   ragmedx_access_token  — short-lived JWT access token
 *   ragmedx_refresh_token — long-lived refresh token
 *   ragmedx_session       — AuthUser object (id, name, email, isGuest)
 */

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  isGuest: boolean;
}

// ── Storage helpers ──────────────────────────────────────────────────────────

const ACCESS_KEY = 'ragmedx_access_token';
const REFRESH_KEY = 'ragmedx_refresh_token';
const SESSION_KEY = 'ragmedx_session';

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_KEY);
}

function saveTokens(access: string, refresh: string): void {
  localStorage.setItem(ACCESS_KEY, access);
  localStorage.setItem(REFRESH_KEY, refresh);
}

function clearTokens(): void {
  localStorage.removeItem(ACCESS_KEY);
  localStorage.removeItem(REFRESH_KEY);
  localStorage.removeItem(SESSION_KEY);
}

function saveSession(user: AuthUser): void {
  localStorage.setItem(SESSION_KEY, JSON.stringify(user));
}

export function getCurrentUser(): AuthUser | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

// ── API base ─────────────────────────────────────────────────────────────────

const BASE = '/api/v1';

async function apiPost<T>(path: string, body: unknown, token?: string): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  const json = await res.json();
  if (!res.ok) {
    // Backend returns { message } or { detail }
    throw new Error(json.message || json.detail || `Request failed (${res.status})`);
  }
  return json as T;
}

// ── Backend response shapes ──────────────────────────────────────────────────

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  user: {
    id: string;
    email: string;
    username: string;
    full_name: string | null;
    role: string;
    is_active: boolean;
    is_verified: boolean;
    created_at: string;
  };
}

// ── Public API ───────────────────────────────────────────────────────────────

export async function signIn(email: string, password: string): Promise<AuthUser> {
  const data = await apiPost<TokenResponse>('/auth/login', { email, password });
  saveTokens(data.access_token, data.refresh_token);
  const user: AuthUser = {
    id: data.user.id,
    name: data.user.full_name || data.user.username,
    email: data.user.email,
    isGuest: false,
  };
  saveSession(user);
  return user;
}

export async function signUp(name: string, email: string, password: string): Promise<AuthUser> {
  // Backend expects: email, username, password, full_name
  // Derive username from name (lowercase, no spaces)
  const username = name.toLowerCase().replace(/\s+/g, '_');
  const data = await apiPost<TokenResponse>('/auth/register', {
    email,
    username,
    password,
    full_name: name,
    role: 'healthcare_professional',
  });
  saveTokens(data.access_token, data.refresh_token);
  const user: AuthUser = {
    id: data.user.id,
    name: data.user.full_name || data.user.username,
    email: data.user.email,
    isGuest: false,
  };
  saveSession(user);
  return user;
}

export async function signOut(): Promise<void> {
  const refresh = localStorage.getItem(REFRESH_KEY);
  if (refresh) {
    try {
      await apiPost('/auth/logout', { refresh_token: refresh });
    } catch {
      // Logout best-effort — clear local state regardless
    }
  }
  clearTokens();
}

export function continueAsGuest(): AuthUser {
  const guest: AuthUser = { id: 'guest', name: 'Guest', email: '', isGuest: true };
  saveSession(guest);
  return guest;
}
