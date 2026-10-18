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
 *
 * Fix #8: expires_in is now used to schedule a proactive token refresh
 * (at 90% of the token lifetime) so callers never silently hit a 401.
 */

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  isGuest: boolean;
}

// ── Storage helpers ──────────────────────────────────────────────────────────

const ACCESS_KEY  = 'ragmedx_access_token';
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
  cancelScheduledRefresh();
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

// ── Proactive token refresh (Fix #8) ─────────────────────────────────────────

let _refreshTimer: ReturnType<typeof setTimeout> | null = null;

function cancelScheduledRefresh(): void {
  if (_refreshTimer !== null) {
    clearTimeout(_refreshTimer);
    _refreshTimer = null;
  }
}

/**
 * Schedule a silent refresh at 90% of the token's lifetime.
 * If the refresh fails (network down, etc.) we leave the existing token in
 * place so the user stays logged in for as long as the token is valid; the
 * next explicit request will then surface the 401 normally.
 */
function scheduleRefresh(expiresInSeconds: number): void {
  cancelScheduledRefresh();
  const delayMs = Math.max((expiresInSeconds * 0.9) * 1000, 0);
  _refreshTimer = setTimeout(async () => {
    const refreshToken = localStorage.getItem(REFRESH_KEY);
    if (!refreshToken) return;
    try {
      const data = await apiPost<TokenResponse>('/auth/refresh', { refresh_token: refreshToken });
      saveTokens(data.access_token, data.refresh_token);
      scheduleRefresh(data.expires_in);
    } catch {
      // Refresh failed silently; the token will expire naturally and the next
      // request will return 401, prompting the user to log in again.
    }
  }, delayMs);
}

// ── API base ─────────────────────────────────────────────────────────────────

const BASE = '/api/v1';

/**
 * Parse a response body as JSON without throwing on an empty or non-JSON body.
 * A bare `res.json()` throws "Unexpected end of JSON input" when the body is
 * empty (e.g. a proxy hiccup, a 204, or a network-truncated response), which
 * surfaces to the user as a confusing error on an otherwise ordinary failure.
 * We read the text first and only attempt to parse when there is something to
 * parse, returning `{}` otherwise so callers can fall back to a status message.
 */
async function safeJson(res: Response): Promise<Record<string, unknown>> {
  const text = await res.text();
  if (!text) return {};
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return {};
  }
}

async function apiPost<T>(path: string, body: unknown, token?: string): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
  } catch {
    // fetch rejects only on a network-level failure (server down, DNS, CORS).
    throw new Error('Network error — unable to reach the server. Is the backend running?');
  }

  const json = await safeJson(res);
  if (!res.ok) {
    const message =
      (json.message as string) ||
      (json.detail as string) ||
      `Request failed (${res.status})`;
    throw new Error(message);
  }
  return json as T;
}

// ── Backend response shapes ──────────────────────────────────────────────────

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;   // Fix #8: now typed (was ignored before)
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
  scheduleRefresh(data.expires_in);  // Fix #8
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
  const username = name.toLowerCase().replace(/\s+/g, '_');
  const data = await apiPost<TokenResponse>('/auth/register', {
    email,
    username,
    password,
    full_name: name,
    role: 'healthcare_professional',
  });
  saveTokens(data.access_token, data.refresh_token);
  scheduleRefresh(data.expires_in);  // Fix #8
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
  clearTokens();  // also cancels the scheduled refresh timer
}

export function continueAsGuest(): AuthUser {
  const guest: AuthUser = { id: 'guest', name: 'Guest', email: '', isGuest: true };
  saveSession(guest);
  return guest;
}

/**
 * Call once on app startup to re-arm the refresh timer if a token is already
 * present in localStorage (e.g. after a page reload).  Reads the expiry from
 * the JWT exp claim so no extra storage is needed.
 */
export function rehydrateRefreshSchedule(): void {
  const token = getAccessToken();
  if (!token) return;
  try {
    // JWT payload is the second base64-url segment
    const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
    const expiresIn = payload.exp - Math.floor(Date.now() / 1000);
    if (expiresIn > 0) scheduleRefresh(expiresIn);
  } catch {
    // malformed token — leave it; the next API call will get a 401
  }
}
