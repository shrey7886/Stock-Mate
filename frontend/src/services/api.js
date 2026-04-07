const BASE = "/api";

async function request(path, options = {}) {
  const token = localStorage.getItem("sm_token");
  const headers = { "Content-Type": "application/json", ...options.headers };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  if (res.status === 401 && !path.includes("/auth/") && !path.includes("/user/me")) {
    localStorage.removeItem("sm_token");
    localStorage.removeItem("sm_user");
    window.location.href = "/login";
    throw new Error("Session expired");
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}`);
  }

  return res.json();
}

/* ── Auth ─────────────────────────────────────────── */
export const auth = {
  login: (payload) =>
    request("/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  register: (payload) =>
    request("/auth/register", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  me: () => request("/user/me"),
  forgotPassword: (email) =>
    request("/auth/forgot-password", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),
  resetPassword: (token, new_password) =>
    request("/auth/reset-password", {
      method: "POST",
      body: JSON.stringify({ token, new_password }),
    }),
};

/* ── Portfolio ────────────────────────────────────── */
export const portfolio = {
  summary: () => request("/portfolio/summary"),
  verifyBroker: (accountId) =>
    request(`/portfolio/verify-broker${accountId ? `?account_id=${accountId}` : ""}`),
};

/* ── Chat ─────────────────────────────────────────── */
export const chat = {
  send: (message, sessionId) =>
    request("/chat/message", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId }),
    }),
  clearSession: () =>
    request("/chat/clear-session", { method: "POST" }),
  proactiveInsights: () => request("/chat/proactive-insights"),
};

/* ── Zerodha ──────────────────────────────────────── */
export const zerodha = {
  start: () => request("/zerodha/start"),
  status: () => request("/zerodha/status"),
  accounts: () => request("/zerodha/accounts"),
  unlink: (accountId) =>
    request(`/zerodha/unlink${accountId ? `?account_id=${accountId}` : ""}`, {
      method: "POST",
    }),
  setPrimary: (accountId) =>
    request(`/zerodha/accounts/primary?account_id=${encodeURIComponent(accountId)}`, {
      method: "POST",
    }),
};

/* ── Health ────────────────────────────────────────── */
export const health = {
  check: () => request("/health"),
};
