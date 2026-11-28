// Typed client for the FastAPI backend (backend/main.py). Shapes mirror its response models.

export type Role = "admin" | "senior" | "associate";

export interface Me {
  id: string;
  name: string;
  email: string;
  role: Role;
  firm_id: string | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user_name: string;
  user_role: Role;
}

export type ApprovalStatus = "not_required" | "pending" | "approved" | "rejected" | "expired";

export interface QueryResponse {
  session_id: string;
  response: string;
  confidence_score: number;
  legal_domain: string;
  missing_facts: string[];
  uncertainties: string[];
  questions_for_lawyer: string[];
  sources: string[];
  next_steps: string[];
  has_document: boolean;
  document_hindi: string | null;
  document_english: string | null;
  is_emergency: boolean;
  urgency_level: string;
  human_approval_needed: boolean;
  source_status: Record<string, string>;
  missing_fields: string[];
  unverified_citations: string[];
  tokens_used: number;
  run_id: string | null;
  approval_status: ApprovalStatus;
  response_time_ms: number;
}

export interface QueryRequest {
  query: string;
  jurisdiction: string;
  run_id?: string;
  session_id?: string;
  case_id?: string;
}

export interface RunProgress {
  run_id: string;
  status: "running" | "awaiting_approval" | "completed";
  stage: string;
  next: string[];
}

export interface ApprovalRow {
  run_id: string;
  status: ApprovalStatus;
  requester_id: string;
  case_id: string | null;
  document_type: string | null;
  created_at: string | null;
  decided_at: string | null;
  reviewer_id: string | null;
}

export interface CaseRow {
  id: string;
  title: string;
  legal_domain: string;
  status: string;
  created_at: string;
}

export interface Decision {
  decision: "approved" | "rejected";
  notes: string;
  edited_response?: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

// The token lives in sessionStorage: it ends with the tab, which suits shared office machines.
const TOKEN_KEY = "nyaymitra.token";

export function getToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage blocked: the session just won't survive a reload */
  }
}

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(fn: () => void): void {
  onUnauthorized = fn;
}

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

/** FastAPI errors: {"detail": "text"} or {"detail": [{"msg": ...}, ...]} for validation. */
export function errorMessage(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) => (d as { msg?: string })?.msg).filter(Boolean);
    if (msgs.length) return msgs.join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && typeof init.body === "string" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(BASE + path, { ...init, headers });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 401 && token) onUnauthorized();
    throw new ApiError(res.status, errorMessage(body, `Request failed (${res.status})`));
  }
  return body as T;
}

const post = <T>(path: string, data?: unknown) =>
  request<T>(path, { method: "POST", body: data === undefined ? undefined : JSON.stringify(data) });

export const api = {
  login(email: string, password: string) {
    // OAuth2 password form (backend uses OAuth2PasswordRequestForm)
    const form = new URLSearchParams({ username: email, password });
    return request<TokenResponse>("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    });
  },
  register(data: { name: string; email: string; password: string; invite_token: string }) {
    return post<TokenResponse>("/api/auth/register", data);
  },
  me: () => request<Me>("/api/auth/me"),
  logout: () => post<{ status: string }>("/api/auth/logout"),
  query: (data: QueryRequest) => post<QueryResponse>("/api/query", data),
  runProgress: (runId: string) => request<RunProgress>(`/api/runs/${encodeURIComponent(runId)}`),
  approvals: (status: ApprovalStatus | "all" = "pending") =>
    request<ApprovalRow[]>(`/api/approvals?status=${status}`),
  approval: (runId: string) => request<QueryResponse>(`/api/approvals/${encodeURIComponent(runId)}`),
  decide: (runId: string, d: Decision) => post<QueryResponse>(`/api/approvals/${encodeURIComponent(runId)}`, d),
  cases: () => request<CaseRow[]>("/api/cases"),
  createCase: (title: string) => post<{ id: string; title: string }>("/api/cases", { title }),
};
