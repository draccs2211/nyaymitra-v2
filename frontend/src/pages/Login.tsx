import { useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate } from "react-router-dom";
import { LangToggle } from "../components/LangToggle";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

export function Login() {
  const { t } = useI18n();
  const { user, login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await login(email.trim(), password);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthCard title={t.signInTitle}>
      <form onSubmit={submit} className="stack">
        <div className="field">
          <label htmlFor="email">{t.email}</label>
          <input id="email" type="email" autoComplete="username" required value={email}
            onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="password">{t.password}</label>
          <input id="password" type="password" autoComplete="current-password" required value={password}
            onChange={(e) => setPassword(e.target.value)} />
        </div>
        {error && <p className="alert alert-danger" role="alert">{error}</p>}
        <button type="submit" className="btn-primary" disabled={busy}>{t.signIn}</button>
        <p className="muted small">{t.inviteOnly} <Link to="/invite">{t.acceptInvite}</Link></p>
      </form>
    </AuthCard>
  );
}

export function AuthCard({ title, children }: { title: string; children: ReactNode }) {
  const { t } = useI18n();
  return (
    <div className="auth-page">
      <div className="auth-card card">
        <div className="row-between">
          <span className="auth-brand">{t.appName}</span>
          <LangToggle light />
        </div>
        <h1 className="h-serif">{title}</h1>
        {children}
      </div>
    </div>
  );
}
