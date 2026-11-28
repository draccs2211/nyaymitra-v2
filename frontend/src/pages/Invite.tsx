import { useState, type FormEvent } from "react";
import { Navigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";
import { AuthCard } from "./Login";

/** /invite (optionally ?token=...). The token comes from `maintenance create-firm` or POST /api/firms/invites. */
export function Invite() {
  const { t } = useI18n();
  const { user, acceptInvite } = useAuth();
  const [params] = useSearchParams();
  const [token, setTokenValue] = useState(params.get("token") ?? "");
  const [name, setName] = useState("");
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
      await acceptInvite({ name: name.trim(), email: email.trim(), password, invite_token: token });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthCard title={t.acceptInvite}>
      <form onSubmit={submit} className="stack">
          <p className="muted">{t.acceptInviteHint}</p>
          <div className="field">
            <label htmlFor="token">{t.inviteToken}</label>
            <input id="token" required autoComplete="off" spellCheck={false} maxLength={200} value={token}
              onChange={(e) => setTokenValue(e.target.value.trim())} />
          </div>
          <div className="field">
            <label htmlFor="name">{t.name}</label>
            <input id="name" autoComplete="name" required value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="email">{t.email}</label>
            <input id="email" type="email" autoComplete="username" required value={email}
              onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="password">{t.password}</label>
            <input id="password" type="password" autoComplete="new-password" required minLength={8} maxLength={72}
              value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          {error && <p className="alert alert-danger" role="alert">{error}</p>}
          <button type="submit" className="btn-primary" disabled={busy}>{t.createAccount}</button>
        </form>
    </AuthCard>
  );
}
