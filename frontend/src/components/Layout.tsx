import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";
import { LangToggle } from "./LangToggle";

export function Layout() {
  const { t } = useI18n();
  const { user, logout } = useAuth();
  const [pending, setPending] = useState<number | null>(null);
  const location = useLocation();

  useEffect(() => {
    api.approvals("pending").then((rows) => setPending(rows.length)).catch(() => setPending(null));
  }, [location.pathname]);

  const initials = (user?.name ?? "")
    .split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]!.toUpperCase()).join("");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <ScalesIcon />
          <span className="brand-name">{t.appName}</span>
        </div>
        <div className="topbar-right">
          <LangToggle />
          <span className="avatar" title={user?.name} aria-hidden="true">{initials}</span>
          <button type="button" className="btn-ghost-dark" onClick={() => void logout()}>{t.signOut}</button>
        </div>
      </header>
      <div className="shell">
        <nav className="sidenav" aria-label={t.mainNav}>
          <NavLink to="/" end className="navlink"><SearchIcon />{t.research}</NavLink>
          <NavLink to="/review" className="navlink">
            <CheckIcon />{t.review}
            {pending ? <span className="badge" aria-label={`${pending}`}>{pending}</span> : null}
          </NavLink>
        </nav>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function ScalesIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#D9B86A" strokeWidth="1.6"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3v18" /><path d="M5 21h14" /><path d="M4 7h16" />
      <path d="M7 7l-3 7a3 3 0 0 0 6 0z" /><path d="M17 7l-3 7a3 3 0 0 0 6 0z" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></svg>
  );
}

function CheckIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 11l3 3 8-8" /><path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9" />
    </svg>
  );
}
