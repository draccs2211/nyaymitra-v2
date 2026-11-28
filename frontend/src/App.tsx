import { useEffect, type ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AuthProvider, useAuth } from "./lib/auth";
import { I18nProvider, useI18n } from "./lib/i18n";
import { Invite } from "./pages/Invite";
import { Login } from "./pages/Login";
import { Review } from "./pages/Review";
import { Workspace } from "./pages/Workspace";

function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const { t } = useI18n();
  if (loading) return <p className="muted pad">{t.loading}</p>;
  return user ? <>{children}</> : <Navigate to="/login" replace />;
}

function DocumentLang() {
  const { lang } = useI18n();
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  return null;
}

export function App() {
  return (
    <I18nProvider>
      <DocumentLang />
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/invite" element={<Invite />} />
            <Route element={<RequireAuth><Layout /></RequireAuth>}>
              <Route index element={<Workspace />} />
              <Route path="review" element={<Review />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </I18nProvider>
  );
}
