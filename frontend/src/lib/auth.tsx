import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, getToken, setToken, setUnauthorizedHandler, type Me } from "./api";

interface AuthState {
  user: Me | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  acceptInvite: (data: { name: string; email: string; password: string; invite_token: string }) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Me | null>(null);
  const [loading, setLoading] = useState(() => getToken() !== null);

  const clear = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(clear);           // expired or revoked token -> back to sign-in
    if (!getToken()) return;
    api.me().then(setUser).catch(clear).finally(() => setLoading(false));
  }, [clear]);

  const start = async (token: string) => {
    setToken(token);
    setUser(await api.me());
  };

  const value: AuthState = {
    user,
    loading,
    login: async (email, password) => start((await api.login(email, password)).access_token),
    acceptInvite: async (data) => start((await api.register(data)).access_token),
    logout: async () => {
      try {
        await api.logout();                  // revokes this token server-side
      } finally {
        clear();
      }
    },
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}

/** Mirrors can_review in backend/main.py: solo users review their own; in a firm, admins/seniors. */
export function canDecide(user: Me, requesterId: string): boolean {
  if (user.firm_id === null) return user.id === requesterId;
  return user.role === "admin" || user.role === "senior";
}
