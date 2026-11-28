import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Dev: /api is proxied to the FastAPI server, so the browser sees one origin (no CORS).
// Production: serve dist/ behind the same reverse proxy as the API.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
