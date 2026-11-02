import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8000 (uv run uvicorn api.server:create_app_from_env --factory).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000", "/files": "http://127.0.0.1:8000" } },
});
