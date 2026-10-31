import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    open: false,
    proxy: {
      // Forward every /api request to the FastAPI backend.
      // The frontend can call fetch('/api/v1/...') without CORS issues.
      // `ws: true` also upgrades WebSocket connections (/api/v1/ws/query-stream)
      // through the dev proxy, so the dev origin is single-origin like prod and
      // buildWsUrl can default to window.location.host.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
