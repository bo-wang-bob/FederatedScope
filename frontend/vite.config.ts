import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': process.env.FS_API_PROXY || 'http://127.0.0.1:8001',
    },
  },
  preview: { port: 4173 },
  build: { sourcemap: true },
});
