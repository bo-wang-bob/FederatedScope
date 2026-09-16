import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => ({
  plugins: [react(), ...(mode === 'design' ? [{
    name: 'design-preview-no-api',
    configureServer(server: import('vite').ViteDevServer) {
      // A design session must never forward an accidental request to a live service.
      server.middlewares.use('/api', (_request, response) => {
        response.statusCode = 403;
        response.setHeader('Content-Type', 'application/json');
        response.end(JSON.stringify({ error: { message: '设计预览不连接后端' } }));
      });
    },
  }] : [])],
  server: {
    port: mode === 'design' ? 5174 : 5173,
    proxy: mode === 'design' ? undefined : {
      // Preserve the browser's Host/Origin pair; keep backend same-origin checks intact.
      '/api': { target: process.env.FS_API_PROXY || 'http://127.0.0.1:8001', changeOrigin: false },
    },
  },
  preview: { port: 4173 },
  build: { manifest: true, sourcemap: true, outDir: mode === 'design' ? 'dist-design' : 'dist' },
}));
