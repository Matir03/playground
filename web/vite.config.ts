import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Rust engines speak gRPC-Web. We proxy them under same-origin paths
// (/c4, /go9, /a, /hex, /mi) so the whole setup runs behind a single origin with no CORS. The
// frontend uses these relative paths as its engine base URLs (see App.tsx).
// Local-only: bound to localhost (not exposed publicly).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/c4": {
        target: "http://127.0.0.1:7100",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/c4/, ""),
      },
      "/go9": {
        target: "http://127.0.0.1:7101",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/go9/, ""),
      },
      "/a": {
        target: "http://127.0.0.1:7102",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/a/, ""),
      },
      "/hex": {
        target: "http://127.0.0.1:7104",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/hex/, ""),
      },
      "/mi": {
        target: "http://127.0.0.1:7105",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/mi/, ""),
      },
    },
  },
});
