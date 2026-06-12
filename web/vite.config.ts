import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The two Rust engines speak gRPC-Web. We proxy them under same-origin paths
// (/c4, /go9) so the whole setup is reachable behind a single origin (and a
// single public tunnel) with no CORS. The frontend uses these relative paths
// as its engine base URLs (see App.tsx).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Allow any Host header so a Cloudflare quick-tunnel domain can reach the
    // dev server (it would otherwise be rejected by Vite's host check).
    allowedHosts: true,
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
    },
  },
});
