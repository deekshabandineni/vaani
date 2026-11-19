import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const shared = fileURLToPath(new URL("../shared", import.meta.url));

export default defineConfig({
  resolve: { alias: { "@shared": shared } },
  server: { port: 5173, fs: { allow: [".", shared] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
