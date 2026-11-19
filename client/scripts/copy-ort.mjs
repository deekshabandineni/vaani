// Copy onnxruntime-web's WASM runtime into public/ort so the app can run
// inference fully offline (no CDN), and the service worker can cache it.
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const dist = path.dirname(require.resolve("onnxruntime-web"));
const out = new URL("../public/ort/", import.meta.url);
mkdirSync(out, { recursive: true });
const files = readdirSync(dist).filter((f) => /^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f));
for (const f of files) copyFileSync(path.join(dist, f), new URL(f, out));
console.log(`copied ${files.length} onnxruntime-web files to public/ort`);
