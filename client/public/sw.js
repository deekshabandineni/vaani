// Offline support: cache the app shell, the ONNX model and the WASM runtime.
// API calls always go to the network (the app queues attempts itself when offline).
const CACHE = "vaani-v2"; // bump to drop caches from older versions
const PRECACHE = ["/", "/index.html", "/manifest.webmanifest",
  "/models/vaani_int8.onnx", "/models/model_meta.json", "/models/calibration.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => Promise.allSettled(PRECACHE.map((u) => c.add(u)))));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api")) return;
  // cache-first for heavy immutable assets, network-first for the page itself
  const heavy = /^\/(ort|assets)\//.test(url.pathname); // content-hashed or versioned: cache-first
  e.respondWith(heavy
    ? caches.match(e.request).then((hit) => hit ?? fetch(e.request).then((r) => cache(e.request, r)))
    : fetch(e.request).then((r) => cache(e.request, r)).catch(() => caches.match(e.request).then((h) => h ?? caches.match("/index.html"))));
});

function cache(req, res) {
  // never store an HTML fallback under a model/asset URL
  const html = (res.headers.get("content-type") || "").includes("text/html");
  if (res.ok && !(html && /^\/(models|ort|assets)\//.test(new URL(req.url).pathname))) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
  return res;
}
