// Offline support: cache the app shell, the ONNX model and the WASM runtime.
// API calls always go to the network (the app queues attempts itself when offline).
const CACHE = "vaani-v1";
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
  const heavy = /^\/(models|ort|assets)\//.test(url.pathname);
  e.respondWith(heavy
    ? caches.match(e.request).then((hit) => hit ?? fetch(e.request).then((r) => cache(e.request, r)))
    : fetch(e.request).then((r) => cache(e.request, r)).catch(() => caches.match(e.request).then((h) => h ?? caches.match("/index.html"))));
});

function cache(req, res) {
  if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
  return res;
}
