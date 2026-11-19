// API client, offline queue (IndexedDB) and WebSocket sync.
const API = (import.meta.env.VITE_API_URL ?? "http://localhost:3000").replace(/\/$/, "");

const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* private mode */ } },
};

export const deviceId = store.get("vaani.device") ?? (() => {
  const id = `${/Mobi/.test(navigator.userAgent) ? "phone" : "desktop"}-${crypto.randomUUID().slice(0, 6)}`;
  store.set("vaani.device", id);
  return id;
})();

export const session = {
  get token() { return store.get("vaani.token"); },
  set token(v) { store.set("vaani.token", v); },
};

export async function api(path, { method = "GET", body, form } = {}) {
  const headers = { "x-device-id": deviceId };
  if (session.token) headers.authorization = `Bearer ${session.token}`;
  if (body) headers["content-type"] = "application/json";
  const r = await fetch(API + path, { method, headers, body: form ?? (body && JSON.stringify(body)) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error ?? `HTTP ${r.status}`), { status: r.status });
  return data;
}

// ---------- offline queue ----------
function idb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("vaani", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("queue", { keyPath: "clientAttemptId" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function tx(mode, fn) {
  const db = await idb();
  return new Promise((resolve, reject) => {
    const t = db.transaction("queue", mode);
    const out = fn(t.objectStore("queue"));
    t.oncomplete = () => resolve(out?.result ?? out);
    t.onerror = () => reject(t.error);
  });
}
export const queue = {
  add: (item) => tx("readwrite", (s) => s.put(item)),
  remove: (id) => tx("readwrite", (s) => s.delete(id)),
  all: () => tx("readonly", (s) => s.getAll()),
};

/** Send one attempt: device report first (instant progress), then audio for cloud scoring. */
export async function submitAttempt({ clientAttemptId, drillId, text, wav, deviceReport, createdAt }) {
  if (deviceReport?.ok) {
    await api("/api/attempts", { method: "POST",
      body: { clientAttemptId, drillId, text, source: "device", report: deviceReport, createdAt } });
  }
  const form = new FormData();
  form.append("audio", new Blob([wav], { type: "audio/wav" }), "attempt.wav");
  for (const [k, v] of Object.entries({ text, drillId: drillId ?? "", clientAttemptId, createdAt })) form.append(k, v);
  return api("/api/assess", { method: "POST", form });
}

export async function flushQueue(onEach) {
  for (const item of await queue.all()) {
    try {
      const res = await submitAttempt(item);
      await queue.remove(item.clientAttemptId);
      onEach?.(item, res);
    } catch (err) {
      if (err.status && err.status < 500 && err.status !== 401) await queue.remove(item.clientAttemptId); // unrecoverable
      else break; // still offline; try later
    }
  }
}

// ---------- live sync ----------
export function connectSync(onMessage, onStatus) {
  let ws, retry = 1000, closed = false;
  const open = () => {
    if (!session.token) return;
    ws = new WebSocket(`${API.replace(/^http/, "ws")}/ws?token=${encodeURIComponent(session.token)}&device=${deviceId}`);
    ws.onopen = () => { retry = 1000; onStatus(true); };
    ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data)); } catch { /* ignore */ } };
    ws.onclose = () => { onStatus(false); if (!closed) setTimeout(open, (retry = Math.min(retry * 2, 30000))); };
  };
  open();
  return {
    send: (msg) => ws?.readyState === 1 && ws.send(JSON.stringify(msg)),
    close: () => { closed = true; ws?.close(); },
  };
}
