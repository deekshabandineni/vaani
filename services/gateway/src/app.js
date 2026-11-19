// HTTP API for the Vaani client. The gateway owns users, history and the
// adaptive schedule; pronunciation scoring is delegated to the FastAPI ML service.
import bcrypt from "bcryptjs";
import express from "express";
import jwt from "jsonwebtoken";
import multer from "multer";

import { correctStat, initialStat, phoneScores, pickDrills, updateStat } from "./scheduler.js";

const L1S = new Set(["te", "hi", "other"]);
const SOURCES = new Set(["device", "local", "azure", "fused"]);
const isEmail = (s) => typeof s === "string" && /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(s);

export function createApp({ db, hub, drills, profiles, jwtSecret, mlApiUrl, fetchImpl = fetch }) {
  const app = express();
  const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 5 * 1024 * 1024 } });
  app.use(express.json({ limit: "1mb" }));
  app.use((req, res, next) => {
    res.set("Access-Control-Allow-Origin", "*");
    res.set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Device-Id");
    res.set("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS");
    if (req.method === "OPTIONS") return res.sendStatus(204);
    next();
  });

  const sign = (user) => jwt.sign({ sub: user.id }, jwtSecret, { expiresIn: "30d" });
  const auth = async (req, res, next) => {
    try {
      const claims = jwt.verify((req.get("authorization") ?? "").replace(/^Bearer /, ""), jwtSecret);
      req.user = await db.userById(claims.sub);
      if (!req.user) throw new Error("no user");
      req.deviceId = req.get("x-device-id") ?? null;
      next();
    } catch {
      res.status(401).json({ error: "unauthorized" });
    }
  };
  const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

  app.get("/health", (_req, res) => res.json({ status: "ok" }));

  // ---- auth ----
  app.post("/api/auth/register", wrap(async (req, res) => {
    const { email, password, name, l1 = "other" } = req.body ?? {};
    if (!isEmail(email) || typeof password !== "string" || password.length < 8 || !name)
      return res.status(400).json({ error: "email, name and a password of 8+ characters are required" });
    if (!L1S.has(l1)) return res.status(400).json({ error: "l1 must be te, hi or other" });
    if (await db.userByEmail(email)) return res.status(409).json({ error: "email already registered" });
    const user = await db.createUser({ email, name, l1, passwordHash: await bcrypt.hash(password, 10) });
    res.status(201).json({ token: sign(user), user });
  }));

  app.post("/api/auth/login", wrap(async (req, res) => {
    const { email, password } = req.body ?? {};
    const row = isEmail(email) ? await db.userByEmail(email) : null;
    if (!row || !(await bcrypt.compare(String(password ?? ""), row.password_hash)))
      return res.status(401).json({ error: "wrong email or password" });
    const user = { id: row.id, email: row.email, name: row.name, l1: row.l1 };
    res.json({ token: sign(user), user });
  }));

  app.get("/api/me", auth, (req, res) => res.json(req.user));
  app.patch("/api/me", auth, wrap(async (req, res) => {
    const { name, l1 } = req.body ?? {};
    if (l1 && !L1S.has(l1)) return res.status(400).json({ error: "l1 must be te, hi or other" });
    res.json(await db.updateUser(req.user.id, { name, l1 }));
  }));

  // ---- drills ----
  app.get("/api/drills", (_req, res) => res.json(drills));

  // phonemes for a custom sentence, so the client can score it on-device too
  app.get("/api/phonemize", auth, wrap(async (req, res) => {
    const text = String(req.query.text ?? "").slice(0, 300);
    if (!text.trim()) return res.status(400).json({ error: "text is required" });
    try {
      const r = await fetchImpl(`${mlApiUrl}/phonemize?${new URLSearchParams({ text })}`);
      res.status(r.status).json(await r.json());
    } catch (err) {
      res.status(502).json({ error: `ML service unreachable: ${err.message}` });
    }
  }));
  app.get("/api/drills/next", auth, wrap(async (req, res) => {
    const count = Math.min(10, Math.max(1, Number(req.query.count) || 3));
    const recentIds = (await db.recentAttempts(req.user.id, 15)).map((a) => a.drillId).filter(Boolean);
    res.json(pickDrills({ drills, statsByPhone: await db.stats(req.user.id), recentIds,
                          l1: req.user.l1, profiles, count }));
  }));

  // ---- attempts ----
  async function recordAttempt(user, deviceId, { clientAttemptId, drillId, text, source, report, createdAt }) {
    const overall = Number(report?.overall?.accuracy ?? NaN);
    const entry = { clientAttemptId, drillId, text, source, report,
                    overall: Number.isFinite(overall) ? overall : null,
                    createdAt: createdAt && !Number.isNaN(Date.parse(createdAt)) ? new Date(createdAt).toISOString() : new Date().toISOString() };
    const existing = await db.attemptByClientId(user.id, clientAttemptId);
    const stats = await db.stats(user.id);
    const changed = [];
    let attempt;

    if (!existing) {
      attempt = await db.insertAttempt(user.id, entry);
      for (const [phone, score] of Object.entries(phoneScores(report))) {
        const s = updateStat(stats[phone] ?? initialStat(phone, user.l1, profiles), score);
        await db.saveStat(user.id, s);
        changed.push(s);
      }
    } else if (existing.source === "device" && source !== "device") {
      // cloud result for an attempt first scored on-device: correct the stats, don't double count
      const before = phoneScores(existing.report);
      attempt = await db.replaceAttemptReport(user.id, clientAttemptId, entry);
      for (const [phone, score] of Object.entries(phoneScores(report))) {
        if (!stats[phone]) continue;
        const s = before[phone] != null ? correctStat(stats[phone], before[phone], score) : stats[phone];
        await db.saveStat(user.id, s);
        changed.push(s);
      }
    } else {
      return { attempt: existing, changed, duplicate: true };
    }
    hub?.broadcast(user.id, { type: "attempt", attempt }, deviceId);
    if (changed.length) hub?.broadcast(user.id, { type: "stats", stats: changed }, deviceId);
    return { attempt, changed, duplicate: false };
  }

  app.post("/api/attempts", auth, wrap(async (req, res) => {
    const b = req.body ?? {};
    if (!b.clientAttemptId || !b.text || !SOURCES.has(b.source) || !Array.isArray(b.report?.words))
      return res.status(400).json({ error: "clientAttemptId, text, source and report.words are required" });
    const out = await recordAttempt(req.user, req.deviceId, b);
    res.status(out.duplicate ? 200 : 201).json(out);
  }));

  // Upload audio for cloud scoring (Azure + server model); stores the result.
  app.post("/api/assess", auth, upload.single("audio"), wrap(async (req, res) => {
    const { text, drillId, clientAttemptId, createdAt } = req.body ?? {};
    if (!req.file || !text || !clientAttemptId)
      return res.status(400).json({ error: "audio, text and clientAttemptId are required" });
    const form = new FormData();
    form.append("audio", new Blob([req.file.buffer], { type: req.file.mimetype || "audio/wav" }), "audio.wav");
    form.append("reference_text", text);
    form.append("l1", req.user.l1);
    let upstream;
    try {
      upstream = await fetchImpl(`${mlApiUrl}/assess`, { method: "POST", body: form });
    } catch (err) {
      return res.status(502).json({ error: `ML service unreachable: ${err.message}` });
    }
    const report = await upstream.json().catch(() => ({ detail: "invalid response" }));
    if (!upstream.ok) return res.status(upstream.status).json({ error: report.detail ?? "scoring failed" });
    const out = await recordAttempt(req.user, req.deviceId,
      { clientAttemptId, drillId, text, source: report.source ?? "local", report, createdAt });
    res.json({ report, attempt: out.attempt });
  }));

  // ---- progress ----
  app.get("/api/progress", auth, wrap(async (req, res) => {
    const stats = Object.values(await db.stats(req.user.id)).sort((a, b) => a.ema - b.ema);
    const days = await db.practiceDays(req.user.id);
    // consecutive practice days ending today (or yesterday, if today hasn't started yet)
    const key = (d) => d.toISOString().slice(0, 10);
    const day = new Date();
    if (!days.includes(key(day))) day.setUTCDate(day.getUTCDate() - 1);
    let streak = 0;
    while (days.includes(key(day))) { streak++; day.setUTCDate(day.getUTCDate() - 1); }
    const recent = await db.recentAttempts(req.user.id, 20);
    res.json({ stats, streak, practiceDays: days.length,
               recent: recent.map(({ report, ...a }) => ({ ...a, feedback: report?.feedback ?? [] })) });
  }));

  app.use((err, _req, res, _next) => {
    console.error(err);
    res.status(err.status ?? 500).json({ error: err.expose ? err.message : "internal error" });
  });
  return app;
}
