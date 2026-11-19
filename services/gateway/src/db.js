// Postgres data access. Works with a real `pg` Pool or pg-mem in tests.
import { randomUUID } from "node:crypto";

export const SCHEMA = `
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL,
  l1 TEXT NOT NULL DEFAULT 'other',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS attempts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id),
  client_attempt_id TEXT NOT NULL,
  drill_id TEXT,
  text TEXT NOT NULL,
  source TEXT NOT NULL,
  overall REAL,
  report JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  UNIQUE (user_id, client_attempt_id)
);
CREATE INDEX IF NOT EXISTS attempts_user_time ON attempts (user_id, created_at);
CREATE TABLE IF NOT EXISTS phoneme_stats (
  user_id TEXT NOT NULL REFERENCES users(id),
  phone TEXT NOT NULL,
  ema REAL NOT NULL,
  attempts INTEGER NOT NULL,
  ease REAL NOT NULL,
  interval_h REAL NOT NULL,
  due_at TIMESTAMPTZ NOT NULL,
  last_score REAL,
  PRIMARY KEY (user_id, phone)
);`;

const rowToStat = (r) => ({ phone: r.phone, ema: Number(r.ema), attempts: Number(r.attempts), ease: Number(r.ease),
  intervalH: Number(r.interval_h), dueAt: new Date(r.due_at).getTime(), lastScore: r.last_score == null ? null : Number(r.last_score) });

const rowToAttempt = (r) => ({ id: r.id, clientAttemptId: r.client_attempt_id, drillId: r.drill_id, text: r.text,
  source: r.source, overall: r.overall == null ? null : Number(r.overall),
  report: typeof r.report === "string" ? JSON.parse(r.report) : r.report, createdAt: new Date(r.created_at).toISOString() });

export function createDb(pool) {
  const q = (text, params) => pool.query(text, params);
  return {
    async migrate() { for (const stmt of SCHEMA.split(";").map((s) => s.trim()).filter(Boolean)) await q(stmt); },

    async createUser({ email, passwordHash, name, l1 }) {
      const id = randomUUID();
      await q("INSERT INTO users (id, email, password_hash, name, l1) VALUES ($1, $2, $3, $4, $5)",
        [id, email.toLowerCase(), passwordHash, name, l1]);
      return { id, email: email.toLowerCase(), name, l1 };
    },
    async userByEmail(email) {
      return (await q("SELECT * FROM users WHERE email = $1", [email.toLowerCase()])).rows[0] ?? null;
    },
    async userById(id) {
      const r = (await q("SELECT id, email, name, l1 FROM users WHERE id = $1", [id])).rows[0];
      return r ?? null;
    },
    async updateUser(id, { name, l1 }) {
      if (name) await q("UPDATE users SET name = $2 WHERE id = $1", [id, name]);
      if (l1) await q("UPDATE users SET l1 = $2 WHERE id = $1", [id, l1]);
      return this.userById(id);
    },

    async attemptByClientId(userId, clientAttemptId) {
      const r = (await q("SELECT * FROM attempts WHERE user_id = $1 AND client_attempt_id = $2", [userId, clientAttemptId])).rows[0];
      return r ? rowToAttempt(r) : null;
    },
    async insertAttempt(userId, a) {
      const id = randomUUID();
      await q(`INSERT INTO attempts (id, user_id, client_attempt_id, drill_id, text, source, overall, report, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
        [id, userId, a.clientAttemptId, a.drillId ?? null, a.text, a.source, a.overall, JSON.stringify(a.report), a.createdAt]);
      return { id, ...a };
    },
    async replaceAttemptReport(userId, clientAttemptId, { source, overall, report }) {
      await q("UPDATE attempts SET source = $3, overall = $4, report = $5 WHERE user_id = $1 AND client_attempt_id = $2",
        [userId, clientAttemptId, source, overall, JSON.stringify(report)]);
      return this.attemptByClientId(userId, clientAttemptId);
    },
    async recentAttempts(userId, limit = 20) {
      const r = await q("SELECT * FROM attempts WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2", [userId, limit]);
      return r.rows.map(rowToAttempt);
    },
    async practiceDays(userId) {
      const r = await q("SELECT created_at FROM attempts WHERE user_id = $1", [userId]);
      return [...new Set(r.rows.map((x) => new Date(x.created_at).toISOString().slice(0, 10)))].sort();
    },

    async stats(userId) {
      const r = await q("SELECT * FROM phoneme_stats WHERE user_id = $1", [userId]);
      return Object.fromEntries(r.rows.map((x) => [x.phone, rowToStat(x)]));
    },
    async saveStat(userId, s) {
      const params = [userId, s.phone, s.ema, s.attempts, s.ease, s.intervalH, new Date(s.dueAt).toISOString(), s.lastScore];
      const exists = (await q("SELECT 1 FROM phoneme_stats WHERE user_id = $1 AND phone = $2", [userId, s.phone])).rows.length;
      if (exists) {
        await q(`UPDATE phoneme_stats SET ema = $3, attempts = $4, ease = $5, interval_h = $6, due_at = $7, last_score = $8
                 WHERE user_id = $1 AND phone = $2`, params);
      } else {
        await q(`INSERT INTO phoneme_stats (user_id, phone, ema, attempts, ease, interval_h, due_at, last_score)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`, params);
      }
    },
  };
}
