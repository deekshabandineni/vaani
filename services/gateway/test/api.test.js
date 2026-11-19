import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import http from "node:http";
import { after, before, test } from "node:test";

import jwt from "jsonwebtoken";
import { newDb } from "pg-mem";
import WebSocket from "ws";

import { createApp } from "../src/app.js";
import { createDb } from "../src/db.js";
import { attachWebSocket, Hub } from "../src/hub.js";

const shared = new URL("../../../shared/", import.meta.url);
const drills = JSON.parse(readFileSync(new URL("drills.json", shared)));
const profiles = JSON.parse(readFileSync(new URL("l1_profiles.json", shared)));
const SECRET = "test-secret-0123456789";

const report = (vScore) => ({
  ok: true, source: "local", overall: { accuracy: (vScore + 90 * 3) / 4 },
  words: [{ word: "very", score: 0, phones: [
    { phone: "V", score: vScore }, { phone: "EH", score: 90 }, { phone: "R", score: 90 }, { phone: "IY", score: 90 }] }],
  feedback: [],
});

let server, base, mlCalls = [];
before(async () => {
  const { Pool } = newDb().adapters.createPg();
  const db = createDb(new Pool());
  await db.migrate();
  const fakeFetch = async (url, opts) => {
    mlCalls.push({ url, text: opts.body.get("reference_text"), l1: opts.body.get("l1") });
    return new Response(JSON.stringify({ ...report(85), source: "fused" }), { status: 200 });
  };
  const hub = new Hub();
  server = http.createServer(createApp({ db, hub, drills, profiles, jwtSecret: SECRET, mlApiUrl: "http://ml", fetchImpl: fakeFetch }));
  attachWebSocket(server, hub, (t) => jwt.verify(t, SECRET));
  await new Promise((r) => server.listen(0, r));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => server.close());

const api = async (path, { token, device, ...opts } = {}) => {
  const headers = { "content-type": "application/json", ...(token && { authorization: `Bearer ${token}` }),
                    ...(device && { "x-device-id": device }) };
  const r = await fetch(base + path, { ...opts, headers: { ...headers, ...opts.headers },
                                       body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body });
  return { status: r.status, body: await r.json().catch(() => null) };
};

async function register(email, l1 = "te") {
  const r = await api("/api/auth/register", { method: "POST", body: { email, password: "password123", name: "Deeksha", l1 } });
  assert.equal(r.status, 201, JSON.stringify(r.body));
  return r.body.token;
}

test("register, login, validation", async () => {
  await register("a@uf.edu");
  assert.equal((await api("/api/auth/register", { method: "POST", body: { email: "a@uf.edu", password: "password123", name: "x" } })).status, 409);
  assert.equal((await api("/api/auth/register", { method: "POST", body: { email: "b@uf.edu", password: "short", name: "x" } })).status, 400);
  assert.equal((await api("/api/auth/login", { method: "POST", body: { email: "a@uf.edu", password: "wrongpass1" } })).status, 401);
  const ok = await api("/api/auth/login", { method: "POST", body: { email: "A@uf.edu", password: "password123" } });
  assert.equal(ok.status, 200);
  assert.equal((await api("/api/me", { token: ok.body.token })).body.l1, "te");
  assert.equal((await api("/api/me")).status, 401);
});

test("device attempt updates stats; cloud result corrects without double counting", async () => {
  const token = await register("c@uf.edu");
  const first = await api("/api/attempts", { token, method: "POST",
    body: { clientAttemptId: "att-1", drillId: "d001", text: "very", source: "device", report: report(20) } });
  assert.equal(first.status, 201);
  let stats = (await api("/api/progress", { token })).body.stats;
  const v1 = stats.find((s) => s.phone === "V");
  assert.equal(v1.attempts, 1);
  assert.equal(stats[0].phone, "V"); // weakest first

  const dup = await api("/api/attempts", { token, method: "POST",
    body: { clientAttemptId: "att-1", text: "very", source: "device", report: report(20) } });
  assert.equal(dup.status, 200);
  assert.ok(dup.body.duplicate);

  // cloud scoring for the same attempt via /api/assess
  const form = new FormData();
  form.append("audio", new Blob([new Uint8Array(100)], { type: "audio/wav" }), "a.wav");
  form.append("text", "very"); form.append("clientAttemptId", "att-1"); form.append("drillId", "d001");
  const r = await fetch(`${base}/api/assess`, { method: "POST", body: form, headers: { authorization: `Bearer ${token}` } });
  assert.equal(r.status, 200);
  assert.equal((await r.json()).attempt.source, "fused");
  assert.deepEqual(mlCalls.at(-1), { url: "http://ml/assess", text: "very", l1: "te" });

  const prog = (await api("/api/progress", { token })).body;
  const v2 = prog.stats.find((s) => s.phone === "V");
  assert.equal(v2.attempts, 1, "cloud correction must not add an attempt");
  assert.ok(v2.ema > v1.ema);
  assert.equal(prog.recent.length, 1);
  assert.equal(prog.streak, 1);
});

test("next drills adapt to weak phones", async () => {
  const token = await register("d@uf.edu", "other");
  const weakZH = { ok: true, overall: { accuracy: 20 }, words: [{ word: "measure", phones: [
    { phone: "M", score: 95 }, { phone: "EH", score: 95 }, { phone: "ZH", score: 5 }, { phone: "ER", score: 90 }] }] };
  for (let i = 0; i < 3; i++) {
    await api("/api/attempts", { token, method: "POST", body: { clientAttemptId: `z${i}`, text: "measure", source: "device", report: weakZH } });
  }
  const next = (await api("/api/drills/next?count=2", { token })).body;
  assert.equal(next.length, 2);
  assert.ok(next[0].targets.includes("ZH"), next[0].text);
});

test("websocket relays attempts and session state to the user's other devices", async () => {
  const token = await register("e@uf.edu");
  const open = (device) => new Promise((resolve, reject) => {
    const ws = new WebSocket(`${base.replace("http", "ws")}/ws?token=${token}&device=${device}`);
    const inbox = [];
    ws.on("message", (m) => inbox.push(JSON.parse(m)));
    ws.on("open", () => resolve({ ws, inbox }));
    ws.on("error", reject);
  });
  const phone = await open("phone");
  const laptop = await open("laptop");
  const wait = (box, type) => new Promise((resolve, reject) => {
    const t0 = Date.now();
    const iv = setInterval(() => {
      const m = box.find((x) => x.type === type);
      if (m) { clearInterval(iv); resolve(m); } else if (Date.now() - t0 > 2000) { clearInterval(iv); reject(new Error(`no ${type}`)); }
    }, 10);
  });
  await wait(phone.inbox, "presence");

  phone.ws.send(JSON.stringify({ type: "session", state: { drillId: "d007" } }));
  assert.deepEqual((await wait(laptop.inbox, "session")).state, { drillId: "d007" });

  await api("/api/attempts", { token, device: "phone", method: "POST",
    body: { clientAttemptId: "w1", text: "very", source: "device", report: report(50) } });
  assert.equal((await wait(laptop.inbox, "attempt")).attempt.clientAttemptId, "w1");
  await wait(laptop.inbox, "stats");
  assert.ok(!phone.inbox.some((m) => m.type === "attempt"), "sender should not get its own echo");

  const bad = new WebSocket(`${base.replace("http", "ws")}/ws?token=nope`);
  await new Promise((resolve) => bad.on("error", resolve));
  phone.ws.close(); laptop.ws.close();
});
