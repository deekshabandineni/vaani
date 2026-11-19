// Browser end-to-end test: real Chrome, fake microphone, on-device ONNX
// inference, gateway + ML service, offline queue and two-device sync.
import assert from "node:assert/strict";

import puppeteer from "puppeteer-core";

const APP = process.env.APP_URL ?? "http://localhost:4173";
const AUDIO = process.env.FAKE_AUDIO;
const email = `e2e-${Date.now()}@uf.edu`;
const log = (...a) => console.log("  ✓", ...a);

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
         `--use-file-for-fake-audio-capture=${AUDIO}`, "--autoplay-policy=no-user-gesture-required"],
});

const visible = (page, sel) => page.waitForFunction((s) => { const n = document.querySelector(s); return n && !n.hidden; }, { timeout: 20000 }, sel);
const text = (page, sel) => page.$eval(sel, (n) => n.textContent);

async function openApp(ctx) {
  const page = await ctx.newPage();
  page.on("pageerror", (e) => console.error("    page error:", e.message));
  await page.goto(APP, { waitUntil: "networkidle0" });
  return page;
}

async function practice(page, sentence) {
  await page.$eval("details.custom", (d) => (d.open = true));
  await page.$eval("#custom-form input", (i) => (i.value = ""));
  await page.type("#custom-form input", sentence);
  await page.click("#custom-form button");
  await page.waitForFunction((s) => document.querySelector("#sentence").textContent === s, {}, sentence);
}

async function recordAndWait(page, source) {
  await page.$eval("#result", (r) => (r.hidden = true));
  await page.click("#record");
  await page.waitForFunction((src) => !document.querySelector("#result").hidden &&
    document.querySelector("#result-source").textContent.startsWith(src), { timeout: 30000 }, source);
}

try {
  // ---- device 1: sign up ----
  const ctx1 = await browser.createBrowserContext();
  const p1 = await openApp(ctx1);
  await page1Ready(p1);
  async function page1Ready(p) {
    await visible(p, "#view-auth");
    await p.waitForFunction(() => document.querySelector("#model-status").textContent.includes("on-device"), { timeout: 20000 });
  }
  log("app loaded, on-device model:", await text(p1, "#model-status"));

  await p1.click("#auth-toggle");
  await p1.type("[name=name]", "Deeksha");
  await p1.type("[name=email]", email);
  await p1.type("[name=password]", "password123");
  await p1.select("[name=l1]", "te");
  await p1.click("#auth-submit");
  await visible(p1, "#view-practice");
  log("registered; first drill:", await text(p1, "#sentence"), "| targets:", await text(p1, "#targets"));

  // ---- on-device result, then cloud result ----
  await practice(p1, "very well");
  await recordAndWait(p1, "On-device");
  const device = await p1.evaluate(() => ({
    score: document.querySelector("#score").textContent,
    extra: document.querySelector("#result-extra").textContent,
    words: [...document.querySelectorAll("#scored-words .word")].map((w) => [w.textContent, w.className]),
    tip: document.querySelector("#tips li")?.textContent }));
  log("on-device result:", JSON.stringify(device));
  assert.match(device.words[0][1], /bad/, "'very' should be flagged (V said as W)");
  assert.match(device.words[1][1], /good/, "'well' should be fine");
  assert.match(device.tip, /lower lip/);

  await p1.waitForFunction(() => document.querySelector("#result-source").textContent.startsWith("Server"), { timeout: 20000 });
  log("cloud result:", await text(p1, "#result-source"), "| score", await text(p1, "#score"));

  // ---- device 2: handoff + live attempts ----
  const ctx2 = await browser.createBrowserContext();
  const p2 = await openApp(ctx2);
  await visible(p2, "#view-auth");
  await p2.type("[name=email]", email);
  await p2.type("[name=password]", "password123");
  await p2.click("#auth-submit");
  await visible(p2, "#view-practice");
  await p1.waitForFunction(() => document.querySelector("#sync-status").textContent === "live sync");
  await p2.waitForFunction(() => document.querySelector("#sync-status").textContent === "live sync");

  await practice(p1, "very well");
  await visible(p2, "#handoff");
  log("device 2 handoff banner:", await text(p2, "#handoff span"));
  await p2.click("#handoff button");
  assert.equal(await text(p2, "#sentence"), "very well");

  await recordAndWait(p1, "On-device");
  await p2.waitForFunction(() => /another device/.test(document.querySelector("#toast").textContent), { timeout: 15000 });
  log("device 2 toast:", await text(p2, "#toast"));

  // ---- offline: score on device, queue, sync when back online ----
  await p1.waitForFunction(() => document.querySelector("#result-source").textContent.startsWith("Server"), { timeout: 20000 });
  await p1.setOfflineMode(true);
  await recordAndWait(p1, "On-device");
  const queued = await p1.evaluate(async () => {
    const db = await new Promise((r) => { const q = indexedDB.open("vaani", 1); q.onsuccess = () => r(q.result); });
    return new Promise((r) => { const q = db.transaction("queue").objectStore("queue").count(); q.onsuccess = () => r(q.result); });
  });
  assert.equal(queued, 1);
  log("offline: scored on-device, 1 attempt queued");
  await p1.setOfflineMode(false);
  await p1.waitForFunction(() => /Synced an offline attempt/.test(document.querySelector("#toast").textContent), { timeout: 20000 });
  log("back online:", await text(p1, "#toast"));

  // ---- progress ----
  await p1.click('[data-view="progress"]');
  await p1.waitForFunction(() => document.querySelectorAll("#recent li").length >= 3, { timeout: 10000 });
  const prog = await p1.evaluate(() => ({
    streak: document.querySelector("#streak").textContent,
    weakest: document.querySelector("#phone-bars .bar .label")?.textContent,
    attempts: document.querySelectorAll("#recent li").length }));
  log("progress:", JSON.stringify(prog));
  assert.equal(prog.streak, "1");
  assert.equal(prog.attempts, 3);
  assert.match(prog.weakest, /V/);
  await p1.screenshot({ path: "screenshot-progress.png" });
  await p1.click('[data-view="practice"]');
  await p1.setViewport({ width: 900, height: 1000 });
  await p1.screenshot({ path: "screenshot-practice.png", fullPage: true });
  console.log("\nE2E PASSED");
} finally {
  await browser.close();
}
