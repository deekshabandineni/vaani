import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { correctStat, initialStat, phoneScores, pickDrills, updateStat, weakness } from "../src/scheduler.js";

const shared = new URL("../../../shared/", import.meta.url);
const drills = JSON.parse(readFileSync(new URL("drills.json", shared)));
const profiles = JSON.parse(readFileSync(new URL("l1_profiles.json", shared)));
const NOW = Date.UTC(2026, 10, 18);

test("L1 prior makes known confusions weaker from the start", () => {
  const v = initialStat("V", "te", profiles, NOW);
  const k = initialStat("K", "te", profiles, NOW);
  assert.ok(v.ema < k.ema);
  assert.ok(weakness(v, NOW) > weakness(k, NOW));
});

test("passing grows the interval, failing resets it", () => {
  let s = initialStat("TH", "hi", profiles, NOW);
  s = updateStat(s, 90, NOW);
  assert.equal(s.intervalH, 4);
  s = updateStat(s, 90, NOW);
  assert.ok(s.intervalH > 4 && s.dueAt > NOW);
  const failed = updateStat(s, 30, NOW);
  assert.equal(failed.intervalH, 0);
  assert.ok(failed.dueAt <= NOW);
  assert.ok(failed.ease < s.ease && failed.ema < s.ema);
  assert.ok(weakness(failed, NOW + 3600e3) > weakness(s, NOW + 3600e3));
});

test("correctStat shifts EMA without touching the schedule", () => {
  const s = updateStat(initialStat("Z", "te", profiles, NOW), 40, NOW);
  const c = correctStat(s, 40, 70);
  assert.ok(c.ema > s.ema);
  assert.equal(c.dueAt, s.dueAt);
  assert.equal(c.attempts, s.attempts);
});

test("phoneScores averages repeated phones", () => {
  const r = { words: [{ phones: [{ phone: "V", score: 20 }, { phone: "EH", score: 90 }] },
                      { phones: [{ phone: "V", score: 60 }] }] };
  assert.deepEqual(phoneScores(r), { V: 40, EH: 90 });
});

test("pickDrills targets the weakest phones and diversifies", () => {
  const strong = (ph) => ({ ...initialStat(ph, "other", profiles, NOW), ema: 95, dueAt: NOW + 1e9 });
  const statsByPhone = {};
  for (const ph of ["V", "W", "TH", "DH", "Z", "AE", "IH", "IY"]) statsByPhone[ph] = strong(ph);
  statsByPhone.ZH = { ...initialStat("ZH", "other", profiles, NOW), ema: 10, dueAt: NOW - 1 };
  const picks = pickDrills({ drills, statsByPhone, profiles, l1: "other", count: 3, now: NOW });
  assert.equal(picks.length, 3);
  assert.ok(picks[0].targets.includes("ZH"), `first pick should target ZH: ${picks[0].text}`);
  assert.equal(new Set(picks.map((p) => p.id)).size, 3);
});

test("pickDrills skips recent drills and new users get V/W-style drills", () => {
  const first = pickDrills({ drills, statsByPhone: {}, profiles, l1: "te", count: 1, now: NOW })[0];
  assert.ok(first.focus.some((f) => ["V", "W", "TH", "DH"].includes(f)), first.text);
  const next = pickDrills({ drills, statsByPhone: {}, profiles, l1: "te", count: 1, now: NOW, recentIds: [first.id] })[0];
  assert.notEqual(next.id, first.id);
});
