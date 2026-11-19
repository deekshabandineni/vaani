// Adaptive drill scheduling: per-phoneme spaced repetition.
//
// Every (user, phoneme) pair has a stat:
//   ema       exponential moving average of the phone's score (0-100)
//   ease      SM-2 style growth factor for the review interval
//   intervalH hours until the phoneme is due again
//   dueAt     when it becomes due
// A new user's stats start from an L1 prior (shared/l1_profiles.json), so a
// Telugu speaker is drilled on V/W and TH from the first session. After a few
// attempts the user's own scores take over.

const HOUR = 3600 * 1000;
export const PASS_SCORE = 75;

export function priorFor(phone, l1, profiles) {
  const base = profiles.contrasts[phone]?.prior ?? profiles.default_prior;
  const boost = profiles.profiles[l1]?.boost?.[phone] ?? 0;
  return Math.min(1, base + boost);
}

export function initialStat(phone, l1, profiles, now = Date.now()) {
  return { phone, ema: 100 - 50 * priorFor(phone, l1, profiles), attempts: 0, ease: 2.0,
           intervalH: 0, dueAt: now, lastScore: null };
}

/** Apply one attempt's mean score for this phone. Returns a new stat. */
export function updateStat(stat, score, now = Date.now()) {
  const alpha = Math.max(0.25, 1 / (stat.attempts + 1));
  const s = { ...stat, ema: stat.ema + alpha * (score - stat.ema), attempts: stat.attempts + 1, lastScore: score };
  if (score >= PASS_SCORE) {
    s.intervalH = stat.intervalH === 0 ? 4 : Math.min(24 * 30, stat.intervalH * stat.ease);
    s.ease = Math.min(2.8, stat.ease + 0.1);
  } else {
    s.intervalH = 0; // failed: due again right away (relearning restarts at 4 h)
    s.ease = Math.max(1.3, stat.ease - 0.2);
  }
  s.dueAt = now + s.intervalH * HOUR;
  return s;
}

/** A later, better score for the same attempt (cloud replacing on-device).
 *  Shift the EMA by the same step the first update used; leave the schedule alone. */
export function correctStat(stat, oldScore, newScore) {
  const alpha = Math.max(0.25, 1 / Math.max(1, stat.attempts));
  return { ...stat, ema: Math.max(0, Math.min(100, stat.ema + alpha * (newScore - oldScore))), lastScore: newScore };
}

export function weakness(stat, now = Date.now()) {
  const due = stat.dueAt <= now;
  return (0.15 + (100 - stat.ema) / 100) * (due ? 1 : 0.35);
}

/** Mean score per phone in a pronunciation report. */
export function phoneScores(report) {
  const acc = new Map();
  for (const w of report?.words ?? []) {
    for (const p of w.phones ?? []) {
      if (typeof p.score !== "number" || !p.phone) continue;
      const a = acc.get(p.phone) ?? [0, 0];
      acc.set(p.phone, [a[0] + p.score, a[1] + 1]);
    }
  }
  return Object.fromEntries([...acc].map(([ph, [sum, n]]) => [ph, sum / n]));
}

/**
 * Choose `count` drills that cover the user's weakest due phonemes.
 * statsByPhone: { PHONE: stat } (missing phones get the L1 prior)
 * recentIds: drill ids practiced recently (skipped unless nothing else is left)
 */
export function pickDrills({ drills, statsByPhone, recentIds = [], l1 = "other", profiles, count = 3, now = Date.now() }) {
  const stat = (ph) => statsByPhone[ph] ?? initialStat(ph, l1, profiles, now);
  const w = {};
  for (const d of drills) for (const word of d.words) for (const ph of word.phones) w[ph] ??= weakness(stat(ph), now);

  const recent = new Set(recentIds);
  let pool = drills.filter((d) => !recent.has(d.id));
  if (pool.length < count) pool = drills.slice();

  const chosen = [];
  while (chosen.length < count && pool.length) {
    let best = null;
    for (const d of pool) {
      const phones = d.words.flatMap((x) => x.phones);
      const uniq = [...new Set(phones)];
      // squared weakness: one phone the user really fails outweighs several "maybe weak" prior guesses;
      // counting every occurrence favours sentences that repeat the target sound
      const coverage = phones.reduce((s, ph) => s + w[ph] ** 2, 0) / Math.sqrt(phones.length);
      const focus = d.focus.reduce((s, ph) => s + (w[ph] ?? 0) ** 2, 0);
      const value = coverage + 0.6 * focus;
      if (!best || value > best.value) best = { d, value, uniq };
    }
    const targets = best.uniq.sort((a, b) => w[b] - w[a]).slice(0, 3);
    chosen.push({ ...best.d, targets, priority: Number(best.value.toFixed(3)) });
    for (const ph of targets) w[ph] *= 0.5; // diversify the next pick
    pool = pool.filter((d) => d !== best.d);
  }
  return chosen;
}
