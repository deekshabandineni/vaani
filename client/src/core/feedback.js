// L1-aware coaching tips for on-device results (mirrors services/ml-api/app/fusion.py:attach_feedback).
import { MISPRONOUNCED_BELOW } from "./gop.js";

export function priorFor(phone, l1, profiles) {
  const base = profiles.contrasts[phone]?.prior ?? profiles.default_prior;
  return Math.min(1, base + (profiles.profiles[l1]?.boost?.[phone] ?? 0));
}

export function attachFeedback(report, l1, profiles, maxTips = 3) {
  const weak = [];
  for (const w of report.words ?? [])
    for (const p of w.phones ?? [])
      if (p.score < MISPRONOUNCED_BELOW) weak.push([p.score - 20 * priorFor(p.phone, l1, profiles), w.word, p]);
  weak.sort((a, b) => a[0] - b[0]);
  const tips = [], seen = new Set();
  for (const [, word, p] of weak) {
    const c = profiles.contrasts[p.phone];
    if (!c || seen.has(p.phone)) continue;
    seen.add(p.phone);
    tips.push({ phone: p.phone, heard: p.heard, tip: c.tip, word, score: p.score,
                known_confusion: Boolean(p.heard && c.often_heard_as.includes(p.heard)) });
    if (tips.length >= maxTips) break;
  }
  return { ...report, feedback: tips, l1 };
}
