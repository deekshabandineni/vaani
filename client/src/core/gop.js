// GOP scoring: a line-by-line port of ml/vaani_ml/gop.py (keep them in sync).
export const MISPRONOUNCED_BELOW = 60;
const NEG_INF = -1e30;

export function calibratedScore(gop, cal = { a: 3, b: 3 }) {
  return 100 / (1 + Math.exp(-(cal.a * gop + cal.b)));
}

/** In-place row-wise log-softmax over columns [from, C). Returns new Float64Array [T*(C-from)]. */
function logSoftmaxRows(x, T, C, from = 0) {
  const W = C - from, out = new Float64Array(T * W);
  for (let t = 0; t < T; t++) {
    let m = -Infinity;
    for (let c = from; c < C; c++) m = Math.max(m, x[t * C + c]);
    let s = 0;
    for (let c = from; c < C; c++) s += Math.exp(x[t * C + c] - m);
    const lse = m + Math.log(s);
    for (let c = from; c < C; c++) out[t * W + c - from] = x[t * C + c] - lse;
  }
  return out;
}

/** CTC Viterbi forced alignment. lp: Float64Array [T*C]. Returns Int32Array of target index per frame (-1 = blank) or null. */
export function ctcForceAlign(lp, T, C, targets, blank = 0) {
  const N = targets.length;
  if (N === 0) return null;
  const S = 2 * N + 1;
  const ext = new Int32Array(S).fill(blank);
  for (let i = 0; i < N; i++) ext[2 * i + 1] = targets[i];
  let repeats = 0;
  for (let i = 1; i < N; i++) if (targets[i] === targets[i - 1]) repeats++;
  if (T < N + repeats) return null;

  const canSkip = new Uint8Array(S);
  for (let s = 2; s < S; s++) canSkip[s] = ext[s] !== blank && ext[s] !== ext[s - 2] ? 1 : 0;
  let alpha = new Float64Array(S).fill(NEG_INF);
  alpha[0] = lp[ext[0]];
  alpha[1] = lp[ext[1]];
  const back = new Int8Array(T * S);
  for (let t = 1; t < T; t++) {
    const next = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      let best = alpha[s], choice = 0;
      const p1 = s >= 1 ? alpha[s - 1] : NEG_INF;
      if (p1 > best) { best = p1; choice = 1; }
      const p2 = s >= 2 && canSkip[s] ? alpha[s - 2] : NEG_INF;
      if (p2 > best) { best = p2; choice = 2; }
      next[s] = best + lp[t * C + ext[s]];
      back[t * S + s] = choice;
    }
    alpha = next;
  }
  let s = alpha[S - 1] >= alpha[S - 2] ? S - 1 : S - 2;
  if (alpha[s] <= NEG_INF / 2) return null;
  const assign = new Int32Array(T);
  for (let t = T - 1; t >= 0; t--) {
    assign[t] = s % 2 === 1 ? (s - 1) / 2 : -1;
    s -= back[t * S + s];
  }
  return assign;
}

/**
 * logits: Float32Array [T*C] from the model; words: [{word, phones}];
 * tokens: ["<b>", "AA", ...]; ipa: {AA: "ɑ", ...}
 */
export function scoreUtterance(logits, T, C, words, { tokens, ipa = {}, calibration } = {}) {
  const lp = logSoftmaxRows(logits, T, C, 0);
  const id = Object.fromEntries(tokens.map((p, i) => [p, i]));
  const flat = words.flatMap((w) => w.phones);
  const assign = ctcForceAlign(lp, T, C, flat.map((p) => id[p]));
  if (!assign) return { ok: false, error: "audio too short for the reference text", words: [] };

  const nb = logSoftmaxRows(lp, T, C, 1); // renormalise over the 39 phones
  const W = C - 1;
  const spans = flat.map(() => [Infinity, -1]);
  for (let t = 0; t < T; t++) {
    const i = assign[t];
    if (i >= 0) { spans[i][0] = Math.min(spans[i][0], t); spans[i][1] = t + 1; }
  }

  let k = 0;
  const all = [];
  const outWords = words.map((w) => {
    const phones = w.phones.map((p) => {
      const [s, e] = spans[k++];
      const pid = id[p] - 1;
      let gop = 0;
      const sums = new Float64Array(W);
      for (let t = s; t < e; t++) {
        let mx = -Infinity;
        for (let c = 0; c < W; c++) { const v = nb[t * W + c]; sums[c] += v; if (v > mx) mx = v; }
        gop += nb[t * W + pid] - mx;
      }
      gop /= e - s;
      let best = 0;
      for (let c = 1; c < W; c++) if (sums[c] > sums[best]) best = c;
      const score = calibratedScore(gop, calibration);
      all.push(score);
      return { phone: p, ipa: ipa[p] ?? p, gop: Math.round(gop * 1e4) / 1e4, score: Math.round(score * 10) / 10,
               heard: best !== pid && score < MISPRONOUNCED_BELOW ? tokens[best + 1] : null, start: s, end: e };
    });
    const ws = phones.reduce((a, p) => a + p.score, 0) / phones.length;
    return { word: w.word, score: Math.round(ws * 10) / 10, phones };
  });
  const acc = all.reduce((a, b) => a + b, 0) / all.length;
  return { ok: true, source: "device", overall: { accuracy: Math.round(acc * 10) / 10 }, words: outWords };
}
