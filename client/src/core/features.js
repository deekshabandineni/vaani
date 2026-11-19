// Log-mel features: a line-by-line port of ml/vaani_ml/features.py.
// Must stay numerically identical. ml/tests/test_parity.py checks it.
export const SAMPLE_RATE = 16000;
export const N_FFT = 512;
export const WIN_LENGTH = 400;
export const HOP_LENGTH = 160;
export const N_MELS = 80;
const F_MIN = 20, F_MAX = 7600, PREEMPH = 0.97, LOG_EPS = 1e-6;
const N_BINS = N_FFT / 2 + 1;

const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

let FILTERS = null, WINDOW = null, TWIDDLE = null, BITREV = null;

function init() {
  if (FILTERS) return;
  const lo = hzToMel(F_MIN), hi = hzToMel(F_MAX);
  const hz = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  FILTERS = [];
  for (let m = 0; m < N_MELS; m++) {
    const w = new Float64Array(N_BINS);
    let first = -1, last = -1;
    for (let k = 0; k < N_BINS; k++) {
      const f = (k * SAMPLE_RATE) / N_FFT;
      const v = Math.max(0, Math.min((f - hz[m]) / (hz[m + 1] - hz[m]), (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1])));
      w[k] = v;
      if (v > 0) { if (first < 0) first = k; last = k; }
    }
    FILTERS.push({ w, first: Math.max(0, first), last });
  }
  WINDOW = new Float64Array(WIN_LENGTH);
  for (let n = 0; n < WIN_LENGTH; n++) WINDOW[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / WIN_LENGTH);
  TWIDDLE = { re: new Float64Array(N_FFT / 2), im: new Float64Array(N_FFT / 2) };
  for (let k = 0; k < N_FFT / 2; k++) {
    TWIDDLE.re[k] = Math.cos((-2 * Math.PI * k) / N_FFT);
    TWIDDLE.im[k] = Math.sin((-2 * Math.PI * k) / N_FFT);
  }
  BITREV = new Uint16Array(N_FFT);
  const bits = Math.log2(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    BITREV[i] = r;
  }
}

// in-place iterative radix-2 FFT
function fft(re, im) {
  for (let i = 0; i < N_FFT; i++) {
    const j = BITREV[i];
    if (j > i) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1, step = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let k = 0; k < half; k++) {
        const tr = TWIDDLE.re[k * step], ti = TWIDDLE.im[k * step];
        const a = start + k, b = a + half;
        const xr = re[b] * tr - im[b] * ti, xi = re[b] * ti + im[b] * tr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

export const numFrames = (n) => 1 + Math.floor(Math.max(0, n - WIN_LENGTH) / HOP_LENGTH);

/** audio: Float32Array at 16 kHz in [-1, 1]. Returns { data: Float32Array(T*80), frames: T }. */
export function logMel(audio, normalize = true) {
  init();
  let x = Float64Array.from(audio);
  if (x.length < WIN_LENGTH) { const p = new Float64Array(WIN_LENGTH); p.set(x); x = p; }
  const y = new Float64Array(x.length);
  y[0] = x[0];
  for (let i = 1; i < x.length; i++) y[i] = x[i] - PREEMPH * x[i - 1];

  const T = numFrames(y.length);
  const out = new Float64Array(T * N_MELS);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), power = new Float64Array(N_BINS);
  for (let t = 0; t < T; t++) {
    re.fill(0); im.fill(0);
    const off = t * HOP_LENGTH;
    for (let n = 0; n < WIN_LENGTH; n++) re[n] = y[off + n] * WINDOW[n];
    fft(re, im);
    for (let k = 0; k < N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      const { w, first, last } = FILTERS[m];
      let s = 0;
      for (let k = first; k <= last; k++) s += power[k] * w[k];
      out[t * N_MELS + m] = Math.log(s + LOG_EPS);
    }
  }
  if (normalize) {
    for (let m = 0; m < N_MELS; m++) {
      let mean = 0;
      for (let t = 0; t < T; t++) mean += out[t * N_MELS + m];
      mean /= T;
      let v = 0;
      for (let t = 0; t < T; t++) v += (out[t * N_MELS + m] - mean) ** 2;
      const sd = Math.sqrt(v / T) + 1e-5;
      for (let t = 0; t < T; t++) out[t * N_MELS + m] = (out[t * N_MELS + m] - mean) / sd;
    }
  }
  return { data: Float32Array.from(out), frames: T };
}

/** Energy-based trim of leading/trailing silence (same rule as features.trim_silence). */
export function trimSilence(audio, thresholdDb = -40, padMs = 150) {
  const frame = 320;
  if (audio.length < frame) return audio;
  const n = Math.floor(audio.length / frame);
  const db = new Float64Array(n);
  let max = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < frame; j++) s += audio[i * frame + j] ** 2;
    db[i] = Math.sqrt(s / frame + 1e-12);
    if (db[i] > max) max = db[i];
  }
  let first = -1, last = -1;
  for (let i = 0; i < n; i++) {
    if (20 * Math.log10(db[i] / (max + 1e-12)) > thresholdDb) { if (first < 0) first = i; last = i; }
  }
  if (first < 0) return audio;
  const pad = Math.trunc(padMs / 20);
  return audio.subarray(Math.max(0, (first - pad) * frame), Math.min(audio.length, (last + 1 + pad) * frame));
}
