// On-device scoring with the INT8 ONNX model via onnxruntime-web (WASM).
// Everything here runs in the browser and works offline once cached.
import * as ort from "onnxruntime-web/wasm";

import { logMel, N_MELS, trimSilence } from "./core/features.js";
import { scoreUtterance } from "./core/gop.js";

ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`;
ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation

export class OnDeviceScorer {
  static async load(base = `${import.meta.env.BASE_URL}models/`) {
    const meta = await fetch(`${base}model_meta.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error("no model"))));
    const calibration = await fetch(`${base}calibration.json`).then((r) => (r.ok ? r.json() : undefined)).catch(() => undefined);
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(`${base}vaani_int8.onnx`, { executionProviders: ["wasm"] });
    return new OnDeviceScorer(session, meta, calibration, performance.now() - t0);
  }

  constructor(session, meta, calibration, loadMs) {
    Object.assign(this, { session, meta, calibration, loadMs });
  }

  async assess(audio16k, words) {
    const t0 = performance.now();
    const { data, frames } = logMel(trimSilence(audio16k));
    const out = await this.session.run({ feats: new ort.Tensor("float32", data, [1, frames, N_MELS]) });
    const logits = out.logits;
    const [, T, C] = logits.dims;
    const report = scoreUtterance(logits.data, T, C, words,
      { tokens: this.meta.tokens, ipa: this.meta.ipa, calibration: this.calibration });
    report.timings = { device_ms: Math.round(performance.now() - t0) };
    return report;
  }
}
