// Microphone capture with a simple energy VAD: recording stops by itself
// ~1.2 s after the learner finishes speaking. Output is 16 kHz mono Float32.
import { resampleLinear } from "./core/wav.js";

const WORKLET = `
class Capture extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor("vaani-capture", Capture);`;

export async function record({ onLevel, maxSeconds = 15, silenceMs = 1200, signal } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "vaani-capture");
  src.connect(node);

  const chunks = [];
  let total = 0, speech = false, quietFor = 0, noise = 0.003;
  const sr = ctx.sampleRate;

  await new Promise((resolve) => {
    const stop = () => { node.port.onmessage = null; resolve(); };
    signal?.addEventListener("abort", stop, { once: true });
    node.port.onmessage = ({ data }) => {
      chunks.push(data);
      total += data.length;
      let s = 0;
      for (let i = 0; i < data.length; i++) s += data[i] * data[i];
      const rms = Math.sqrt(s / data.length);
      if (!speech) noise = 0.95 * noise + 0.05 * rms;
      const loud = rms > Math.max(0.012, noise * 4);
      onLevel?.(Math.min(1, rms * 8), loud);
      const ms = (data.length / sr) * 1000;
      if (loud) { speech = true; quietFor = 0; } else if (speech) quietFor += ms;
      if ((speech && quietFor > silenceMs) || total > maxSeconds * sr) stop();
    };
  });

  stream.getTracks().forEach((t) => t.stop());
  src.disconnect();
  await ctx.close();
  const raw = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { raw.set(c, off); off += c.length; }
  return { audio: await to16k(raw, sr), heardSpeech: speech };
}

async function to16k(raw, sr) {
  if (sr === 16000) return raw;
  if (typeof OfflineAudioContext === "undefined") return resampleLinear(raw, sr);
  const off = new OfflineAudioContext(1, Math.ceil((raw.length * 16000) / sr), 16000);
  const buf = off.createBuffer(1, raw.length, sr);
  buf.copyToChannel(raw, 0);
  const s = off.createBufferSource();
  s.buffer = buf;
  s.connect(off.destination);
  s.start();
  return (await off.startRendering()).getChannelData(0);
}
