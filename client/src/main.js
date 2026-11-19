import allDrills from "@shared/drills.json";
import profiles from "@shared/l1_profiles.json";
import phonemeInfo from "@shared/phonemes.json";

import { attachFeedback } from "./core/feedback.js";
import { encodeWav } from "./core/wav.js";
import { api, connectSync, deviceId, flushQueue, queue, session, submitAttempt } from "./net.js";
import { OnDeviceScorer } from "./ondevice.js";
import { record } from "./recorder.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) n.append(k);
  return n;
};
const scoreClass = (s) => (s >= 80 ? "good" : s >= 60 ? "ok" : "bad");

const state = { user: null, drills: [], current: null, scorer: null, sync: null, recording: null, signup: false };

// ---------------- boot ----------------
if ("serviceWorker" in navigator && import.meta.env.PROD) navigator.serviceWorker.register("/sw.js");

OnDeviceScorer.load()
  .then((s) => { state.scorer = s; setPill($("model-status"), `on-device ✓ (${Math.round(s.loadMs)} ms load)`, "on"); })
  .catch(() => setPill($("model-status"), "cloud-only (no on-device model)", "off"));

window.addEventListener("online", () => syncQueue());
boot();

async function boot() {
  if (!session.token) return show("auth");
  try {
    state.user = await api("/api/me");
  } catch (err) {
    if (err.status === 401) { session.token = null; return show("auth"); }
    state.user = { name: "offline", l1: "other" }; // offline start: keep practising locally
  }
  startSession();
}

function startSession() {
  $("nav").hidden = false;
  $("logout").hidden = false;
  state.sync = connectSync(onSyncMessage, (on) => setPill($("sync-status"), on ? "live sync" : "offline", on ? "on" : "off"));
  syncQueue();
  loadNextDrills();
  show("practice");
}

function show(view) {
  for (const v of ["auth", "practice", "progress"]) $(`view-${v}`).hidden = v !== view;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  if (view === "progress") loadProgress();
}
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => show(t.dataset.view)));

function setPill(node, text, cls) {
  node.textContent = text;
  node.className = `pill ${cls ?? ""}`;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 3500);
}

// ---------------- auth ----------------
$("auth-toggle").onclick = () => {
  state.signup = !state.signup;
  $("name-row").hidden = $("l1-row").hidden = !state.signup;
  $("auth-submit").textContent = state.signup ? "Create account" : "Sign in";
  $("auth-toggle").textContent = state.signup ? "Have an account? Sign in" : "New here? Create an account";
};
$("auth-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = Object.fromEntries(new FormData(e.target));
  try {
    const res = state.signup
      ? await api("/api/auth/register", { method: "POST", body: f })
      : await api("/api/auth/login", { method: "POST", body: { email: f.email, password: f.password } });
    session.token = res.token;
    state.user = res.user;
    $("auth-error").hidden = true;
    startSession();
  } catch (err) {
    $("auth-error").textContent = err.message;
    $("auth-error").hidden = false;
  }
};
$("logout").onclick = () => { session.token = null; state.sync?.close(); location.reload(); };

// ---------------- drills ----------------
async function loadNextDrills() {
  try {
    state.drills = await api("/api/drills/next?count=5");
  } catch {
    // offline: rotate through the bundled drill bank
    const start = Math.floor(Math.random() * allDrills.length);
    state.drills = [...allDrills.slice(start), ...allDrills.slice(0, start)].slice(0, 5)
      .map((d) => ({ ...d, targets: d.focus }));
  }
  nextDrill();
}

function nextDrill(broadcast = false) {
  if (!state.drills.length) return loadNextDrills();
  openDrill(state.drills.shift(), broadcast);
}

function openDrill(drill, broadcast) {
  state.current = drill;
  $("sentence").textContent = drill.text;
  $("phones").textContent = drill.words.map((w) => `/${w.phones.map((p) => phonemeInfo.ipa[p]).join("")}/`).join(" ");
  const chips = $("targets");
  chips.replaceChildren(...(drill.targets ?? drill.focus ?? []).map((p) =>
    el("span", { className: "chip", textContent: `${p} /${phonemeInfo.ipa[p]}/` })));
  $("result").hidden = true;
  $("handoff").hidden = true;
  if (broadcast) state.sync?.send({ type: "session", state: { drill: { id: drill.id, text: drill.text, words: drill.words, targets: drill.targets } } });
}
$("next").onclick = () => nextDrill(true);

$("custom-form").onsubmit = async (e) => {
  e.preventDefault();
  const text = new FormData(e.target).get("text").trim();
  if (!text) return;
  try {
    const { words, oov } = await api(`/api/phonemize?${new URLSearchParams({ text })}`);
    if (!words.length) return toast("None of those words are in the dictionary.");
    if (oov.length) toast(`Skipping words I can't score: ${oov.join(", ")}`);
    openDrill({ id: null, text, words, targets: [] }, true);
  } catch {
    toast("Custom sentences need a connection (the dictionary lives on the server).");
  }
};

// ---------------- recording + scoring ----------------
$("record").onclick = async () => {
  if (state.recording) return state.recording.abort();
  const ctrl = new AbortController();
  state.recording = ctrl;
  $("record").classList.add("on");
  $("rec-hint").textContent = "Listening…";
  let result;
  try {
    result = await record({ signal: ctrl.signal, onLevel: (lvl, loud) => {
      $("level").style.width = `${lvl * 100}%`;
      $("level").classList.toggle("loud", loud);
    } });
  } catch (err) {
    toast(`Microphone unavailable: ${err.message}`);
  } finally {
    state.recording = null;
    $("record").classList.remove("on");
    $("level").style.width = "0";
    $("rec-hint").textContent = "Tap and read the sentence aloud. Recording stops when you pause.";
  }
  if (!result) return;
  if (!result.heardSpeech) return toast("I didn't hear anything. Check your microphone and try again.");
  await scoreAttempt(result.audio);
};

async function scoreAttempt(audio) {
  const drill = state.current;
  const clientAttemptId = crypto.randomUUID();
  const createdAt = new Date().toISOString();
  const wav = encodeWav(audio);
  let deviceReport = null;

  if (state.scorer) {
    try {
      deviceReport = attachFeedback(await state.scorer.assess(audio, drill.words), state.user?.l1 ?? "other", profiles);
      if (deviceReport.ok) renderResult(deviceReport);
    } catch (err) {
      console.warn("on-device scoring failed", err);
    }
  }
  const item = { clientAttemptId, drillId: drill.id, text: drill.text, wav, deviceReport, createdAt };

  if (!navigator.onLine) {
    await queue.add(item);
    if (!deviceReport?.ok) toast("Saved. You'll get your score when you're back online.");
    return;
  }
  if (deviceReport?.ok) $("result-extra").textContent = "Checking with the cloud…";
  try {
    const { report } = await submitAttempt(item);
    if (state.current === drill) renderResult(report);
  } catch (err) {
    if (!err.status || err.status >= 500) {
      await queue.add(item);
      $("result-extra").textContent = "Cloud check queued; it'll sync automatically.";
    } else {
      toast(err.message);
    }
  }
}

function renderResult(r) {
  $("result").hidden = false;
  const acc = Math.round(r.overall.accuracy);
  $("score").textContent = acc;
  $("score-ring").style.setProperty("--p", acc);
  $("score-ring").className = `score-ring ${scoreClass(acc)}`;
  const label = { device: "On-device", local: "Server model", azure: "Azure", fused: "Cloud-verified (Azure + model)" }[r.source] ?? r.source;
  setPill($("result-source"), label, r.source === "device" ? "" : "on");
  const extra = [];
  if (r.timings?.device_ms != null) extra.push(`scored on this device in ${r.timings.device_ms} ms`);
  if (r.overall.fluency != null) extra.push(`fluency ${Math.round(r.overall.fluency)}`);
  if (r.overall.completeness != null) extra.push(`completeness ${Math.round(r.overall.completeness)}`);
  $("result-extra").textContent = extra.join(" · ");

  $("phone-detail").hidden = true;
  $("scored-words").replaceChildren(...r.words.flatMap((w) => {
    // a word is only as good as its worst sound: one wrong phone shouldn't hide behind a good average
    const shown = Math.min(w.score, ...w.phones.map((p) => p.score + 20));
    const b = el("button", { className: `word ${scoreClass(shown)}`, textContent: w.word, title: `${Math.round(w.score)}` });
    b.onclick = () => showPhones(w);
    return [b, " "];
  }));
  $("tips").replaceChildren(...(r.feedback ?? []).map((t) => el("li", {},
    el("b", { textContent: `/${phonemeInfo.ipa[t.phone]}/ in “${t.word}”` }),
    t.heard ? el("span", { className: "muted", textContent: ` sounded like /${phonemeInfo.ipa[t.heard]}/` }) : "",
    el("p", { textContent: t.tip }))));
  if (!r.feedback?.length) $("tips").append(el("li", { className: "muted", textContent: "No problem sounds this time. Nice work." }));
}

function showPhones(w) {
  const d = $("phone-detail");
  d.hidden = false;
  d.replaceChildren(el("b", { textContent: w.word }), ...w.phones.map((p) => el("span",
    { className: `ph ${scoreClass(p.score)}`, title: `${p.phone}: ${Math.round(p.score)}${p.heard ? ` (heard ${p.heard})` : ""}`,
      textContent: `/${p.ipa}/ ${Math.round(p.score)}` })));
}

// ---------------- sync ----------------
async function syncQueue() {
  if (!navigator.onLine || !session.token) return;
  await flushQueue((item, res) => {
    toast(`Synced an offline attempt: ${Math.round(res.report.overall.accuracy)}`);
    if (state.current?.text === item.text) renderResult(res.report);
  });
  $("queued").textContent = (await queue.all()).length;
}

function onSyncMessage(msg) {
  if (msg.type === "hello") syncQueue();
  if (msg.type === "attempt") {
    toast(`${msg.attempt.source === "device" ? "New attempt" : "Scored"} on another device: ${Math.round(msg.attempt.overall ?? 0)}`);
    if (!$("view-progress").hidden) loadProgress();
  }
  if (msg.type === "session" && msg.state?.drill) {
    const d = msg.state.drill;
    const h = $("handoff");
    h.replaceChildren(el("span", { textContent: `Continue on this device: “${d.text}”` }),
      Object.assign(el("button", { className: "primary small", textContent: "Continue here" }),
        { onclick: () => openDrill(d, false) }));
    h.hidden = false;
  }
  if (msg.type === "presence") toast(`${msg.devices} devices connected`);
}

// ---------------- progress ----------------
async function loadProgress() {
  $("queued").textContent = (await queue.all()).length;
  let p;
  try { p = await api("/api/progress"); } catch { return toast("Progress needs a connection."); }
  $("streak").textContent = p.streak;
  $("days").textContent = p.practiceDays;
  $("phone-bars").replaceChildren(...p.stats.slice(0, 15).map((s) => el("div", { className: "bar" },
    el("span", { className: "label", textContent: `/${phonemeInfo.ipa[s.phone]}/ ${s.phone}` }),
    Object.assign(el("div", { className: "track" }, Object.assign(el("div", { className: `fill ${scoreClass(s.ema)}` }), { style: `width:${s.ema}%` }))),
    el("span", { className: "val", textContent: `${Math.round(s.ema)}${s.dueAt <= Date.now() ? " · due" : ""}` }))));
  if (!p.stats.length) $("phone-bars").append(el("p", { className: "muted", textContent: "Record a few sentences to see your sounds here." }));
  $("recent").replaceChildren(...p.recent.map((a) => el("li", {},
    el("span", { className: `dot ${scoreClass(a.overall ?? 0)}`, textContent: Math.round(a.overall ?? 0) }),
    el("span", { textContent: a.text }),
    el("small", { className: "muted", textContent: ` ${a.source} · ${new Date(a.createdAt).toLocaleString()}` }))));
}

console.info("Vaani device", deviceId);
