// Operator screen: big OK/NOK, start/stop, teach-in assistant, finish batch, reference check.
// Loaded last: also starts the page (view selection + status polling).
// ======================================================== operator view
let view = "simple", wiz = null, sSel = null, sShown = null, sLastId = null, sModelsKey = "";

function setView(v) {
  view = v;
  $("simple").classList.toggle("hidden", v !== "simple");
  $("expert").classList.toggle("hidden", v !== "expert");
  const on = v === "simple" ? $("sLive") : $("live"), off = v === "simple" ? $("live") : $("sLive");
  off.removeAttribute("src"); on.src = "stream.mjpg?" + Date.now();       // only one camera stream at a time
  if (v === "simple") {
    if (setupToken) fetch("api/setup/lock", { method: "POST", headers: { "X-Setup-Token": setupToken, "X-QC-Token": token } });
    setupToken = ""; try { sessionStorage.removeItem("qc-setup"); } catch (e) {}
  }
}
$("btnOperator").onclick = () => setView("simple");
$("sLock").onclick = async () => {
  const c = await (await fetch("api/setup/check", { headers: { "X-Setup-Token": setupToken } })).json();
  if (c.unlocked) { if (c.pin_required === false) setupToken = ""; setView("expert"); return; }
  $("pinInput").value = ""; $("pinModal").classList.remove("hidden"); $("pinInput").focus();
};
$("pinCancel").onclick = () => $("pinModal").classList.add("hidden");
$("pinInput").onkeydown = (e) => { if (e.key === "Enter") $("pinOk").click(); };
$("pinOk").onclick = async () => {
  const r = await api("api/setup/unlock", "POST", { pin: $("pinInput").value }).catch(() => null);
  if (!r) return;
  setupToken = r.token; try { sessionStorage.setItem("qc-setup", r.token); } catch (e) {}
  $("pinModal").classList.add("hidden"); setView("expert");
};

async function sLoadModels(s) {
  const models = await (await fetch("api/models")).json();
  const active = s.model?.slug;
  $("sModel").innerHTML = models.length
    ? (active ? "" : `<option value="">– select part –</option>`) +
      models.map((m) => `<option value="${m.slug}" ${m.slug === active ? "selected" : ""}>${esc(m.name)}</option>`).join("")
    : `<option value="">– no part taught in yet –</option>`;
  window._sModels = models;
}
$("sModel").onchange = async (e) => { if (e.target.value) { await api(`api/models/${e.target.value}/select`, "POST"); sModelsKey = ""; } };

function sShow(r) {
  sShown = r;
  const big = $("sBig");
  if (!r) { big.className = "big"; big.innerHTML = `–<small>${status?.mode === "inspecting" ? "Waiting for the next part" : "Press Start to begin"}</small>`;
            $("sWhy").innerHTML = ""; $("sLastWrap").classList.add("hidden"); $("sWrongRow").classList.add("hidden"); return; }
  const txt = r.status === "NO_PART" ? "NO PART" : r.status;
  big.className = "big " + r.status;
  big.innerHTML = `${txt}<small>${esc(r.model)} · ${r.timestamp.slice(11, 19)}${r.selftest ? " · reference check" : ""}</small>`;
  $("sWhy").innerHTML = plainDefects(r.defects).map((t) => `<li><b>${esc(t)}</b></li>`).join("");
  $("sLastWrap").classList.toggle("hidden", !(r.channels || []).length);
  if ((r.channels || []).length) $("sLast").src = `api/results/${r.image_id}/0/overlay.jpg`;
  const canFb = (r.status === "OK" || r.status === "NOK") && !r.selftest;
  $("sWrongRow").classList.toggle("hidden", !canFb);
  if (canFb) {
    $("sWrong").textContent = r.status === "NOK" ? "Part is actually GOOD" : "Part is actually DEFECTIVE";
    $("sWrong").disabled = r.feedback === "false_alarm" || r.feedback === "missed";
    $("sWrongState").textContent = r.feedback === "false_alarm" || r.feedback === "missed" ? "✓ noted – thank you" : "";
  }
}
$("sWrong").onclick = async () => {
  const r = sShown; if (!r) return;
  const q = r.status === "NOK" ? "This part was rated NOK but is GOOD?\nIt will be used to improve the model."
                               : "This part was rated OK but is DEFECTIVE?\nTake it out of the good parts.";
  if (!confirm(q)) return;
  const res = await api(`api/results/${r.image_id}/feedback`, "POST",
                        { verdict: r.status === "NOK" ? "ok" : "nok", add_reference: r.status === "NOK" });
  r.feedback = res.outcome; sShow(r); toast("Noted – thank you", "ok");
};

const STEP_LABEL = { starting: "starting", "white balance": "white balance", exposure: "brightness", focus: "focus",
                     measuring: "size and position of the part" };
// defects in plain words for the supervisor (grouped, no scores/thresholds)
const PLAIN = { MISSING_HOLE: "Hole missing", BLIND_HOLE: "Hole not drilled through", HOLE_POSITION: "Hole in the wrong place",
  HOLE_SHAPE: "Hole has the wrong shape", EXTRA_HOLE: "Extra hole or cut-out", OUTLINE: "Outline / edge damaged",
  SURFACE: "Surface defect (scratch, dent, stain)", ANOMALY: "Surface defect (scratch, dent, stain)" };
function plainDefects(defects) {
  const n = new Map();
  for (const d of defects) {
    let t = PLAIN[d.kind] || d.label;
    if (d.kind === "HOLE_DIAMETER") t = /\(\s*-/.test(d.detail) ? "Hole too small" : /\(\s*\+/.test(d.detail) ? "Hole too big" : "Wrong hole size";
    n.set(t, (n.get(t) || 0) + 1);
  }
  return [...n].map(([t, k]) => k > 1 && !t.startsWith("Surface") ? `${t} (×${k})` : t);
}

async function sDots() {
  const h = await (await fetch("api/results")).json();
  window._sHist = h;
  $("sDots").innerHTML = h.slice(0, 24).map((r, i) => `<i class="${r.status} ${sSel && sSel.image_id === r.image_id ? "sel" : ""}"
     data-i="${i}" title="${r.timestamp.slice(11, 19)} ${r.status}"></i>`).join("");
}
$("sDots").onclick = (e) => {
  const i = e.target.dataset?.i; if (i === undefined) return;
  sSel = window._sHist[+i]; sShow(sSel); sDots();
  clearTimeout(window._sSelT); window._sSelT = setTimeout(() => { sSel = null; sShow(status?.last_result); sDots(); }, 20000);
};

$("sStart").onclick = async () => {
  if (status.mode === "inspecting" || status.mode === "selftest") return api("api/inspect/stop", "POST");
  const slug = $("sModel").value;
  if (!slug) return toast("Select a part first (or teach in a new one).");
  await api("api/inspect/start", "POST", { model: slug });
};
$("sCapture").onclick = () => api("api/capture", "POST");
$("sCheck").onclick = async () => {
  if (!status.model) return toast("Select a part first.");
  if (status.mode === "inspecting") await api("api/inspect/stop", "POST");
  api("api/selftest/start", "POST");
};
$("sBatch").onclick = async () => {
  const parts = status.counts.OK + status.counts.NOK;
  const label = prompt(`Finish batch?\n${parts} part(s) will be saved as a ZIP file and the counters reset.\n\nBatch name:`,
                       status.model?.name || "batch");
  if (label === null) return;
  if (status.mode === "inspecting" || status.mode === "selftest") await api("api/inspect/stop", "POST");
  const r = await api("api/archives", "POST", { label });
  sSel = null; sShow(null); sDots(); lastImageId = null; toast(`Batch saved: ${r.file}`, "ok");
};

// ---- teach-in wizard (right panel)
$("sTeach").onclick = async () => {
  if (status.mode === "inspecting") {
    if (!confirm("The inspection will be stopped to teach in a new part. Continue?")) return;
    await api("api/inspect/stop", "POST");
  }
  wiz = { step: "name" }; renderWiz(status);
};
function renderWiz(s) {
  const el = $("sWiz");
  // keep the wizard in sync with the machine state (e.g. after a page reload)
  if (s.mode === "learning" && (!wiz || wiz.step === "name" || wiz.step === "camera")) wiz = { step: "feed", name: s.learning.name };
  if (s.mode === "camera_setup" && !wiz) wiz = { step: "camera", name: "new part", started: true };
  if (s.mode === "training" && wiz?.step !== "train") wiz = { step: "train", name: s.learning.name || wiz?.name, seen: true };
  if (wiz?.step === "train" && s.mode === "learning") wiz = { step: "feed", name: wiz.name, failed: true, seen: true };
  if (wiz && ["learning", "training"].includes(s.mode)) wiz.seen = true;
  if (wiz?.seen && ["train", "feed"].includes(wiz.step) && s.mode === "idle")      // finished (or cancelled elsewhere)
    wiz = s.model && s.model.name === wiz.name && !s.learning.count ? { step: "done", name: wiz.name } : null;
  const on = !!wiz;
  $("sRun").classList.toggle("hidden", on); el.classList.toggle("hidden", !on);
  if (!on) { el.dataset.step = ""; return; }
  const L = s.learning;
  if (wiz.step === "name" && !el.dataset.step?.startsWith("name")) {
    el.innerHTML = `<h2>Teach in a new part – step 1 of ${s.camera_controls ? 3 : 2}</h2>
      <p>Enter a name for the part:</p>
      <input type="text" id="wName" placeholder="e.g. triangle plate 3 holes" autocomplete="off">
      <p class="s-note">Then only <b>good</b> parts go over the conveyor – no defective ones.</p>
      <div class="bigbtns"><button id="wCancel">Cancel</button><button class="go" id="wNext">Next →</button></div>`;
    $("wName").focus();
    $("wName").onkeydown = (e) => { if (e.key === "Enter") $("wNext").click(); };
    $("wCancel").onclick = () => { wiz = null; el.dataset.step = ""; renderWiz(status); };
    $("wNext").onclick = async () => {
      const name = $("wName").value.trim();
      if (!name) return toast("Please enter a name.");
      if ((window._sModels || []).some((m) => m.name.toLowerCase() === name.toLowerCase()) &&
          !confirm(`A part called “${name}” already exists. Replace it?`)) return;
      if (status.camera_controls) { wiz = { step: "camera", name }; renderWiz(status); return; }
      await api("api/learn/start", "POST", { name, auto_finish: true });
      wiz = { step: "feed", name }; renderWiz(status);
    };
  } else if (wiz.step === "camera") {
    const cs = s.camera_setup;
    const state = s.mode === "camera_setup" ? "running" : !wiz.started || !cs ? "ready" : cs.error ? "failed" : "done";
    if (el.dataset.step !== "camera-" + state) {
      const r = cs?.result;
      const STEP = { "white balance": "white balance", exposure: "brightness", focus: "focus", measuring: "size and position" };
      el.innerHTML = `<h2>Teach in “${esc(wiz.name)}” – step 2 of 3: camera</h2>` + ({
        ready: `<p>Stop the belt and lay <b>one good part</b> flat under the camera, roughly in the middle of the picture.</p>
          <p class="s-note">The camera then sets brightness, white balance${s.camera_controls ? " and focus (if it has a focus motor)" : ""} by itself.
          Zoom and belt direction follow with the first part that runs.</p>
          <div class="bigbtns"><button class="go wide" id="wCam">Adjust camera automatically</button>
          <button id="wCancel">Cancel</button><button id="wSkip">Skip</button></div>`,
        running: `<p>Adjusting the camera … please do not move the part.</p><div class="progress"><i style="width:100%"></i></div>
          <p class="s-note" id="wCamStep"></p>`,
        failed: `<div class="notices"><div class="error">${esc(cs?.error || "")}</div></div>
          <div class="bigbtns"><button class="go wide" id="wCam">Try again</button><button id="wCancel">Cancel</button><button id="wSkip">Skip</button></div>`,
        done: `<div class="big OK" style="font-size:30px">✓ CAMERA ADJUSTED</div>
          <ul class="why" style="font-size:15px">
            <li>Brightness: exposure ${r?.exposure_ms} ms${r?.gain > 1.01 ? `, gain ${r.gain.toFixed(1)}` : ""}</li>
            ${r?.autofocus ? `<li>Focus: ${r.lens_position ? "set automatically" : "<b>not found</b>"}</li>` :
              `<li>Sharpness: ${r?.edge_width_px == null ? "–" : r.edge_width_px <= 3 ? "good" : "<b>blurry – focus the lens once</b>"}</li>`}
          </ul>
          ${(r?.notes || []).map((n) => `<div class="notices"><div class="warn">⚠ ${esc(n)}</div></div>`).join("")}
          <p>Now start the belt and let <b>${s.learning.target} good parts</b> run. The first part measures the belt direction
             and zooms in – it is not counted.</p>
          <div class="bigbtns"><button class="go wide" id="wNext2">Next →</button><button class="wide" id="wCam">Adjust again</button></div>`,
      }[state]);
      $("wCam") && ($("wCam").onclick = async () => { wiz.started = true; await api("api/camera-setup/start", "POST"); });
      $("wSkip") && ($("wSkip").onclick = async () => {
        await api("api/camera-setup/cancel", "POST");
        await api("api/learn/start", "POST", { name: wiz.name, auto_finish: true }); wiz = { step: "feed", name: wiz.name }; renderWiz(status); });
      $("wCancel") && ($("wCancel").onclick = async () => { await api("api/camera-setup/cancel", "POST"); wiz = null; renderWiz(status); });
      $("wNext2") && ($("wNext2").onclick = async () => {
        await api("api/learn/start", "POST", { name: wiz.name, auto_finish: true }); wiz = { step: "feed", name: wiz.name }; renderWiz(status); });
      el.dataset.step = "camera-" + state;
    }
    if (state === "running" && $("wCamStep")) $("wCamStep").textContent = "Now: " + (STEP_LABEL[cs?.step] || cs?.step || "");
    return;
  } else if (wiz.step === "feed") {
    const pct = Math.min(100, 100 * L.count / L.target);
    if (el.dataset.step !== "feed") {
      el.innerHTML = `<h2>Teach in “${esc(wiz.name)}” – ${s.camera_controls ? "step 3 of 3" : "step 2 of 2"}</h2>
        <div id="wPath" class="notices"></div>
        <div id="wFailed" class="notices"></div>
        <p>Now run <b>${L.target} good parts</b> over the conveyor, each one placed a bit differently.
           Every part is captured automatically in the middle of the picture.</p>
        <div class="count" id="wCount"></div>
        <div class="progress"><i id="wBar"></i></div>
        <div class="thumbs" id="wThumbs" title="Tap a picture to remove it (e.g. a defective part slipped in)"></div>
        <p class="s-note">The model is created automatically after ${L.target} parts. Wrong part captured? Tap its picture to remove it.</p>
        <div class="bigbtns"><button id="wCancel">Cancel</button><button id="wFinish">Finish now</button></div>`;
      $("wCancel").onclick = async () => { if (confirm("Cancel teaching in? The captured pictures are lost.")) { await api("api/learn/cancel", "POST"); wiz = null; renderWiz(status); } };
      $("wFinish").onclick = () => api("api/learn/finish", "POST", {});
      $("wThumbs").onclick = (e) => { if (e.target.dataset.i !== undefined) api("api/learn/undo", "POST", { index: +e.target.dataset.i }).then(() => $("wThumbs").innerHTML = ""); };
    }
    const err = (s.messages.find((m) => m.level === "error") || {}).text || "";
    $("wFailed").innerHTML = wiz.failed ? `<div class="error">Creating the model failed: ${esc(err)}<br>Run more good parts and press “Finish now”, or cancel.</div>` : "";
    $("wPath").innerHTML = L.path_pending ? `<div class="warn">▶ Start the belt. The first part measures the belt direction
      and zooms in – it is not counted. (Belt stopped? Press “Capture now”.)</div>` : "";
    $("wCount").innerHTML = `${L.count} <span>/ ${L.target}</span>`;
    $("wBar").style.width = pct + "%";
    $("wFinish").disabled = L.count < L.minimum || s.mode !== "learning";
    $("wFinish").textContent = L.count < L.minimum ? `Finish now (min. ${L.minimum})` : "Finish now";
    const th = $("wThumbs");
    if (th.children.length !== L.count)
      th.innerHTML = Array.from({ length: L.count }, (_, i) => `<img src="api/learn/thumb/${i}.jpg?v=${L.count}" data-i="${i}">`).join("");
  } else if (wiz.step === "train" && el.dataset.step !== "train") {
    el.innerHTML = `<h2>Creating the model …</h2><p>Please wait a moment.</p><div class="progress"><i style="width:100%"></i></div>`;
  } else if (wiz.step === "done" && el.dataset.step !== "done") {
    el.innerHTML = `<div class="big OK" style="font-size:40px">✓ READY<small>“${esc(wiz.name)}” has been taught in</small></div>
      <p>Tip: check it once with a known good and a known defective part (“Reference part check”).</p>
      <div class="bigbtns"><button class="go wide" id="wGo">▶ Start inspection</button><button class="wide" id="wClose">Close</button></div>`;
    $("wGo").onclick = async () => { await api("api/inspect/start", "POST", {}); wiz = null; sModelsKey = ""; renderWiz(status); };
    $("wClose").onclick = () => { wiz = null; sModelsKey = ""; renderWiz(status); };
  }
  el.dataset.step = wiz.step;
}

function renderSimple(s) {
  const busy = { camera_setup: "Adjusting camera", learning: "Teaching in", training: "Creating model", selftest: "Reference check" }[s.mode];
  $("sState").textContent = s.mode === "inspecting" ? "● Inspection running" : busy || "Stopped";
  $("sState").className = "s-state " + (s.mode === "inspecting" ? "run" : busy ? "busy" : "");
  const notes = [...(s.notices || [])];
  if (s.mode === "selftest" && s.selftest.active)
    notes.unshift({ level: "warn", text: s.selftest.active.step === "good" ? "Reference check 1/2: put the known GOOD reference part on the conveyor."
                                                                             : "Reference check 2/2: put the known DEFECTIVE reference part on the conveyor." });
  $("sNotices").innerHTML = notes.map((n) => `<div class="${n.level}">${n.level === "info" ? "ℹ" : "⚠"} ${esc(n.text)}</div>`).join("");
  $("sModel").disabled = s.mode !== "idle";
  const running = s.mode === "inspecting" || s.mode === "selftest";
  $("sStart").textContent = running ? (s.mode === "selftest" ? "■ Cancel reference check" : "■ Stop inspection") : "▶ Start inspection";
  $("sStart").className = (running ? "stop" : "go") + " wide";
  $("sStart").disabled = !running && (s.mode !== "idle" || !s.model);
  $("sTeach").disabled = !["idle", "inspecting"].includes(s.mode);
  $("sCheck").disabled = s.mode !== "idle" && s.mode !== "inspecting" || !s.model;
  $("sBatch").disabled = ["training", "learning"].includes(s.mode) || !(s.counts.OK + s.counts.NOK);
  const active = ["learning", "inspecting", "selftest"].includes(s.mode);
  $("sCapture").classList.toggle("hidden", !active);
  $("sTrig").textContent = !active ? "" : s.continuous ? (s.armed ? "Waiting for the next part …" : "Part captured") : "";
  const c = s.counts, total = c.OK + c.NOK;
  $("sOK").textContent = c.OK; $("sNOK").textContent = c.NOK;
  $("sRate").textContent = total ? `${(100 * c.NOK / total).toFixed(1)} %` : "–";
  const m = s.model, st = s.settings;
  $("sLearnInfo").textContent = !m || !st.self_learning ? "" : s.self_learning_running ? "⟳ The model is improving itself right now – inspection continues."
    : st.self_learning_auto ? `The model improves itself with good parts (${m.collected}/${st.self_learning_every} collected).` : "";
  const key = (m?.slug || "") + s.mode + (m?.learned ?? "");
  if (key !== sModelsKey) { sModelsKey = key; sLoadModels(s); }
  if (s.last_result && s.last_result.image_id !== sLastId) {
    sLastId = s.last_result.image_id; if (!sSel) sShow(s.last_result); sDots();
  } else if (!s.last_result && sLastId !== null) { sLastId = null; sSel = null; sShow(null); sDots(); }
  else if (!s.last_result && !sShown) sShow(null);
  renderWiz(s);
}

async function poll() {
  try {
    const s = await (await fetch("api/status")).json();
    render(s); renderSimple(s);
  } catch (e) { console.error(e); $("mode").textContent = "offline"; $("sState").textContent = "offline"; }
  setTimeout(poll, 700);
}
(async () => {
  opts = await (await fetch("api/options")).json();
  methodSettings();
  let unlocked = false;
  if (setupToken) unlocked = (await (await fetch("api/setup/check", { headers: { "X-Setup-Token": setupToken } })).json()).unlocked;
  setView(unlocked ? "expert" : "simple");
  poll();
})();
