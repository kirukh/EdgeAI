// Setup area (PIN): live view with scores, history, models, settings, calibration, GPIO.
// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll("[data-pane]").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== b.dataset.tab));
  try { localStorage.setItem("qc-tab", b.dataset.tab); } catch (e) {}
});
try { const t = localStorage.getItem("qc-tab"); if (t) document.querySelector(`.tabs button[data-tab="${t}"]`)?.click(); } catch (e) {}

// ------------------------------------------------------ method settings
function methodSettings() {
  const d = opts.defaults;
  const checks = Object.entries(opts.methods).map(([k, v]) =>
    `<label class="check"><input type="checkbox" name="method" value="${k}" ${d.methods.includes(k) ? "checked" : ""}> ${esc(v)}</label>`).join("");
  const sel = (id, cur) => `<select id="${id}">` + Object.entries(opts.representations).map(([k, v]) =>
    `<option value="${k}" ${k === cur ? "selected" : ""}>${esc(v)}</option>`).join("") + "</select>";
  $("methodSettings").innerHTML = checks +
    `<label>Representation for reference comparison ${sel("diffRep", d.diff_representation)}</label>` +
    `<label>Representation for ML anomaly detection ${sel("pcaRep", d.pca_representation)}</label>` +
    `<p class="hint">Decision: ${opts.decision === "fusion" ? "fusion – a method decides alone only above its level (ML ×" +
      opts.standalone.pca + "), otherwise two methods must agree" : "OR – any method ≥ 1 → NOK"}. In dual-light mode the backlight channel always uses geometry.</p>`;
}
function currentMethodSettings() {
  return { methods: [...document.querySelectorAll('input[name=method]:checked')].map((e) => e.value),
           diff_representation: $("diffRep").value, pca_representation: $("pcaRep").value };
}

// ---------------------------------------------------------------- result
function showResult(r) {
  if (!r) return;
  shownResult = r;
  $("resBadge").className = "badge " + r.status;
  $("resBadge").textContent = r.status === "NO_PART" ? "no part" : r.status;
  $("resTitle").textContent = `Model: ${r.model}` + (r.selftest ? ` · self-test (${r.selftest} part)` : "");
  const gt = r.ground_truth ? ` · Expected: ${r.ground_truth}` : "";
  $("resMeta").textContent = `#${r.image_id} · ${r.timestamp.replace("T", " ").slice(0, 19)} · ${r.time_ms} ms${gt}`;
  $("resShots").innerHTML = (r.shots || []).length > 1 ? r.shots.map((s) => `<span class="${s}">${s === "NO_PART" ? "–" : s}</span>`).join("") : "";
  const chans = r.channels && r.channels.length ? r.channels : [];
  $("resImgs").innerHTML = chans.map((c, i) => `<div class="imgs">
      ${chans.length > 1 ? `<div class="cap">${esc(CH[c] || c)}</div>` : ""}
      <img src="api/results/${r.image_id}/${i}/overlay.jpg" alt=""><img src="api/results/${r.image_id}/${i}/detail.jpg" alt=""></div>`).join("");
  $("resDefects").innerHTML = r.defects.map((d) =>
    `<li><b>${esc(d.label)}</b> ${d.channel ? `<span class="ch">${esc(CH[d.channel] || d.channel)}</span>` : ""}
      <span class="d">– ${esc(d.detail)} (${esc(opts.methods[d.method] || d.method)})</span></li>`).join("");
  $("resScores").innerHTML = r.methods.map((m) => {
    const w = Math.min(100, m.score * 50);
    const cls = m.decisive ? "bad" : m.score >= 1 ? "over" : "";
    const marks = `<b style="left:50%"></b>` + (m.standalone > 1 ? `<b style="left:${Math.min(99, m.standalone * 50)}%"></b>` : "");
    const note = m.score >= 1 && !m.decisive ? `<div class="note">below its stand-alone level ${m.standalone} – not confirmed by a 2nd method</div>` : "";
    return `<span>${esc(m.label)}${m.channel ? ` <span class="ch">${esc(CH[m.channel] || m.channel)}</span>` : ""}${note}</span>
      <div class="bar"><i class="${cls}" style="width:${w}%"></i>${marks}</div><span>${m.score.toFixed(2)}</span>`;
  }).join("");
  // feedback
  const canFb = (r.status === "OK" || r.status === "NOK") && !r.selftest;
  $("feedback").classList.toggle("hidden", !canFb);
  if (canFb) {
    $("fbWrong").textContent = r.status === "NOK" ? "✗ Was OK – false alarm" : "✗ Was NOK – missed defect";
    $("fbRefWrap").classList.toggle("hidden", r.status !== "NOK");
    const f = FB[r.feedback];
    $("fbState").textContent = f ? f[0] + (r.reference_added ? " · queued as reference" : "") : "";
    $("fbState").className = "fb " + (f ? f[1] : "");
  }
}

async function sendFeedback(correct) {
  const r = shownResult; if (!r) return;
  const verdict = correct ? r.status.toLowerCase() : (r.status === "NOK" ? "ok" : "nok");
  const addRef = !correct && r.status === "NOK" && $("fbRef").checked;
  const res = await api(`api/results/${r.image_id}/feedback`, "POST", { verdict, add_reference: addRef });
  r.feedback = res.outcome; if (addRef) r.reference_added = true;
  showResult(r); loadHistory(); lastModelsKey = "";
  if (res.pending) toast(`Queued as reference (${res.pending} pending) – retrain the model in the Models tab`, "ok");
}
$("fbCorrect").onclick = () => sendFeedback(true);
$("fbWrong").onclick = () => sendFeedback(false);

// ---------------------------------------------------------------- render
function render(s) {
  status = s;
  $("mode").textContent = MODE[s.mode] || s.mode;
  $("mode").className = "pill " + s.mode;
  $("cam").textContent = `${s.camera} · ${s.fps} fps`;
  $("activeModel").textContent = s.model ? `Model: ${s.model.name}` : "no model loaded";
  const cal = s.calibration;
  $("calPill").textContent = cal.calibrated ? `${cal.mm_per_px} mm/px` : "not calibrated (px)";
  $("calPill").className = "pill " + (cal.calibrated ? "good" : "");
  const st = s.selftest.last;
  const today = new Date().toISOString().slice(0, 10);
  if (!st) { $("stPill").textContent = "no self-test"; $("stPill").className = "pill warn"; }
  else {
    const t = st.time.replace("T", " ").slice(0, 16);
    $("stPill").textContent = `self-test ${st.passed ? "passed" : "FAILED"} ${st.time.slice(0, 10) === today ? t.slice(11) : t}`;
    $("stPill").className = "pill " + (!st.passed ? "bad" : st.time.slice(0, 10) === today ? "good" : "warn");
  }
  const dw = s.drift.warnings;
  $("driftPill").classList.toggle("hidden", !dw.length);
  $("driftPill").textContent = `Drift: ${dw.length} warning${dw.length > 1 ? "s" : ""}`;
  $("driftPill").className = "pill warn" + (dw.length ? "" : " hidden");
  const banner = [...s.model_warnings, ...dw.map((w) => "Image quality drift: " + w + " – check lighting/camera.")];
  $("banner").innerHTML = banner.map((b) => `<div>⚠ ${esc(b)}</div>`).join("");
  $("banner").classList.toggle("hidden", !banner.length);

  $("triggerHint").textContent = s.continuous
    ? (s.mode === "idle" ? "Trigger inactive (start learning or inspection mode first)" : s.armed ? "Trigger armed – waiting for the next part" : "Part captured – waiting until it leaves the image")
    : "Image folder: automatic in inspection mode, click to capture in learning mode";
  $("btnCapture").disabled = !["learning", "inspecting", "selftest"].includes(s.mode);

  const c = s.counts, total = c.OK + c.NOK, f = s.feedback;
  $("cOK").textContent = c.OK; $("cNOK").textContent = c.NOK;
  $("cRate").textContent = total ? `${(100 * c.NOK / total).toFixed(1)} %` : "–";
  const rev = f.confirmed + f.false_alarm + f.missed;
  $("fbStats").textContent = rev ? `Supervisor review of ${rev} result(s): ${f.confirmed} confirmed, ${f.false_alarm} false alarm(s), ` +
    `${f.missed} missed defect(s) → ${(100 * f.confirmed / rev).toFixed(0)} % correct` : "No supervisor review yet – use “Supervisor check” below a result.";

  $("shotsHint").textContent = `Always ${s.settings.shots_per_part} images per part with majority vote (fixed). `
    + "The further images are only evaluated when the first one is not clearly good.";
  // setup
  if (document.activeElement?.closest("[data-pane=setup]") == null) {
    $("setLight").value = s.settings.lighting_mode;
    $("setSelfLearn").checked = s.settings.self_learning;
    $("setSelfAuto").checked = s.settings.self_learning_auto;
  }
  $("setLight").disabled = s.mode !== "idle";
  $("selfLearnHint").textContent = `Only parts with every method below ${Math.round(s.settings.self_learning_margin * 100)} % of its threshold, `
    + `max. ${s.settings.self_learning_max_pool} per model, not while a drift warning is active. `
    + (s.settings.self_learning_auto ? `Trained in automatically every ${s.settings.self_learning_every} parts (safety checks; rejected pools are discarded).`
                                     : "Trained in only with the button in the Models tab.");
  renderCalibration(cal, s);
  const cst = s.camera_state || {};
  $("camInfo").textContent = (s.camera_info || s.camera) + (cst.exposure_us ? ` · now: exposure ${(cst.exposure_us / 1000).toFixed(2)} ms, gain ${(+cst.gain).toFixed(2)}` : "")
    + ` · zoom ${cst.zoom}× · belt axis ${cst.axis}` + (cst.auto ? " (automatic camera profile of this part)" : "");
  const fa = s.focus_assist;
  $("btnFocus").textContent = fa ? "Focus assistant off" : "Focus assistant on";
  $("btnFocus").className = fa ? "" : "primary";
  $("focusVal").textContent = fa ? `sharpness ${fa.value.toFixed(0)} · best ${fa.best.toFixed(0)} · ${fa.best ? Math.round(100 * fa.value / fa.best) : 0} %` : "";
  $("driftRows").innerHTML = s.drift.rows.length ? s.drift.rows.map((r) => `<tr><td>${esc(r.label)}</td><td>${r.reference}</td>
      <td>${r.current}</td><td class="${r.warning ? "warn" : ""}">${esc(r.deviation)}${r.warning ? " ⚠" : ""}</td></tr>`).join("")
    : `<tr><td colspan="4" class="hint">No data yet – inspect some parts.</td></tr>`;
  const io = s.io;
  const pin = (p) => p == null ? "–" : `GPIO ${p}`;
  $("ioInfo").innerHTML = `<span>Backend</span><span>${io.backend === "gpio" ? "Raspberry Pi GPIO" : "simulated (no GPIO)"}</span>
    <span>Reject</span><span>${pin(io.reject_pin)}${io.reject_pin != null ? `, ${io.reject_delay_ms} ms after capture` : ""} · ${io.rejects} rejected</span>
    <span>OK / NOK lamp</span><span>${pin(io.ok_lamp_pin)} / ${pin(io.nok_lamp_pin)}</span>
    <span>Front / back light</span><span>${pin(io.front_light_pin)} / ${pin(io.back_light_pin)} · on: ${esc(io.light || "–")}</span>`;
  $("ioEvents").innerHTML = io.events.map((e) => `<div>${e.t} · ${esc(e.text)}</div>`).join("");

  $("simCard").classList.toggle("hidden", !s.sim);
  $("simBoard").classList.toggle("hidden", !s.sim);
  $("gtHead").classList.toggle("hidden", !s.sim);
  if (s.sim && document.activeElement?.closest("#simCard") == null) {
    $("simLight").value = s.sim.lighting; $("simPart").value = s.sim.part_type; $("simAngle").value = s.sim.any_angle ? "1" : "0";
    $("simRate").value = s.sim.defect_rate; $("simRateVal").textContent = Math.round(s.sim.defect_rate * 100) + " %";
  }

  $("msgs").innerHTML = s.messages.map((m) => `<div class="${m.level}">${m.t} · ${esc(m.text)}</div>`).join("");

  if (s.last_result && s.last_result.image_id !== lastImageId && !selectedResult) {
    lastImageId = s.last_result.image_id; showResult(s.last_result); loadHistory();
  }
  const key = (s.model?.slug || "") + s.mode + (s.model?.pending ?? "") + JSON.stringify(s.model?.sensitivity || {})
    + (s.model?.learned ?? "") + (s.model?.known_bad ?? "") + (s.model?.collected ? "c" : "") + (s.last_self_learning?.time || "");
  if (s.model) {        // live counter without re-rendering the model list
    const c = $(`coll-${s.model.slug}`), lb = $(`lbtn-${s.model.slug}`);
    if (c) c.textContent = s.model.collected;
    if (lb) lb.textContent = `Learn collected good parts (+${s.model.collected})`;
  }
  if (key !== lastModelsKey) { lastModelsKey = key; loadModels(); loadArchives(); }
}

function renderCalibration(cal, s) {
  $("calInfo").innerHTML = cal.calibrated
    ? `<span>Scale</span><span><b>${cal.mm_per_px} mm/px</b> (1 px ≈ ${(cal.mm_per_px * 1000).toFixed(0)} µm)</span>
       <span>Lens distortion</span><span>${cal.distortion ? `corrected (RMS ${cal.rms_px} px)` : "not corrected"}</span>
       <span>Board / images</span><span>${cal.board.cols} × ${cal.board.rows}, ${cal.board.square_mm} mm · ${cal.n_images} image(s)</span>
       <span>Calibrated</span><span>${esc(cal.id)}</span>` + (cal.note ? `<span>Note</span><span style="color:var(--warn)">${esc(cal.note)}</span>` : "")
    : `<span>Status</span><span>not calibrated – dimensions in pixels, learned tolerances</span>`;
  const sess = cal.session;
  $("calSession").classList.toggle("hidden", !sess); $("calIdle").classList.toggle("hidden", !!sess);
  if (sess) {
    $("calCount").textContent = sess.count;
    $("calInstr").textContent = sess.count === 0
      ? `Image 1: lay the ${sess.cols} × ${sess.rows} board (${sess.square_mm} mm squares) flat on the belt, then “Capture image”.`
      : sess.count < 5 ? `Scale can be computed now. For lens distortion: ${5 - sess.count} more image(s), board tilted 20–40° at different positions.`
      : "Enough images – capture more (up to 15) or compute.";
    $("btnCalCompute").disabled = sess.count < 1;
  } else if (document.activeElement?.closest("#calIdle") == null) {
    const d = cal.defaults; $("calCols").value ||= d.cols; $("calRows").value ||= d.rows; $("calSq").value ||= d.square_mm;
  }
  $("btnCalDelete").disabled = !cal.calibrated;
  $("btnCalStart").disabled = s.mode !== "idle";
}

async function loadHistory() {
  const h = await (await fetch("api/results")).json();
  $("history").innerHTML = h.map((r, i) => {
    const f = FB[r.feedback];
    return `<tr data-i="${i}">
      <td>${r.timestamp.slice(11, 19)}</td><td><span class="tag ${r.status}">${r.selftest ? "ST " : ""}${r.status}</span></td>
      <td>${esc(r.defects.map((d) => d.label).join(", "))}</td>
      <td>${f ? `<span class="fb ${f[1]}">${f[0]}</span>` : ""}</td>
      <td class="${status?.sim ? "" : "hidden"}">${esc(r.ground_truth || "")}</td><td>${r.time_ms.toFixed(0)}</td></tr>`;
  }).join("");
  $("history").onclick = (e) => {
    const tr = e.target.closest("tr"); if (!tr) return;
    selectedResult = h[+tr.dataset.i]; showResult(selectedResult);
    clearTimeout(window._selT); window._selT = setTimeout(() => { selectedResult = null; }, 30000);
  };
}

const fmtSize = (b) => b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB";
async function loadArchives() {
  const d = await (await fetch("api/archives")).json();
  $("archiveList").innerHTML = d.archives.map((a) => {
    const s = a.summary;
    const info = s ? `${s.inspected} parts · ${s.nok} NOK${s.nok_rate_percent != null ? ` (${s.nok_rate_percent} %)` : ""} · ` : "";
    return `<div class="a">
      <div><div class="n">${esc(a.file)}</div><div class="s">${info}${fmtSize(a.size_bytes)} · ${a.created.replace("T", " ")}</div></div>
      <div class="row"><a class="btn" href="api/archives/${encodeURIComponent(a.file)}" download>Download</a>
        <button class="danger" data-f="${esc(a.file)}">Delete</button></div></div>`;
  }).join("");
}
$("archiveList").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-f]"); if (!b) return;
  if (confirm(`Permanently delete the archive ${b.dataset.f}?`)) { await api(`api/archives/${encodeURIComponent(b.dataset.f)}`, "DELETE"); loadArchives(); }
});
// ---------------------------------------------------------------- models
async function loadModels() {
  const models = await (await fetch("api/models")).json();
  const active = status?.model?.slug;
  $("models").innerHTML = models.length ? models.map((m) => {
    const meth = Object.entries(m.methods).map(([ch, ms]) => (m.mode === "dual" ? `${CH[ch]}: ` : "") + ms.map((x) => esc(opts.methods[x] || x)).join(", ")).join(" · ");
    const sens = ["geometry", "diff", "pca"].map((k) => `<span>${esc(opts.methods[k])}</span>
        <input type="range" min="0.5" max="2" step="0.05" value="${m.sensitivity[k] ?? 1}" data-sens="${k}" data-s="${m.slug}">
        <span id="sv-${m.slug}-${k}">×${(m.sensitivity[k] ?? 1).toFixed(2)}</span>`).join("");
    return `<div class="m ${m.slug === active ? "active" : ""}">
      <div class="n">${esc(m.name)} ${m.mode === "dual" ? '<span class="ch">dual light</span>' : ""}</div>
      <div class="s">${m.n_samples} references · ${meth}<br>comparison: ${esc(m.diff_representation)} · ML: ${esc(m.pca_representation)}
        · ${m.mm_per_px ? `${m.mm_per_px.toFixed(4)} mm/px` : "px (uncalibrated)"} · ${m.created.replace("T", " ")}</div>
      <div class="sens">${sens}</div>
      <div class="sl"><b>Self-learning</b> · <span id="coll-${m.slug}">${m.collected}</span> collected good part(s) waiting ·
        ${m.learned} learned · ${m.anchor} taught-in anchor refs · ${m.known_bad} known defective part(s) for the safety check
        ${m.known_bad ? "" : `<div class="hint" style="margin:2px 0 0">Tip: run the self-test or confirm NOK results (“✓ Correct”) –
          these parts are kept and a new model must still detect them. Without them only the threshold check is active.</div>`}
        ${slLast(m.last_self_learning)}
        <div class="row" style="margin-top:6px">
          <button class="small ${m.collected ? "primary" : ""}" data-a="learn" data-s="${m.slug}" id="lbtn-${m.slug}" ${m.collected ? "" : "disabled"}
            title="Train the collected good parts in – only applied if the safety checks pass">Learn collected good parts (+${m.collected})</button>
          <button class="small" data-a="discard" data-s="${m.slug}" ${m.collected ? "" : "disabled"}>Discard collected</button>
        </div></div>
      <div class="row" style="margin-top:8px">
        <button class="small" data-a="sens" data-s="${m.slug}">Save sensitivity</button>
        <button class="small" data-a="report" data-s="${m.slug}">Training report</button>
        <button class="small ${m.pending ? "primary" : ""}" data-a="retrain" data-s="${m.slug}"
          title="Retrain with the method settings above${m.pending ? " and the queued feedback references" : ""}">Retrain${m.pending ? ` (+${m.pending} feedback ref.)` : ""}</button>
        ${m.has_previous ? `<button class="small" data-a="restore" data-s="${m.slug}"
          title="Go back to the version before the last teach-in, retraining or self-learning">Undo last update</button>` : ""}
        <button class="small danger" data-a="delete" data-s="${m.slug}">Delete</button>
      </div></div>`;
  }).join("") : `<p class="hint">No models yet.</p>`;
}
function slLast(r) {
  if (!r) return "";
  const when = r.time.replace("T", " ").slice(0, 16);
  if (r.accepted) {
    const pca = r.thresholds.find((t) => t.method === "pca");
    return `<div class="res good">✓ ${when}: ${r.collected} part(s) learned${pca ? `, ML threshold ${pca.change_percent > 0 ? "+" : ""}${pca.change_percent} %` : ""}, `
      + `known defective parts ${r.known_bad.still_detected}/${r.known_bad.checked} still detected</div>`;
  }
  return `<div class="res bad">✗ ${when}: rejected – ${esc(r.reasons.join(" "))}</div>`;
}

$("models").addEventListener("input", (e) => {
  const i = e.target.closest("input[data-sens]"); if (!i) return;
  $(`sv-${i.dataset.s}-${i.dataset.sens}`).textContent = "×" + (+i.value).toFixed(2);
});
$("models").addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b) return;
  const slug = b.dataset.s;
  if (b.dataset.a === "report") {
    const r = await (await fetch(`api/models/${slug}`)).json();
    $("report").textContent = (r.warnings.length ? "Warnings:\n- " + r.warnings.join("\n- ") + "\n\n" : "") + JSON.stringify(r.report, null, 2);
    $("report").classList.remove("hidden");
  } else if (b.dataset.a === "sens") {
    const vals = {};
    b.closest(".m").querySelectorAll("input[data-sens]").forEach((i) => vals[i.dataset.sens] = +i.value);
    await api(`api/models/${slug}/sensitivity`, "POST", vals); toast("Sensitivity saved", "ok");
  } else if (b.dataset.a === "learn") {
    await api(`api/models/${slug}/learn-collected`, "POST"); toast("Learning collected good parts … see Messages", "ok");
  } else if (b.dataset.a === "discard" && confirm("Discard all collected good parts of this model?")) {
    await api(`api/models/${slug}/discard-collected`, "POST"); lastModelsKey = "";
  } else if (b.dataset.a === "retrain") {
    await api(`api/models/${slug}/retrain`, "POST", currentMethodSettings());
  } else if (b.dataset.a === "restore" && confirm("Restore the previous version of this model? (This can be undone the same way.)")) {
    await api(`api/models/${slug}/restore`, "POST"); lastModelsKey = ""; toast("Previous version restored", "ok");
  } else if (b.dataset.a === "delete" && confirm("Really delete this model?")) {
    await api(`api/models/${slug}`, "DELETE"); lastModelsKey = "";
  }
});

// ---------------------------------------------------------------- actions
$("btnCapture").onclick = () => api("api/capture", "POST");
$("btnReset").onclick = () => api("api/counters/reset", "POST").then(() => { lastImageId = null; $("history").innerHTML = ""; });
$("methodSettings").addEventListener("change", () => api("api/method-settings", "POST", currentMethodSettings())
  .then((r) => { opts.defaults = { ...opts.defaults, ...r }; toast("Method settings saved", "ok"); }).catch(() => methodSettings()));
$("setLight").onchange = (e) => api("api/settings", "POST", { lighting_mode: e.target.value }).catch(() => {});
$("setSelfLearn").onchange = (e) => api("api/settings", "POST", { self_learning: e.target.checked });
$("setSelfAuto").onchange = (e) => api("api/settings", "POST", { self_learning_auto: e.target.checked });
$("btnTestReject").onclick = () => api("api/io/test-reject", "POST");
$("btnFocus").onclick = () => api("api/focus", "POST", { on: !status.focus_assist });
$("btnCalStart").onclick = () => api("api/calibration/start", "POST", { cols: +$("calCols").value, rows: +$("calRows").value, square_mm: +$("calSq").value });
$("btnCalCapture").onclick = async () => {
  try { await api("api/calibration/capture", "POST"); } finally { $("calPreview").src = "api/calibration/preview.jpg?" + Date.now(); $("calPreview").classList.remove("hidden"); }
};
$("btnCalCompute").onclick = async () => { await api("api/calibration/compute", "POST"); $("calPreview").classList.add("hidden"); toast("Calibration saved – retrain your models", "ok"); };
$("btnCalCancel").onclick = () => { api("api/calibration/cancel", "POST"); $("calPreview").classList.add("hidden"); };
$("btnCalDelete").onclick = () => confirm("Delete the calibration? Dimensions will be in pixels again.") && api("api/calibration", "DELETE");
$("btnBoardShow").onclick = () => api("api/sim", "POST", { board: true });
$("btnBoardHide").onclick = () => api("api/sim", "POST", { board: false });
$("simLight").onchange = (e) => api("api/sim", "POST", { lighting: e.target.value });
$("simPart").onchange = (e) => api("api/sim", "POST", { part_type: e.target.value });
$("simAngle").onchange = (e) => api("api/sim", "POST", { any_angle: e.target.value === "1" });
$("simRate").oninput = (e) => $("simRateVal").textContent = Math.round(e.target.value * 100) + " %";
$("simRate").onchange = (e) => api("api/sim", "POST", { defect_rate: +e.target.value });
