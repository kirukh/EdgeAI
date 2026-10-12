// Setup area (PIN): live view with scores, history, models (sensitivity, report), camera/focus, drift, simulator.
// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll("[data-pane]").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== b.dataset.tab));
  try { localStorage.setItem("qc-tab", b.dataset.tab); } catch (e) {}
});
try { const t = localStorage.getItem("qc-tab"); if (t) document.querySelector(`.tabs button[data-tab="${t}"]`)?.click(); } catch (e) {}

const METHOD_KEYS = ["geometry", "diff", "pca"];

// ---------------------------------------------------------------- result
function showResult(r) {
  if (!r) return;
  shownResult = r;
  $("resBadge").className = "badge " + r.status;
  $("resBadge").textContent = r.status === "NO_PART" ? "no part" : r.status;
  $("resTitle").textContent = `Model: ${r.model}`;
  const gt = r.ground_truth ? ` · Expected: ${r.ground_truth}` : "";
  $("resMeta").textContent = `#${r.image_id} · ${r.timestamp.replace("T", " ").slice(0, 19)} · ${r.time_ms} ms${gt}`;
  $("resShots").innerHTML = (r.shots || []).length > 1 ? r.shots.map((s) => `<span class="${s}">${s === "NO_PART" ? "–" : s}</span>`).join("") : "";
  $("resImgs").innerHTML = r.has_image ? `<div class="imgs"><img src="api/results/${r.image_id}/overlay.jpg" alt="">
      <img src="api/results/${r.image_id}/detail.jpg" alt=""></div>` : "";
  $("resDefects").innerHTML = r.defects.map((d) =>
    `<li><b>${esc(d.label)}</b> <span class="d">– ${esc(d.detail)} (${esc(opts.methods[d.method] || d.method)})</span></li>`).join("");
  $("resScores").innerHTML = r.methods.map((m) => {
    const w = Math.min(100, m.score * 50);
    const cls = m.decisive ? "bad" : m.score >= 1 ? "over" : "";
    const marks = `<b style="left:50%"></b>` + (m.standalone > 1 ? `<b style="left:${Math.min(99, m.standalone * 50)}%"></b>` : "");
    const note = m.score >= 1 && !m.decisive ? `<div class="note">below its stand-alone level ${m.standalone} – not confirmed by a 2nd method</div>` : "";
    return `<span>${esc(m.label)}${note}</span><div class="bar"><i class="${cls}" style="width:${w}%"></i>${marks}</div><span>${m.score.toFixed(2)}</span>`;
  }).join("");
  const canFb = r.status === "OK" || r.status === "NOK";
  $("feedback").classList.toggle("hidden", !canFb);
  if (canFb) {
    $("fbWrong").textContent = r.status === "NOK" ? "✗ Within tolerance – accept as OK (learn)" : "✗ Was NOK – missed defect";
    const f = FB[r.feedback];
    $("fbState").textContent = f ? f[0] : "";
    $("fbState").className = "fb " + (f ? f[1] : "");
  }
}

async function sendFeedback(correct) {
  const r = shownResult; if (!r) return;
  const verdict = correct ? r.status.toLowerCase() : (r.status === "NOK" ? "ok" : "nok");
  const res = await api(`api/results/${r.image_id}/feedback`, "POST", { verdict });
  r.feedback = res.outcome;
  showResult(r); loadHistory(); lastModelsKey = "";
  if (res.outcome === "false_alarm" && res.waiting) toast(`Accepted – ${res.waiting} part(s) waiting for the next model update`, "ok");
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
  $("scalePill").textContent = s.mm_per_px ? `${s.mm_per_px.toFixed(4)} mm/px` : "scale: px";
  $("scalePill").className = "pill " + (s.mm_per_px ? "good" : "");
  const dw = s.drift.warnings;
  $("driftPill").textContent = `Drift: ${dw.length} warning${dw.length > 1 ? "s" : ""}`;
  $("driftPill").className = "pill warn" + (dw.length ? "" : " hidden");
  const banner = dw.map((w) => "Image quality drift: " + w + " – check lighting/camera.");
  $("banner").innerHTML = banner.map((b) => `<div>⚠ ${esc(b)}</div>`).join("");
  $("banner").classList.toggle("hidden", !banner.length);

  $("triggerHint").textContent = s.continuous
    ? (s.mode === "idle" ? "Trigger inactive (start learning or inspection mode first)" : s.armed ? "Trigger armed – waiting for the next part" : "Part captured – waiting until it leaves the image")
    : "Image folder: automatic in inspection mode, click to capture in learning mode";
  $("btnCapture").disabled = !["learning", "inspecting"].includes(s.mode);

  const c = s.counts, total = c.OK + c.NOK, f = s.feedback;
  $("cOK").textContent = c.OK; $("cNOK").textContent = c.NOK;
  $("cRate").textContent = total ? `${(100 * c.NOK / total).toFixed(1)} %` : "–";
  const rev = f.confirmed + f.false_alarm + f.missed;
  $("fbStats").textContent = rev ? `Supervisor review of ${rev} result(s): ${f.confirmed} confirmed, ${f.false_alarm} false alarm(s), ` +
    `${f.missed} missed defect(s) → ${(100 * f.confirmed / rev).toFixed(0)} % correct` : "No supervisor review yet – use “Supervisor check” below a result.";

  $("inspInfo").textContent = `Always ${s.settings.shots_per_part} images per part with majority vote; the further images are only `
    + `evaluated when the first one is not clearly good. Rotation search: ${s.settings.rotation_search}. Methods and tolerances: config.json → "method".`;
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

  $("simCard").classList.toggle("hidden", !s.sim);
  $("gtHead").classList.toggle("hidden", !s.sim);
  if (s.sim && document.activeElement?.closest("#simCard") == null) {
    $("simLight").value = s.sim.lighting; $("simPart").value = s.sim.part_type; $("simAngle").value = s.sim.any_angle ? "1" : "0";
    $("simRate").value = s.sim.defect_rate; $("simRateVal").textContent = Math.round(s.sim.defect_rate * 100) + " %";
  }
  $("msgs").innerHTML = s.messages.map((m) => `<div class="${m.level}">${m.t} · ${esc(m.text)}</div>`).join("");

  if (s.last_result && s.last_result.image_id !== lastImageId && !selectedResult) {
    lastImageId = s.last_result.image_id; showResult(s.last_result); loadHistory();
  }
  const imp = s.improvement || {};
  const key = (s.model?.slug || "") + s.mode + JSON.stringify(s.model?.sensitivity || {}) + (s.model?.version ?? "")
    + imp.running + (imp.last?.time || "") + (s.feedback.false_alarm + s.feedback.missed + s.feedback.confirmed);
  if (key !== lastModelsKey && view === "expert") { lastModelsKey = key; loadModels(); loadArchives(); }
}

async function loadHistory() {
  const h = await (await fetch("api/results")).json();
  $("history").innerHTML = h.map((r, i) => {
    const f = FB[r.feedback];
    return `<tr data-i="${i}">
      <td>${r.timestamp.slice(11, 19)}</td><td><span class="tag ${r.status}">${r.status}</span></td>
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
    const meth = m.methods.map((x) => esc(opts.methods[x] || x)).join(", ");
    const sens = METHOD_KEYS.map((k) => `<span>${esc(opts.methods[k])}</span>
        <input type="range" min="0.5" max="2" step="0.05" value="${m.sensitivity[k] ?? 1}" data-sens="${k}" data-s="${m.slug}">
        <span id="sv-${m.slug}-${k}">×${(m.sensitivity[k] ?? 1).toFixed(2)}</span>`).join("");
    const excl = m.excluded.length ? ` · reference(s) ${m.excluded.map((i) => i + 1).join(", ")} left out` : "";
    return `<div class="m ${m.slug === active ? "active" : ""}">
      <div class="n">${esc(m.name)} <span class="hint">· version ${m.version}</span></div>
      <div class="s">${m.n_samples} references (${m.accepted} accepted by the operator)${excl} · ${meth}<br>comparison: ${esc(m.diff_representation)} · ML: ${esc(m.pca_representation)}
        · ${m.mm_per_px ? `${m.mm_per_px.toFixed(4)} mm/px` : "px"} · ${m.created.replace("T", " ")}</div>
      <div class="sens">${sens}</div>
      ${improvementCard(m)}
      <div class="row" style="margin-top:8px">
        <button class="small" data-a="sens" data-s="${m.slug}">Save sensitivity</button>
        <button class="small" data-a="report" data-s="${m.slug}">Training report</button>
        <button class="small danger" data-a="delete" data-s="${m.slug}">Delete</button>
      </div></div>`;
  }).join("") : `<p class="hint">No models yet.</p>`;
}

// Continuous improvement (qc/learning.py): accepted parts waiting, test parts, version history
function improvementCard(m) {
  const running = status?.improvement?.running;
  const hist = (m.history || []).slice(-5).reverse().map((h) => `<div class="res good">v${h.version_after} · ${h.time.replace("T", " ").slice(0, 16)}
      · ${h.accepted_parts} accepted part(s) learned · test parts ${h.known_bad_still_detected}/${h.known_bad_checked} still detected
      ${h.still_nok.length ? ` · <span style="color:var(--warn)">${h.still_nok.length} still NOK</span>` : ""}</div>`).join("");
  const last = status?.improvement?.last;
  const rejected = last && !last.released && last.model === m.name
    ? `<div class="res bad">✗ ${last.time.replace("T", " ").slice(0, 16)}: update rejected – ${esc(last.reasons.join(" "))}</div>` : "";
  return `<div class="sl"><b>Continuous improvement</b> · ${m.waiting} accepted part(s) waiting · ${m.known_bad} test part(s) (confirmed defective)
    ${running ? `<div class="res">⟳ update running …</div>` : ""}${rejected}${hist}
    <div class="row" style="margin-top:6px">
      <button class="small ${m.waiting ? "primary" : ""}" data-a="update" data-s="${m.slug}" ${m.waiting && !running ? "" : "disabled"}
        title="Learn the accepted parts now – the new version is only used if every test part is still detected">Update now (+${m.waiting})</button>
      <button class="small" data-a="waiting" data-s="${m.slug}" ${m.waiting ? "" : "disabled"}>Show accepted parts</button>
      ${m.has_previous ? `<button class="small" data-a="undo" data-s="${m.slug}" title="Back to the previous version (can be undone the same way)">Undo last update</button>` : ""}
    </div><div class="thumbs hidden" id="w-${m.slug}"></div></div>`;
}

// thumbnails need the setup token → fetched with headers and shown as blob URLs
async function showWaiting(slug) {
  const box = $(`w-${slug}`), parts = await api(`api/models/${slug}/waiting`);
  box.innerHTML = parts.map((p) => `<figure style="margin:0"><img data-n="${esc(p.name)}" alt="">
    <figcaption class="hint">${esc((p.defects || []).join(", ") || p.result)}<br>
    <button class="small danger" data-a="discard" data-s="${slug}" data-n="${esc(p.name)}">Discard</button></figcaption></figure>`).join("");
  box.classList.remove("hidden");
  for (const img of box.querySelectorAll("img")) {
    const r = await fetch(`api/models/${slug}/waiting/${encodeURIComponent(img.dataset.n)}.png`, { headers: headers() });
    if (r.ok) img.src = URL.createObjectURL(await r.blob());
  }
}

$("models").addEventListener("input", (e) => {
  const i = e.target.closest("input[data-sens]"); if (!i) return;
  $(`sv-${i.dataset.s}-${i.dataset.sens}`).textContent = "×" + (+i.value).toFixed(2);
});
$("models").addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b) return;
  const slug = b.dataset.s;
  if (b.dataset.a === "report") {
    const r = await api(`api/models/${slug}`);
    $("report").textContent = (r.warnings.length ? "Warnings:\n- " + r.warnings.join("\n- ") + "\n\n" : "") + JSON.stringify(r.report, null, 2);
    $("report").classList.remove("hidden");
  } else if (b.dataset.a === "sens") {
    const vals = {};
    b.closest(".m").querySelectorAll("input[data-sens]").forEach((i) => vals[i.dataset.sens] = +i.value);
    await api(`api/models/${slug}/sensitivity`, "POST", vals); toast("Sensitivity saved", "ok");
  } else if (b.dataset.a === "update") {
    await api(`api/models/${slug}/update`, "POST"); toast("Model update started – see Messages", "ok");
  } else if (b.dataset.a === "waiting") {
    showWaiting(slug);
  } else if (b.dataset.a === "discard" && confirm("Discard this accepted part? It will not be learned.")) {
    await api(`api/models/${slug}/waiting/${encodeURIComponent(b.dataset.n)}`, "DELETE"); lastModelsKey = "";
  } else if (b.dataset.a === "undo" && confirm("Go back to the previous version of this model? (This can be undone the same way.)")) {
    await api(`api/models/${slug}/undo`, "POST"); lastModelsKey = ""; toast("Previous version restored", "ok");
  } else if (b.dataset.a === "delete" && confirm("Really delete this model?")) {
    await api(`api/models/${slug}`, "DELETE"); lastModelsKey = "";
  }
});

// ---------------------------------------------------------------- actions
$("btnCapture").onclick = () => api("api/capture", "POST");
$("btnReset").onclick = () => api("api/counters/reset", "POST").then(() => { lastImageId = null; $("history").innerHTML = ""; });
$("btnFocus").onclick = () => api("api/focus", "POST", { on: !status.focus_assist });
$("simLight").onchange = (e) => api("api/sim", "POST", { lighting: e.target.value });
$("simPart").onchange = (e) => api("api/sim", "POST", { part_type: e.target.value });
$("simAngle").onchange = (e) => api("api/sim", "POST", { any_angle: e.target.value === "1" });
$("simRate").oninput = (e) => $("simRateVal").textContent = Math.round(e.target.value * 100) + " %";
$("simRate").onchange = (e) => api("api/sim", "POST", { defect_rate: +e.target.value });
