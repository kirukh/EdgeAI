// Shared helpers: API calls (setup token + CSRF header), toasts, labels.
const token = new URLSearchParams(location.search).get("token") || "";
let setupToken = "";
try { setupToken = sessionStorage.getItem("qc-setup") || ""; } catch (e) {}
const $ = (id) => document.getElementById(id);
let opts = null, lastImageId = null, selectedResult = null, shownResult = null, lastModelsKey = "", status = null;
// X-QC-Client: required by the server for every modifying request (protection against cross-site requests)
const headers = () => ({ "Content-Type": "application/json", "X-QC-Client": "1", "X-QC-Token": token, "X-Setup-Token": setupToken });

async function api(path, method = "GET", body) {
  const r = await fetch(path, { method, headers: headers(), body: body ? JSON.stringify(body) : undefined });
  const data = await r.json().catch(() => ({}));
  if (r.status === 403 && /locked/i.test(data.error || "")) setView("simple");
  if (!r.ok || data.ok === false) { toast(data.error || `Error ${r.status}`); throw new Error(data.error); }
  return data;
}
function toast(text, kind) {
  const t = document.createElement("div"); t.className = "toast" + (kind === "ok" ? " ok" : ""); t.textContent = text;
  document.body.appendChild(t); setTimeout(() => t.remove(), 4500);
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const MODE = { idle: "Ready", camera_setup: "Adjusting camera", learning: "Learning", training: "Training", inspecting: "Inspection running" };
const FB = { false_alarm: ["false alarm", "bad"], missed: ["missed defect", "bad"], confirmed_ok: ["✓ confirmed", "good"],
             confirmed_nok: ["✓ confirmed", "good"] };
