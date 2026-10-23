"""Archiving of inspection results.

When a batch is finished, the complete results folder (inspection logs and
saved images) is packed into a ZIP file together with a summary, and the
results folder is emptied for the next batch::

    data/archive/<batch-name>_<YYYY-MM-DD_HHMMSS>.zip
        summary.json          counts, defect types, models, time range
        summary.txt           the same, human-readable
        results/<date>/...    inspection_log.csv, feedback_log.csv, saved images
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .model import slugify


class ArchiveError(ValueError):
    pass


def has_results(results_dir: str | Path) -> bool:
    root = Path(results_dir)
    return root.exists() and any(p.is_file() for p in root.rglob("*"))


def _version_key(row: dict[str, Any]) -> str:
    return f"{row.get('model', '')} v{row.get('model_version') or 1}"


def _read_inspections(results_dir: Path) -> tuple[Counter[str], Counter[str], Counter[str], list[str], dict[str, Any]]:
    """inspection_log.csv files → (results, defect types, models, times, per model version)."""
    results: Counter[str] = Counter()
    defects: Counter[str] = Counter()
    models: Counter[str] = Counter()
    times: list[str] = []
    per_version: dict[str, Counter[str]] = {}
    for log in sorted(results_dir.rglob("inspection_log.csv")):
        with log.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results[row.get("result", "")] += 1
                models[row.get("model", "")] += 1
                per_version.setdefault(_version_key(row), Counter())[row.get("result", "")] += 1
                if row.get("time"):
                    times.append(row["time"])
                for d in filter(None, (row.get("defects") or "").split(" | ")):
                    defects[d.split(": ", 1)[0]] += 1
    return results, defects, models, times, per_version


def _read_feedback(results_dir: Path) -> tuple[Counter[str], dict[str, Counter[str]]]:
    """feedback_log.csv files → (outcomes, outcomes per model version). The last word counts."""
    latest: dict[str, dict[str, Any]] = {}
    for log in sorted(results_dir.rglob("feedback_log.csv")):
        with log.open(newline="", encoding="utf-8") as f:
            latest.update({row["image_id"]: row for row in csv.DictReader(f)})
    fb: Counter[str] = Counter(row["outcome"] for row in latest.values())
    per_version: dict[str, Counter[str]] = {}
    for row in latest.values():
        per_version.setdefault(_version_key(row), Counter())[row["outcome"]] += 1
    return fb, per_version


def summarize(results_dir: str | Path) -> dict[str, Any]:
    """Evaluates all inspection logs in the results folder."""
    results, defects, models, times, insp_v = _read_inspections(Path(results_dir))
    fb, fb_v = _read_feedback(Path(results_dir))
    ok, nok = results.get("OK", 0), results.get("NOK", 0)
    reviewed = sum(fb.values())
    correct = fb.get("confirmed_ok", 0) + fb.get("confirmed_nok", 0)
    # continuous improvement: how each model version performed (false alarms = parts the operator
    # accepted as good although they were rated NOK)
    versions = {}
    for key in sorted(insp_v):
        n = insp_v[key].get("OK", 0) + insp_v[key].get("NOK", 0)
        fa = fb_v.get(key, Counter()).get("false_alarm", 0)
        versions[key] = {
            "inspected": n,
            "nok": insp_v[key].get("NOK", 0),
            "false_alarms": fa,
            "missed_defects": fb_v.get(key, Counter()).get("missed", 0),
            "false_alarm_rate_percent": round(100.0 * fa / n, 2) if n else None,
        }
    return {
        "supervisor_feedback": {
            "reviewed": reviewed,
            "false_alarms": fb.get("false_alarm", 0),
            "missed_defects": fb.get("missed", 0),
            "confirmed": correct,
            "accuracy_of_reviewed_percent": round(100.0 * correct / reviewed, 1) if reviewed else None,
        },
        "inspected": ok + nok,
        "ok": ok,
        "nok": nok,
        "nok_rate_percent": round(100.0 * nok / (ok + nok), 2) if ok + nok else None,
        "no_part": results.get("NO_PART", 0),
        "not_inspected": results.get("NOT_INSPECTED", 0),  # Pi could not keep up (see results.py)
        "defect_types": dict(defects.most_common()),
        "models": dict(models.most_common()),
        "model_versions": versions,
        "first_inspection": min(times) if times else None,
        "last_inspection": max(times) if times else None,
    }


def _summary_text(label: str, s: dict[str, Any]) -> str:
    lines = [
        f"Batch:            {label}",
        f"Archived:         {datetime.now().isoformat(timespec='seconds')}",
        f"Model(s):         {', '.join(s['models']) or '-'}",
        f"Period:           {s['first_inspection'] or '-'}  to  {s['last_inspection'] or '-'}",
        f"Inspected parts:  {s['inspected']}",
        f"OK:               {s['ok']}",
        f"NOK:              {s['nok']}"
        + (f"  ({s['nok_rate_percent']} %)" if s["nok_rate_percent"] is not None else ""),
        f"Not inspected:    {s['not_inspected']}",
        "",
        "Defect types:",
    ]
    lines += [f"  {k}: {v}" for k, v in s["defect_types"].items()] or ["  none"]
    fb = s.get("supervisor_feedback", {})
    if fb.get("reviewed"):
        lines += [
            "",
            "Supervisor review:",
            f"  reviewed results: {fb['reviewed']}",
            f"  false alarms:     {fb['false_alarms']}",
            f"  missed defects:   {fb['missed_defects']}",
            f"  confirmed:        {fb['confirmed']}  ({fb['accuracy_of_reviewed_percent']} % of reviewed)",
        ]
    if len(s.get("model_versions", {})) > 1:  # continuous improvement: compare the versions
        lines += ["", "Model versions (false alarms = parts accepted as good by the operator):"]
        for key, v in s["model_versions"].items():
            rate = f"{v['false_alarm_rate_percent']} %" if v["false_alarm_rate_percent"] is not None else "-"
            lines.append(
                f"  {key}: {v['inspected']} parts, {v['nok']} NOK, {v['false_alarms']} false alarms ({rate}), "
                f"{v['missed_defects']} missed defects"
            )
    return "\n".join(lines) + "\n"


def create_archive(results_dir: str | Path, archive_dir: str | Path, label: str = "") -> dict[str, Any]:
    """Packs the results folder into a ZIP file and empties it afterwards."""
    results = Path(results_dir)
    if not has_results(results):
        raise ArchiveError("There are no results to archive.")
    label = label.strip() or "batch"
    out_dir = Path(archive_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    target = out_dir / f"{slugify(label)}_{stamp}.zip"
    summary = summarize(results)
    summary["batch"] = label

    tmp = target.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
        zf.writestr("summary.txt", _summary_text(label, summary))
        for p in sorted(results.rglob("*")):
            if p.is_file():
                zf.write(p, Path("results") / p.relative_to(results))
    # Only empty the results folder once the archive has been written completely
    with zipfile.ZipFile(tmp) as zf:
        if zf.testzip() is not None:
            tmp.unlink()
            raise ArchiveError("Archive verification failed – results were not deleted.")
    tmp.rename(target)
    for p in results.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    return {"file": target.name, "size_bytes": target.stat().st_size, "summary": summary}


def list_archives(archive_dir: str | Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(Path(archive_dir).glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True):
        entry = {
            "file": p.name,
            "size_bytes": p.stat().st_size,
            "created": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
        }
        try:
            with zipfile.ZipFile(p) as zf:
                entry["summary"] = json.loads(zf.read("summary.json"))
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            entry["summary"] = None
        out.append(entry)
    return out


def archive_path(archive_dir: str | Path, name: str) -> Path:
    """Resolves an archive file name safely (no path traversal)."""
    base = Path(archive_dir).resolve()
    p = (base / name).resolve()
    if p.parent != base or p.suffix != ".zip" or not p.is_file():
        raise ArchiveError("Archive not found.")
    return p


def delete_archive(archive_dir: str | Path, name: str) -> None:
    archive_path(archive_dir, name).unlink()
