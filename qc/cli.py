"""Command line (``python main.py …`` or, after ``pip install -e .``, ``qc …``).

qc web                       web UI (default: simulator)
qc web --camera pi           … with the Raspberry Pi camera
qc train  --name X --images folder/
qc inspect --model X img1.png img2.png --out results/
qc archive --name "Batch 42"  zip results + empty the results folder
qc benchmark [--model X --images folder/]   run time per step (e.g. on the Pi)
qc generate --out data/synth --lighting front
qc evaluate --synthetic      compare methods
qc evaluate --train good/ --test test/   (test/ok, test/nok/<defect_type>)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import AppConfig, ConfigError

if TYPE_CHECKING:
    from flask import Flask


def _setup_logging(cfg: AppConfig) -> Path:
    """Log file next to the models (data/logs/qc.log, 3 × 1 MB rotating) + console."""
    log_dir = Path(cfg.storage.models_dir).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("qc")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = RotatingFileHandler(log_dir / "qc.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        log.addHandler(ch)
    return log_dir / "qc.log"


def _serve(app: Flask, host: str, port: int) -> None:
    """waitress (production WSGI server, if installed) – otherwise Flask's built-in server."""
    try:
        from waitress import serve
    except ImportError:
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
        return
    serve(app, host=host, port=port, threads=8, channel_timeout=60, ident=None)


def _cfg(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load(args.config)
    if getattr(args, "camera", None):
        cfg.camera.source = args.camera
    if getattr(args, "methods", None):
        cfg.method.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if getattr(args, "diff_rep", None):
        cfg.method.diff_representation = args.diff_rep
    if getattr(args, "pca_rep", None):
        cfg.method.pca_representation = args.pca_rep
    if getattr(args, "lighting", None) and hasattr(cfg.camera, "sim_lighting"):
        cfg.camera.sim_lighting = args.lighting
    return cfg


def cmd_web(args: argparse.Namespace) -> None:
    from .camera import open_camera
    from .system import QCSystem
    from .web import create_app

    cfg = _cfg(args)
    if args.part_type:
        cfg.camera.sim_part_type = args.part_type
    if args.any_angle:
        cfg.camera.sim_any_angle = True
    log_file = _setup_logging(cfg)
    system = QCSystem(cfg, open_camera(cfg.camera))
    system.start()
    print(f"Web UI: http://{args.host}:{args.port}/  (camera: {system.camera.name}, log: {log_file})")
    try:
        _serve(create_app(system), args.host, args.port)
    finally:
        system.stop()


def cmd_train(args: argparse.Namespace) -> None:
    from .evaluate import load_folder
    from .model import QCModel

    cfg = _cfg(args)
    frames = load_folder(args.images)
    mm = cfg.camera.mm_per_px
    print(f"{len(frames)} reference images from {args.images}" + (f" ({mm} mm/px)" if mm else ""))
    model = QCModel.train(args.name, frames, cfg.localization, cfg.method, mm)
    path = model.save(cfg.storage.models_dir)
    print(f"Model saved: {path}")
    print(json.dumps(model.report, indent=2, ensure_ascii=False))
    for w in model.warnings():
        print("WARNING:", w)


def cmd_inspect(args: argparse.Namespace) -> int:
    from .model import QCModel

    cfg = _cfg(args)
    model = QCModel.load(Path(cfg.storage.models_dir) / args.model)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    n_nok = 0
    for f in args.images:
        img = cv2.imread(f)
        if img is None:
            print(f"{f}: cannot be read")
            continue
        res = model.inspect(img)
        n_nok += res.status != "OK"
        defects = "; ".join(f"{d.label} ({d.detail})" for d in res.defects)
        print(f"{Path(f).name:30s} {res.status:9s} {res.time_ms:6.0f} ms  {defects}")
        if out and res.overlay is not None and res.detail is not None:
            cv2.imwrite(str(out / f"{Path(f).stem}_{res.status}.jpg"), res.overlay)
            cv2.imwrite(str(out / f"{Path(f).stem}_{res.status}_detail.jpg"), res.detail)
    return 1 if n_nok else 0


def cmd_archive(args: argparse.Namespace) -> int:
    from . import archive as qa

    cfg = _cfg(args)
    if args.list:
        for a in qa.list_archives(cfg.storage.archive_dir):
            s = a["summary"] or {}
            parts = f"{s.get('inspected', '?')} parts, {s.get('nok', '?')} NOK"
            print(f"{a['file']:50s} {a['size_bytes'] / 1024:8.0f} KB  {parts}")
        return 0
    info = qa.create_archive(cfg.storage.results_dir, cfg.storage.archive_dir, args.name or "batch")
    s = info["summary"]
    print(f"Archived: {Path(cfg.storage.archive_dir) / info['file']}  ({s['inspected']} parts, {s['nok']} NOK)")
    print("Results folder cleared.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Measures the time of every processing step – run it on the Raspberry Pi."""
    from . import synthetic as S
    from .alignment import align_part
    from .evaluate import load_folder
    from .model import QCModel, _prepare, decide, merge_defects
    from .visualize import annotate

    cfg = _cfg(args)
    if args.model:
        model = QCModel.load(Path(cfg.storage.models_dir) / args.model)
        frames = load_folder(args.images) if args.images else model.reference_frames()
    else:
        rng = np.random.default_rng(0)
        size = (cfg.camera.width, cfg.camera.height)
        frames = [S.compose(S.make_part("A", None, rng), S.random_pose(rng), "front", size, rng=rng) for _ in range(15)]
        t0 = time.perf_counter()
        model = QCModel.train("benchmark", frames, cfg.localization, cfg.method)
        print(f"Training (15 images {size[0]}×{size[1]}): {(time.perf_counter() - t0) * 1000:.0f} ms")
    frames = (frames * (args.n // max(1, len(frames)) + 1))[: args.n]
    t: dict[str, list[float]] = {"detect": [], "align (rotation search + ECC)": [], "visualise": [], "total": []}
    for f in frames:
        t_start = time.perf_counter()
        work, det = _prepare(f, model.loc_cfg)
        t1 = time.perf_counter()
        t["detect"].append(t1 - t_start)
        if det is None:
            continue
        a = align_part(work, det, model.loc_cfg, model.canvas, model.ref_norm)
        t2 = time.perf_counter()
        t["align (rotation search + ECC)"].append(t2 - t1)
        results = []
        for k, m in model.methods.items():
            tm = time.perf_counter()
            results.append(m.score(a, model.sensitivity.get(k, 1.0), model.mm_per_px))
            t.setdefault(f"method {k}", []).append(time.perf_counter() - tm)
        ok, dec = decide(results, model.method_cfg)
        t3 = time.perf_counter()
        annotate(a, ok, merge_defects(dec), None)
        t4 = time.perf_counter()
        t["visualise"].append(t4 - t3)
        t["total"].append(t4 - t_start)
    print(
        f"\n{len(frames)} inspections, input {frames[0].shape[1]}×{frames[0].shape[0]}, "
        f"working width {model.loc_cfg.work_width} px, rotation search '{model.loc_cfg.rotation_search}':"
    )
    for k, v in t.items():
        if v:
            print(f"  {k:32s} {np.mean(v) * 1000:7.1f} ms  (max {np.max(v) * 1000:.1f})")
    tot = np.mean(t["total"])
    shots = max(3, cfg.inspection.shots_per_part)
    print(
        f"\n→ about {60 / tot:.0f} parts/minute for clearly good parts (1st image is final), "
        f"{60 / (tot * shots):.0f}/minute if every part needs all {shots} images (plus camera/trigger time)."
    )
    print("Faster: smaller localization.work_width, rotation_search 'flip'/'off' for guided parts, fewer methods.")


def cmd_generate(args: argparse.Namespace) -> None:
    from .synthetic import generate_dataset

    out = generate_dataset(
        args.out, args.part_type, args.lighting, args.n_train, args.n_good, args.n_defect, args.seed, args.any_angle
    )
    print(f"Dataset created: {out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from . import evaluate as E

    cfg = _cfg(args)
    grid = E.default_grid(cfg.method)
    if args.only:
        wanted = [s.strip() for s in args.only.split(";")]
        grid = [g for g in grid if g[0] in wanted]
    md_parts = []
    if args.synthetic:
        for light in args.lightings.split(","):
            for pt in args.part_types.split(","):
                title = f"Part {pt}, {'front light' if light == 'front' else 'backlight'}"
                print(f"\n== {title} ==")
                train, test = E.synthetic_sets(
                    pt, light, n_good=args.n_good, n_per_defect=args.n_defect, seed=args.seed, any_angle=args.any_angle
                )
                rows = E.evaluate(train, test, grid, cfg.localization)
                md_parts.append(E.to_markdown(rows, title))
                if args.csv:
                    E.to_csv(rows, args.csv, {"part_type": pt, "lighting": light})
    else:
        if not (args.train and args.test):
            sys.exit("Specify --train and --test, or use --synthetic")
        train, test = E.load_folder(args.train), E.load_labeled_folder(args.test)
        print(f"{len(train)} training images, {len(test)} test images")
        rows = E.evaluate(train, test, grid, cfg.localization)
        md_parts.append(E.to_markdown(rows, "Own images"))
        if args.csv:
            E.to_csv(rows, args.csv)
    md = "\n".join(md_parts)
    print("\n" + md)
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
        print(f"Table saved: {args.md}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qc", description="AI-assisted quality inspection of metal parts (Raspberry Pi)")
    p.add_argument("--config", default="config.json", help="JSON configuration (optional)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def method_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--methods", help="e.g. geometry,diff,pca")
        sp.add_argument("--diff-rep", help="image representation for the reference comparison")
        sp.add_argument("--pca-rep", help="image representation for the ML anomaly detection")

    w = sub.add_parser("web", help="start the web UI")
    w.add_argument("--camera", help="sim | pi | usb:0 | folder:<path>")
    w.add_argument("--host", default="0.0.0.0")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--lighting", choices=["front", "back"], help="simulator only")
    w.add_argument("--part-type", choices=["A", "B", "C", "D"], help="simulator only (C = triangle, D = square)")
    w.add_argument("--any-angle", action="store_true", help="simulator only: parts arrive in any rotation")
    method_args(w)
    w.set_defaults(func=cmd_web)

    t = sub.add_parser("train", help="learn a model from a folder of good-part images")
    t.add_argument("--name", required=True)
    t.add_argument("--images", required=True)
    method_args(t)
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("inspect", help="inspect images with a saved model")
    i.add_argument("--model", required=True, help="folder name of the model")
    i.add_argument("--out", help="folder for annotated result images")
    i.add_argument("images", nargs="+")
    i.set_defaults(func=cmd_inspect)

    a = sub.add_parser("archive", help="zip all results and empty the results folder (batch finished)")
    a.add_argument("--name", help="batch name used in the file name")
    a.add_argument("--list", action="store_true", help="only list existing archives")
    a.set_defaults(func=cmd_archive)

    b = sub.add_parser("benchmark", help="run time of every processing step (run it on the Pi)")
    b.add_argument("--model", help="model folder name (default: synthetic test part)")
    b.add_argument("--images", help="images to inspect (default: the model's references)")
    b.add_argument("--n", type=int, default=30)
    method_args(b)
    b.set_defaults(func=cmd_benchmark)

    g = sub.add_parser("generate", help="generate a synthetic test dataset")
    g.add_argument("--out", default="data/synth")
    g.add_argument("--part-type", default="A", choices=["A", "B", "C", "D"])
    g.add_argument("--lighting", default="front", choices=["front", "back"])
    g.add_argument("--n-train", type=int, default=15)
    g.add_argument("--n-good", type=int, default=20)
    g.add_argument("--n-defect", type=int, default=5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--any-angle", action="store_true", help="parts in any rotation (0–360°)")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="compare methods × representations")
    e.add_argument("--synthetic", action="store_true")
    e.add_argument("--lightings", default="front,back")
    e.add_argument("--part-types", default="A,B", help="comma-separated, from A,B,C,D")
    e.add_argument("--any-angle", action="store_true", help="parts in any rotation (0–360°)")
    e.add_argument("--n-good", type=int, default=40)
    e.add_argument("--n-defect", type=int, default=10)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--train")
    e.add_argument("--test")
    e.add_argument("--only", help="only these configurations, separated by ';'")
    e.add_argument("--csv", help="additionally append results to a CSV file")
    e.add_argument("--md", help="save the result table as Markdown")
    method_args(e)
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    try:
        return args.func(args) or 0
    except ConfigError as e:
        print(f"{args.config}: {e}", file=sys.stderr)
        return 2
