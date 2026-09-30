#!/usr/bin/env python3
"""Entry point – command line.

    python main.py web                       web UI (default: simulator)
    python main.py web --camera pi           … with the Raspberry Pi camera
    python main.py train  --name X --images folder/
    python main.py inspect --model X img1.png img2.png --out results/
    python main.py archive --name "Batch 42"  zip results + empty the results folder
    python main.py generate --out data/synth --lighting front
    python main.py evaluate --synthetic      compare methods
    python main.py evaluate --train good/ --test test/   (test/ok, test/nok/<defect_type>)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from qc.config import AppConfig


def _cfg(args) -> AppConfig:
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


def cmd_web(args):
    from qc.camera import open_camera
    from qc.system import QCSystem
    from qc.webapp import create_app

    cfg = _cfg(args)
    if args.part_type:
        cfg.camera.sim_part_type = args.part_type
    if args.any_angle:
        cfg.camera.sim_any_angle = True
    system = QCSystem(cfg, open_camera(cfg.camera))
    system.start()
    print(f"Web UI: http://{args.host}:{args.port}/  (camera: {system.camera.name})")
    try:
        create_app(system).run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        system.stop()


def cmd_train(args):
    from qc.evaluate import load_folder
    from qc.model import QCModel

    cfg = _cfg(args)
    frames = load_folder(args.images)
    print(f"{len(frames)} reference images from {args.images}")
    model = QCModel.train(args.name, frames, cfg.localization, cfg.method)
    path = model.save(cfg.storage.models_dir)
    print(f"Model saved: {path}")
    print(json.dumps(model.report, indent=2, ensure_ascii=False))


def cmd_inspect(args):
    from qc.model import QCModel

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
        if out and res.overlay is not None:
            cv2.imwrite(str(out / f"{Path(f).stem}_{res.status}.jpg"), res.overlay)
            cv2.imwrite(str(out / f"{Path(f).stem}_{res.status}_detail.jpg"), res.detail)
    return 1 if n_nok else 0


def cmd_archive(args):
    from qc import archive as qa

    cfg = _cfg(args)
    if args.list:
        for a in qa.list_archives(cfg.storage.archive_dir):
            s = a["summary"] or {}
            print(f"{a['file']:50s} {a['size_bytes'] / 1024:8.0f} KB  {s.get('inspected', '?')} parts, {s.get('nok', '?')} NOK")
        return 0
    info = qa.create_archive(cfg.storage.results_dir, cfg.storage.archive_dir, args.name or "batch")
    s = info["summary"]
    print(f"Archived: {Path(cfg.storage.archive_dir) / info['file']}  ({s['inspected']} parts, {s['nok']} NOK)")
    print("Results folder cleared.")
    return 0


def cmd_generate(args):
    from qc.synthetic import generate_dataset

    out = generate_dataset(args.out, args.part_type, args.lighting, args.n_train, args.n_good, args.n_defect, args.seed,
                           args.any_angle)
    print(f"Dataset created: {out}")


def cmd_evaluate(args):
    from qc import evaluate as E

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
                train, test = E.synthetic_sets(pt, light, n_good=args.n_good, n_per_defect=args.n_defect, seed=args.seed,
                                               any_angle=args.any_angle)
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


def main(argv=None):
    p = argparse.ArgumentParser(description="AI-assisted quality inspection of metal parts (Raspberry Pi)")
    p.add_argument("--config", default="config.json", help="JSON configuration (optional)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def method_args(sp):
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
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
