# Project 1 – AI-assisted quality inspection of metal parts

Camera-based OK/NOK inspection of metal parts on a conveyor belt, running on a Raspberry Pi.
The system is taught per part type with 10–20 **good parts**. It needs no defect examples and no
predefined defect categories. It inspects with a combination of classical image processing (OpenCV),
reference image comparison and unsupervised machine learning. For NOK parts, the defect location is
marked in the image.

![User interface](docs/screenshot_ui.png)

## Quick start (no hardware)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py web            # → http://localhost:8000
```

The simulator generates a conveyor with synthetic plates. You can set the lighting
(front light/backlight), the part type (A = plate with 2 holes, B = hole + slot, C = triangle with 3 holes,
D = square with 4 holes), the rotation on the belt (roughly aligned or any angle) and the defect rate. Workflow in the UI:

1. **Learning mode**: enter a name, then *Start*. Each part is captured automatically when it passes the
   centre of the image. In learning mode the simulator only sends good parts. Click a thumbnail to
   remove a bad capture.
2. **Finish & create model**. Before that you can optionally choose the methods and image representation.
3. **Inspection mode**: select a model, then *Start*. Results, marked defect locations, per-method scores and
   the history appear live. In the simulator the expected class is shown for comparison.

## Inspection pipeline

```
Camera image
  → Trigger: part completely visible and in the image centre?          (system.py)
  → Localisation: Otsu threshold, largest contour, minAreaRect         (alignment.py)
  → Alignment: centroid onto a fixed canvas, 360° rotation search against
    the reference (decided by the hole pattern), ECC fine alignment (sub-pixel)
  → Photometric normalisation: background = 0, part = 1
  → Image representation per method: gray, CLAHE, edges, threshold,
    LAB, HSV …                                                          (representations.py)
  → Inspection methods (combined with OR)                               (methods.py)
       geometry  holes + outline vs. learned nominal geometry
       diff      z-score difference map vs. mean/spread image of the good parts
       pca       ML: PCA subspace of the good parts, reconstruction error = anomaly
  → OK / NOK + defect type + marking in the original image             (visualize.py)
```

| Defect | Detected by | Reported as |
|---|---|---|
| missing hole | geometry | Hole missing |
| hole not drilled through | geometry (edge ring at the nominal location, but no see-through) | Hole not drilled through |
| mispositioned hole | geometry (Hungarian matching, tolerance from the learning phase) | Hole mispositioned |
| wrong diameter | geometry (equivalent diameter) | Wrong hole diameter |
| wrong shape (slot/cut-out) | geometry (aspect ratio) | Wrong cut-out shape |
| extra hole | geometry | Extra hole/cut-out |
| damaged outline | geometry (mask XOR) | Outline deviation |
| scratches, discoloration, anything unexpected | diff, pca | Deviation from reference image / ML anomaly |

**Thresholds without defect examples:** Geometry tolerances come from the spread of the good parts:
max(minimum tolerance, k·σ). For `diff` and `pca`, leave-one-out measures how far each good part deviates
from the others, and the threshold is set above that with a safety margin. After learning, the system
inspects its own references. It warns if one of them stands out, which usually means a defective part was
taught in by mistake.

## Evaluation: which combination works?

```bash
python main.py evaluate --synthetic --md docs/evaluation_synthetic.md
python main.py evaluate --train images/good --test images/test    # own photos: test/ok, test/nok/<defect_type>
```

Results on synthetic data (15 training images, 40 good + 80 defective parts per scenario; full table in
[`docs/evaluation_synthetic.md`](docs/evaluation_synthetic.md)):

- **The combination `geometry + diff[norm] + pca[edges]`** detects all 8 defect types under front light
  with 0 % false alarms. Each single method covers only part of them: `geometry` misses surface defects,
  `diff[edges]` misses holes that are not drilled through, and the colour spaces (`lab_ab`, `hsv_s`) only find
  discoloration.
- **Photometric normalisation beats raw grayscale** (98 % vs. 82 %). It cancels brightness changes without
  losing sensitivity.
- **Backlight** gives very clean hole contours but makes surface defects invisible. It also cannot tell
  "not drilled through" apart from "missing": both are NOK, but reported as "missing".
  → Recommendation for the setup: **combine backlight and front light**, e.g. two captures or switchable
  lighting. This would be a good extension.
- **New shapes in any rotation** (triangle C, square D, parts arriving at 0–360°): the combination again
  detects all hole and outline defects with 0 % false alarms. Scratches on the triangle drop to 70 %, because
  the brushed texture rotates with the part (see "Surface texture" above).
- `pca[binary]` and `pca[norm]` cause false alarms in some scenarios (up to 30 %). ML is not a sure thing
  here; the preprocessing is decisive.

> **Important:** The synthetic defects are pronounced and self-modelled. The 100 % values only show that the
> pipeline works, not how well it performs on real parts. For reliable statements, repeat the evaluation
> with real photos (folder structure above), and include borderline defects in particular: small offsets,
> slightly wrong diameters, fine scratches.

## Teaching a new part type

The system does not assume any particular shape or number of holes. Outline, hole count, hole positions
and sizes are all learned from the good parts. A triangular plate with 3 holes, a square with 4 holes, an
L-bracket or a disc is taught in exactly like the rectangular plate: start learning mode, run 10–20 good
parts, finish. Each part type gets its own model, which you select before inspection.

The alignment searches the full 360° and decides between look-alike rotations (e.g. 0°/120°/240° for a
triangle) by the hole pattern. Parts can therefore arrive on the conveyor in any rotation.

What the part and the setup must satisfy:

| Requirement | Why |
|---|---|
| Clear contrast between the part and the belt (or backlight) | The part is found with an automatic threshold. Shiny metal on a matt dark belt, or backlight, works well. |
| Part completely in the image, same camera height as when learning | Everything is measured in pixels; a different distance changes all dimensions. |
| Flat part lying on the belt | The method is 2D. Tall parts, parts that can lie on different sides or tilted parts need their own model per resting position. |
| Holes are through holes, i.e. you can see the belt/backlight through them | Only then are they detected as holes. Blind holes are recognised by their edge (front light only). |
| 10–20 good parts, each placed slightly differently | The system learns the normal variation from them. Too few or identical placements make it over-sensitive. |
| Holes at least ~6 px in diameter in the image | Smaller holes are filtered as noise (`hole_min_area_px`). |

**Symmetric parts:** If the part *and* its hole pattern are fully symmetric (e.g. a square with 4 identical
holes in the corners), all matching rotations are equivalent and any of them is fine. If only the outline is
symmetric but the hole pattern is not, the hole pattern decides.

**Parts lying upside down:** A flat part turned over is a mirror image. By default it is rated NOK, because
the mirrored part may be a different part (left/right version). Set `"localization": {"allow_mirror": true}` in
`config.json` if a turned-over part is still a good part.

**Warnings after learning:** The UI warns if a reference image looks like a defective part, if the hole count
differs between references, or if hole positions scatter strongly between references. In the last case the
model would be insensitive to position errors. Check lighting and contrast, then teach in again.

**Surface texture:** Brushed or grinding textures rotate with the part. If parts arrive in random rotations,
the learned texture variation is higher and fine scratches are detected less reliably than with aligned
parts. Hole and outline defects are not affected.

## Running on the Raspberry Pi

**Recommended hardware:** Raspberry Pi 5 (or a Pi 4 with 4 GB). With a moving conveyor, use a
**global-shutter camera**, because a rolling shutter distorts moving parts. Otherwise use a Camera Module 3 or
HQ Camera with a short exposure time. Mount the camera perpendicular above the belt at a fixed distance and
shield it from ambient light.

**1. Install (once).** Use Raspberry Pi OS (Bookworm or newer); it includes the camera stack.

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-scipy python3-flask
rpicam-hello -t 5000          # camera test: a preview must appear (or no error on a headless Pi)
```

**2. Copy the project** to the Pi (e.g. unzip it to `/home/pi/pi_qc`) and create the configuration:

```bash
cd ~/pi_qc
python3 -m venv --system-site-packages .venv   # system-site-packages because of picamera2
cp config.example.json config.json             # already contains "source": "pi"
```

**3. Start:**

```bash
.venv/bin/python main.py --config config.json web --camera pi
```

**4. Open the UI** from a laptop/tablet on the same network: `http://<IP of the Pi>:8000`
(find the IP with `hostname -I`).

**5. Set up the image** before teaching in:
- Look at the live view. The part must be clearly visible, sharp and completely in the image.
- Adjust `exposure_us` in `config.json` so the metal is bright but not blown out (white). Shorter exposure =
  less motion blur on a moving belt. Restart after changing the configuration.
- If the belt runs from top to bottom in the image instead of left to right, set `"trigger": {"axis": "y"}`.
- When the part passes the centre, the trigger lines turn grey for a moment (= captured). If parts are not
  captured, the contrast to the belt is usually too low. For a stopped belt use *Manual capture*.

**6. Teach in and inspect** exactly as described in the quick start.

To start automatically at boot, use `deploy/qc.service` (instructions inside the file). Other camera sources:
`--camera usb:0`, or `--camera folder:/path/to/photos` to inspect an image folder one image at a time.

**Camera settings** (`config.json → camera`): fixed exposure (`exposure_us`), fixed gain and fixed white
balance are preset on purpose. Automatic settings would change the brightness from part to part and
disturb the reference comparisons.

**Run time:** On an x86 laptop an inspection takes about 60–90 ms (including the 360° rotation search),
and training takes about 2–3 s
(15 images, 1280×960). This has not been measured on the Pi; expect several times that. The main tuning knob
is `localization.work_width` (default 640 px).

**Security:** The web UI has no login by default. Only run it on an isolated network, or set `QC_TOKEN`.
All modifying actions then require the token; open the page with `?token=…`. Model names are checked
against path traversal.

## Command line

```bash
python main.py train   --name PlateA --images photos/good/ [--methods geometry,diff --diff-rep clahe]
python main.py inspect --model PlateA photos/new/*.png --out results/
python main.py generate --out data/synth --lighting back --part-type B
python -m pytest -q                        # 18 tests, incl. the complete web workflow with the simulator
```

## Data storage

```
data/models/<name>/     meta.json (settings, training report), arrays.npz, refs/*.png
data/results/<date>/    inspection_log.csv (all inspections), NOK images (marked + original)
data/archive/*.zip      archived batches
```

### Archiving a finished batch

When a batch or part type is finished, **stop the inspection** and click **Archive & clear results** in the
*Results archive* card. Optionally enter a batch name first; the default is the model name. The system then:

1. packs the complete results folder into `data/archive/<batch-name>_<date>_<time>.zip`, together with
   `summary.txt` / `summary.json` (parts inspected, OK/NOK, NOK rate, count per defect type, model, time range),
2. verifies the ZIP and only then empties the results folder,
3. resets counters and history, ready for the next part type.

Archives are listed in the UI with their key figures and can be downloaded or deleted there. The same from the
command line: `python main.py archive --name "Plate A batch 1"` (or `--list`).

The reference images are stored inside the model, so a model can be **retrained** at any time with different
methods or image representations (button in the UI). This is handy for comparisons.

## Project structure

```
main.py                  command line (web, train, inspect, generate, evaluate)
qc/config.py             all parameters (dataclasses, overridable via config.json)
qc/camera.py             Pi camera, USB, image folder, simulator
qc/alignment.py          localisation, alignment, photometric normalisation
qc/representations.py    image representations (gray, CLAHE, edges, threshold, LAB, HSV …)
qc/methods.py            inspection methods geometry / diff / pca
qc/model.py              learning, inspection, save/load, retraining
qc/visualize.py          defect location marking, heatmap
qc/system.py             process control, conveyor trigger, log
qc/webapp.py + static/   supervisor UI (Flask, no external dependencies)
qc/synthetic.py          synthetic parts with defects
qc/evaluate.py           comparison of method × representation × lighting
qc/archive.py            zip + summary of finished batches, clear results
tests/                   pytest
```

## Open points / extensions

- **Real images**: fine-tune the thresholds (`method` section of the configuration) on real parts.
- **Handling of NOK parts** (reject via GPIO/PLC): the hook is `QCSystem._inspect`.
- **Dimensions in mm**: everything is currently in pixels. A calibration (scale in px/mm) would allow direct
  conversion.
- **Stronger ML**: pretrained features (e.g. MobileNet via TFLite) plus PatchCore-style nearest-neighbour
  comparison could be added as another method in `methods.py`.
- **Combined lighting**, front light and backlight (see evaluation).
