# Changelog

## 2.1.4 – part detection next to rails, UI theme (2026-10-20)

- **Part detection ignores structures at the image border.** Bright guide rails along the belt (or any bright or dark
  structure that runs out of the image) made the automatic polarity take the dark belt for the part: it touched the
  image border, was never "complete", the trigger never fired (teach-in only with "Capture now") and a model taught
  that way rated every part NOK. Now the polarity is the one in which a large object lies completely inside the
  image, and the part is the largest such object; the old border rule remains the fallback for parts entering or
  leaving the image. Such an object must also differ from its surroundings by at least
  `localization.min_part_contrast` grey levels (default 25), so belt texture or noise on an empty belt never counts
  as a part. Tested with a conveyor video (rails top and bottom): before 0 of 20 teach-in parts and 0 of 100
  test parts captured, now 20/20 and 100/100 (also with `trigger.detect_width` 240 as on the Pi). New test
  `test_rails_at_the_image_border_are_not_the_part`. The synthetic evaluation is unchanged (identical table).
- **UI theme "measuring table"**: graphite machine bar with a safety-yellow edge, faint measuring grid, viewfinder
  corners around the camera image, monospace for labels and numbers. OK/NOK as solid colour blocks with white text
  (readable from a distance); NOK is also striped, so it differs from OK not only by colour. Light and dark scheme,
  system fonts only (offline, CSP unchanged). Screenshots in `docs/` updated.

## 2.1.3 – CI, faster tests, code clean-up (2026-10-14)

No change to the inspection: the synthetic evaluation (`qc evaluate --synthetic`, 4 scenarios × 15 configurations)
gives exactly the same table as 2.1.2.

- **Continuous integration** (`.github/workflows/ci.yml`): `ruff format --check`, `ruff check`, `mypy` and
  `pytest --cov` on Python 3.11 (Raspberry Pi OS Bookworm) and 3.13 (Trixie) for every pull request and every push
  to `main`.
- **Tests 3.5× faster** (158 s → ~45 s on 4 cores): they run in parallel (`pytest-xdist`, one thread per worker –
  NumPy/OpenCV threads on top of the workers made a parallel run slower than a serial one), and the simulator reuses
  8 pre-generated sensor noise frames instead of drawing new random numbers for every frame (33 → 11 ms per frame;
  this also lowers the CPU load of the simulator in the web UI). Training and test data (`synthetic.compose`) are
  unchanged.
- **Test coverage 91 % → 94 %** (77 tests): all image representations, `qc evaluate` with own photo folders and CSV
  output, the Pi camera with focus motor (Camera Module 3) against the picamera2 stand-in, the camera sources from
  the config, the operator endpoints (live image, PIN state, result images, feedback).
- **`qc/improvement.py`**: the continuous-improvement logic (operator decisions, background update, release, undo,
  accepted parts, notices) moved out of `QCSystem` into its own component `ModelImprovement`, like `CameraSetup`
  (`system.py` 791 → 638 lines). The setup API calls `system.improve.…`.
- **Type check with current libraries:** `mypy` reported 17 errors against the type information of NumPy 2.5 and
  OpenCV 5 (earlier runs had not seen it). Fixed with annotations; `read_image()` gives a clear error for an
  unreadable image in `qc evaluate` instead of failing later.
- Test fix: `test_operator_acceptance_updates_the_model_during_inspection` failed about once in 10 runs – a part
  that was already being inspected with version 1 when version 2 was released is correctly logged as version 1;
  the test now waits for the first part of version 2.

## 2.1.2 – Raspberry Pi OS Trixie (2026-10-12)

- `deploy/install.sh`: no fixed `numpy<2` any more. On Raspberry Pi OS Trixie (Debian 13, Python 3.13) `picamera2`
  is built for NumPy 2 and NumPy 1.x does not exist for Python 3.13 – pip failed with *Could not find a version that
  satisfies the requirement numpy<2*. The installer now pins the NumPy version of the system (`numpy==<system>`,
  Bookworm 1.x, Trixie 2.x), so pip only picks OpenCV/SciPy versions that fit it. A NumPy inside `.venv` that hides
  the system NumPy is removed; the final check compares the major version with the system's.
- `pyproject.toml`: plain `numpy>=1.24` (the ARM marker `<2` was wrong for Trixie).
- Tested: failure reproduced with Python 3.13 + system NumPy 2.2.4 (as on the Pi); fixed installer on the same
  state → NumPy 2.2.4 kept, OpenCV, SciPy, Flask, waitress installed, `qc` registered, `config.json` valid.

## 2.1.1 – installation without sudo (2026-10-12)

- `deploy/install.sh`: without sudo, missing packages (OpenCV, SciPy, Flask, waitress) are installed into `.venv`
  with pip – ready-made packages only (no compiling on the Pi) and NumPy < 2, because `picamera2` from apt is built
  for NumPy 1.x. Clear messages for a missing `picamera2` or `python3-venv`; final check imports the modules the
  program really uses. New option `--no-camera` (laptop/simulator). Registers the command `qc` (`pip install -e .`).
  Can be run again at any time (also after an update).
- `pyproject.toml`: NumPy < 2 on ARM Linux (Raspberry Pi) via environment markers, so `pip install -e .` is safe there.
- `docs/INSTALL.de.md`: step-by-step installation guide (German).
- Tested: installer with Python 3.11 + system NumPy 1.24 without sudo (→ NumPy 1.26, OpenCV 4.11), second run,
  missing `picamera2`, laptop mode; all 72 tests pass on Python 3.11 + NumPy 1.26 + OpenCV 4.11 as on the Pi.

## 2.1.0 – continuous improvement (2026-10-12)

Requirement of the supervising professor: the model must improve during operation. Implemented as controlled
learning from the operator's decisions (human in the loop), not as unsupervised self-learning.

### Added
- Operator screen: *Within our tolerance – accept as OK* on a NOK result; the part waits for the next model update.
  *Part is actually DEFECTIVE* (and confirmed NOK results) become test parts for the release check.
- `qc/learning.py`: model update after `learning.update_after` accepted parts (default 5) or by button, trained in
  the background while the inspection continues; release check (every known defective part must still be detected);
  version number + history; *Undo last update* (swap, can itself be undone); accepted parts can be viewed and
  discarded in the setup area.
- Local tolerances instead of global thresholds (`methods.py`): accepted hole deviations widen only that hole's
  tolerance (deviation + k·σ); grey-value and ML deviations become tolerance maps only where the accepted part was
  conspicuous. Nominal geometry, mean image and thresholds are learned from the normal good parts only.
  Test: offset 8–9 px at the accepted hole 100 % → 0 % NOK; 18 px, other hole, scratches and good parts unchanged.
- Model version in the inspection and feedback logs; the batch summary compares the versions (false alarm rate).
- 11 tests (`tests/test_learning.py`); 72 tests in total, 91 % coverage; ruff 0 findings, mypy 0 errors,
  no function above complexity 15.

### Unchanged
- Without accepted parts the inspection is identical to 2.0.0 (golden master, 250 parts: 0 differences).
- Models of 2.0 load (no learned tolerances).

## 2.0.0 – core version (2026-10-08)

Goal: "as compact as possible, but as robust and accurate as possible". The inspection itself is unchanged –
verified with a golden-master comparison of 250 synthetic parts (4 part types, front and backlight): identical
OK/NOK decisions, defect types and reference screening; scores differ by at most 1·10⁻⁶ (floating-point noise
of the parallel matrix maths – version 1 differs from itself by the same amount between two runs).

### Removed (not needed for the task – KISS/YAGNI)
- Dual light (backlight + front light per part), incl. light switching
- Self-learning during operation (collected parts, background training, safety checks, undo)
- Reference part self-test
- GPIO outputs (reject, lamps) – handling of NOK parts comes later
- Checkerboard calibration → replaced by one scale value `camera.mm_per_px`
- Retraining with other methods from the UI (use `qc evaluate` to compare methods)

### Structure
- `QCSystem` (700 lines + 3 mixins sharing ~105 attributes) → coordinator plus components with one
  responsibility each: `CameraSetup`, `TeachIn`, `inspect_part`/`vote`, `ResultStore`, `DriftMonitor`
- `Recipe` merged into `QCModel` (sensitivity and camera profile are stored with the model)
- Web UI: 370-line `create_app` → package `qc/web` with two Flask blueprints (operator / setup) and `auth.py`
- `QCModel.train` split into prepare → screen → report (`FitResult` instead of temporary attributes)
- `GeometryMethod.fit/score` split into named steps
- Command line moved to `qc/cli.py`; `main.py` stays as entry point (autostart unchanged)
- Inspection runs outside the lock → the web UI no longer waits for an inspection on the Pi 3

### Security
- CSRF protection: modifying requests need the `X-QC-Client` header
- PIN brute-force lockout per device instead of global (no lock-out of the technician)
- Security headers (CSP, X-Frame-Options, nosniff, Referrer-Policy, no-store for the API)
- Training report and all setup functions require the setup PIN

### Robustness
- Parts that could not be inspected (Pi overloaded) are logged as `NOT_INSPECTED` and counted in the batch summary
- `config.json` and single-light models of version 1 still load; removed config sections are reported, not fatal

### Code standards
- `pyproject.toml` (PEP 621) with pytest, coverage, ruff and mypy configuration
- PEP 8 formatting (ruff format), 0 lint findings, 0 mypy errors, cyclomatic complexity ≤ 15 everywhere
- 61 tests (was 67, of which 16 covered removed features), 90 % coverage

| Metric | 1.x | 2.0 |
|---|---|---|
| Python statements | 4 603 | 3 607 (−22 %) |
| Lint findings (same rule set) | 330 | 0 |
| mypy errors | 353 | 0 |
| Functions rated D/E/F (radon) | 9 | 0 |
| Largest function | 369 lines | 77 lines |
| Test coverage | – | 90 % |

## 1.x – full version
Reference screening (leave-one-out) added; see README, section "Reference screening".
