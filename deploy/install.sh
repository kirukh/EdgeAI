#!/usr/bin/env bash
# Installation on the Raspberry Pi (Raspberry Pi OS Bookworm or newer) – WITH or WITHOUT sudo rights.
#
#   cd ~/pi_qc
#   bash deploy/install.sh                   # install + autostart (service "qc")
#   bash deploy/install.sh --no-autostart    # install only; start by hand
#   bash deploy/install.sh --no-camera       # e.g. on a laptop: simulator / image folders only
#
# Steps:
#   1. checks Python (≥ 3.11) and the camera library picamera2 (only available via apt – it is
#      pre-installed on Raspberry Pi OS). With sudo the system packages are installed via apt.
#   2. creates the Python environment .venv – with access to the system packages (picamera2)
#   3. installs what is still missing into .venv with pip (no sudo needed): OpenCV, SciPy, Flask,
#      waitress – with NumPy < 2, because picamera2 of the system is built for NumPy 1.x.
#      Only ready-made packages are used (never hours of compiling on the Pi).
#   4. creates config.json from the Raspberry Pi 3 + Camera Module v2 preset and asks for the setup PIN
#   5. autostart (unless --no-autostart): sudo → system service; without → user service or crontab
#
# The script can be run again at any time (e.g. after an update); it only does what is missing.
# Afterwards: open http://<IP of the Pi>:8000 on a laptop, tablet or phone.
set -euo pipefail

NO_AUTOSTART=0
NO_CAMERA=0
for arg in "$@"; do
  case "$arg" in
    --no-autostart) NO_AUTOSTART=1 ;;
    --no-camera) NO_CAMERA=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
PYTHON="${QC_PYTHON:-python3}"           # QC_PYTHON: use another Python interpreter
cd "$DIR"
if [ "${QC_NO_SUDO:-0}" != 1 ] && sudo -n true 2>/dev/null; then HAVE_SUDO=1; else HAVE_SUDO=0; fi   # QC_NO_SUDO=1 forces the user mode
echo "== Project: $DIR (user: $RUN_USER, sudo: $([ $HAVE_SUDO = 1 ] && echo yes || echo no))"

fail() { echo; echo "!! $*"; exit 1; }

echo "== 1/5 Python and camera library"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || fail "Python ≥ 3.11 required ($("$PYTHON" --version 2>&1)). Use Raspberry Pi OS Bookworm or newer."
if [ $HAVE_SUDO = 1 ]; then
  # all from apt: ready-made packages that fit together (pip then has nothing left to install)
  PKGS="python3-venv python3-numpy python3-opencv python3-scipy python3-flask python3-waitress"
  [ $NO_CAMERA = 0 ] && PKGS="python3-picamera2 $PKGS"
  sudo apt-get update -q
  # shellcheck disable=SC2086
  sudo apt-get install -y -q $PKGS
elif [ $NO_CAMERA = 0 ] && ! "$PYTHON" -c "import picamera2" 2>/dev/null; then
  fail "The camera library picamera2 is missing. It only exists as a system package – ask an administrator:
     sudo apt install python3-picamera2
   (Without a Pi camera, e.g. on a laptop: bash deploy/install.sh --no-camera)"
fi
"$PYTHON" -c "import venv, ensurepip" 2>/dev/null \
  || fail "Python cannot create environments – ask an administrator:  sudo apt install python3-venv"
echo "   $("$PYTHON" --version)$([ $NO_CAMERA = 0 ] && echo ", picamera2 present")"

echo "== 2/5 Python environment .venv"
# --system-site-packages: the environment also sees the system packages (picamera2 is not on pip).
# Running it on an existing .venv only fixes its configuration – installed packages stay.
"$PYTHON" -m venv --system-site-packages .venv
PY="$DIR/.venv/bin/python"
"$PY" -m pip install -q --upgrade pip 2>/dev/null || true

echo "== 3/5 Python packages"
MISSING=()
for pair in cv2:opencv-python-headless scipy:scipy flask:flask waitress:waitress numpy:numpy; do
  "$PY" -c "import ${pair%%:*}" 2>/dev/null || MISSING+=("${pair##*:}")
done
NUMPY_LIMIT=()
if [ $NO_CAMERA = 0 ]; then
  NUMPY_LIMIT=("numpy<2")   # picamera2 (system) is built for NumPy 1.x – NumPy 2 breaks the camera
fi
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "   installing into .venv: ${MISSING[*]}  (ready-made packages only, a few minutes)"
  "$PY" -m pip install -q --only-binary=numpy,scipy,opencv-python-headless "${MISSING[@]}" "${NUMPY_LIMIT[@]}" \
    || fail "pip could not install ${MISSING[*]}. If there is no ready-made package for this Pi, ask an administrator:
     sudo apt install python3-opencv python3-scipy python3-flask python3-waitress
   and run this script again. (Details: uname -m → $(uname -m), $("$PYTHON" --version))"
else
  echo "   all packages present"
fi
# registers the project in .venv (command "qc"); the code stays here – a new version just replaces the folder
"$PY" -m pip install -q --no-deps -e . 2>/dev/null \
  || echo "   (command 'qc' not installed – start with: .venv/bin/python main.py …)"
"$PY" - "$NO_CAMERA" <<'PY' || fail "The Python environment is not usable (see the message above)."
import sys
from importlib.metadata import version
import cv2, flask, numpy, scipy, waitress  # noqa: F401
import scipy.optimize  # noqa: F401 - used by the program; fails if SciPy and NumPy do not fit together
print(f"   NumPy {numpy.__version__}, OpenCV {cv2.__version__}, SciPy {scipy.__version__}, Flask {version('flask')}")
if sys.argv[1] == "0":
    import picamera2  # noqa: F401
    if int(numpy.__version__.split(".")[0]) >= 2:
        sys.exit("   NumPy 2 is installed in .venv – the camera library needs NumPy 1.x. Fix:\n"
                 "     .venv/bin/pip install 'numpy<2'")
PY

echo "== 4/5 Configuration"
if [ ! -f config.json ]; then
  cp config.pi3_imx219.json config.json
  PIN=""
  if [ -t 0 ]; then
    read -r -p "   Choose a PIN for the setup area (digits, e.g. 4711; empty = random): " PIN || true
  fi
  if [ -z "$PIN" ]; then PIN="$(shuf -i 1000-9999 -n 1)"; fi
  "$PY" - "$PIN" <<'PY'
import json, sys
p = "config.json"
c = json.load(open(p))
c.setdefault("ui", {})["setup_pin"] = sys.argv[1]
json.dump(c, open(p, "w"), indent=2)
PY
  echo "   config.json created – setup PIN: $PIN  (write it down; change it in config.json)"
else
  echo "   config.json exists – left unchanged"
fi
"$PY" -c "from qc.config import AppConfig; AppConfig.load('config.json'); print('   config.json is valid')"

CAMERA_ARG="--camera pi"
[ $NO_CAMERA = 1 ] && CAMERA_ARG="--camera sim"
if [ $NO_AUTOSTART = 1 ]; then
  echo "== Done (no autostart). Start by hand:"
  echo "     cd $DIR && .venv/bin/python main.py web $CAMERA_ARG      (stop with Ctrl+C)"
  echo "   Then open in the browser of a laptop/tablet/phone in the same network:"
  for ip in $(hostname -I 2>/dev/null || echo "<IP-of-the-Pi>"); do echo "     http://$ip:8000"; done
  exit 0
fi

echo "== 5/5 Autostart"
UNIT="[Unit]
Description=AI quality inspection of metal parts
After=network-online.target

[Service]
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python main.py --config config.json web $CAMERA_ARG --port 8000
Restart=always
RestartSec=3
"
if [ $HAVE_SUDO = 1 ]; then
  printf '%s\nUser=%s\n\n[Install]\nWantedBy=multi-user.target\n' "$UNIT" "$RUN_USER" | sudo tee /etc/systemd/system/qc.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now qc
  CTL="sudo systemctl"; LOG="journalctl -u qc"
elif mkdir -p ~/.config/systemd/user && systemctl --user daemon-reload 2>/dev/null; then
  printf '%s\n[Install]\nWantedBy=default.target\n' "$UNIT" > ~/.config/systemd/user/qc.service
  systemctl --user daemon-reload
  systemctl --user enable --now qc
  # Optional: keep the service running without a login (works on many Pis without sudo)
  if loginctl enable-linger "$RUN_USER" 2>/dev/null; then
    echo "   starts at boot, also without login"
  else
    echo "   starts when $RUN_USER is logged in (Pi Desktop with auto-login: at boot)"
  fi
  CTL="systemctl --user"; LOG="journalctl --user -u qc"
else
  # last resort without systemd user services: crontab @reboot + restart loop
  chmod +x deploy/run.sh
  if command -v crontab >/dev/null; then
    (crontab -l 2>/dev/null | grep -v "deploy/run.sh" || true; echo "@reboot $DIR/deploy/run.sh") | crontab -
    echo "   autostart via crontab @reboot (no systemd user services available)"
  else
    echo "   no autostart possible here – after a reboot start it with:  nohup $DIR/deploy/run.sh &"
  fi
  pgrep -f "deploy/run.sh" >/dev/null || nohup "$DIR/deploy/run.sh" >/dev/null 2>&1 &
  CTL=""; LOG="tail -n 50 $DIR/data/logs/console.log"
fi
sleep 8
if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)" 2>/dev/null; then
  echo "== Done. Open on a laptop/tablet/phone in the same network:"
  for ip in $(hostname -I); do echo "     http://$ip:8000"; done
  if [ -n "$CTL" ]; then
    echo "   Restart: $CTL restart qc   |   Log: $LOG -f   |   Stop autostart: $CTL disable --now qc"
  else
    echo "   Log: $LOG   |   Stop: pkill -f deploy/run.sh; pkill -f 'main.py --config'   |   Autostart: crontab -e"
  fi
else
  echo "== The program is not reachable yet – show the error with:  $LOG"
fi
