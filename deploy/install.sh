#!/usr/bin/env bash
# One-time installation on the Raspberry Pi (Raspberry Pi OS Bookworm or newer).
# Works WITH or WITHOUT sudo rights – the script detects which and adapts.
#
#   cd ~/pi_qc            # the unzipped project folder (any name works)
#   bash deploy/install.sh
#
# What it does:
#   1. system packages: installs them with sudo, otherwise only checks they are present
#      (camera library, OpenCV, SciPy, Flask – Raspberry Pi OS Desktop has most of them)
#   2. creates the Python environment .venv; the web server waitress is installed into it
#   3. creates config.json from the Raspberry Pi 3 + Camera Module v2 preset (if missing)
#      and asks for the setup PIN
#   4. autostart:  with sudo    → system service "qc" (starts at boot)
#                  without sudo → user service "qc" (starts when your user is logged in –
#                                 on the Pi Desktop with auto-login: at boot)
#
# Afterwards: open http://<IP of the Pi>:8000 on a laptop, tablet or phone.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
cd "$DIR"
if [ "${QC_NO_SUDO:-0}" != 1 ] && sudo -n true 2>/dev/null; then HAVE_SUDO=1; else HAVE_SUDO=0; fi   # QC_NO_SUDO=1 forces the user mode
echo "== Project: $DIR (user: $RUN_USER, sudo: $([ $HAVE_SUDO = 1 ] && echo yes || echo no))"

echo "== 1/4 System packages"
if [ $HAVE_SUDO = 1 ]; then
  sudo apt-get update -q
  sudo apt-get install -y -q python3-picamera2 python3-opencv python3-scipy python3-flask python3-waitress python3-pil
else
  MISSING=""
  for mod in picamera2:python3-picamera2 cv2:python3-opencv scipy:python3-scipy numpy:python3-numpy; do
    python3 -c "import ${mod%%:*}" 2>/dev/null || MISSING="$MISSING ${mod##*:}"
  done
  if [ -n "$MISSING" ]; then
    echo "   Missing system packages:$MISSING"
    echo "   Ask an administrator to run:  sudo apt install$MISSING"
    exit 1
  fi
  echo "   camera library, OpenCV, SciPy: present"
fi

echo "== 2/4 Python environment"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv       # uses the system packages (picamera2 is not on pip)
fi
for pkg in flask waitress; do
  if ! .venv/bin/python -c "import $pkg" 2>/dev/null; then
    echo "   installing $pkg into .venv"
    .venv/bin/pip install -q "$pkg"
  fi
done

echo "== 3/4 Configuration"
if [ ! -f config.json ]; then
  cp config.pi3_imx219.json config.json
  PIN=""
  if [ -t 0 ]; then
    read -r -p "Choose a PIN for the setup area (digits, e.g. 4711; empty = random): " PIN || true
  fi
  if [ -z "$PIN" ]; then PIN="$(shuf -i 1000-9999 -n 1)"; fi
  .venv/bin/python - "$PIN" <<'PY'
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
.venv/bin/python -c "from qc.config import AppConfig; AppConfig.load('config.json'); print('   config.json is valid')"

echo "== 4/4 Autostart"
UNIT="[Unit]
Description=AI quality inspection of metal parts
After=network-online.target

[Service]
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python main.py --config config.json web --camera pi --port 8000
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
if .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)" 2>/dev/null; then
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
