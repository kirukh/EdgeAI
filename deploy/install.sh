#!/usr/bin/env bash
# One-time installation on the Raspberry Pi (Raspberry Pi OS Bookworm or newer).
#
#   cd ~/pi_qc            # the unzipped project folder (any name works)
#   bash deploy/install.sh
#
# What it does:
#   1. installs the system packages (camera library, OpenCV, web server)
#   2. creates the Python environment .venv
#   3. creates config.json from the Raspberry Pi 3 + Camera Module v2 preset (if missing)
#      and asks for the setup PIN
#   4. installs the autostart service "qc": the inspection starts by itself at every boot
#      and restarts automatically if it ever stops
#
# Afterwards: open http://<IP of the Pi>:8000 on a laptop, tablet or phone.
# Useful later:  sudo systemctl restart qc   |   journalctl -u qc -f   |   sudo systemctl disable --now qc
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
cd "$DIR"
echo "== Project: $DIR (user: $RUN_USER)"

echo "== 1/4 System packages"
sudo apt-get update -q
sudo apt-get install -y -q python3-picamera2 python3-opencv python3-scipy python3-flask python3-waitress python3-pil

echo "== 2/4 Python environment"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv       # uses the apt packages (picamera2 is not on pip)
fi

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

echo "== 4/4 Autostart service"
sudo tee /etc/systemd/system/qc.service >/dev/null <<EOF
[Unit]
Description=AI quality inspection of metal parts
After=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python main.py --config config.json web --camera pi --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now qc
sleep 3
if systemctl is-active --quiet qc; then
  echo "== Done. Open on a laptop/tablet/phone in the same network:"
  for ip in $(hostname -I); do echo "     http://$ip:8000"; done
else
  echo "== The service did not start – show the error with:  journalctl -u qc -n 50"
fi
