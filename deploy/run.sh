#!/usr/bin/env bash
# Fallback autostart without systemd (started by crontab @reboot from deploy/install.sh):
# runs the inspection and restarts it 3 s after it stops for any reason.
# Arguments are passed on (e.g. --camera pi); without, the camera comes from config.json.
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/logs
while true; do
  .venv/bin/python main.py --config config.json web "$@" --port 8000 >> data/logs/console.log 2>&1
  sleep 3
done
