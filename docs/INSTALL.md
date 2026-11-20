# Installation on a new Raspberry Pi – step by step

Applies from version 2.5.0. Raspberry Pi 5 + USB camera (Logitech C310) or Raspberry Pi 3/4/5 + Camera Module.
The install script without sudo is tested with Python 3.13 + NumPy 2.2 (as on Raspberry Pi OS **Trixie**) and
Python 3.11 + NumPy 1.24 (as on **Bookworm**); the sudo path (apt) could not be tested outside a real Pi.

## 0. What you need

| | |
|---|---|
| Hardware | Raspberry Pi 5 (or 3 Model B / 4), power supply, microSD ≥ 16 GB |
| Camera | USB camera, e.g. Logitech C310 – or Camera Module v2 (IMX219) on the camera connector |
| Optional | Grove-LCD RGB Backlight (16 × 2, I²C) for the piece counter next to the conveyor |
| Operating system | Raspberry Pi OS **Trixie** or **Bookworm** (Desktop or Lite; 64-bit recommended) |
| Network | Pi and laptop/tablet in the same network (Wi-Fi or LAN) – or a screen, keyboard and mouse on the Pi |

With **sudo rights** the script installs missing system packages itself. **Without sudo** only the camera library
must be there (`picamera2` for the Camera Module – usually preinstalled on Raspberry Pi OS; `v4l-utils` is
recommended for a USB camera); everything else goes into a Python environment of its own, without sudo.

## 1. Prepare the Pi (fresh Pi only)

1. **Connect the camera.** USB camera: any USB port (blue USB 3 ports are best). Camera Module: **with the Pi
   switched off**, ribbon cable contacts to the correct side, latch closed.
2. **Install the operating system** with the *Raspberry Pi Imager* on the laptop: device, Raspberry Pi OS (64-bit)
   and SD card. Under *Edit settings*: hostname (e.g. `qc-pi`), user + password, Wi-Fi and **enable SSH**.
3. Start the Pi and open a terminal on it – or connect from the laptop:
   ```bash
   ssh <user>@qc-pi.local               # or ssh <user>@<IP of the Pi>
   ```
4. Quick check:
   ```bash
   python3 --version                    # 3.11 or newer
   v4l2-ctl --list-devices              # USB camera: e.g. "UVC Camera … /dev/video0"
   rpicam-hello --list-cameras          # Camera Module: must show "imx219"
   hostname -I                          # IP address of the Pi – note it for the browser
   ```
   No camera shown: check the cable (Camera Module: switch the Pi off first).

## 2. Get the project onto the Pi

From GitHub (with the access you were given):
```bash
cd ~
git clone https://github.com/kirukh/EdgeAI.git
cd ~/EdgeAI
git checkout claude/dazzling-rubin-ww9wch      # the branch with the current version
git log --oneline -1                            # shows the version you have
```
`git clone` asks for a user name and password: on GitHub the password is an access **token**, not the account
password.

Alternatively from a ZIP: copy it with `scp EdgeAI.zip <user>@<IP of the Pi>:~`, then `unzip EdgeAI.zip` on the Pi
and `cd` into the folder.

## 3. Install

```bash
bash deploy/install.sh --usb --no-autostart    # USB camera, e.g. Raspberry Pi 5 + Logitech C310
bash deploy/install.sh --no-autostart          # Camera Module on the camera connector
```
`unknown option --usb`: the folder holds an older version – `git checkout claude/dazzling-rubin-ww9wch` (section 2).

**USB camera:** `--usb` installs the camera tools `v4l-utils` instead of `picamera2` and uses the preset
`config.pi5_usb_c310.json` (camera `usb:0`, recording 1280 × 960 MJPG, working size 960 × 720, 50 Hz, fixed white
balance, flicker-free exposure, stop-and-go capture mode, LCD display on). Which number the camera has is shown by
`v4l2-ctl --list-devices` (entry *UVC Camera*, `/dev/video0` → `usb:0`); enter another number in `config.json`
under `"camera": {"source": …}`.

What you see (roughly):
```
== 1/5 Python and camera library
   Python 3.13.5
== 2/5 Python environment .venv
== 3/5 Python packages
   installing into .venv: opencv-python-headless scipy flask waitress  (ready-made packages only, a few minutes)
   keeping the system NumPy 2.2.4
== 4/5 Configuration
   Choose a PIN for the setup area (digits, e.g. 4711; empty = random): ____
   preset: config.pi5_usb_c310.json
   config.json created – setup PIN: 4711
   config.json is valid
== Done (no autostart). Start by hand: …
```

**Write the PIN down** – it opens the setup area. It can be changed later in `config.json`
(`"ui": {"setup_pin": "…"}`).

What the script does – and why:

| Step | What | Why |
|---|---|---|
| 1 | checks Python ≥ 3.11 and the camera library; with sudo: installs missing system packages | `picamera2` only exists as a system package (apt), not via pip |
| 2 | creates `.venv` with `--system-site-packages` | an environment of its own (no sudo needed) that still sees the system's camera library |
| 3 | installs missing packages into `.venv` – ready-made packages only, **NumPy stays the system's** | `picamera2` is built for the system NumPy (Trixie 2.x, Bookworm 1.x); another major version would break the camera. No hours of compiling on the Pi |
| 4 | `config.json` from the preset for the camera, asks for the PIN | matching camera settings |
| 5 | autostart – skipped with `--no-autostart` | see section 9 |

The script may run again at any time (e.g. after an update); it only does what is still missing. An existing
`config.json` is left unchanged.

## 4. LCD display (optional)

Grove-LCD RGB Backlight on the I²C pins of the Pi (SDA, SCL, 5 V, GND). Switch I²C on once:
`sudo raspi-config` → *Interface Options* → *I2C*. Then:
```bash
sudo apt install python3-smbus i2c-tools      # if not installed yet
i2cdetect -y 1                                # should show 3e (text) and 30 (backlight; 62 on older displays)
.venv/bin/qc lcd-test                         # runs through every state with its colour
```
In operation: line 1 `Part <n>` (piece counter), line 2 `OK` (green) / `NOT OK` (red); blue = ready / nothing
happening / stopped. The display is reset every time `qc web` starts and goes dark when it ends. Switch it on
or off and change the texts in `config.json`:
```json
"lcd": { "enabled": true, "result_seconds": 5, "ok_text": "OK", "nok_text": "NOT OK" }
```

## 5. Start

```bash
cd ~/EdgeAI
.venv/bin/qc web                        # USB camera: the camera is set in config.json (usb:0)
.venv/bin/qc web --camera pi            # Camera Module
```
The terminal shows among others `Configuration: /home/<user>/EdgeAI/config.json`, `Camera: …`, `LCD display: on …`
and `Web UI: http://0.0.0.0:8000/`. **Stop with `Ctrl+C`.**

In the browser of a **laptop, tablet or phone** (or on the Pi itself):
```
http://<IP of the Pi>:8000
```

## 6. First set-up at the machine

1. **Mount the camera:** vertically above the belt, fixed distance, even light, shield stray light. A matt, dark
   belt under bright metal parts is best. The Logitech C310 has a fixed focus – it is sharp from about 40 cm.
2. **Capture mode** (🔒 *Setup* → PIN → tab *Setup* → *Take the picture*):
   - *while the part passes the centre* – continuous belt,
   - *when the part rests in the centre* – stop-and-go belt (sharpest image),
   - *only with the capture button* – no belt: lay a part under the camera, press *Capture* (or the space bar).
3. **Sharpness** (Camera Module v2 only, it has no autofocus): tab *Setup* → *Focus assistant on*. Lay a part under
   the camera and turn the lens slowly with the plastic tool until the live image shows ≈ 100 %. Then
   *Focus assistant off* and *← Operator view · lock*.
4. **Teach in a part:** *＋ Teach in new part* → name → *Next* → lay **one** good part in the middle →
   *Adjust camera automatically* → *Next* → feed **15–30 good parts** (belt running, or one by one with *Capture* in
   the manual mode). The first part measures belt direction and zoom and is not counted. The model is then created
   by itself.
5. **Inspect:** *▶ Start inspection*.

## 7. Check that it works

```bash
# stop the server with Ctrl+C first – only one program can use the camera at a time
.venv/bin/qc benchmark                  # time per part on this Pi
```
Then run a few known good and defective parts and compare the results.

## 8. In operation

| Situation | What to do |
|---|---|
| NOK, but the part is still within the company's tolerance | *Within our tolerance – accept as OK* → the model learns this deviation at the next update (automatically after 5 accepted parts, inspection keeps running) |
| OK, but the part is defective | *Part is actually DEFECTIVE* → becomes a test part that every new model version must detect |
| Batch finished | *✓ Finish batch* → ZIP with log, NOK images and figures per model version |
| An update did not fit | 🔒 *Setup* → *Models* → *Undo last update* |

When the update starts automatically: `config.json` → `"learning": {"update_after": 5}` (0 = only with the button).

## 9. Autostart (optional, later)

```bash
cd ~/EdgeAI
bash deploy/install.sh --usb            # without --no-autostart (--usb only for a USB camera)
```
With sudo: system service `qc` (starts at boot). Without sudo: user service `qc` (starts as soon as the user is
logged in – with desktop auto-login that is at boot), as a last resort `crontab @reboot`.

| | with sudo | without sudo |
|---|---|---|
| Restart after changing `config.json` | `sudo systemctl restart qc` | `systemctl --user restart qc` |
| Live log | `journalctl -u qc -f` | `journalctl --user -u qc -f` |
| Remove the autostart | `sudo systemctl disable --now qc` | `systemctl --user disable --now qc` |

The log is always also written to `~/EdgeAI/data/logs/qc.log`.

## 10. Update to a new version

```bash
systemctl --user stop qc 2>/dev/null; sudo systemctl stop qc 2>/dev/null   # only if the autostart is running
cd ~/EdgeAI
git pull                                                     # models, results, config.json and .venv stay
bash deploy/install.sh --usb --no-autostart                  # adds what the new version needs
```
A new setting of the preset (e.g. the LCD) is not copied into an existing `config.json` – add it by hand, or
`cp config.pi5_usb_c310.json config.json` (then choose the PIN in `config.json` again).

**Never delete the folder `~/EdgeAI/data`** – it holds the taught part types (`data/models`) and the results.

## 11. Troubleshooting

| Message / problem | Cause and fix |
|---|---|
| `unknown option --usb` | Older version in the folder → `git checkout claude/dazzling-rubin-ww9wch`, then run the script again |
| `The camera library picamera2 is missing` | System package missing. With sudo: `sudo apt install python3-picamera2`, otherwise ask an administrator |
| `Python cannot create environments` | `sudo apt install python3-venv` |
| `pip could not install …` | No ready-made package for this Pi: `sudo apt install python3-opencv python3-scipy python3-flask python3-waitress`, then run the script again |
| `.venv uses NumPy …, the camera library is built for NumPy …` | `.venv/bin/pip uninstall -y numpy`, then run the script again |
| `Configuration error …` at start | The message names the wrong entry in `config.json` |
| `Camera 'usb:0' could not be opened` | Camera plugged in? `v4l2-ctl --list-devices`; another program using it? |
| `Corrupt JPEG data …` in the terminal | Harmless message of the C310 (versions from 2.3.0 hide it) |
| No camera / black image (Camera Module) | Switch the Pi off, check the cable; `rpicam-hello --list-cameras` (stop the program first) |
| `Address already in use` | An instance is already running (autostart?) → `systemctl --user stop qc` or `pgrep -af "[m]ain.py\|qc web"` |
| Page does not load | Same network? Check the IP with `hostname -I`; port 8000 |
| LCD shows old text / stays dark | Terminal at start: `LCD display: off` → `"lcd": {"enabled": true}` in `config.json`; `LCD display not reachable` → cable, I²C on, `i2cdetect -y 1` |
| PIN forgotten | In `~/EdgeAI/config.json` under `"setup_pin"` |
| "The image is getting blurry" | Clean the lens, check the focus (Camera Module) or the distance (C310), teach the part in again |

## Appendix: installation by hand (without the script)

For understanding – at its core the script does exactly this:
```bash
cd ~/EdgeAI
python3 -m venv --system-site-packages .venv        # environment that sees the system's camera library
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e . "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')"   # keep the system NumPy
cp config.pi5_usb_c310.json config.json             # or config.pi3_imx219.json; then change "setup_pin"
.venv/bin/qc web
```
On a laptop without a camera: `bash deploy/install.sh --no-camera` and then `.venv/bin/qc web --camera sim`
(simulator).
