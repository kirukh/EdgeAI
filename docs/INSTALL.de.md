# Installation auf einem neuen Raspberry Pi – Schritt für Schritt

Gilt für Version 2.1. Getestet ist das Installationsskript mit Python 3.11 (wie Raspberry Pi OS Bookworm),
mit und ohne fehlende Pakete; der sudo-Weg (apt) konnte außerhalb eines echten Pi nicht getestet werden.

## 0. Was du brauchst

| | |
|---|---|
| Hardware | Raspberry Pi 3 Model B (oder neuer), Camera Module v2 (IMX219), Netzteil 5,1 V / 2,5 A, microSD ≥ 16 GB |
| Betriebssystem | Raspberry Pi OS **Bookworm** oder neuer (Desktop oder Lite, 32- oder 64-bit) |
| Netzwerk | Pi und Laptop/Tablet im selben Netz (WLAN oder LAN) |
| Projekt | `EdgeAi_v2.1_learning.zip` (bzw. die neueste Version) |

Mit **sudo-Rechten** installiert das Skript fehlende Systempakete selbst. **Ohne sudo** muss nur die
Kamera-Bibliothek `picamera2` vorhanden sein – auf Raspberry Pi OS ist sie normalerweise vorinstalliert; alles
andere installiert das Skript ohne sudo in eine eigene Python-Umgebung.

## 1. Pi vorbereiten (nur bei einem frischen Pi)

1. **Kamera anschließen – Pi dabei ausgeschaltet.** Flachbandkabel: Kontakte zur richtigen Seite, Verriegelung zu.
2. **Betriebssystem aufspielen** mit dem *Raspberry Pi Imager* auf dem Laptop: Gerät, Raspberry Pi OS (Bookworm)
   und SD-Karte wählen. Unter *Einstellungen bearbeiten*: Hostname (z. B. `qc-pi`), Benutzer + Passwort,
   WLAN und **SSH aktivieren**.
3. Pi starten und vom Laptop aus verbinden:
   ```bash
   ssh <benutzer>@qc-pi.local          # oder ssh <benutzer>@<IP-des-Pi>
   ```
4. Kurz prüfen:
   ```bash
   python3 --version                    # 3.11 oder höher
   rpicam-hello --list-cameras          # muss "imx219" zeigen
   hostname -I                          # IP-Adresse des Pi – für den Browser merken
   ```
   Zeigt `rpicam-hello` keine Kamera: Pi ausschalten, Kabel prüfen.

## 2. Projekt auf den Pi kopieren

Auf dem **Laptop**, im Ordner mit der ZIP:
```bash
scp EdgeAi_v2.1_learning.zip <benutzer>@<IP-des-Pi>:~
```
Auf dem **Pi**:
```bash
cd ~
unzip EdgeAi_v2.1_learning.zip          # → Ordner ~/pi_qc
cd ~/pi_qc
```

## 3. Installieren

```bash
bash deploy/install.sh --no-autostart
```

Was du siehst (ohne sudo, ungefähr):
```
== 1/5 Python and camera library
   Python 3.11.2, picamera2 present
== 2/5 Python environment .venv
== 3/5 Python packages
   installing into .venv: opencv-python-headless scipy flask waitress  (ready-made packages only, a few minutes)
   NumPy 1.26.4, OpenCV 4.11.0, SciPy 1.x, Flask 3.x
== 4/5 Configuration
   Choose a PIN for the setup area (digits, e.g. 4711; empty = random): ____
   config.json created – setup PIN: 4711
   config.json is valid
== Done (no autostart). Start by hand: …
```

**Die PIN aufschreiben** – sie öffnet den Einrichter-Bereich. Ändern lässt sie sich später in `config.json`
(`"ui": {"setup_pin": "…"}`).

Was das Skript tut – und warum:

| Schritt | Was | Warum |
|---|---|---|
| 1 | prüft Python ≥ 3.11 und `picamera2` | `picamera2` gibt es nur als Systempaket (apt), nicht über pip |
| 2 | legt `.venv` mit `--system-site-packages` an | eigene Umgebung (ohne sudo), die trotzdem `picamera2` des Systems sieht |
| 3 | installiert fehlende Pakete in `.venv` – nur fertige Pakete, **NumPy < 2** | `picamera2` ist für NumPy 1.x gebaut; NumPy 2 würde die Kamera zerstören. Kein stundenlanges Kompilieren auf dem Pi |
| 4 | `config.json` aus der Vorlage für Pi 3 + Camera Module v2, fragt die PIN | passende Kamera-Einstellungen (voller Bildausschnitt, 640×480, 15 fps) |
| 5 | Autostart – entfällt mit `--no-autostart` | siehe Abschnitt 8 |

Das Skript darf jederzeit erneut laufen (z. B. nach einem Update); es macht nur, was noch fehlt.

## 4. Starten

```bash
cd ~/pi_qc
.venv/bin/qc web --camera pi            # gleichwertig: .venv/bin/python main.py web --camera pi
```
Im Terminal erscheint u. a. `Camera: sensor imx219, readout 1640×1232 …` und `Web UI: http://0.0.0.0:8000/`.
**Beenden mit `Ctrl+C`.**

Im Browser eines **Laptops, Tablets oder Handys** (nicht auf dem Pi selbst – das kostet dem Pi 3 zu viel Leistung):
```
http://<IP-des-Pi>:8000
```

## 5. Erste Einrichtung an der Anlage

1. **Kamera montieren:** senkrecht über dem Band, fester Abstand, gleichmäßiges Licht, Streulicht abschirmen.
   Am besten mattes, dunkles Band unter blanken Metallteilen.
2. **Schärfe einstellen** (die Camera Module v2 hat keinen Autofokus): 🔒 *Setup* → PIN → Reiter *Setup* →
   *Focus assistant on*. Ein Teil unter die Kamera legen, das Objektiv mit dem Plastikwerkzeug langsam drehen, bis
   der Wert im Livebild ≈ 100 % zeigt. Danach *Focus assistant off* und *← Operator view · lock*.
3. **Teil einlernen:** *＋ Teach in new part* → Name → *Next* → Band anhalten, **ein** Gutteil mittig hinlegen →
   *Adjust camera automatically* → *Next* → Band starten und **15 Gutteile** laufen lassen. Das erste Teil misst
   Bandrichtung und Zoom und zählt nicht. Danach entsteht das Modell von selbst.
4. **Prüfen:** *▶ Start inspection*.

## 6. Funktion prüfen

```bash
# Server vorher mit Ctrl+C stoppen – die Kamera kann nur ein Programm gleichzeitig nutzen
.venv/bin/qc benchmark                  # Zeit pro Teil auf diesem Pi
```
Dann ein paar bekannte Gut- und Schlechtteile laufen lassen und die Ergebnisse vergleichen.

## 7. Im Betrieb

| Situation | Was tun |
|---|---|
| NOK, aber das Teil ist für das Unternehmen noch in Toleranz | *Within our tolerance – accept as OK* → das Modell lernt diese Abweichung beim nächsten Update (nach 5 akzeptierten Teilen automatisch, Prüfung läuft weiter) |
| OK, aber das Teil ist defekt | *Part is actually DEFECTIVE* → wird Testteil, das jede neue Modellversion erkennen muss |
| Charge fertig | *✓ Finish batch* → ZIP mit Protokoll, NOK-Bildern und Kennzahlen pro Modellversion |
| Update hat nicht gepasst | 🔒 *Setup* → *Models* → *Undo last update* |

Wann das Update automatisch startet: `config.json` → `"learning": {"update_after": 5}` (0 = nur per Knopf).

## 8. Autostart (optional, später)

```bash
cd ~/pi_qc
bash deploy/install.sh                  # ohne --no-autostart
```
Mit sudo: Systemdienst `qc` (startet beim Booten). Ohne sudo: Benutzerdienst `qc` (startet, sobald der Benutzer
angemeldet ist – mit Auto-Login am Desktop also beim Booten), notfalls `crontab @reboot`.

| | mit sudo | ohne sudo |
|---|---|---|
| Neustart nach Änderung an `config.json` | `sudo systemctl restart qc` | `systemctl --user restart qc` |
| Log live | `journalctl -u qc -f` | `journalctl --user -u qc -f` |
| Autostart wieder entfernen | `sudo systemctl disable --now qc` | `systemctl --user disable --now qc` |

Das Log steht immer zusätzlich in `~/pi_qc/data/logs/qc.log`.

## 9. Update auf eine neue Version

```bash
systemctl --user stop qc 2>/dev/null; sudo systemctl stop qc 2>/dev/null   # nur falls Autostart läuft
mv ~/pi_qc ~/pi_qc_old                                       # alte Version als Sicherung
unzip ~/EdgeAi_<neue-version>.zip -d ~                       # → neuer Ordner ~/pi_qc
mv ~/pi_qc_old/data ~/pi_qc_old/config.json ~/pi_qc_old/.venv ~/pi_qc/
cd ~/pi_qc && bash deploy/install.sh --no-autostart          # ergänzt, was die neue Version braucht
```
Modelle, Ergebnisse, PIN und Python-Umgebung bleiben erhalten. Läuft alles, kann `~/pi_qc_old` gelöscht werden.

## 10. Fehlerbehebung

| Meldung / Problem | Ursache und Lösung |
|---|---|
| `The camera library picamera2 is missing` | Systempaket fehlt. Mit sudo: `sudo apt install python3-picamera2`, sonst Administrator fragen |
| `Python cannot create environments` | `sudo apt install python3-venv` (Administrator) |
| `pip could not install …` | Kein fertiges Paket für diesen Pi. Administrator: `sudo apt install python3-opencv python3-scipy python3-flask python3-waitress`, dann Skript erneut starten |
| `NumPy 2 is installed in .venv` | `.venv/bin/pip install "numpy<2"` |
| `Configuration error …` beim Start | Die Meldung nennt den falschen Eintrag in `config.json` |
| `section "self_learning" … is ignored` | Nur ein Hinweis (Konfiguration einer älteren Version) – kann bleiben |
| Keine Kamera / schwarzes Bild | Pi aus, Kabel prüfen; `rpicam-hello --list-cameras` (Programm vorher stoppen) |
| `Address already in use` | Läuft schon eine Instanz (Autostart?) → `systemctl --user stop qc` bzw. `pkill -f "web --camera"` |
| Seite lädt nicht | Gleiches Netz? IP mit `hostname -I` prüfen; Port 8000 |
| PIN vergessen | steht in `~/pi_qc/config.json` unter `"setup_pin"` |
| „The image is getting blurry“ | Objektiv reinigen, mit dem Fokus-Assistenten nachstellen, Teil neu einlernen |

## Anhang: Installation von Hand (ohne Skript)

Zum Verständnis – das Skript macht im Kern genau das:
```bash
cd ~/pi_qc
python3 -m venv --system-site-packages .venv        # Umgebung, die picamera2 des Systems sieht
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .                          # Abhängigkeiten aus pyproject.toml; auf dem Pi automatisch NumPy < 2
cp config.pi3_imx219.json config.json               # danach "setup_pin" in config.json ändern
.venv/bin/qc web --camera pi
```
Auf einem Laptop ohne Pi-Kamera: `bash deploy/install.sh --no-camera` und dann `.venv/bin/qc web` (Simulator).
