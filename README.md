# Gaussian Splatting: Android-Aufnahme + GPU-Training am Desktop

Ein Objekt mit dem Handy fotografieren, die Kamerapositionen per **ARCore** mitschreiben, die
Aufnahme über **Tailscale** an einen Desktop-PC schicken, dort auf der **Grafikkarte** ein
3D-Gaussian-Splatting-Modell trainieren und das Ergebnis in der App im **3D-Viewer** ansehen.

```
 Android-App (Kotlin)                                Desktop-PC (reines C11 + CUDA)
┌─────────────────────────────┐   Tailscale (WireGuard)  ┌──────────────────────────────────┐
│ ARCore-Aufnahme             │   HTTP  100.x.y.z:8080   │ gsplat-server                    │
│  • Fotos (CPU-Bild, JPEG)   │ ───────────────────────► │  • REST-API, Job-Verwaltung      │
│  • Kamera-Pose je Foto      │  transforms.json,        │  • Trainer auf der GPU (CUDA):   │
│  • Intrinsik (fx,fy,cx,cy)  │  images/, points3d.ply   │    Projektion, Tile-Rasterizer,  │
│  • Punktwolke (Tiefe+AR)    │                          │    L1+SSIM-Loss, Backprop, Adam, │
│ Upload / Status / Download  │ ◄─────────────────────── │    Densification (clone/split/   │
│ 3D-Viewer (OpenGL ES 3)     │        model.ply         │    prune)                        │
└─────────────────────────────┘                          └──────────────────────────────────┘
```

| Verzeichnis | Inhalt |
|---|---|
| `android/` | Kotlin-App: Aufnahme mit ARCore, Upload/Status/Download, Splat-Viewer |
| `desktop/` | `gsplat-server` in C11: HTTP-Server, Datensatz-Loader, GPU-Trainer, PLY-Export |
| `desktop/kernels/gsplat_kernels.h` | GPU-Kernel (Vorwärts- und Rückwärtspass des differenzierbaren Rasterizers), eine Quelle für CUDA und OpenCL |
| `desktop/kernels/gsplat.cu` | CUDA-Einstieg für die Kernel (NVRTC zur Laufzeit oder `nvcc` → PTX) |
| `desktop/tools/make_synthetic.py` | erzeugt eine synthetische Test-Aufnahme im App-Format |

---

## 1. Desktop-Server (`desktop/`)

Der Server ist in reinem C11 geschrieben und rechnet auf einer NVIDIA-GPU mit **CUDA**
(ausgelegt auf die RTX 3080, sm_86):

- Das Host-Programm nutzt die **CUDA Driver API**. `libcuda.so` bzw. `nvcuda.dll` gehört zum
  NVIDIA-Treiber und wird zur Laufzeit geladen. Zum Bauen ist daher kein CUDA-SDK nötig, nur ein
  C-Compiler und CMake.
- Die Kernel (`kernels/gsplat_kernels.h`) werden beim Start mit **NVRTC** genau für die
  eingebaute GPU übersetzt (RTX 3080 → `compute_86`). NVRTC kommt mit dem CUDA Toolkit. Ist beim
  Bauen `nvcc` vorhanden, entsteht zusätzlich ein vorgebautes `gsplat.ptx` als Ausweichlösung.
- Im Rückwärtspass werden native `atomicAdd`-Operationen auf `float` verwendet.
- Optional gibt es ein **OpenCL**-Backend aus derselben Kernel-Quelle (AMD/Intel-GPUs oder
  CPU-Tests). Es wird nur gebaut, wenn das OpenCL-SDK gefunden wird.

Voraussetzung am PC: ein aktueller NVIDIA-Treiber und das
[CUDA Toolkit 12.x oder 13.x](https://developer.nvidia.com/cuda-downloads) (für NVRTC, ggf. `nvcc`).

### Bauen unter Windows (empfohlen für den Desktop-PC)

Visual Studio 2022 (Workload „Desktopentwicklung mit C++“), CMake und das CUDA Toolkit
installieren, dann in der „x64 Native Tools Command Prompt“:

```bat
cd desktop
cmake -B build
cmake --build build --config Release
build\Release\gsplat-server.exe --list-devices
```

`nvrtc64_*.dll` wird über `%CUDA_PATH%\bin` gefunden, das der CUDA-Installer setzt.

### Bauen unter Linux

```bash
sudo apt install build-essential cmake
# NVIDIA-Treiber + CUDA Toolkit (liefert libnvrtc, nvcc) installieren
cd desktop
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gsplat-server --list-devices
```

Der Build kopiert die Kernel-Quellen nach `build/kernels/`, weil sie zur Laufzeit übersetzt
werden. CMake-Optionen: `-DGS_WITH_OPENCL=OFF` (nur CUDA), `-DGS_CUDA_ARCH=86` (Architektur des
vorgebauten PTX).

### Starten

```bash
./build/gsplat-server --list-devices        # sollte die RTX 3080 unter "CUDA devices" zeigen
./build/gsplat-server --selftest            # prüft die GPU-Gradienten (Finite Differenzen)
./build/gsplat-server --port 8080 --data ./gsplat-data
# nur über Tailscale erreichbar machen (empfohlen):
./build/gsplat-server --bind $(tailscale ip -4) --port 8080
# optional zusätzlich mit Token (in der App unter „Token“ eintragen):
./build/gsplat-server --bind $(tailscale ip -4) --token MEIN-GEHEIMNIS
```

Wichtige Optionen (`--help` zeigt alle):

| Option | Standard | Bedeutung |
|---|---|---|
| `--backend B` | `auto` | `cuda`, `opencl` oder `auto` (CUDA, falls eine NVIDIA-GPU da ist) |
| `--device N` | 0 | Gerät innerhalb des Backends (`--list-devices`) |
| `--iterations N` | 30000 | Standard-Iterationen (die App kann pro Job andere vorgeben) |
| `--max-dim N` | 1600 | Bilder werden auf diese längste Seite verkleinert |
| `--sh-degree N` | 3 | Grad der Spherical Harmonics (blickabhängige Farbe) |
| `--max-gaussians N` | 3000000 | Obergrenze bei der Densification (GPU-Speicher) |
| `--white-background` | aus | gegen weißen statt schwarzen Hintergrund trainieren |

Es läuft immer ein Training gleichzeitig; weitere Jobs warten in der Warteschlange. Unter
`http://<ip>:8080/` zeigt der Server eine kleine Statusseite mit allen Jobs.

### Offline trainieren (ohne App)

```bash
./build/gsplat-server --train /pfad/zur/aufnahme --out /pfad/ausgabe --iterations 30000 \
    --preview /pfad/vorschau      # optional: Foto/Render-Vergleichsbilder (PPM)
```

Das Eingabeformat ist kompatibel zu nerfstudio (`transforms.json` mit Kamera-zu-Welt-Matrizen
in OpenGL-Konvention, `images/`, optional `points3d.ply`).

---

## 2. Tailscale einrichten

1. Tailscale auf dem Desktop-PC und dem Handy installieren und mit demselben Konto anmelden
   (gleiches Tailnet).
2. Am PC die Tailscale-IP anzeigen: `tailscale ip -4` (z. B. `100.101.102.103`). MagicDNS-Namen
   wie `mein-pc` oder `mein-pc.tailnet-xyz.ts.net` funktionieren ebenfalls.
3. In der App unter „Desktop-Server“ die IP und den Port (8080) eintragen und „Verbindung
   testen“ tippen. Angezeigt wird die erkannte GPU.
4. Windows: Beim ersten Start fragt die Firewall nach; den Zugriff im privaten Netz erlauben
   (Tailscale-Adapter).

Die App spricht HTTP. Verschlüsselung und Authentifizierung der Geräte übernimmt der
WireGuard-Tunnel von Tailscale. Mit `--bind <tailscale-ip>` ist der Server im lokalen Netz
nicht erreichbar.

---

## Downloads

Fertige Builds (APK und Windows-Server) baut GitHub Actions bei jedem Push. Sie liegen unter
**Releases → „latest-build“**:

- `GaussianSplatting.apk`: Android-App. Zum Installieren auf dem Handy „Installation aus
  unbekannten Quellen“ erlauben. Die APK ist debug-signiert, vor einem Update muss eine
  vorherige Version deshalb ggf. deinstalliert werden.
- `gsplat-server-windows.zip`: Desktop-Server für Windows mit CUDA. Entpacken und starten;
  nötig ist nur ein NVIDIA-Treiber ab Version 550 (NVRTC liegt bei).
- `gsplat-server-linux.tar.gz`: Desktop-Server für Linux.

## 3. Android-App (`android/`)

Voraussetzungen: ein [ARCore-fähiges Gerät](https://developers.google.com/ar/devices),
Android 8.0 oder neuer (Cardboard SDK), Android Studio (Ladybug oder neuer) bzw. JDK 17 und
Android SDK 35.

Das [Google Cardboard SDK](https://github.com/googlevr/cardboard) gibt es nicht als
Maven-Paket. Vor dem ersten Build baut ein Skript es einmalig aus dem offiziellen Release
(v1.36.0). Unter Windows geht das in der Git Bash:

```bash
cd android
bash cardboard/build_cardboard_sdk.sh   # einmalig: Cardboard SDK bauen (braucht ANDROID_HOME)
./gradlew assembleDebug                 # APK: app/build/outputs/apk/debug/app-debug.apk
./gradlew installDebug                  # direkt auf das angeschlossene Handy
```

Alternativ `android/` in Android Studio öffnen und starten.

### Ablauf

1. **Neue Aufnahme** fragt zuerst alle Optionen ab: Auslösen (manuell oder automatisch),
   Foto-Auflösung, ob die gemachten Fotos im Raum und der Abdeckungskreis angezeigt werden und
   ob ARCore-Tiefendaten aufgezeichnet werden. Danach zeigt der **Fotomodus** nur noch das
   Kamerabild, immer im Hochformat: oben eine Statuszeile (Anzahl Fotos oder was gerade nicht
   stimmt), unten links **Letztes Foto verwerfen**, in der Mitte der Auslöser, rechts **Fertig**.
   Langsam um das Objekt herumgehen, aus mehreren Höhen und Winkeln, 60 bis 200 Fotos.
   Im **Automatik**-Modus entsteht ein Foto, sobald die Überlappung mit dem letzten Foto unter
   70 % fällt und das Handy ruhig gehalten wird, höchstens eines pro Sekunde.
   **Jedes Foto bleibt im Kamerabild an seiner Aufnahmeposition stehen** als Kamerarahmen mit
   dem Vorschaubild, das letzte ist gelb markiert. Jeder Rahmen hängt an einem ARCore-Anker:
   korrigiert ARCore seine Karte, wandert der Rahmen mit der echten Szene mit.

   ARCore läuft mit derselben Konfiguration wie SceneView: Flächenerkennung horizontal und
   vertikal, HDR-Lichtschätzung, Autofokus, auf dem Hauptthread im Takt des Displays. Mit
   abgeschalteter Flächenerkennung und Lichtschätzung bekam ARCore auf dem Galaxy A56 nur etwa
   6 Sensorwerte pro Sekunde statt 125, und das Tracking fiel mehrmals pro Sekunde aus. Das Log
   enthält jede Sekunde eine Zeile `tracking stats: x of y camera frames tracked`, die Zahl der
   Sensorwerte steht in ARCores Zeilen `[Sensor][SENSOR_TYPE_ACCELEROMETER_0] Accumulative count`.
   Fotos, deren Anker ARCore gerade nicht verfolgt, werden ausgeblendet.

   **Abdeckung**: Der grüne Bogen um den Auslöser zeigt, welcher Anteil der Blickrichtungen
   schon fotografiert ist. Der Kreis oben rechts zeigt das genauer, von oben gesehen: der äußere
   Ring von unten bzw. auf Objekthöhe (bis 15°), der mittlere schräg von oben (15–45°), die
   Mitte steil von oben. Jeder Ring ist in Sektoren geteilt (12, 12 und 6), grün heißt
   abgedeckt. Der gelbe Punkt ist die aktuelle Position, das erste Foto liegt unten. Den
   Objektmittelpunkt schätzt die App als den Punkt, auf den alle Blickachsen zeigen. Ziel: alle
   30 Richtungen grün.

   **Schärfe**: Jedes Foto wird auf Bewegungsunschärfe geprüft (Varianz des Laplace-Operators
   auf der Helligkeit). Weil der Wert von der Szene abhängt, vergleicht die App mit dem Median
   der letzten Kamerabilder derselben Aufnahme. Liegt ein Foto unter 60 % davon, wird es
   verworfen und beim nächsten Kamerabild erneut versucht. Ein Tipp auf den Auslöser wartet
   bis zu 1,5 Sekunden auf ein scharfes Bild und speichert danach notfalls ein unscharfes (mit
   Hinweis in der Statuszeile, das sich mit **Letztes Foto verwerfen** wieder entfernen lässt).
2. Pro Foto werden gespeichert: das CPU-Kamerabild (JPEG, Auflösung in den Aufnahme-Optionen,
   Standard: die größte, die ARCore anbietet, meist 1920×1080), die
   Kamera-Pose (`Camera.getPose()`, an der Sensor-Ausrichtung des Bildes ausgerichtet), die
   Intrinsik des Bildes (`getImageIntrinsics()`) und, falls das Gerät die ARCore Depth API
   kann und es eingeschaltet ist, eingefärbte 3D-Punkte aus dem Tiefenbild. Dazu kommen die
   ARCore-Feature-Punkte. Daraus entsteht `points3d.ply`, die Startpunktwolke für das Training.
3. **Fertig** → Projektseite: **Hochladen & trainieren** legt am Server einen Job an, lädt
   alle Dateien hoch und startet das Training. Fortschritt, PSNR und Anzahl der Gaussians
   werden live angezeigt. Der Upload läuft weiter, auch wenn man den Bildschirm verlässt.
4. **Modell herunterladen** (während des Trainings: „Zwischenstand laden“) → **3D-Modell
   ansehen**. Mit **Modell teilen** lässt sich die `.ply` an andere Apps oder Viewer
   weitergeben (z. B. SuperSplat).
5. **Modelle auf dem Server** listet alle Jobs am PC (auch von anderen Geräten). Tippen lädt
   das Modell und öffnet es, langes Drücken löscht den Job.

### 3D-Viewer

Echtzeit-Splat-Renderer mit OpenGL ES 3.0: pro Splat ein instanziertes Quad, die
3D-Kovarianz wird im Vertex-Shader auf eine 2D-Ellipse projiziert (EWA-Splatting wie im
3DGS-Paper). Die Splats werden in einem Hintergrund-Thread von hinten nach vorne sortiert,
dort wird auch die blickabhängige Farbe (Spherical Harmonics) berechnet. Er öffnet `.ply`
(3DGS-Format oder normale Punktwolke) und `.splat`, auch aus anderen Apps.

| Bedienung | Aktion |
|---|---|
| 1 Finger ziehen | um das Objekt drehen |
| 2 Finger ziehen | verschieben |
| Pinch | zoomen |
| Doppeltipp / „Ansicht zurücksetzen“ | Startansicht |
| „Auto-Rotation“ | Modell dreht sich automatisch |
| „Hintergrund“ | Schwarz / Grau / Weiß |
| „Hochachse“ | +Y / −Y / +Z / −Z (für Modelle aus anderen Tools, z. B. COLMAP: −Y) |
| „Splat-Größe“ | Splats skalieren (0.1× bis 2×) |
| „Sichtfeld“ | Brennweite (20° bis 110°) |
| „AR / VR“ | Modell im Raum platzieren (siehe unten) |

**Einstellungen bleiben pro Modell erhalten.** Kameraposition und Drehpunkt (Zentrum),
Hochachse, Sichtfeld, Splat-Größe, Hintergrund und Auto-Rotation werden bei jeder Änderung
für dieses Modell gespeichert und beim nächsten Öffnen wiederhergestellt. In der AR-Ansicht gilt
das für die Höhe des Modellzentrums, den Maßstab und die Drehung. Erst **„Ansicht
zurücksetzen“** (oder ein Doppeltipp) löscht die gespeicherten Werte und lädt das Original.

### AR- und Cardboard-Ansicht

„AR / VR“ im 3D-Viewer öffnet das Modell im Raum. Der Bildschirm läuft im Querformat, weil
Cardboard das braucht.

1. **Boden suchen**: Das Handy langsam über den Boden bewegen, bis der weiße Zielkreis
   erscheint. Er liegt dort auf dem Boden, wohin die Bildmitte zeigt.
2. **Platzieren**: Der Bodenpunkt wird fixiert. Mit dem Finger **nach oben wischen** lässt einen
   senkrechten Strich nach oben wachsen. Er markiert, wo das **Zentrum** des Modells sitzen soll
   (die Höhe wird angezeigt, das Modell ist als Vorschau zu sehen). Nach unten wischen senkt ihn.
3. **Platzieren** erneut: Das Modellzentrum wird mit einem ARCore-Anker an der Spitze des
   Strichs verankert. Ab jetzt ist **nur noch das Modell** zu sehen, kein Kamerabild mehr.
   ARCore verfolgt das Handy aber weiter, man kann also um das Modell herumlaufen, und es
   bleibt an seiner Stelle.
4. **Maßstab**: Der Regler skaliert das Modell von 1:100 bis 20:1 (auch per Pinch). **1:1** ist
   echte Größe, denn Aufnahmen dieser App sind in Metern. Angezeigt wird die ungefähre
   Modellgröße. Waagrecht wischen dreht das Modell um die senkrechte Achse.
5. **Cardboard**: Das Bild wird stereoskopisch für eine Google-Cardboard-Brille dargestellt,
   mit Linsenentzerrung aus dem offiziellen Cardboard SDK. Die Kopfbewegung kommt dabei von
   ARCore (volle Raumposition, nicht nur Drehung). Beim ersten Mal startet der Scan des
   **QR-Codes** auf der Brille, damit die Darstellung zur Linse passt. Über das Zahnrad lässt
   sich jederzeit eine andere Brille scannen, das X beendet den Cardboard-Modus.

„Neu platzieren“ beginnt wieder bei Schritt 1. Höhe, Maßstab und Drehung der letzten Platzierung
werden beim nächsten Mal vorgeschlagen; der Strich steht nach dem ersten „Platzieren“ also schon
auf der gespeicherten Höhe. Die Drehung bezieht sich auf die Blickrichtung beim Platzieren,
weil ARCore die Himmelsrichtung bei jeder Sitzung neu festlegt.

---

### Zahnradmenü (Hauptbildschirm)

- **Logs exportieren**: schreibt das Log der App (inkl. der ARCore-Meldungen) mit Gerät,
  Android- und ARCore-Version in eine Textdatei und öffnet das Teilen-Menü.

Die Foto-Auflösung (Standard 640×480, Mittel ab 1280×720, Maximal) wird vor jeder Aufnahme
mit den anderen Optionen abgefragt.

### Worauf es bei Gaussian Splatting ankommt

Kurz zusammengefasst, was die App dafür tut und was (noch) nicht:

| Anforderung | Stand |
|---|---|
| Kameraposen pro Foto | ARCore-Posen, ohne Nachoptimierung (siehe unten) |
| Rundum-Abdeckung aus mehreren Höhen | Abdeckungskreis mit 30 Richtungen |
| 60–80 % Überlappung | Fotos als verankerte Rahmen im Raum, Automatik bei < 70 % |
| Scharfe Fotos | Schärfeprüfung, unscharfe Bilder werden verworfen |
| Hohe Auflösung | größte ARCore-CPU-Auflösung (Kamera-Vollauflösung parallel zu ARCore ginge nur über die Shared-Camera-API) |
| Startpunktwolke | ARCore-Feature-Punkte und optional Tiefenpunkte |
| Gleichbleibende Belichtung/Weißabgleich | **fehlt**: ARCore regelt automatisch. Sperren ginge nur über die Shared-Camera-API; alternativ eine Belichtungskorrektur pro Bild im Trainer |
| Floater entfernen | **fehlt** in der App; z. B. in SuperSplat nachbearbeiten |
| Kompaktes Austauschformat | `.ply` und `.splat`; **SPZ/SOG fehlen** noch |

## 4. REST-API

Alle Pfade unter `/api`. Wenn der Server mit `--token` gestartet wurde, muss jeder Request
`Authorization: Bearer <token>` (oder `?token=`) mitschicken.

| Methode & Pfad | Beschreibung |
|---|---|
| `GET /api/info` | Server-Version, Backend (`CUDA`/`OpenCL`), GPU, Speicher |
| `GET /api/jobs` | alle Jobs |
| `POST /api/jobs` | Job anlegen: `{"name", "iterations", "max_dim", "sh_degree"}` → Job |
| `GET /api/jobs/{id}` | Status: `state` (`uploading`, `queued`, `training`, `done`, `failed`, `cancelled`), `iteration`, `psnr`, `gaussians`, `model_available`, … |
| `PUT /api/jobs/{id}/files/{pfad}` | Datei hochladen (`transforms.json`, `points3d.ply`, `images/frame_00000.jpg`, …) |
| `POST /api/jobs/{id}/start` | Training starten bzw. neu starten (optional `{"iterations", "max_dim"}`) |
| `POST /api/jobs/{id}/cancel` | Training abbrechen |
| `DELETE /api/jobs/{id}` | Job und Dateien löschen |
| `GET /api/jobs/{id}/model.ply` | Modell im Standard-3DGS-PLY-Format (inkl. SH) |
| `GET /api/jobs/{id}/model.splat` | kompaktes `.splat`-Format (32 Byte/Splat) |

Beispiel mit `curl`:

```bash
ID=$(curl -s -X POST -d '{"name":"Test","iterations":7000}' http://100.101.102.103:8080/api/jobs | jq -r .id)
curl -X PUT --data-binary @transforms.json http://100.101.102.103:8080/api/jobs/$ID/files/transforms.json
...
curl -X POST http://100.101.102.103:8080/api/jobs/$ID/start
```

## 5. Datenformat einer Aufnahme

```
transforms.json   {"camera_model":"OPENCV", "ply_file_path":"points3d.ply",
                   "frames":[{"file_path":"images/frame_00000.jpg",
                              "transform_matrix":[[4x4 Kamera→Welt, OpenGL-Achsen]],
                              "fl_x","fl_y","cx","cy","w","h","timestamp_ns",
                              "sharpness"}, ...]}
images/           Fotos (Sensor-Ausrichtung, wie von ARCore geliefert)
points3d.ply      ASCII-PLY: x y z red green blue (Meter, ARCore-Weltkoordinaten, Y nach oben)
```

## 6. Trainer: Details

Der Trainer folgt Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
(SIGGRAPH 2023):

- Parameter pro Gaussian: Position, log-Skalierung, Rotation (Quaternion), Opazität (Logit),
  Spherical Harmonics bis Grad 3. Initialisierung aus `points3d.ply`, Skalierung aus dem
  mittleren Abstand der 3 nächsten Nachbarn.
- Pro Iteration läuft alles auf der GPU: Projektion und EWA-Kovarianz; Tiefensortierung
  (stabiler Radix-Sort); Erzeugen der (Tile, Gaussian)-Paare über einen Prefix-Scan;
  Sortierung nach 16×16-Tile und Bestimmung der Tile-Bereiche; Alpha-Compositing pro Tile;
  Loss `0.8·L1 + 0.2·(1−SSIM)`; analytischer Rückwärtspass durch Rasterizer, Projektion,
  Kovarianz und SH; Adam. Pro Iteration liest der Host nur die Anzahl der Paare zurück.
- Die Trainingsfotos liegen dauerhaft im GPU-Speicher, wenn sie in 30 % davon passen (bei
  10 GB etwa 750 Fotos in 1600×1200). Sonst wird pro Iteration ein Foto hochgeladen.
- Densification alle 100 Iterationen (500 bis 15000), ebenfalls komplett auf der GPU: Klonen
  kleiner bzw. Teilen großer Gaussians mit großem Bildraum-Gradienten, Entfernen transparenter
  oder zu großer, Kompaktierung per Prefix-Scan. Alle 3000 Iterationen wird die Opazität
  zurückgesetzt. Die Positions-Lernrate fällt exponentiell und ist an die Szenengröße angepasst.
- Zwischenstände werden alle 5000 Iterationen als `model.ply` geschrieben und können schon
  während des Trainings geladen werden.

`--selftest` vergleicht die analytischen GPU-Gradienten mit finiten Differenzen. Außerdem prüft
er Scan, Radix-Sort, Binning und Densification der GPU gegen CPU-Referenzen.

## 7. Getesteter Stand und Grenzen

- **CUDA**: Die Kernel werden in der Entwicklungsumgebung mit NVRTC 12.9 für `compute_86`
  übersetzt und von `ptxas` für sm_86 (RTX 3080) assembliert, ohne Register-Spills. Weil dort keine
  NVIDIA-GPU verfügbar war, liefen sie noch **nicht auf echter Hardware**. Bitte zuerst
  `--list-devices` und `--selftest` auf der RTX 3080 laufen lassen.
- **Kernel-Logik**: CUDA und OpenCL übersetzen dieselbe Quelle. Über das OpenCL-Backend (PoCL auf
  der CPU) sind der Gradienten-Selbsttest, die GPU-Sortierung und -Densification, das Training
  einer synthetischen Szene
  (`tools/make_synthetic.py`) und der komplette HTTP-Ablauf getestet. Wie schnell das Training
  auf der RTX 3080 tatsächlich ist, ist deshalb noch nicht gemessen.
- **Windows**: Der Code kompiliert und linkt mit MinGW, wurde aber nicht unter Windows ausgeführt.
- **Android-App**: Der Kotlin-Code wurde gegen die Android-Framework-Klassen typgeprüft, die
  Viewer-Shader liefen in WebGL2 gegen echte Trainer-Ausgaben. Ein vollständiger Gradle-Build und
  ein Test auf einem Gerät waren in der Umgebung nicht möglich (kein Zugriff auf Google Maven /
  Android SDK). Beim ersten Build in Android Studio können daher noch kleinere Korrekturen nötig
  sein.
- **Auf der CPU bleiben** nur Dinge außerhalb der Trainingsschleife: das Laden und Dekodieren
  der Fotos, die Initialisierung (k-nächste-Nachbarn) und das Speichern des Modells (alle 5000
  Iterationen).
- Die ARCore-Posen werden direkt verwendet, ohne Nachoptimierung (z. B. per COLMAP).
  Ungenaues Tracking macht das Modell unscharf. Tipps: gleichmäßiges Licht, strukturierte
  Umgebung, langsam bewegen.
- Der Gradient der SH-Farbe nach der Blickrichtung (und damit nach der Position) wird beim
  Training vernachlässigt, wie bei vielen schnellen Implementierungen. Die SH-Koeffizienten
  selbst werden voll optimiert.
