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

## 3. Android-App (`android/`)

Voraussetzungen: ein [ARCore-fähiges Gerät](https://developers.google.com/ar/devices),
Android 7.0 oder neuer, Android Studio (Ladybug oder neuer) bzw. JDK 17 und Android SDK 35.

```bash
cd android
./gradlew assembleDebug          # APK: app/build/outputs/apk/debug/app-debug.apk
./gradlew installDebug           # direkt auf das angeschlossene Handy
```

Alternativ `android/` in Android Studio öffnen und starten.

### Ablauf

1. **Neue Aufnahme**: ARCore verfolgt die Position des Handys. Langsam um das Objekt herumgehen,
   aus mehreren Höhen und Winkeln. Im **Auto**-Modus wird automatisch ein Foto gemacht, sobald
   das Handy ≥ 7 cm bewegt oder ≥ 10° gedreht wurde *und* ruhig gehalten wird (gegen
   Bewegungsunschärfe). Mit **Foto** wird manuell ausgelöst. 60 bis 200 Fotos sind ein guter Wert.
2. Pro Foto werden gespeichert: das CPU-Kamerabild in voller Auflösung (JPEG), die
   Kamera-Pose (`Camera.getPose()`, an der Sensor-Ausrichtung des Bildes ausgerichtet), die
   Intrinsik des Bildes (`getImageIntrinsics()`) und, falls das Gerät die ARCore Depth API
   kann (**Tiefe**), eingefärbte 3D-Punkte aus dem Tiefenbild. Dazu kommen die
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

---

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
                              "fl_x","fl_y","cx","cy","w","h","timestamp_ns"}, ...]}
images/           Fotos (Sensor-Ausrichtung, wie von ARCore geliefert)
points3d.ply      ASCII-PLY: x y z red green blue (Meter, ARCore-Weltkoordinaten, Y nach oben)
```

## 6. Trainer: Details

Der Trainer folgt Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
(SIGGRAPH 2023):

- Parameter pro Gaussian: Position, log-Skalierung, Rotation (Quaternion), Opazität (Logit),
  Spherical Harmonics bis Grad 3. Initialisierung aus `points3d.ply`, Skalierung aus dem
  mittleren Abstand der 3 nächsten Nachbarn.
- Pro Iteration (GPU): Projektion und EWA-Kovarianz, Tiefensortierung und Zuordnung zu
  16×16-Tiles (CPU, Radix-Sort), Alpha-Compositing pro Tile, Loss `0.8·L1 + 0.2·(1−SSIM)`,
  analytischer Rückwärtspass durch Rasterizer, Projektion, Kovarianz und SH, dann Adam.
- Densification alle 100 Iterationen (500 bis 15000): Klonen kleiner bzw. Teilen großer Gaussians
  mit großem Bildraum-Gradienten, Entfernen transparenter oder zu großer. Alle 3000 Iterationen
  wird die Opazität zurückgesetzt. Die Positions-Lernrate fällt exponentiell und ist an die
  Szenengröße angepasst.
- Zwischenstände werden alle 5000 Iterationen als `model.ply` geschrieben und können schon
  während des Trainings geladen werden.

`--selftest` vergleicht die analytischen GPU-Gradienten mit finiten Differenzen.

## 7. Getesteter Stand und Grenzen

- **CUDA**: Die Kernel werden in der Entwicklungsumgebung mit NVRTC 12.9 für `compute_86`
  übersetzt und von `ptxas` für sm_86 (RTX 3080) assembliert, ohne Register-Spills. Weil dort keine
  NVIDIA-GPU verfügbar war, liefen sie noch **nicht auf echter Hardware**. Bitte zuerst
  `--list-devices` und `--selftest` auf der RTX 3080 laufen lassen.
- **Kernel-Logik**: CUDA und OpenCL übersetzen dieselbe Quelle. Über das OpenCL-Backend (PoCL auf
  der CPU) sind der Gradienten-Selbsttest, das Training einer synthetischen Szene
  (`tools/make_synthetic.py`) und der komplette HTTP-Ablauf getestet.
- **Windows**: Der Code kompiliert und linkt mit MinGW, wurde aber nicht unter Windows ausgeführt.
- **Android-App**: Der Kotlin-Code wurde gegen die Android-Framework-Klassen typgeprüft, die
  Viewer-Shader liefen in WebGL2 gegen echte Trainer-Ausgaben. Ein vollständiger Gradle-Build und
  ein Test auf einem Gerät waren in der Umgebung nicht möglich (kein Zugriff auf Google Maven /
  Android SDK). Beim ersten Build in Android Studio können daher noch kleinere Korrekturen nötig
  sein.
- **Geschwindigkeit**: Tiefensortierung und Tile-Zuordnung laufen pro Iteration auf der CPU
  (Radix-Sort). Bei sehr vielen Gaussians (> 1 Mio.) ist das der größte Zeitanteil. Eine
  Sortierung direkt auf der GPU wäre der nächste Optimierungsschritt.
- Die ARCore-Posen werden direkt verwendet, ohne Nachoptimierung (z. B. per COLMAP).
  Ungenaues Tracking macht das Modell unscharf. Tipps: gleichmäßiges Licht, strukturierte
  Umgebung, langsam bewegen.
- Der Gradient der SH-Farbe nach der Blickrichtung (und damit nach der Position) wird beim
  Training vernachlässigt, wie bei vielen schnellen Implementierungen. Die SH-Koeffizienten
  selbst werden voll optimiert.
