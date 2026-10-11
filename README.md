# Hard hat detection: SHWD + Roboflow YOLOv8s

This repository contains **one** trained two-class detector: the SHWD + Roboflow fine-tuned YOLOv8s checkpoint at [`models/best.pt`](models/best.pt). It detects head regions with a helmet (`helmet`, class 0) or without one (`no_helmet`, class 1), and draws boxes with class names and confidence scores. It does not classify whole people or determine whether a worksite is safe.

The 22.5 MB checkpoint is stored directly in Git, not Git LFS. A normal `git clone` downloads the actual model. SHA-256: `4e8b6d6fbc53e56958e9096d23857c07f49bdd0d11bedee30fc9a63d7f76cb7f`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `models/best.pt` | Only published checkpoint, selected during fine-tuning |
| `scripts/detect.py` | Webcam or video inference; optional window and MP4 recording |
| `config/training_config.json`, `config/ultralytics_args.yaml` | Recorded training settings; source paths refer to the original laptop |
| `reports/DATASET_PREPARATION_REPORT.md`, `reports/dataset_report.json` | Data preparation and split audit |
| `reports/validation_history.csv`, `reports/EVALUATION.md` | Recorded validation results and their scope |
| `reports/TRAINING_STATUS.md` | Original training completion report |
| `requirements-laptop.txt`, `requirements-pi.txt` | Separate inference dependency entry points |

No raw images, labels, original SHWD checkpoint, other checkpoints, or training run directory are included.

## Training data and approach

The starting point was a separate YOLOv8s model trained on [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), initialized in that earlier run from Ultralytics `yolov8s.pt`. This release fine-tuned that SHWD checkpoint using a combined SHWD and [Hard Hat Workers v1](https://universe.roboflow.com/srinjoy-bahuti/hard-hat-workers-6pipi) Roboflow dataset. The earlier SHWD checkpoint is **not** included here.

SHWD `hat` head boxes became `helmet`; SHWD `person` boxes represent uncovered heads in this dataset and became `no_helmet`. Roboflow's original classes were `head` (mapped to `no_helmet`), `helmet` (mapped to `helmet`), and `person` (excluded). Images left without boxes were kept as negative examples. The prepared data contained 10,818 training images, 1,886 validation images, a preserved 1,205-image SHWD test split, and a separate 706-image Roboflow test split. Only train and validation were used by the fine-tuning run. The exact split counts, class counts, and checks are in the dataset reports.

Training used 800-pixel input, batch size 4, mixed precision, seed 42, up to 100 epochs, and early stopping patience 15. It completed 51 epochs on an NVIDIA RTX 5050 Laptop GPU using Ultralytics 8.4.174. The full recorded settings are in `config/`. The local absolute dataset and checkpoint paths there document the original run; this release is for inference and does not include the training data.

## Evaluation and limitations

The best recorded **validation** row, at epoch 36, has overall box precision **0.94507**, recall **0.93647**, mAP@0.5 **0.96854**, and mAP@0.5:0.95 **0.61751**. These are training-run validation metrics, not held-out test or webcam results. **No held-out test evaluation was run for this fine-tuned checkpoint.** See [`reports/EVALUATION.md`](reports/EVALUATION.md) and the original CSV for context.

The data mix is imbalanced, and SHWD combines imagery from different sources. Validation metrics may not transfer to a particular worksite, camera, helmet style, lighting, viewpoint, blurred frame, or small/occluded head. `no_helmet` means an annotated uncovered head, not a guarantee about every person in frame. Review representative local footage before using the output for any decision. Raspberry Pi 5 speed and accuracy have not been benchmarked here; lowering `--imgsz` may improve speed but can affect small-head detection.

## Laptop installation and inference

Use Python 3.10+ and a local environment. The dependency file installs CPU-capable inference packages; for NVIDIA CUDA, first install the [PyTorch build compatible with your driver](https://pytorch.org/get-started/locally/) in this environment. The training machine used PyTorch 2.14.1+cu130 and Ultralytics 8.4.174; the CUDA training environment is not required on a Pi.

```bash
git clone https://github.com/srinjoybahuti/Hard-Hat-Detection.git
cd Hard-Hat-Detection
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-laptop.txt
sha256sum models/best.pt
```

USB webcam with a live window (`0` is the first camera; use `2` if your external camera is `/dev/video2`):

```bash
.venv/bin/python scripts/detect.py 0 --show --device cpu
.venv/bin/python scripts/detect.py 2 --show --device 0
```

Video file, with an annotated output:

```bash
.venv/bin/python scripts/detect.py /path/to/input.mp4 --device cpu --output /path/to/annotated.mp4
```

The default threshold is `--conf 0.25` and input size is `--imgsz 800`. Add `--show` for a window; press **q** to stop it. Without `--show`, press Ctrl+C to stop a webcam run. The script refuses to overwrite an existing output file.

## Raspberry Pi 5 (64-bit Raspberry Pi OS)

Use a 64-bit OS with internet access, a USB webcam, and a display if you want the live window. The Pi runs this PyTorch checkpoint on **CPU**; install its dependencies separately from any NVIDIA laptop environment. PyTorch and Ultralytics installation on ARM may take time and depend on the current OS/Python wheel availability. See the [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) for platform-specific notes.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip libgl1 libglib2.0-0
git clone https://github.com/srinjoybahuti/Hard-Hat-Detection.git
cd Hard-Hat-Detection
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-pi.txt
sha256sum models/best.pt
ls -l /dev/video*
```

Plug in the USB webcam and choose its capture index from `/dev/video*` (often `0`; if the Pi has another camera, try `2`). For a desktop session:

```bash
.venv/bin/python scripts/detect.py 0 --device cpu --imgsz 640 --show
```

For a headless Pi, record an MP4 and stop with Ctrl+C:

```bash
.venv/bin/python scripts/detect.py 0 --device cpu --imgsz 640 --output usb_camera.mp4
```

For an existing video, with no display required:

```bash
.venv/bin/python scripts/detect.py /path/to/input.mp4 --device cpu --imgsz 640 --output annotated.mp4
```

If a selected `/dev/videoN` is a metadata node rather than a capture node, choose the paired video capture index. For repeatable camera mapping, inspect `ls -l /dev/v4l/by-id/`.

## Source rights and licensing

The [SHWD repository](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) displays an MIT license. The downloaded Roboflow export's `data.yaml` states `license: Public Domain` for the creator's [Hard Hat Workers v1 dataset](https://universe.roboflow.com/srinjoy-bahuti/hard-hat-workers-6pipi). No source dataset files are redistributed here. Ultralytics [licenses its code and models under AGPL-3.0 or an Enterprise license](https://github.com/ultralytics/ultralytics/blob/main/README.md); this repository publishes its model and code under AGPL-3.0 (see [`LICENSE`](LICENSE)). Review those terms for your use, especially commercial deployment.
