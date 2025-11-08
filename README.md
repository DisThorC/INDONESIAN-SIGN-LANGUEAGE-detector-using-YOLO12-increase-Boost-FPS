# YOLO12 Indonesian Sign Language Detector (GPU-Accelerated)

Final scope: Real‑time Indonesian Sign Language (ISL) gesture detection using Ultralytics YOLOv12 with optional GPU acceleration to raise FPS while retaining accuracy. The project provides:
- Training & evaluation pipeline (mAP metrics, class distribution)
- Realtime inference (threaded capture, stride skipping, dynamic resolution, optional tracking & smoothing)
- Temporal features: semantic transition smoothing, Markov overlay, dwell‑based sequence token logging & aggregation
- Export & optimization: ONNX, TensorRT (FP16), quantization (INT8), benchmarking scripts
- Performance tuning helpers (camera benchmark, torch.compile, resolution adaptation)

Target: Achieve higher FPS on GPU hardware while using the provided 50‑epoch weights (no retraining required). Current pure PyTorch FPS depends on image size, stride, and tracking flags.

---
# yolo12 — Indonesian Sign Language (ISL) with Ultralytics YOLO


## Quick start (Windows PowerShell)

1) Create a virtual environment and install requirements

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# IMPORTANT: Install PyTorch separately according to your CUDA/CPU setup
# For example (CUDA 12.1):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2) Run realtime (uses 50‑epoch best weights provided)

```powershell
.\.venv\Scripts\Activate.ps1
python .\realtime.py --weights runs\train\exp_y12s_50e_640\weights\best.pt --device cuda --duration 15
```

Training is not required for the final setup (we use the existing 50‑epoch checkpoint). You can ignore dataset scripts and training commands.

3) Inference (single image / video)

```powershell
# Image or video file
python -m src.yolo12.infer --weights runs/train/exp/weights/best.pt --source .\sample.jpg

# Webcam (0)
python -m src.yolo12.infer --weights runs/train/exp/weights/best.pt --source 0
```

4) Extended evaluation & metrics (optional)

Use the extended evaluator for per-class mAP@50:95 and optional confusion matrix:

```powershell
python -m src.yolo12.evaluation --weights runs/train/exp/weights/best.pt --data configs/data.yaml --split val --show-matrix --save-json runs/eval/metrics.json
```

Or the convenience wrapper:

```powershell
python .\scripts\eval_metrics.py --weights runs/train/exp/weights/best.pt --split val
```

View class distribution (imbalance insight):

```powershell
python .\scripts\class_distribution.py --data configs/data.yaml --split train --save runs\analysis\train_dist.csv
```

```powershell
python -m src.yolo12.evaluation --weights runs/train/exp/weights/best.pt --data configs/data.yaml
```

Export/ONNX/TensorRT steps have been removed from the minimal runtime. If you plan to optimize further, reintroduce those scripts and toolchains.

```powershell
# Basic ONNX
python .\scripts\export_onnx.py --weights runs/train/exp/weights/best.pt --imgsz 640

# ONNX + TensorRT engine build (requires trtexec in PATH)
python .\scripts\export_onnx.py --weights runs/train/exp/weights/best.pt --tensorrt --fp16

# Quantize ONNX dynamically (INT8 weights) -> best.int8.onnx
python .\scripts\quantize_onnx.py --onnx runs\train\exp\weights\best.onnx --mode dynamic
```

Benchmark pure model latency (no webcam):

```powershell
python .\scripts\benchmark_inference.py --weights runs/train/exp_y12s_50e_640/weights/best.pt --imgsz 320 512 640
```

Compare torch.compile speedup:

```powershell
python .\scripts\torch_compile_benchmark.py --weights runs/train/exp_y12s_50e_640/weights/best.pt
```

## Project layout

- `src/yolo12/`: training, inference, dataset tools, evaluation, utils
- `configs/`: data specification and training hyperparameters
- `data/datasets/isl_yolo/`: expected dataset location
- `scripts/`: helper scripts (download, split, export)
- `tests/`: smoke tests
- `.vscode/`: editor setup and tasks

## Troubleshooting

- Torch GPU: Ensure you install a torch build that matches your NVIDIA driver and CUDA version. See https://pytorch.org/get-started/locally/
- OpenCV issues on Windows: If ImportError occurs, try `pip install opencv-python-headless`.
- If `ultralytics` complains about missing dependencies, run `pip install ultralytics[export]`.

## Realtime with GPU (webcam)

1) Ensure Torch with CUDA is installed and recognized:

```powershell
python - <<'PY'
import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
PY
```

2) Run real-time inference with CUDA, FP16, and on-screen display:

Simplest run:

```powershell
python .\realtime.py --weights runs\train\exp_y12s_50e_640\weights\best.pt --duration 15
```

Alternative module style:

```powershell
# Module style
python -m src.yolo12.realtime --weights runs\train\exp_y12s_50e_640\weights\best.pt --duration 15

# Direct file under src
python .\src\yolo12\realtime.py --weights runs\train\exp_y12s_50e_640\weights\best.pt --duration 15
```

Useful flags:

```text
--threaded            Use capture thread (default)
--vid-stride N        Skip frames (2..3 for speed)
--smooth-window N     Temporal smoothing of class/conf per track
--semantic-smooth     Enable semantic transition smoothing
--semantic-threshold  Minimum transition probability to accept class change (default 0.15)
--semantic-map PATH   JSON of transition probs: {"1->3": 0.8, ...}
--max-det N           Limit detections to reduce postprocess cost
--torch-compile       Enable PyTorch 2.x compile (if available)
--cam-width/--cam-height  Force capture resolution
--save-dir PATH       Save annotated frames periodically
--log-fps PATH.csv    Log per-frame FPS/detections
```

If initial camera index fails the app will fallback to other indices (0..4). To enumerate cameras:

```powershell
python .\scripts\list_cameras.py --max-index 10
```

Notes:
- Press 'q' to quit the video window.
- Increase `--vid-stride` (e.g., 2 or 3) to skip frames for higher FPS on slower GPUs/CPUs.
- If you see CUDA errors, reinstall Torch for your exact CUDA version from the PyTorch website.

### Semantic smoothing example (optional)

```powershell
python .\realtime.py --duration 15 --threaded --vid-stride 2 --smooth-window 5 --semantic-smooth --semantic-threshold 0.15 --semantic-map configs\semantic_prior.json
```

Adjust probabilities in `configs\semantic_prior.json` to reflect realistic gesture transitions; raise threshold to make class changes stricter.

If you need “YOLO12” weights
- This repo uses the Ultralytics YOLO runtime, and it can load any compatible `.pt` weights.
- Place your weights at `weights\yolo12.pt`, then run:

```powershell
python -m src.yolo12.infer --weights .\weights\yolo12.pt --source 0 --device cuda --realtime --show --half
```

You can also use the VS Code task “Infer Realtime (GPU - yolo12.pt)” which expects the file at `weights\yolo12.pt`.

## Use YOLOv12 base weights
## Future: Gesture sequence aggregation

Prototype (see `src/yolo12/sequence.py`) for converting frame-wise class predictions into dwell-time tokens that can later be fed into a sequence model (LSTM/TCN) for phrase-level interpretation.

## Dataset classes

| ID | Name         |
|----|--------------|
| 0  | Ayah         |
| 1  | Halo         |
| 2  | Kakak        |
| 3  | Minum        |
| 4  | Rumah        |
| 5  | Sama-sama    |
| 6  | Sehat        |
| 7  | Teman        |
| 8  | Terima kasih |
| 9  | Tidur        |


Ultralytics can load YOLOv12 base checkpoints by name (it will download if needed). To train quickly on GPU with YOLOv12n:

```powershell
python -m src.yolo12.train --data configs/data.yaml --model yolo12n.pt --epochs 1 --imgsz 640 --device cuda
```

Then run realtime with the newly trained weights:

```powershell
python -m src.yolo12.infer --weights runs\train\exp\weights\best.pt --source 0 --device cuda --realtime --show --half
```

## License

MIT License — see `LICENSE`.
