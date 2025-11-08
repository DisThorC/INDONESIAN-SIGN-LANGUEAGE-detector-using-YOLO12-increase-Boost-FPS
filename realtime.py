import argparse
import csv
import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict

import cv2
import torch
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("runs/train/exp_y12s_50e_640/weights/best.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple realtime webcam inference")
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to .pt weights")
    p.add_argument("--source", type=str, default="0", help="Webcam index or video path (e.g., 0)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")
    p.add_argument("--vid-stride", type=int, default=2, help="Process every Nth frame (>=2 boosts FPS, adds motion gaps)")
    p.add_argument("--threaded", action="store_true", help="Use a separate capture thread (drops frames for lower latency)")
    p.add_argument("--no-track", action="store_true", help="Disable tracking")
    p.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        help="Tracker config yaml (botsort.yaml or bytetrack.yaml); ignored if --no-track",
    )
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    p.add_argument("--duration", type=float, default=0.0, help="Auto-stop after N seconds (0 = until 'q')")
    p.add_argument("--warmup-frames", type=int, default=20, help="Frames to ignore for FPS stats (model warmup)")
    p.add_argument("--max-det", type=int, default=50, help="Maximum number of detections per image to keep (lower = faster postprocess)")
    p.add_argument("--cam-width", type=int, default=0, help="Force camera capture width (0 = leave default)")
    p.add_argument("--cam-height", type=int, default=0, help="Force camera capture height (0 = leave default)")
    p.add_argument("--smooth-window", type=int, default=0, help="Temporal smoothing window for class/conf per track (0=off)")
    p.add_argument("--smooth-min-fps", type=float, default=0.0, help="Enable smoothing only when avg FPS >= this (0=always follow --smooth-window)")
    p.add_argument("--torch-compile", action="store_true", help="Use torch.compile for potential speed-up (PyTorch 2.x)")
    # Hysteresis to stabilize class switching
    p.add_argument("--hysteresis", action="store_true", help="Enable hysteresis on class switching per track (requires tracking)")
    p.add_argument("--hyst-margin", type=float, default=0.15, help="If --threshold-map provided, set high threshold = low + margin (per class)")
    p.add_argument("--hyst-low", type=float, default=0.30, help="Global low threshold if --threshold-map not provided")
    p.add_argument("--hyst-high", type=float, default=0.60, help="Global high threshold if --threshold-map not provided")
    # Dynamic resolution adaptation
    p.add_argument("--target-fps", type=float, default=0.0, help="Auto-adapt imgsz towards this FPS (0=disabled)")
    p.add_argument("--min-imgsz", type=int, default=320, help="Lower bound for dynamic imgsz")
    p.add_argument("--max-imgsz", type=int, default=640, help="Upper bound for dynamic imgsz")
    p.add_argument("--resize-step", type=int, default=32, help="Step size when changing imgsz dynamically")
    p.add_argument("--adapt-interval", type=float, default=1.0, help="Seconds between adaptation checks")
    # Saving & logging (disabled by default unless flags provided)
    p.add_argument("--save-dir", type=str, default="", help="Directory to save annotated frames; created if missing")
    p.add_argument("--save-detections-only", action="store_true", help="Save frames only when detections exist")
    p.add_argument("--save-interval", type=int, default=1, help="Save every Nth eligible frame (>=1)")
    p.add_argument("--log-fps", type=str, default="", help="Path to CSV file to log per-frame FPS and counts")
    # Sequence aggregation (per-track gesture tokens)
    p.add_argument("--seq-log", type=str, default="", help="Path to JSON file to save aggregated gesture tokens per track")
    p.add_argument("--seq-min-dwell", type=float, default=0.6, help="Minimum seconds holding the same class to emit a token")
    # Semantic smoothing (Markov-like prior on class transitions)
    p.add_argument("--semantic-smooth", action="store_true", help="Enable semantic transition smoothing (uses prior to suppress unlikely class jumps)")
    p.add_argument("--semantic-threshold", type=float, default=0.15, help="Minimum transition probability required to accept class change when semantic smoothing active")
    p.add_argument("--semantic-map", type=str, default="", help="Optional JSON file with transition probabilities: {\"prev->next\": prob, ...}")
    p.add_argument("--markov-display", action="store_true", help="Overlay Markov-decoded class (using transition prior) next to raw track class")
    # Per-class threshold override map
    p.add_argument("--threshold-map", type=str, default="", help="Optional JSON file produced by scripts/per_class_thresholds.py to override per-class confidence thresholds")
    # Console logging (per-interval summary)
    p.add_argument("--console-log-interval", type=float, default=1.0, help="Interval seconds for console stats (set 0 to disable)")
    p.add_argument("--console-log-verbose", action="store_true", help="Print extended per-interval metrics (class histogram, status)")
    p.add_argument("--console-log-topk", type=int, default=3, help="Top-K classes to print in verbose console logs")
    # Default to threaded mode by default for stability on Windows webcams
    p.set_defaults(threaded=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device auto-detect
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Half precision only on CUDA
    half = device == "cuda"

    # Perf knobs
    try:
        cv2.setUseOptimized(True)
        # Reduce OpenCV CPU thread contention with torch
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        if device == "cuda":
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
    except Exception:
        pass

    cfg = {
        "weights": args.weights,
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "track": not args.no_track,
        "tracker": None if args.no_track else args.tracker,
        "device": device,
        "half": half,
        "vid_stride": args.vid_stride,
    }

    if args.dry_run:
        print("[realtime] DRY RUN config:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        return

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)
    # Optional torch.compile
    if args.torch_compile:
        try:
            if hasattr(torch, "compile") and callable(getattr(torch, "compile")):
                model.model = torch.compile(model.model)  # type: ignore[attr-defined]
                print("[realtime] Enabled torch.compile for the model backbone")
            else:
                print("[realtime] torch.compile not available in this PyTorch version; skipping")
        except Exception as e:  # noqa: BLE001
            print(f"[realtime] torch.compile failed: {e}")

    # Explicitly move model to target device (Ultralytics defers until first predict if not forced)
    try:
        if device == "cuda":
            model.to(device)
            # Defer half precision; rely on Ultralytics internal half conversion in predict() to avoid dtype mismatch issues.
            print(f"[realtime] Model placed on {device} (requested half={half}, internal auto-half will manage)")
        else:
            model.to(device)
            print(f"[realtime] Model placed on {device} (FP32)")
    except Exception as e:  # noqa: BLE001
        print(f"[realtime] Failed to move model to {device}: {e}")

    # Prepare save/log destinations (only if user requested)
    save_dir: Optional[Path] = Path(args.save_dir) if args.save_dir else None

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    log_fp = None
    log_writer = None
    if args.log_fps:
        log_path = Path(args.log_fps)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, mode="w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_fp)
        log_writer.writerow(["time_s","frame_all","frame_measured","fps_inst","fps_avg","n_dets","imgsz","vid_stride","threaded","conf","iou"])  # header

    # Threaded capture implementation (manual inference loop) -----------------
    class FrameGrabber:
        def __init__(self, src: str):
            self.cap = None
            tried = []
            def try_open(index: int) -> Optional[cv2.VideoCapture]:
                tried.append(index)
                cap = None
                try:
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if not cap or not cap.isOpened():
                        if cap:
                            cap.release()
                        cap = cv2.VideoCapture(index)
                except Exception:
                    cap = cv2.VideoCapture(index)
                if cap and cap.isOpened():
                    return cap
                if cap:
                    cap.release()
                return None

            try:
                cam_index = int(src)
                self.cap = try_open(cam_index)
                if self.cap is None:
                    # fallback scan a few indices
                    for alt in range(0, 5):
                        if alt == cam_index:
                            continue
                        self.cap = try_open(alt)
                        if self.cap is not None:
                            print(f"[realtime] Fallback to camera index {alt} (primary {cam_index} failed)")
                            break
            except ValueError:
                self.cap = cv2.VideoCapture(src)
            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError(f"Failed to open source: {src}. Tried indices: {tried}")
            # Apply camera dimension preferences if provided
            if args.cam_width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.cam_width))
            if args.cam_height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.cam_height))
            # Reduce internal buffering if supported
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.lock = threading.Lock()
            self.frame: Optional[cv2.typing.MatLike] = None
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

        def _loop(self) -> None:
            while self.running:
                ok, f = self.cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = f
                # Small sleep to yield; reduce CPU spin
                time.sleep(0.001)

        def get(self) -> Optional[cv2.typing.MatLike]:
            with self.lock:
                return None if self.frame is None else self.frame.copy()

        def release(self) -> None:
            self.running = False
            self.thread.join(timeout=0.5)
            self.cap.release()

    def frames_stream():
        if args.threaded:
            grabber = FrameGrabber(args.source)
            try:
                stride_counter = 0
                while True:
                    frame = grabber.get()
                    if frame is None:
                        time.sleep(0.005)
                        continue
                    stride_counter += 1
                    # Apply stride by yielding None frames quickly to avoid blocking predict loop
                    if args.vid_stride > 1 and (stride_counter % args.vid_stride) != 0:
                        continue
                    yield frame
            finally:
                grabber.release()
        else:
            # Use built-in streaming (predict/track) for efficiency
            if not args.no_track:
                try:
                    yield from model.track(
                        source=args.source,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        device=device,
                        half=half,
                        max_det=args.max_det,
                        vid_stride=args.vid_stride,
                        stream=True,
                        persist=True,
                        save=False,
                        tracker=args.tracker,
                        project="runs/infer",
                        name="realtime",
                        exist_ok=True,
                    )
                    return
                except Exception as e:  # noqa: BLE001
                    print(f"[realtime] Tracker failed ({e}). Falling back to predict().")

            yield from model.predict(
                source=args.source,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                half=half,
                max_det=args.max_det,
                vid_stride=args.vid_stride,
                stream=True,
                save=False,
                project="runs/infer",
                name="realtime",
                exist_ok=True,
            )

    win = "yolo12-realtime"
    t0 = time.time()
    frame_count = 0          # counted (post-warmup) frames
    all_frames = 0           # total frames seen
    avg_fps = 0.0
    adapt_last = time.time()
    dynamic_imgsz = args.imgsz
    # Track measurement window separate from warmup so short runs still report sensible FPS
    measure_started = False
    measure_t0 = t0
    # Load per-class threshold map if provided
    per_class_thresholds: Optional[Dict[int, float]] = None
    if args.threshold_map:
        try:
            with open(args.threshold_map, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept either {name:th} or nested structure like produced by script
            if "thresholds_by_id" in data:
                # keys may be str(int)
                pct: Dict[int, float] = {}
                for k, v in data["thresholds_by_id"].items():
                    try:
                        pct[int(k)] = float(v)
                    except Exception:
                        pass
                per_class_thresholds = pct if pct else None
            elif isinstance(data, dict):
                # name keyed
                # need model names to map later; store temporarily as name->value using -1 sentinel id map performed after model load
                per_class_thresholds = {}
                # We'll remap names after model is created below
                # Temporarily stash raw map in this variable and reprocess below
                per_class_thresholds = None  # placeholder; will rebuild after model init
                name_map_raw = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            else:
                per_class_thresholds = None
        except Exception as e:  # noqa: BLE001
            print(f"[realtime] Failed to load threshold map: {e}")
            per_class_thresholds = None
    else:
        name_map_raw = None  # type: ignore

    if args.threaded:
        # Lightweight IOU tracker for threaded mode
        from src.yolo12.utils.tracker import SimpleTracker
        from src.yolo12.sequence import SequenceAggregator
        from src.yolo12.markov import MarkovDecoder
        # Load semantic transition prior if requested
        semantic_prior = None
        if args.semantic_smooth or args.markov_display:
            if args.semantic_map:
                try:
                    import json as _json
                    with open(args.semantic_map, "r", encoding="utf-8") as fp:
                        raw = _json.load(fp)
                    # Expect keys like "a->b": prob
                    sem: Dict[tuple, float] = {}
                    for k, v in raw.items():
                        if "->" in k:
                            a, b = k.split("->", 1)
                            try:
                                sem[(int(a), int(b))] = float(v)
                            except Exception:
                                pass
                    semantic_prior = sem if sem else None
                except Exception as e:  # noqa: BLE001
                    print(f"[realtime] Failed to load semantic map: {e}")
            if semantic_prior is None:
                # Default tiny prior (uniform low probability -> only large threshold disables changes aggressively)
                semantic_prior = {}
        # Markov decoder for overlay (per track)
        markov = MarkovDecoder(transition=semantic_prior or {}) if args.markov_display else None

        # If conditional smoothing is requested, start with smoothing off; it will toggle on when FPS condition met.
        init_sw = 0 if (not args.no_track and args.smooth_min_fps > 0 and args.smooth_window > 0) else args.smooth_window
        # Build hysteresis maps if requested
        hyst_low_map = None
        hyst_high_map = None
        if args.hysteresis:
            if per_class_thresholds:
                hyst_low_map = {int(k): float(v) for k, v in per_class_thresholds.items()}
                hyst_high_map = {k: max(float(v) + float(args.hyst_margin), float(v)) for k, v in hyst_low_map.items()}
            else:
                hyst_low_map = args.hyst_low
                hyst_high_map = args.hyst_high

        tracker = (
            SimpleTracker(
                iou_thresh=0.4,
                max_miss=15,
                smooth_window=init_sw,
                semantic_prior=semantic_prior if args.semantic_smooth else None,
                semantic_threshold=float(args.semantic_threshold) if args.semantic_smooth else -1.0,
                hysteresis=bool(args.hysteresis),
                thresh_low=hyst_low_map,
                thresh_high=hyst_high_map,
            )
            if not args.no_track
            else None
        )
        # Sequence aggregators per track id
        seq_aggs: Dict[int, SequenceAggregator] = {}
        seq_enabled = bool(args.seq_log) and (tracker is not None)

        # Per-interval console logging accumulators
        sec_t0 = time.time()
        sec_frames = 0
        sec_detected_frames = 0
        sec_dets_total = 0
        sec_conf_sum = 0.0
        sec_conf_count = 0
        sec_max_conf = 0.0
        sec_max_conf_name = ""
        sec_cls_count: Dict[int, int] = {}
        last_any_detected = False
        last_max_conf = 0.0
        last_max_name = ""

        for frame in frames_stream():
            all_frames += 1
            try:
                results = model.predict(
                    source=frame,
                    imgsz=dynamic_imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    half=half,
                    max_det=args.max_det,
                    verbose=False,
                )
            except Exception as e:  # noqa: BLE001
                # Fallback to CPU on CUDA-related errors
                if device == "cuda":
                    print(f"[realtime] CUDA inference error '{e}'. Falling back to CPU...")
                    try:
                        results = model.predict(
                            source=frame,
                            imgsz=dynamic_imgsz,
                            conf=args.conf,
                            iou=args.iou,
                            device="cpu",
                            half=False,
                            max_det=args.max_det,
                            verbose=False,
                        )
                        device = "cpu"
                    except Exception as ee:  # noqa: BLE001
                        print(f"[realtime] CPU fallback failed: {ee}")
                        break
                else:
                    print(f"[realtime] Inference error: {e}")
                    break
            res = results[0]
            # If per-class threshold map loaded, filter detections manually
            if per_class_thresholds is None and 'name_map_raw' in locals() and name_map_raw:
                # Build id map using res.names
                try:
                    id_map = {name: idx for idx, name in res.names.items()}  # type: ignore[attr-defined]
                    pct: Dict[int, float] = {}
                    for name, th in name_map_raw.items():
                        if name in id_map:
                            try:
                                pct[int(id_map[name])] = float(th)
                            except Exception:
                                pass
                    per_class_thresholds = pct if pct else None
                except Exception:
                    pass
            im = frame.copy()
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                dets_raw = [(xyxy[i].tolist(), float(confs[i]), int(clss[i])) for i in range(len(xyxy))]
                if per_class_thresholds:
                    dets = []
                    for b, c, k in dets_raw:
                        th = per_class_thresholds.get(k, args.conf)
                        if c >= th:
                            dets.append((b, c, k))
                else:
                    dets = dets_raw
            else:
                dets = []

            # Per-interval stats update
            sec_frames += 1
            if dets:
                sec_detected_frames += 1
                sec_dets_total += len(dets)
                # confidence aggregates
                for _, c, k in dets:
                    sec_conf_sum += float(c)
                    sec_conf_count += 1
                    if c > sec_max_conf:
                        sec_max_conf = float(c)
                        try:
                            sec_max_conf_name = res.names.get(k, str(k))  # type: ignore[attr-defined]
                        except Exception:
                            sec_max_conf_name = str(k)
                    sec_cls_count[int(k)] = sec_cls_count.get(int(k), 0) + 1
                # remember last frame status
                last_any_detected = True
                top = max(dets, key=lambda x: x[1])
                last_max_conf = float(top[1])
                try:
                    last_max_name = res.names.get(int(top[2]), str(int(top[2])))  # type: ignore[attr-defined]
                except Exception:
                    last_max_name = str(int(top[2]))
            else:
                last_any_detected = False
                last_max_conf = 0.0
                last_max_name = ""

            if tracker is not None and dets:
                assigned = tracker.update(dets)
                # draw tracked boxes with IDs
                for tid, (b, conf, cls) in assigned.items():
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = f"ID {tid} {res.names.get(cls, str(cls))}:{conf:.2f}"
                    if markov is not None:
                        # For a simple display, consider candidate = current raw cls with score=conf
                        prev_cls = seq_aggs.get(tid).tokens[-1].cls if (tid in seq_aggs and seq_aggs[tid].tokens) else cls
                        decoded = markov.step(int(prev_cls), [(int(cls), float(conf))])
                        if decoded != cls:
                            label += f" -> M:{res.names.get(decoded, str(decoded))}"
                    cv2.putText(im, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    # sequence aggregation push
                    if seq_enabled:
                        agg = seq_aggs.get(tid)
                        if agg is None:
                            agg = SequenceAggregator(min_dwell=float(args.seq_min_dwell))
                            seq_aggs[tid] = agg
                        agg.push(int(cls), float(time.time() - t0))
            else:
                # fallback: use model's plotted output
                im = res.plot()

            if all_frames <= args.warmup_frames:
                current_fps = 0.0
            else:
                frame_count += 1
                now = time.time()
                if not measure_started:
                    measure_started = True
                    measure_t0 = now
                elapsed = now - measure_t0
                if elapsed > 0:
                    current_fps = frame_count / elapsed
                    avg_fps = avg_fps * 0.9 + current_fps * 0.1
                else:
                    current_fps = 0.0
            overlay = f"FPS: {current_fps:.1f} (avg {avg_fps:.1f})"
            if args.target_fps > 0:
                overlay += f" | dyn {dynamic_imgsz}"
            if tracker is not None:
                overlay += f" | S:{'on' if tracker.smooth_window and tracker.smooth_window>0 else 'off'}"
            cv2.putText(
                im,
                overlay,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # Save frame if requested (per frame)
            n_dets = len(dets)
            should_save = save_dir is not None and (all_frames % max(1, args.save_interval) == 0)
            if should_save:
                if args.save_detections_only and n_dets == 0:
                    pass
                else:
                    out_path = save_dir / f"frame_{all_frames:06d}.jpg"
                    try:
                        cv2.imwrite(str(out_path), im)
                    except Exception:
                        pass

            # Log per-frame if requested
            if log_writer is not None:
                try:
                    log_writer.writerow([f"{time.time():.3f}", all_frames, frame_count, f"{current_fps:.3f}", f"{avg_fps:.3f}", n_dets, args.imgsz, args.vid_stride, True, args.conf, args.iou])
                except Exception:
                    pass

            # Dynamic resolution adaptation
            if args.target_fps > 0 and (time.time() - adapt_last) >= args.adapt_interval and frame_count > 5:
                adapt_last = time.time()
                # Compare rolling avg_fps to target
                if avg_fps < args.target_fps * 0.92 and dynamic_imgsz > args.min_imgsz:
                    dynamic_imgsz = max(args.min_imgsz, dynamic_imgsz - args.resize_step)
                elif avg_fps > args.target_fps * 1.08 and dynamic_imgsz < args.max_imgsz:
                    dynamic_imgsz = min(args.max_imgsz, dynamic_imgsz + args.resize_step)

            # Conditional smoothing toggle based on FPS threshold
            if (
                tracker is not None
                and args.smooth_window > 0
                and args.smooth_min_fps > 0
                and (time.time() - adapt_last) >= 0.0  # reuse adapt cadence
                and frame_count > 5
            ):
                if avg_fps >= args.smooth_min_fps and tracker.smooth_window == 0:
                    tracker.set_smooth_window(args.smooth_window)
                elif avg_fps < args.smooth_min_fps and tracker.smooth_window != 0:
                    tracker.set_smooth_window(0)

            cv2.imshow(win, im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break
            # Console interval logging
            if args.console_log_interval > 0 and (time.time() - sec_t0) >= args.console_log_interval:
                sec_elapsed = max(1e-6, time.time() - sec_t0)
                fps_inst = sec_frames / sec_elapsed if sec_frames > 0 else 0.0
                avg_conf = (sec_conf_sum / sec_conf_count) if sec_conf_count > 0 else 0.0
                det_rate = sec_detected_frames / sec_frames if sec_frames > 0 else 0.0
                base_msg = (
                    f"[log] fps={fps_inst:.1f} | {sec_frames}f | det={sec_detected_frames} ({det_rate*100:.1f}%) "
                    f"| boxes={sec_dets_total} | conf(avg/max)={avg_conf:.3f}/{sec_max_conf:.3f} {sec_max_conf_name}"
                )
                if args.console_log_verbose and sec_cls_count:
                    # Top-K classes by count
                    try:
                        sorted_cls = sorted(sec_cls_count.items(), key=lambda x: x[1], reverse=True)[: max(1, args.console_log_topk)]
                        parts = []
                        for cid, cnt in sorted_cls:
                            try:
                                cname = res.names.get(cid, str(cid))  # type: ignore[attr-defined]
                            except Exception:
                                cname = str(cid)
                            parts.append(f"{cname}:{cnt}")
                        cls_str = ", ".join(parts)
                        base_msg += f" | top[{len(sorted_cls)}]={cls_str}"
                    except Exception:
                        pass
                print(base_msg)
                # reset accumulators
                sec_t0 = time.time()
                sec_frames = 0
                sec_detected_frames = 0
                sec_dets_total = 0
                sec_conf_sum = 0.0
                sec_conf_count = 0
                sec_max_conf = 0.0
                sec_max_conf_name = ""
                sec_cls_count = {}
        # End of threaded loop: dump sequence JSON if requested
        if args.seq_log and tracker is not None:
            try:
                # Flush remaining tokens per track
                out: Dict[str, object] = {"tracks": {}, "meta": {"min_dwell": float(args.seq_min_dwell)}}
                names = getattr(res, "names", {}) if 'res' in locals() else {}
                for tid, agg in seq_aggs.items():
                    tokens = []
                    for tk in agg.flush():
                        item = {
                            "cls": int(tk.cls),
                            "name": names.get(int(tk.cls), str(int(tk.cls))) if isinstance(names, dict) else str(int(tk.cls)),
                            "start": float(tk.start_t),
                            "end": float(tk.end_t),
                            "duration": float(tk.duration),
                        }
                        tokens.append(item)
                    out["tracks"][str(tid)] = tokens
                out_path = Path(args.seq_log)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                print(f"[realtime] Saved sequence tokens -> {out_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[realtime] Failed to save sequence tokens: {e}")
    else:
        for res in frames_stream():
            all_frames += 1
            im = res.plot()
            n_dets = 0
            try:
                n_dets = len(res.boxes) if res.boxes is not None else 0
            except Exception:
                n_dets = 0

            # Non-threaded interval logging accumulators init
            if 'sec_t0' not in locals():
                sec_t0 = time.time()
                sec_frames = 0
                sec_detected_frames = 0
                sec_dets_total = 0
                sec_conf_sum = 0.0
                sec_conf_count = 0
                sec_max_conf = 0.0
                sec_max_conf_name = ""
                sec_cls_count = {}
            sec_frames += 1
            if n_dets > 0 and res.boxes is not None:
                boxes_obj = res.boxes
                try:
                    confs = boxes_obj.conf.cpu().numpy()
                    clss = boxes_obj.cls.cpu().numpy()
                except Exception:
                    confs = []
                    clss = []
                sec_detected_frames += 1
                sec_dets_total += int(n_dets)
                for i in range(len(confs)):
                    c = float(confs[i])
                    sec_conf_sum += c
                    sec_conf_count += 1
                    if c > sec_max_conf:
                        sec_max_conf = c
                        try:
                            sec_max_conf_name = res.names.get(int(clss[i]), str(int(clss[i])))  # type: ignore[attr-defined]
                        except Exception:
                            sec_max_conf_name = str(int(clss[i]))
                    sec_cls_count[int(clss[i])] = sec_cls_count.get(int(clss[i]), 0) + 1

            # Compute instantaneous and rolling FPS
            if all_frames <= args.warmup_frames:
                current_fps = 0.0
            else:
                frame_count += 1
                now = time.time()
                if not measure_started:
                    measure_started = True
                    measure_t0 = now
                elapsed = now - measure_t0
                if elapsed > 0:
                    current_fps = frame_count / elapsed
                    avg_fps = avg_fps * 0.9 + current_fps * 0.1
                else:
                    current_fps = 0.0

            # Save & log
            should_save = save_dir is not None and (all_frames % max(1, args.save_interval) == 0)
            if should_save:
                if args.save_detections_only and n_dets == 0:
                    pass
                else:
                    out_path = save_dir / f"frame_{all_frames:06d}.jpg"
                    try:
                        cv2.imwrite(str(out_path), im)
                    except Exception:
                        pass
            if log_writer is not None:
                try:
                    log_writer.writerow([f"{time.time():.3f}", all_frames, frame_count, f"{current_fps:.3f}", f"{avg_fps:.3f}", n_dets, args.imgsz, args.vid_stride, False, args.conf, args.iou])
                except Exception:
                    pass

            # Overlay text
            overlay = f"FPS: {current_fps:.1f} (avg {avg_fps:.1f})"
            cv2.putText(
                im,
                overlay,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break
            if args.console_log_interval > 0 and (time.time() - sec_t0) >= args.console_log_interval:
                det_rate = sec_detected_frames / sec_frames if sec_frames > 0 else 0.0
                avg_conf = (sec_conf_sum / sec_conf_count) if sec_conf_count > 0 else 0.0
                sec_elapsed = max(1e-6, time.time() - sec_t0)
                fps_inst = sec_frames / sec_elapsed if sec_frames > 0 else 0.0
                msg = (
                    f"[log] fps={fps_inst:.1f} | {sec_frames}f | det={sec_detected_frames} ({det_rate*100:.1f}%) | boxes={sec_dets_total} "
                    f"| conf(avg/max)={avg_conf:.3f}/{sec_max_conf:.3f} {sec_max_conf_name}"
                )
                if args.console_log_verbose and sec_cls_count:
                    try:
                        sorted_cls = sorted(sec_cls_count.items(), key=lambda x: x[1], reverse=True)[: max(1, args.console_log_topk)]
                        parts = []
                        for cid, cnt in sorted_cls:
                            try:
                                cname = res.names.get(cid, str(cid))  # type: ignore[attr-defined]
                            except Exception:
                                cname = str(cid)
                            parts.append(f"{cname}:{cnt}")
                        msg += f" | top[{len(sorted_cls)}]={', '.join(parts)}"
                    except Exception:
                        pass
                print(msg)
                sec_t0 = time.time()
                sec_frames = 0
                sec_detected_frames = 0
                sec_dets_total = 0
                sec_conf_sum = 0.0
                sec_conf_count = 0
                sec_max_conf = 0.0
                sec_max_conf_name = ""
                sec_cls_count = {}

    cv2.destroyAllWindows()
    total_time = time.time() - t0
    if frame_count > 0 and measure_started:
        measured_time = max(1e-6, time.time() - measure_t0)
        final_fps = frame_count / measured_time
        print(
            f"[realtime] Finished. Total frames: {all_frames} (warmup {min(args.warmup_frames, all_frames)}), "
            f"measured frames: {frame_count}, measured time: {measured_time:.2f}s, avg FPS: {final_fps:.1f}"
        )
    else:
        # Fallback: short run entirely within warmup; still provide useful FPS
        final_fps = all_frames / total_time if total_time > 0 else 0.0
        print(
            f"[realtime] Finished (all frames fell into warmup). Total frames: {all_frames}, elapsed: {total_time:.2f}s, "
            f"approx FPS: {final_fps:.1f}. Tip: use --warmup-frames 0 or increase --duration for measured stats."
        )
    if log_fp is not None:
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
