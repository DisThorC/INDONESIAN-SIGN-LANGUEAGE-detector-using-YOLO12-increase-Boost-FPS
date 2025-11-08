import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference with Ultralytics YOLO")
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights .pt")
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image/video or webcam index (e.g., 0)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="Filter by class indices (e.g., 0 2 5)")
    p.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    p.add_argument("--augment", action="store_true", help="Enable test-time augmentation (slower, sometimes better)")
    p.add_argument("--track", action="store_true", help="Use tracker for temporal smoothing in realtime mode")
    p.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config file (BoT-SORT/ByteTrack)")
    p.add_argument("--device", type=str, default="cpu", help='"cpu" or "cuda"')
    p.add_argument("--save-dir", type=str, default="runs/infer", help="Project directory to save results")
    p.add_argument("--realtime", action="store_true", help="Enable low-latency webcam/stream mode")
    p.add_argument("--show", action="store_true", help="Display annotated frames in a window")
    p.add_argument("--half", action="store_true", help="Use FP16 on CUDA for speed")
    p.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame for speed")
    return p


def infer(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    project = args.save_dir or "runs/infer"
    print(f"[yolo12] Inference on {args.source} with weights {args.weights}")

    # Safety: disable half when not on CUDA
    if str(args.device).lower() == "cpu" and args.half:
        print("[yolo12] Warning: --half ignored on CPU. Using FP32.")
        args.half = False

    if args.realtime:
        # Stream results for lower latency; optionally display
        if args.track:
            gen = model.track(
                source=args.source,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=args.classes,
                max_det=args.max_det,
                device=args.device,
                half=args.half,
                vid_stride=args.vid_stride,
                stream=True,
                persist=True,
                save=False,
                tracker=args.tracker,
                project=project,
                name="realtime",
                exist_ok=True,
            )
        else:
            gen = model.predict(
                source=args.source,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=args.classes,
                max_det=args.max_det,
                device=args.device,
                half=args.half,
                vid_stride=args.vid_stride,
                stream=True,
                save=False,
                project=project,
                name="realtime",
                exist_ok=True,
            )
        win = "yolo12-realtime"
        for res in gen:
            im = res.plot()  # annotated frame (numpy array BGR)
            if args.show:
                cv2.imshow(win, im)
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if args.show:
            cv2.destroyAllWindows()
        print(f"[yolo12] Realtime inference finished.")
    else:
        results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            max_det=args.max_det,
            device=args.device,
            half=args.half,
            augment=args.augment,
            save=True,
            project=project,
            name="exp",
            exist_ok=True,
        )
        # results is a list; Ultralytics handles saving visuals
        print(f"[yolo12] Inference complete. Saved under {project}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()

