import sys
from pathlib import Path

# Simple launcher for realtime with preferred defaults
# Usage: python run_realtime.py [optional extra flags to override]
# Example override: python run_realtime.py --imgsz 512 --vid-stride 2

DEFAULT_WEIGHTS = Path("runs/train/exp_y12s_50e_640/weights/best.pt")
DEFAULT_ARGS = [
    "--weights", str(DEFAULT_WEIGHTS),
    "--device", "cuda",
    "--imgsz", "320",
    "--vid-stride", "2",
    "--duration", "153",
    "--smooth-window", "5",
    "--smooth-min-fps", "20",
]


def main():
    try:
        import realtime  # the project's realtime.py (root)
    except Exception as e:
        print(f"[run_realtime] Failed to import realtime module: {e}")
        sys.exit(1)

    # Allow user-provided overrides after defaults
    extras = sys.argv[1:]
    argv = ["realtime.py", *DEFAULT_ARGS, *extras]

    # Temporarily replace sys.argv and call realtime.main()
    old_argv = sys.argv
    try:
        sys.argv = argv
        realtime.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
