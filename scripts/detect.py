#!/usr/bin/env python3
"""Run SHWD + Roboflow helmet detection on a webcam or video file."""

import argparse
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Video path or camera index, such as 0 or 2")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA index, such as 0")
    parser.add_argument("--show", action="store_true", help="Show live annotated frames (requires a desktop)")
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    args = parser.parse_args()
    args.camera = int(args.source) if args.source.isdecimal() else None
    if args.camera is None and not Path(args.source).is_file():
        parser.error(f"video not found: {args.source}")
    if not args.weights.is_file():
        parser.error(f"weights not found: {args.weights}")
    if not 0 < args.conf <= 1 or args.imgsz <= 0:
        parser.error("--conf must be in (0, 1] and --imgsz must be positive")
    if args.output:
        if args.output.suffix.lower() != ".mp4":
            parser.error("--output must be an .mp4 file")
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        if args.camera is None and args.output.resolve() == Path(args.source).resolve():
            parser.error("output must differ from the input video")
        if args.output.resolve() == args.weights.resolve():
            parser.error("output must differ from the model weights")
    return args


def main():
    args = parse_args()
    model = YOLO(str(args.weights))
    if model.task != "detect" or model.names != {0: "helmet", 1: "no_helmet"}:
        raise RuntimeError(f"unexpected model task/classes: {model.task}, {model.names}")

    capture = cv2.VideoCapture(args.camera if args.camera is not None else args.source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    writer = None
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                   device=args.device, verbose=False)[0]
            annotated = result.plot(conf=True, labels=True)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot create output: {args.output}")
                writer.write(annotated)
            frames += 1
            if args.show:
                cv2.imshow("SHWD + Roboflow (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"Processed {frames} frames" + (f"; saved {args.output}" if args.output else ""))


if __name__ == "__main__":
    main()
