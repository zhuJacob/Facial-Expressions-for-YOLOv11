# coding:utf-8
"""
Realtime YOLO inference script for webcam / video / RTSP.
Usage:
  python realtime_yolo_infer.py            # uses default models/best.pt and webcam
  python realtime_yolo_infer.py --source 0  # use camera 0
  python realtime_yolo_infer.py --source test_video.mp4
  python realtime_yolo_infer.py --source rtsp://user:pass@ip:port/stream

Features:
- Auto-detect webcam if multiple indices fail
- Display FPS and inference time
- Press 'q' to quit, 'r' to toggle recording (saved to output.mp4)
- Press 's' to save a snapshot

Requires: ultralytics, opencv-python
"""

import argparse
import time
from pathlib import Path
import cv2
from ultralytics import YOLO


def find_working_camera(max_id=10):
    for i in range(max_id):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret:
            return i
        cap.release()
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None, help='camera index, video file, or rtsp url')
    parser.add_argument('--model', type=str, default='models/best.pt', help='path to model weights')
    parser.add_argument('--save', action='store_true', help='save output video to output.mp4')
    parser.add_argument('--output', type=str, default='output.mp4', help='output video file')
    parser.add_argument('--device', type=str, default=None, help='torch device e.g. 0 or cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    model_path = Path(args.model)
    # If a relative path was provided, resolve it relative to the script directory
    if not model_path.is_absolute():
        model_path = script_dir / model_path
    print(f"Resolved model path: {model_path}")
    if not model_path.exists():
        print(f"模型文件 {model_path} 不存在，请确认路径。")
        return

    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path)) if args.device is None else YOLO(str(model_path), device=args.device)

    source = args.source
    if source is None:
        cam_id = find_working_camera(10)
        if cam_id is None:
            print("未找到可用摄像头，请通过 --source 指定视频或 RTSP 地址")
            return
        source = int(cam_id)
        print(f"自动选择摄像头: {source}")
    else:
        # try to parse numeric camera index
        try:
            if str(source).isdigit():
                source = int(source)
        except Exception:
            pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开来源: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_writer = None
    recording = False

    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        recording = True
        print(f"开始保存到: {args.output}")

    print("按 'q' 退出, 'r' 切换录制, 's' 保存快照")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，退出")
            break

        t0 = time.time()
        results = model(frame)
        t1 = time.time()

        annotated = results[0].plot()

        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            cur_fps = frame_count / (now - prev_time)
            prev_time = now
            frame_count = 0
        else:
            cur_fps = None

        inf_time = (t1 - t0) * 1000

        # overlay text
        txt = f"FPS: {cur_fps:.1f}" if cur_fps else f"inf: {inf_time:.1f}ms"
        cv2.putText(annotated, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('YOLO Realtime', annotated)

        if recording and out_writer is not None:
            out_writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户退出")
            break
        elif key == ord('r'):
            recording = not recording
            if recording and out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
                print(f"开始保存到: {args.output}")
            print(f"录制: {recording}")
        elif key == ord('s'):
            snap_path = script_dir / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(snap_path), annotated)
            print(f"保存快照: {snap_path}")

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
