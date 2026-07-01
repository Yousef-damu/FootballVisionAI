"""
player_tracker.py

Purpose:
    Track soccer players in a match video using YOLO + ByteTrack.

Project:
    FootballVisionAI
"""

from pathlib import Path
import cv2
from ultralytics import YOLO


def main() -> None:
    # 入力動画のパス
    video_path = Path("videos/raw/test.mp4")

    # 出力動画のパス
    output_dir = Path("outputs/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "player_tracking.mp4"

    # パス確認
    print(f"Video path: {video_path.resolve()}")
    print(f"Exists: {video_path.exists()}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # YOLOモデルを読み込む
    model = YOLO("yolov8n.pt")

    # 動画を開く
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # 動画情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}")
    print(f"Frame size: {width}x{height}")

    # 保存用のVideoWriterを作成
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30, (width, height))

    # 追跡ループ
    frame_index = 0
    while True:
        ret, frame = cap.read()

        # 動画の最後まで読んだら終了
        if not ret:
            print("Video ended.")
            break

        frame_index += 1

        # person(class=0)だけを追跡
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # 検出結果が空ならそのまま表示
        if len(results) == 0:
            print(f"Frame {frame_index}: No results")
            cv2.imshow("Player Tracking", frame)
            writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        result = results[0]

        # 追跡IDがある場合はログ出力
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                class_name = model.names[class_id]

                print("-----")
                print(f"Frame: {frame_index}")
                print(f"Track ID: {track_id}")
                print(f"Class ID: {class_id}")
                print(f"Class Name: {class_name}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Bounding Box: {bbox}")

        # 追跡結果を描画
        annotated_frame = result.plot()

        # 画面表示
        cv2.imshow("Player Tracking", annotated_frame)

        # 保存
        writer.write(annotated_frame)

        # qキーで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 後処理
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()