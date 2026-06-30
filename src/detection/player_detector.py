"""
player_detector.py

サッカー画像に対してYOLOで物体検出を行い、検出結果を標準出力に表示しつつ
画像上に描画して表示する研究用スクリプト。

プロジェクト: FootballVisionAI
"""

from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


# 設定値
IMAGE_PATH = Path("videos/raw/test.png")
MODEL_PATH = "yolov8n.pt"


def load_model(model_path: str) -> YOLO:
    """YOLOモデルを読み込む"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"[ERROR] モデルの読み込みに失敗しました: {model_path}")
        print(f"詳細: {e}")
        sys.exit(1)


def load_image(image_path: Path):
    """画像を読み込む（存在確認込み）"""
    print(f"Image path : {image_path}")
    print(f"Exists     : {image_path.exists()}")

    if not image_path.exists():
        print(f"[ERROR] 画像ファイルが存在しません: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] 画像の読み込みに失敗しました（ファイル破損やフォーマット不正の可能性）: {image_path}")
        sys.exit(1)

    return image


def print_detections(boxes, model: YOLO) -> None:
    """検出結果を1件ずつ標準出力に表示する"""
    if boxes is None or len(boxes) == 0:
        print("\n検出されたオブジェクトはありませんでした。")
        return

    print(f"\n検出オブジェクト数: {len(boxes)}")

    for i, box in enumerate(boxes, start=1):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"\nObject #{i}")
        print(f"  Class ID      : {class_id}")
        print(f"  Class Name    : {class_name}")
        print(f"  Confidence    : {confidence:.4f}")
        print(f"  Bounding Box  : ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")


def show_result_image(result) -> None:
    """検出結果を描画した画像をウィンドウ表示する"""
    annotated_image = result.plot()  # YOLOの結果を画像に描画

    cv2.imshow("Detection Result", annotated_image)
    print("\n画像を表示しています。任意のキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    """メイン処理"""
    model = load_model(MODEL_PATH)
    image = load_image(IMAGE_PATH)

    try:
        results = model(image)
    except Exception as e:
        print("[ERROR] 推論中にエラーが発生しました。")
        print(f"詳細: {e}")
        sys.exit(1)

    result = results[0]
    boxes = result.boxes

    print_detections(boxes, model)
    show_result_image(result)


if __name__ == "__main__":
    main()