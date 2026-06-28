from pathlib import Path  # ファイルやフォルダのパスを扱うためのライブラリ
from ultralytics import YOLO  # YOLO（物体検出AI）のライブラリを読み込む

# 画像ファイルのパスを指定する
image_path = Path("videos/raw/test.png")

# パスが正しいか確認する
print(image_path.resolve())
print(image_path.exists())

# YOLOモデルを読み込む
model = YOLO("yolov8n.pt")

# 画像に対して推論を実行する
results = model(str(image_path))

# 読み込みに失敗したときの保険
if len(results) == 0:
    print("No image loaded.")
else:
    results[0].show()