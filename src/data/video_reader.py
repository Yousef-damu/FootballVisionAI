import cv2  # OpenCVライブラリを読み込む（動画や画像を扱うため）

# 読み込みたい動画ファイルのパス
video_path = "videos/raw/test.mp4"

# 動画ファイルを開く
cap = cv2.VideoCapture(video_path)

# 動画のFPS（1秒間に表示されるフレーム数）を取得
fps = cap.get(cv2.CAP_PROP_FPS)

# FPSを確認するために表示
print(f"FPS: {fps}")

# フレーム表示の待ち時間を計算（ミリ秒）
# FPSが取得できない場合は約30FPS（33ms）を使用
delay = int(1000 / fps) if fps > 0 else 33

# 動画が開いている間、フレームを繰り返し読み込む
while cap.isOpened():

    # 動画から1フレーム読み込む
    ret, frame = cap.read()

    # フレームが取得できなければ動画の終わりなので終了
    if not ret:
        break

    # 現在のフレームを画面に表示
    cv2.imshow("Football Video", frame)

    # delayミリ秒待機
    # キーボードで「q」を押したら動画再生を終了
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

# 動画ファイルを閉じる
cap.release()

# 開いているすべてのウィンドウを閉じる
cv2.destroyAllWindows()