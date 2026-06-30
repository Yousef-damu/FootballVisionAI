"""
video_reader.py

サッカー試合動画をOpenCVで1フレームずつ読み込み、再生表示する研究用スクリプト。
FPSに応じて再生速度を調整し、'q'キーで終了できる。

プロジェクト: FootballVisionAI
"""

from pathlib import Path
import sys
import cv2

# 設定値
VIDEO_PATH = Path("videos/raw/test.mp4")

def check_video_exists(video_path: Path) -> None:
    """動画ファイルの存在確認を行う"""
    print(f"Video path : {video_path}")
    print(f"Exists     : {video_path.exists()}")

    if not video_path.exists():
        print(f"[ERROR] 動画ファイルが存在しません: {video_path}")
        sys.exit(1)


def open_video(video_path: Path) -> cv2.VideoCapture:
    """動画を開く（失敗時は分かりやすく終了する）"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] 動画を開けませんでした（コーデックやファイル破損の可能性）: {video_path}")
        sys.exit(1)

    return cap


def get_fps(cap: cv2.VideoCapture) -> float:
    """動画のFPSを取得する（取得失敗時はデフォルト値を使う）"""
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("[WARNING] FPSを取得できませんでした。デフォルト値 30 を使用します。")
        fps = 30.0

    print(f"FPS        : {fps:.2f}")
    return fps


def play_video(cap: cv2.VideoCapture, fps: float) -> None:
    """動画を1フレームずつ読み込み、再生表示する"""
    wait_time_ms = int(1000 / fps)  # FPSに応じた待機時間（ミリ秒）

    while True:
        ret, frame = cap.read()

        # 動画終了（フレームが読み込めない）
        if not ret:
            print("Video ended.")
            break

        cv2.imshow("Video Reader", frame)

        # 'q'キーで終了
        key = cv2.waitKey(wait_time_ms) & 0xFF
        if key == ord("q"):
            print("'q' キーが押されたため終了します。")
            break


def main() -> None:
    """メイン処理"""
    check_video_exists(VIDEO_PATH)
    cap = open_video(VIDEO_PATH)

    try:
        fps = get_fps(cap)
        play_video(cap, fps)
    except Exception as e:
        print("[ERROR] 動画再生中にエラーが発生しました。")
        print(f"詳細: {e}")
    finally:
        # 後処理は必ず実行する
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()