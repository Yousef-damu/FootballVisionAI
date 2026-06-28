import cv2

video_path = "videos/raw/test.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

print(f"FPS: {fps}")

delay = int(1000 / fps) if fps > 0 else 33

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Football Video", frame)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()