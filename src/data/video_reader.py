import cv2
video_path = "videos/raw/test.mp4"
cap = cv2.VideoCapture(video_path)
print(cap.isOpened())
while True:
    ret, frame = cap.read()
    print(ret)
    if not ret:
        print("Video End")
        break
    cv2.imshow("Football Video", frame)
    key = cv2.waitKey(30)
    if key == ord("q"):
        break