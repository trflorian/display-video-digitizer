import cv2

video_file = R"C:\Users\flori\Videos\DisplayVideoDigitizer\WR_B_20220224_3x_RL02_Vergleich.MOV"

capture = cv2.VideoCapture(video_file)
success, image = capture.read()

while success:
    cv2.imshow("Frame", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    success, image = capture.read()
