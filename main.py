import cv2
import numpy as np

video_file = R"C:\Users\flori\Videos\DisplayVideoDigitizer\WR_B_20220224_3x_RL02_Vergleich.MOV"

capture = cv2.VideoCapture(video_file)
success, image = capture.read()

img_width = 800
img_height = 600

image = cv2.resize(image, (img_width, img_height))
blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
b,g,r = cv2.split(blur)
gray = blur

win_name = "Frame"
cv2.namedWindow(win_name)

tb_t1 = "Threshold 1"
tb_t2 = "Threshold 2"
cv2.createTrackbar(tb_t1, win_name, 205, 255, lambda _: None)
cv2.createTrackbar(tb_t2, win_name, 255, 255, lambda _: None)

tb_approx = "Approx Poly DP"
cv2.createTrackbar(tb_approx, win_name, 164, 255, lambda _: None)

tb_morph = "Morphology"
cv2.createTrackbar(tb_morph, win_name, 2, 10, lambda _: None)

while 1:
    t1 = cv2.getTrackbarPos(tb_t1, win_name)
    t2 = cv2.getTrackbarPos(tb_t2, win_name)
    t_approx = cv2.getTrackbarPos(tb_approx, win_name) / 255.0 / 10
    t_morph = cv2.getTrackbarPos(tb_morph, win_name) * 2 + 1

    thresh = cv2.threshold(gray, t1, t2, cv2.THRESH_BINARY_INV)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, np.ones((3, 3)), iterations=t_morph)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for c in cnts:
        area = cv2.contourArea(c)

        if area / (img_width * img_height) < 0.01:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, t_approx * peri, True)
        if len(approx) == 4:
            cv2.drawContours(thresh, [approx], -1, (0, 255, 0), 3)

    cv2.imshow(win_name, thresh)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break



cv2.destroyAllWindows()

# while success:
#     image = cv2.resize(image, (800, 600))
#
#     canny = cv2.Canny(image, 50, 200)
#
#     cv2.imshow("Frame", image)
#     cv2.imshow("Canny", canny)
#
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
#     success, image = capture.read()
