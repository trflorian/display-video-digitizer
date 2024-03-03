import cv2

video_file = R"C:\Users\flori\Videos\DisplayVideoDigitizer\WR_B_20220224_3x_RL02_Vergleich.MOV"

capture = cv2.VideoCapture(video_file)
success, image = capture.read()

image = cv2.resize(image, (800, 600))
blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

win_name = "Frame"
cv2.namedWindow(win_name)

tb_t1 = "Threshold 1"
tb_t2 = "Threshold 2"
cv2.createTrackbar(tb_t1, win_name, 200, 255, lambda x: None)
cv2.createTrackbar(tb_t2, win_name, 255, 255, lambda x: None)

while 1:
    t1 = cv2.getTrackbarPos(tb_t1, win_name)
    t2 = cv2.getTrackbarPos(tb_t2, win_name)

    thresh = cv2.threshold(gray, t1, t2, cv2.THRESH_BINARY_INV)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(thresh, (x, y), (x + w, y + h), (36, 255, 12), 2)

    cv2.imshow(win_name, thresh)

    # canny = cv2.Canny(image, t1, t2)
    # cv2.imshow(win_name, canny)

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
