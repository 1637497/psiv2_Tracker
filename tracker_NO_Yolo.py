import cv2

print("1")
cap = cv2.VideoCapture('output7.mp4')
print("2")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()

    # height, width, _ = frame.shape
    # print(height, width)

    roi = frame[350: 960, 10: 500]

    mask = object_detector.apply(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 3)

    cv2.imshow('ROI', roi)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break
    

cap.release()
cv2.destroyAllWindows()