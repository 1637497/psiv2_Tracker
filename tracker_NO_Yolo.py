import cv2

cap = cv2.VideoCapture('output2.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

while True:
    ret, frame = cap.read()

    # height, width, _ = frame.shape
    # print(height, width)

    roi = frame[350: 960, 10: 500]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('ROI', roi)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break
    

cap.release()
cv2.destroyAllWindows()