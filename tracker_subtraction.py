
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from time import time

# Centroid Tracker Class
class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

# Video input and object detection using background subtraction
cap = cv2.VideoCapture('output7.mp4')
object_detector = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=50)
tracker = CentroidTracker()

t0 = time()
i = 0
prev_bboxes = []  

while True:
    i += 1
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[350:960, 10:500]
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, x+w, y+h))

    if bboxes:
        prev_bboxes = bboxes

    objects = tracker.update(np.array([(int((x + w) / 2), int((y + h) / 2)) for x, y, w, h in prev_bboxes]))

    for (objectID, centroid) in objects.items():
        for (startX, startY, endX, endY) in prev_bboxes:
            adj_startX, adj_startY = startX , startY
            adj_endX, adj_endY = endX, endY 
            cv2.rectangle(roi, (adj_startX, adj_startY), (adj_endX, adj_endY), (0, 255, 0), 2)
        cv2.putText(roi, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(roi, (centroid[0] , centroid[1] ), 4, (0, 255, 0), -1)

    cv2.imshow('ROI', roi)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

t = time()
print("Número total de frames:", i)
print("Ha trigat", t - t0, "segons")
print("Ha trigat", (t - t0) / 60, "minuts")
