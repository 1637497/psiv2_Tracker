from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


import numpy as np
import cv2
import torch
import pathlib
import matplotlib.pyplot as plt
from time import time

# Substituir el Path en cas de Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# cap = cv2.VideoCapture(r"C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\seguiment\output2curt.mp4")
cap = cv2.VideoCapture('output2.mp4')
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Tracker/best50.pt')

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv2.VideoWriter(r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\seguiment\resultattrack.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def detectar(i):
    ll=[]
    results = model1(i)
    ims = np.squeeze(results.render())    
    df = results.pandas().xyxy[0]
    df = df[df['confidence'] > 0.75]      
    df = df.to_dict(orient='records')
    if df:
        for cotxe in df:
            xmin = round(cotxe['xmin'])
            ymin = round(cotxe['ymin'])
            xmax = round(cotxe['xmax'])
            ymax = round(cotxe['ymax'])        
            ll.append((xmin, ymin, xmax, ymax))   
    return ll


class CentroidTracker:
    def __init__(self, maxDisappeared=25,maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        # self.maxDistance=maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
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
                # if D[row,col]>self.maxDistance:
                #     continue            
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects


tracker = CentroidTracker()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

t0 = time()

i = 0
prev_bboxes = []  

while True:
    i += 1
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[340:960, 50:380]
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if i % 4 == 0:
        yolo = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                yolo = True
                break       
        if yolo:
            bboxes = detectar(roi)
            if bboxes:
                prev_bboxes = bboxes 
            objects = tracker.update(bboxes)
        else:
            objects = tracker.update(prev_bboxes)
    else:
        objects = tracker.update(prev_bboxes)
    for (objectID, centroid) in objects.items():
        for (startX, startY, endX, endY) in prev_bboxes:
            adj_startX, adj_startY = startX + 50, startY + 340
            adj_endX, adj_endY = endX + 50, endY + 340
            cv2.rectangle(frame, (adj_startX, adj_startY), (adj_endX, adj_endY), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (centroid[0] + 50 - 10, centroid[1] + 340 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0] + 50, centroid[1] + 340), 4, (0, 255, 0), -1)
    cv2.imshow('mask', mask)
    cv2.imshow('Tracking', frame)
    # out.write(frame)
    key = cv2.waitKey(30)
    if key == 27:
        break


cap.release()
# out.release()
cv2.destroyAllWindows()
t = time()
print("Número total de frames:", i)
print("Ha trigat", t - t0, "segons")
print("Ha trigat", (t - t0) / 60, "minuts")
