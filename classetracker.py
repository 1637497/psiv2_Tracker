from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


import numpy as np
import cv2
import torch
import pathlib
import matplotlib.pyplot as plt

# Substituir el Path en cas de Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

cap = cv2.VideoCapture(r"C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\seguiment\output2curt.mp4")
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Usuario/OneDrive/Escriptori/UAB/4t/psiv/seguiment/model/best25.pt')


def detectar(i):
    ll=[]
    results = model1(i)

    ims = np.squeeze(results.render())
    
    
    df = results.pandas().xyxy[0]

    df = df[df['confidence'] > 0.75]
    
    df = df.sort_values(by='confidence', ascending=False)

   
    df = df.to_dict(orient='records')
    

    if df:
        for cotxe in df:
            xmin = round(cotxe['xmin'])
            ymin = round(cotxe['ymin'])
            xmax = round(cotxe['xmax'])
            ymax = round(cotxe['ymax'])
        
            ll.append((xmin, ymin, xmax - xmin, ymax - ymin))
        
    if len(ll)>1:
        print(ll)
    return ll



class CentroidTracker:
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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[340:500, 50:380]

    bboxes = detectar(roi)

    objects = tracker.update(bboxes)

    for (objectID, centroid) in objects.items():
        for (startX, startY, endX, endY) in bboxes:
            
            # adj_startX, adj_startY = startX + 50, startY + 340
            # adj_endX, adj_endY = endX + 50, endY + 340
            adj_startX, adj_startY = startX + 120, startY + 340
            adj_endX, adj_endY = endX + 120, endY + 340

            cv2.rectangle(frame, (adj_startX, adj_startY), (adj_endX, adj_endY), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {objectID}", (centroid[0] + 50 - 10, centroid[1] + 340 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
