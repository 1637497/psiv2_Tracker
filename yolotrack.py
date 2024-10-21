

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import cv2

cap = cv2.VideoCapture(r"C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\seguiment\output2moltcurt.mp4")
model1=torch.hub.load('ultralytics/yolov5','custom', path='C:/Users/Usuario/OneDrive/Escriptori/UAB/4t/psiv/seguiment/model/best50.pt')#, force_reload=True)


fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv2.VideoWriter(r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\seguiment\resultatyolotrack1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


# def detectar(model, frame):
#     print("Detección de coche")
#     results = model(frame)
    
#     # Procesar resultados
#     df = results.pandas().xyxy[0]
#     df = df[df['confidence'] > 0.1]
#     cotxe = df.sort_values(by='confidence', ascending=False).head(1) 
    


#     xmin = round(cotxe['xmin'])
#     ymin = round(cotxe['ymin'])
#     xmax = round(cotxe['xmax'])
#     ymax = round(cotxe['ymax'])
    
#     return (xmin, ymin, xmax - xmin, ymax - ymin)

def detectar(model,i):
        print("cotxe")
        results=model1(i) 
    
        ims=np.squeeze(results.render())
    
    
        df=results.pandas().xyxy[0]
    
        df=df[df['confidence']>0.49]
    
        df=df.to_dict(orient='records')
    
    
        for cotxe in df:
    
            xmin=round(cotxe['xmin'])
    
            ymin=round(cotxe['ymin'])
    
            xmax=round(cotxe['xmax'])
    
            ymax=round(cotxe['ymax'])
            
            
        
        return (xmin, ymin, xmax - xmin, ymax - ymin)


    
            

detectat=False
tracker = cv2.legacy.TrackerCSRT.create()

while True:
    
    ret, frame = cap.read()    
   
    roi = frame[350:460,50:440]
    
    
    
    if not detectat:
        bbox=detectar(model1, roi)
        bbox=(bbox[0]+50,bbox[1]+350,bbox[2],bbox[3])
        detectat=True
        ok = tracker.init(frame, bbox)
    if detectat:
        ok, bbox = tracker.update(frame)
        if ok:
            drawRectangle(frame, bbox)
        
        
        
    # out.write(frame)
    cv2.imshow("frame",frame)
    

    
    key = cv2.waitKey(30)
    if key & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

