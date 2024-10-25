from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


import numpy as np
import cv2
import torch
import pathlib
import matplotlib.pyplot as plt
from time import time
 

# cap = cv2.VideoCapture(r"C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\seguiment\output2curt.mp4")
cap = cv2.VideoCapture('output2.mp4')
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/aina/Desktop/uni/4rt/psiv/repte2/best50.pt', force_reload=True)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv2.VideoWriter(r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\seguiment\resultattrack.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def detectar(i):
    ll=[]
    results = model1(i)
    ims = np.squeeze(results.render())       
    cv2.imwrite('saquenhojiya.jpg',ims) 
    df = results.pandas().xyxy[0]
    df = df[df['confidence'] > 0.85]      
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
    def __init__(self, maxDisappeared=10,maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance=maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def trobar_minims(self,diccionari):
        resultats = []
        for k, v in diccionari.items():
            # Trobar la posició del valor mínim
            pos_minim = np.argmin(v)
            # Afegir la parella (k, posició del mínim) a la llista de resultats
            resultats.append((k, pos_minim))
        return resultats

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
            print("Distàncies:",D) 
            #Creem diccionari on la clau és l'ID dels centroids que ja 
            #teníem i els valors son la distància a cada un dels centroids nous
            diccionari = {i: D[i] for i in range(D.shape[0])}
            diccionari = dict(sorted(diccionari.items(), key=lambda x: min(x[1])))
            #Apliquem una funció que troba la posició del mínim
            #dels valors per a cada clau
            #Osigui retorna per cada clau la següent parella:
            #(Clau, posició de la distància mínima)
            resultat = self.trobar_minims(diccionari)

            #Creem 2 llistes amb els IDS
            idsactuals=list(range(D.shape[0]))
            idsnous=list(range(D.shape[1]))

            #Iterem sobre les parelles
            for parella in resultat:
                actual=parella[0]
                nou=parella[1]
                
                #Comprovem si la distància és major que maxDistance
                if D[actual,nou]>self.maxDistance:
                    continue
                
                #Mirem si ja hem revisat el centroid
                if actual not in idsactuals or nou not in idsnous:
                    continue
                
                #Aconseguim l'ID del centroid actual
                objectID = objectIDs[actual]
                #Ara actualitzem el centroid actual amb el valor del nou
                self.objects[objectID] = inputCentroids[nou]
                #Posem desaparegut a 0
                self.disappeared[objectID] = 0
                #Borrem de les llistes per indicar que ja els hem revisat
                idsactuals.remove(actual)
                idsnous.remove(nou)
                
            #Ara a les llistes nomes queden aquells centroids que no hem revisat
            #Per tant, si hi ha més idsnous que actuals, haurem de registrar-los
            if len(idsnous)>len(idsactuals):
                for id in idsnous:
                    self.register(inputCentroids[id])
            
            #Però si n'hi ha més d'actuals que de nous,
            #significarà que alguns hauran desaparegut
            else:
                for id in idsactuals:
                    objectID = objectIDs[id]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        #Si porta massa temps desaparegut, l'eliminem
                        self.deregister(objectID) 
        return self.objects


tracker = CentroidTracker()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

t0 = time()

i = 0
prev_bboxes = []  
 # Inicializar contadores para vehículos que suben y bajan
count_up = 0
count_down = 0

# Definir una línea horizontal de referencia para diferenciar entre subida y bajada
line_y = 900 # Por ejemplo, la mitad de la altura del cuadro

# Bucle principal de procesamiento de video
while True:
    i += 1
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[720:960, 50:380]
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
    cv2.line(frame, (0, 900), (frame_width, 900), (0, 0, 255), 2)
    # Actualizar los contadores de subida y bajada
    for (objectID, centroid) in objects.items():
        if objectID in tracker.objects:
            prev_centroid = tracker.objects[objectID]
            # Verificar si el coche cruza la línea de referencia
            if prev_centroid[1] < line_y <= centroid[1]:  # Bajando
                count_down += 1
            elif prev_centroid[1] > line_y >= centroid[1]:  # Subiendo
                count_up += 1

        for (startX, startY, endX, endY) in prev_bboxes:
            adj_startX, adj_startY = startX + 50, startY + 720
            adj_endX, adj_endY = endX + 50, endY + 720
            cv2.rectangle(frame, (adj_startX, adj_startY), (adj_endX, adj_endY), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (centroid[0] + 50 - 10, centroid[1] + 720 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0] + 50, centroid[1] + 720), 4, (0, 255, 0), -1)

    # Mostrar los contadores en la parte baja de la imagen
    cv2.putText(frame, f'Cotxes que pujen: {count_up}', (10, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f'Cotxes que baixen: {count_down}', (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
