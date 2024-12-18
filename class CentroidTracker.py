class CentroidTracker:
    def __init__(self, maxDisappeared=25,maxDistance=50):
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
