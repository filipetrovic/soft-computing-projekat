import cv2
import numpy as np
from scipy import ndimage
from scipy import stats
from vector import distance
from vector import point2line
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
import math
import winsound
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#===================== promenljive
video = "video-8.avi"
vid = cv2.VideoCapture(video)
kernel = np.ones((2,2),np.uint8)
brojevi = []
counter = 0

#=================================================funkcije


    

def tackaNaPravoj(x,y,k,n):
    
    x = float(x)
    
    y = float(y)
    
    k = k.astype(float)
    n = n.astype(float)
    
    fx = k*x+n-y
    
    if (fx == 0):
        return True
    else:
        return False



def odrediPresekPravih(k,n,k1,n1):
    a = np.array([ [k,1],[k1,1] ])
    b = np.array([ n,n1])
    return (np.linalg.solve(a,b))


def linearRegressionTest():
    data = pd.read_csv('lsd.csv')

    data.head()
    X = data['Tissue Concentration'].values[:,np.newaxis]

    y = data['Test Score'].values
    
    model = LinearRegression()
    
    model.fit(X, y)
    plt.scatter(X, y,color='r')

    plt.plot(X, model.predict(X),color='k')
    
    plt.show()

def odrediKoeficijentePrave(x1,y1,x2,y2):
    x1 = x1.astype(float)
    y1 = y1.astype(float)
    x2 = x2.astype(float)
    y2 = y2.astype(float)

    
    k = (y2-y1) / (x2-x1)
    
    n = - ((y2-y1)*x1 - (x2-x1)*y1 )/(x2-x1)
    
   
    return k,n

def makeModelKNNP():
    
    """
    mnist = fetch_mldata('MNIST original')
    data=mnist.data;
    labels = mnist.target.astype('int')
    train_rank = 5000;
    
    #trainData=data;
    #trainLabels=labels;
    train_subset = np.random.choice(data.shape[0], train_rank)
    
    #train dataset
    trainData = data[train_subset]
    trainLabels = labels[train_subset]
    
    """ #ceo data set za ucenje
    mnist = fetch_mldata('MNIST original')
    
    labels = mnist.target.astype('int')
    
   
    
    #train dataset
    trainData =mnist.data
    trainLabels = labels

    
    kVals = range(1, 30, 2)
    #accuracies = []
    ## loop over various values of `k` for the k-Nearest Neighbor classifier
    #    
    #   for k in xrange(1, 30, 2):
    #	# train the k-Nearest Neighbor classifier with the current value of `k`
    #   
    #   	model = KNeighborsClassifier(n_neighbors=k)
    #   	model.fit(trainData, trainLabels)  
    #
    #	# evaluate the model and update the accuracies list
    #   	score = model.score(valData, valLabels)
    #   	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    #   	accuracies.append(score)
    #      
    #
    #
    ## find the value of k that has the largest accuracy
    #   i = np.argmax(accuracies)
    #   print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i] * 100))
    i=1
    
   
   
    for it in range(0,len(trainData)):
        pom=trainData[it];
        #prebacujem iz vektora u matricu
        pom = pom.reshape((28, 28)).astype("uint8")
        pom = exposure.rescale_intensity(pom, out_range=(0, 255))

        ret,pom=cv2.threshold(pom,127, 255, cv2.THRESH_BINARY);

        pom_bin=pom.copy();
        kernel=np.ones((3,3));
        pom=cv2.dilate(pom,kernel,iterations=1);
        kernel=np.ones((1,1));
        pom=cv2.erode(pom,kernel,iterations=2);
        
        img2,contours,hierarchy =cv2.findContours(pom,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
       
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt); 
            pom_cropped=pom_bin[y:y+h,x: x+w];
            pom_cropped = cv2.resize(pom_cropped, (28,28), interpolation = cv2.INTER_CUBIC )
            
                   
                   
            pom_cropped=pom_cropped.flatten();
            pom_cropped = np.reshape(pom_cropped, (1,-1))
            
            cv2.rectangle(pom_bin,(x,y),(x+w,y+h),(0,255,0),2)
            trainData[it]=pom_cropped;
            
                   
        #cv2.imshow('iscrtani reg',pom_cropped_copy);
        #cv2.waitKey()
    
    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)

    return model;

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
#isecenu sliku transformise u oblik pogodan da bi se vrsila predikcija() i vrsi predikciju
def recDigit(cropped_img):
    
    dim = (28, 28)
    pom = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_CUBIC)
    pom=pom.flatten();
    pom = np.reshape(pom, (1,-1))
    prediction=model.predict(pom)[0];
    #plt.figure();        
    #plt.imshow(pom2,'gray');
    #cv2.imshow('naslov',pom2);
    #cv2.waitKey(0)
    return prediction;

id = -1
def idGenerator():
    global id
    id += 1
    return id

def absolute_value(num):
	

	if num >= 0:
		return num
	else:
		return -num

def nadjiNajblizi2(broj,odgovarajuciBrojevi):
    
    
    minDistance = math.hypot(broj['center'][0]-odgovarajuciBrojevi[0]['center'][0], broj['center'][1]-odgovarajuciBrojevi[0]['center'][1] )
    najblizi = odgovarajuciBrojevi[0]
    
    for br in odgovarajuciBrojevi:
        dist = math.hypot(br['center'][0] - broj['center'][0], br['center'][1] - broj['center'][1])
        if dist < minDistance : 
            najblizi = br
            minDistance = distance(br['center'],broj['center'])
            
    
    
   
    
    return najblizi
            

def nadjiNajblizi(broj,brojevi):
    odgovarajuciBrojevi = []
    for br in brojevi:
        dist = math.hypot(br['center'][0] - broj['center'][0], br['center'][1] - broj['center'][1])
        if dist < 20 :
            odgovarajuciBrojevi.append(br)
            
    if not odgovarajuciBrojevi:
        return None
    else :
        return nadjiNajblizi2(broj,odgovarajuciBrojevi)
    

def prepoznajLinije(img):
    kernel = np.ones((2,2),np.uint8)
    gray = cv2.dilate(img,kernel)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
    minLineGap= 8
    minLineLength = 600
    
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength,minLineGap) #minDuzinaLinije, line gap
    
    sveVrednostiXMin = []
    sveVrednostiXMax = []
    
    for i in range(len(lines)):
        sveVrednostiXMin.append(lines[i][0][0])
        
    for i in range(len(lines)):
        sveVrednostiXMax.append(lines[i][0][2])
        
    
    
    index_min = np.argmin(sveVrednostiXMin)
    index_max = np.argmax(sveVrednostiXMax)

    xmin,ymin,xmax,ymax = lines[index_min][0][0] ,lines[index_min][0][1],lines[index_max][0][2],lines[index_max][0][3]
    
    """
    backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    cv2.circle(backtorgb, (xmin,ymin), 4, (25, 25, 255), 1)
    cv2.circle(backtorgb, (xmax,ymax), 4, (25, 25, 255), 1)
    cv2.imshow("test2",backtorgb)
    cv2.waitKey(0)
    """
    return xmin,ymin,xmax,ymax

def filtrirajZelenu(img):
    return cv2.inRange(img,np.array([0,230,0]), np.array([155,255,155]))

def filtrirajPlavu(img):
    return cv2.inRange(img,np.array([230,0,0]), np.array([255,155,155]))


def napraviFrejm():    
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    cv2.imwrite("frame-%d.jpg" % 0, image)     # save frame as JPEG file
    

model=makeModelKNNP();
        
def main():
    napraviFrejm()
    img = cv2.imread('frame-0.jpg')
    
    mask1 = filtrirajZelenu(img)
    mask2 = filtrirajPlavu(img)
    #==========================Kraj maskiranja slika
    ivice1 = prepoznajLinije(mask1)
    ivice2 = prepoznajLinije(mask2)
   
    
    #cv2.imshow('m1', mask1)
    #cv2.waitKey(0)
    #cv2.imshow('m2', mask2)
    #cv2.waitKey(0)
   
    #cv2.circle(img, (ivice1[0],ivice1[1]), 4, (25, 25, 255), 1)
    #cv2.circle(img, (ivice1[2],ivice1[3]), 4, (25, 25, 255), 1)
    #cv2.circle(img, (ivice2[0],ivice2[1]), 4, (25, 25, 255), 1)
    #cv2.circle(img, (ivice2[2],ivice2[3]), 4, (25, 25, 255), 1)
    #==========================Obelezavanje ivica linija
    #cv2.imshow('prikaz',img)
    #cv2.waitKey(0)
    
    #===============================Detekcija regiona brojeva
    #linarRegressionTest()
    suma = 0
    suma1 = 0
    sumaPreostalih = 0;
    frejm = 0
    while  (1) :
        ret, currentFrame = vid.read()
        
        if not ret: 
            print "frejm nije ucitan"
            break
        frejm += 1
        
        for br in brojevi:
           if (frejm - br['frame']) <=5 :
               cv2.circle(currentFrame, (br['center'][0], br['center'][1]), 16, (25, 25, 255), 1)
               cv2.putText(currentFrame,  str(br['id']), (br['center'][0],br['center'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
        
        #cv2.imshow("Current frame",currentFrame)
        #cv2.waitKey(0)
        lower = np.array([220 , 220 , 220],dtype = "uint8")
        upper = np.array([255 , 255 , 255],dtype = "uint8")
        
        trasholdImage = cv2.inRange(currentFrame, lower, upper)
       
        bw_img = np.array(trasholdImage, dtype=float)
        bw_img = cv2.dilate(bw_img,kernel,iterations=2)
        
        labeled, _ = ndimage.label(bw_img)
        
        pictureElements = ndimage.find_objects(labeled)
        
        for i in range(len(pictureElements)) :
            location = pictureElements[i]
            
            center = []
            center.append((location[1].stop + location[1].start) /2)
            center.append((location[0].stop + location[0].start) /2)
            
            dimension = []
            dimension.append(location[1].stop - location[1].start)
            dimension.append(location[0].stop - location[0].start)
            
            if dimension[0]>=9 and dimension[1]>=9 :
                crop_img = bw_img[location[0].start:location[0].stop,location[1].start:location[1].stop]
                
                dimenzijaBroja = (dimension[0], dimension[1])
                centarBroja = (center[0],center[1])
                
                broj = {'center': centarBroja, 'dimension': dimenzijaBroja, 'frame': frejm, 'img' : crop_img}
                
                istiBrojUProslomFrejmu = nadjiNajblizi( broj, brojevi)
                
                
                if istiBrojUProslomFrejmu is None:
                    broj['id'] = idGenerator()
                    broj['prosaoPrvu'] = False
                    broj['prosaoDrugu'] = False
                    broj['tokKretanja'] = [{'center': centarBroja, 'size': dimenzijaBroja, 'frame': frejm}]
                    brojevi.append(broj)
                else:
                    istiBrojUProslomFrejmu['center'] = broj['center']
                    istiBrojUProslomFrejmu['frame'] = frejm
                    istiBrojUProslomFrejmu['tokKretanja'].append({'center': centarBroja, 'size': dimenzijaBroja, 'frame': frejm})
                    
           
        for br in brojevi:
            
            A1 = (ivice1[0],ivice1[1])
            B1 = (ivice1[2],ivice1[3])
            A2 = (ivice2[0],ivice2[1])
            B2 = (ivice2[2],ivice2[3])
            
            
            distancaBrojaOdLinije1 = point2line(br['center'], A1,B1)
            distancaBrojaOdLinije2 = point2line(br['center'], A2,B2)
            
            if (frejm - br['frame'] > 20 ) :
                x = []
                y = []
                for centar in br['tokKretanja']:
                    x.append(centar['center'][0])
                    y.append(centar['center'][1])
                
                slope,intercept,r_value,p_value,std_err = stats.linregress(x,y)
                k,n = odrediKoeficijentePrave(ivice1[0],ivice1[1],ivice1[2],ivice1[3])
                
                m = slope
                b = intercept
                nz, cz  = odrediKoeficijentePrave(ivice1[0], ivice1[1], ivice1[2], ivice1[3])
                npl,cp = odrediKoeficijentePrave(ivice2[0], ivice2[1], ivice2[2], ivice2[3])
                
                tackaPresekaZeleneX = (cz - b) / (m-nz)
                tackaPresekaZeleneY = m*tackaPresekaZeleneX + b
                
                tackaPresekaPlaveX =  (cp - b) / (m-npl)
                tackaPresekaPlaveY = m*tackaPresekaPlaveX+b
                
              
               
                #print tackaPreseka[1]
               
                if ( ( tackaPresekaZeleneX > ivice1[0] and tackaPresekaZeleneX < ivice1[2] ) and (tackaPresekaZeleneY > y[0] ) ):
                    suma -= recDigit(br['img'])
                        
                        
                if(  (tackaPresekaPlaveX > ivice2[0] and tackaPresekaPlaveX < ivice2[2] ) and ( tackaPresekaPlaveY > y[0] ) ):
                    suma += recDigit(br['img'])
                
                brojevi.remove(br)
            
        
        
        
        for br in brojevi:
            
            prepoznat = 0
            
            distancaBrojaOdLinije1 = point2line(br['center'], (ivice1[0], ivice1[1]),(ivice1[2], ivice1[3]))
            distancaBrojaOdLinije2 = point2line(br['center'], (ivice2[0], ivice2[1]),(ivice2[2], ivice2[3]))
                

                
            #cv2.line(img, pnt1, el['center'], (0, 255, 25), 1)
                
            if (distancaBrojaOdLinije1 < 10 and not br['prosaoPrvu']):
                
                    
                if br['prosaoPrvu'] == False:
                    br['prosaoPrvu'] = True
                    prepoznat = recDigit(br['img'])
                    
                    suma1 -= prepoznat
                        
            if (distancaBrojaOdLinije2 < 10 and not br['prosaoDrugu']):
                
                    
                if br['prosaoDrugu'] == False:
                    br['prosaoDrugu'] = True
                    prepoznat = recDigit(br['img'])
                    
                    suma1 += prepoznat
        
    for br in brojevi:
        x = []
        y = []
        for centar in br['tokKretanja']:
            x.append(centar['center'][0])
            y.append(centar['center'][1])      
            
        
        
        slope,intercept,r_value,p_value,std_err = stats.linregress(x,y)
        k,n = odrediKoeficijentePrave(ivice1[0],ivice1[1],ivice1[2],ivice1[3])
                
        m = slope
        b = intercept
        nz, cz  = odrediKoeficijentePrave(ivice1[0], ivice1[1], ivice1[2], ivice1[3])
        npl,cp = odrediKoeficijentePrave(ivice2[0], ivice2[1], ivice2[2], ivice2[3])
                
        tackaPresekaZeleneX = (cz - b) / (m-nz)
        tackaPresekaZeleneY = m*tackaPresekaZeleneX + b
                
        tackaPresekaPlaveX =  (cp - b) / (m-npl)
        tackaPresekaPlaveY = m*tackaPresekaPlaveX+b
            
        if(tackaPresekaZeleneX > ivice1[0] and tackaPresekaZeleneX < ivice1[2] ):
            if( tackaPresekaZeleneY > y[0] ):
                razlika = tackaPresekaZeleneY - y[-1]
                if(razlika < 30):
                    sumaPreostalih -= recDigit(br['img'])
                    #print "preostao je broj: ", recDigit(br['img'])
        
        
        if(tackaPresekaPlaveX > ivice2[0] and tackaPresekaPlaveX < ivice2[2] ):
            if( tackaPresekaPlaveY > y[0] ):
                razlika = tackaPresekaPlaveY - y[-1]
                if(razlika < 30):
                    sumaPreostalih += recDigit(br['img'])
                    #print "preostao je broj: ", recDigit(br['img'])         
            
        
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    print "SUMA JE: ", suma + sumaPreostalih
    print "SUMA1 JE: ", suma1
    
    vid.release()
    cv2.destroyAllWindows()
    
    
    
    
    
main()