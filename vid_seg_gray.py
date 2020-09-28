import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
from sys import stdin, stdout

vs = cv2.VideoCapture(0)

x = 1280
y = 720
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_EXPOSURE,-13) # maxima velocidade do obturador 
camera.set(cv2.CAP_PROP_FRAME_WIDTH, x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, y)
camera.set(cv2.CAP_PROP_FPS, 45)
time.sleep(2) #iniciando o sensor

def nothing(x):
    pass


cv2.namedWindow('Trackbars', flags=cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Trackbars', 600, 600)

cv2.createTrackbar('LH', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('LS', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('LV', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('UH', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('US', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('UV', 'Trackbars', 255, 255, nothing)
i = 0
while (True):
    status = "Liberado"
    a = time.time()
    
    _, frame = vs.read()
    # code = cv2.cvtColor(code, cv2.COLOR_BGR2RGB)
    code = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bluer = cv2.blur(code, (1,1))
  

    l_h = cv2.getTrackbarPos('LH', 'Trackbars')
    l_s = cv2.getTrackbarPos('LS', 'Trackbars')
    l_v = cv2.getTrackbarPos('LV', 'Trackbars')
    u_h = cv2.getTrackbarPos('UH', 'Trackbars')
    u_s = cv2.getTrackbarPos('US', 'Trackbars')
    u_v = cv2.getTrackbarPos('UV', 'Trackbars')

    img1 = np.array([l_h, l_s, l_v])
    img2 = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(bluer, img1, img2)
    result = cv2.bitwise_and(bluer, bluer, mask=mask)
    
    scale_percent = 50 #redimensionando a imagem
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)


    #cv2.namedWindow('Result')
    #cv2.resizeWindow('Result', 300, 300)
    result = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    code = cv2.resize(code, dim, interpolation = cv2.INTER_AREA)
    
    #cv2.resizeWindow('Mask', 300, 300)
    
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    ## aplicação do filtro para posteriormente obter as bordas da imagem isolada
    edges = cv2.Canny(mask,30,200)
    
    #primeiro ele vai procurar contorno
    cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
    #frame = imutils.resize(frame, width=400)
    A = None
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        #print((x,y,w,h))
        if h > 10 and w > 10:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            zoom = frame[int(1.5*y):int(2.5*(y+h)), int(1.5*x):int(2.5*(x+w))]
            zoom = cv2.resize(zoom, (400,400),interpolation = cv2.INTER_AREA)
            #cv2.imwrite("fto/zoom" + str(i) + ".jpg", zoom)
            cv2.imshow("zoom",zoom)
            i = i + 1

     #aplicando texto
    cv2.putText(result, status,(20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    
    
    cv2.namedWindow('Result')
    cv2.imshow('Result', result)
   
    cv2.namedWindow('HSV')
    cv2.imshow('HSV', code)

    cv2.namedWindow('Mask')
    cv2.imshow('Mask', mask)

    cv2.namedWindow('Bordas')
    cv2.imshow('Bordas', edges)
    
    
    b = time.time()
    #stdout.write("\nb:'{0}'".format(b-a))
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
vs.release()
cv2.destroyAllWindows()
