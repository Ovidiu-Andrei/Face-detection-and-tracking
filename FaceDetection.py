import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

profilFacePath = "haarcascade_profileface.xml"
profilFace = cv2.CascadeClassifier(profilFacePath)


frontalFacePath = "haarcascade_frontalface_default.xml"
frontalFace = cv2.CascadeClassifier(frontalFacePath)

log.basicConfig(filename='FaceTraking.log', filemode='w', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
isProfil = 0
isFrontal = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    profilFaces = profilFace.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    frontalFaces = frontalFace.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )  

    if len(profilFaces) == 1:
        isProfil = 1
        
    if len(frontalFaces) == 1:
        isProfil = 1

    if isProfil == 1:
        # Write a text on the frame
        for (x, y, w, h) in profilFaces:
            cv2.putText(frame, 
                    'Face detected from profile', 
                    (55, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_AA)  

    if isFrontal == 1:
        # Draw a circle around the detected face
        for (x, y, w, h) in frontalFaces:
            cv2.circle(frame, (w, h), 150, (255, 0, 0), 4)        

    if anterior != len(profilFaces):
        isProfil = 1
        isFrontal = 0 
        anterior = len(profilFaces)
        log.info("profilFaces: "+str(len(profilFaces))+" at "+str(dt.datetime.now()))
        
    if anterior != len(frontalFaces):
        isProfil = 0 
        isFrontal = 1
        anterior = len(frontalFaces)
        log.info("frontalFaces: "+str(len(frontalFaces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
