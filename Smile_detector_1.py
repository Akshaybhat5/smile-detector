import cv2

face = cv2.CascadeClassifier('face_detector.xml')
smile = cv2.CascadeClassifier('smile_detector.xml')

# webcam feeding
webcam = cv2.VideoCapture(0)

while True:
    successful_frame, frame = webcam.read()

    if not successful_frame:
        break 

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Face detecting
    face_detector = face.detectMultiScale(frame_grey)

    for (x, y, w, h) in face_detector:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200, 50), 4)

        the_face = frame[y:y+h, x:x+w]

        face_grey = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_detector = smile.detectMultiScale(face_grey, scaleFactor=1.7, minNeighbors=20)
        if len(smile_detector) > 0:
            cv2.putText(frame, 'smiling',(x,y+h+40), fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=3, color=(255,255,255)) 
               
    cv2.imshow('smile detector',frame)       
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()








