import cv2 as cv
import time

capture = cv.VideoCapture(0)

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cv.namedWindow('Webcam Feed', cv.WINDOW_NORMAL)
cv.setWindowProperty('Webcam Feed', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

prevCoordinates = (0, 0, 0, 0) # x, y, w, h
TOLERANCE = 5
moveText = ""

def setTimeout(callback, delay):
    time.sleep(delay / 1000)
    callback()

def setMoving():
    moveText = ""

moves = 0

while True:
    ret, frame = capture.read()

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x, y, w, h) in faces: 
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(x, y)
        x1, y1, w1, h1 = prevCoordinates
        if (abs(x - x1) > TOLERANCE or abs(y - y1) > TOLERANCE 
            or abs(w - w1) > TOLERANCE or abs(h - h1) > TOLERANCE):
            # print('moving')
            moves += 1
            if moves > TOLERANCE / 4:
                moveText = "moving..."
                print(moves, moveText)
            # setTimeout(setMoving, 250)
        else:
            moveText = ""
            moves = 0
        prevCoordinates = (x, y, w, h)

    cv.putText(frame, moveText, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow('Webcam Feed', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()