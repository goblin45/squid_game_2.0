import cv2
import numpy as np


def calculate_distance(focal_length, known_width, width_in_frame):
    return (known_width * focal_length) / width_in_frame


def find_distances(focal_length, known_width, detections):
    distances = []
    for detection in detections:
        width = detection[2]
        distances.append(calculate_distance(focal_length, known_width, width))
    return distances


KNOWN_WIDTH = 11.0  
KNOWN_DISTANCE = 24.0  


cap = cv2.VideoCapture(0)


winner_distance = float('inf')
winner_label = None


pause_movement = False

while True:
    if not pause_movement:
        ret, frame = cap.read()
    else:
        
        ret = True
    
    if ret:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        
        focal_length = 500  
        distances = find_distances(focal_length, KNOWN_WIDTH, faces)

        
        for i, ((x, y, w, h), distance) in enumerate(zip(faces, distances)):
            label = f"Distance: {distance:.2f} inches"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        cv2.imshow('Object Distance Estimation', frame)

        
        if pause_movement:
            
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

            
            if min_distance < winner_distance:
                winner_distance = min_distance
                winner_label = f"Winner: {min_distance:.2f} inches"

            
            if winner_label:
                (x, y, w, h) = faces[min_distance_index]
                cv2.putText(frame, winner_label, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        
        pause_movement = not pause_movement


cap.release()
cv2.destroyAllWindows()
