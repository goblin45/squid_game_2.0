import numpy as np
import cv2 as cv
import threading
import time

import audio
import light
import countdown
import distance

def startGame():
    # time.sleep(3)
    timeThread = threading.Thread(target=countdown.countTime, args=())
    timeThread.start()

    capture = cv.VideoCapture(0)
    faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # sizing game window
    cv.namedWindow('Game Window', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Game Window', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    TOLERANCE = 5

    global winner
    winner = ""
    winFlag = False

    players = []
    colors = [
        (0, 255, 255),  # Yellow
        (147, 20, 255),  # Pink
        (0, 100, 0),  # Dark Green
        (0, 0, 128),  # Maroon
        (128, 128, 128)  # Gray
    ]

    FINISH = 25  # to be tuned later

    while True:
        ret, frame = capture.read()
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)) # do whatever you want

        # get distance from finishing line
        distances = distance.get_distances(faces)

        for i, (x, y, w, h) in enumerate(faces): 
            cv.rectangle(frame, (x, y), (x + w, y + h), colors[i % 5], 2)

            # positioning of player id
            text = "Player " + str(i + 1)
            text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 10

            cv.putText(frame, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % 5], 2)

            if distances[i] <= 25:
                winner = str(i + 1)
                winFlag = True
                break

            # positioning of distance measurement
            text = "Distance: " + str(distances[i].round(2))
            text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + (w - text_size[0]) // 2
            text_y = y + h + 20

            cv.putText(frame, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % 5], 1)

            if i + 1 > len(players):
                while len(players) < i + 1:
                    players.append([])
            players[i].append((x, y))

            # we are taking past 20 frames into consideration to detect a movement
            if len(players[i]) > 20: # this parameter should be tuned
                players[i].pop()

            if light.currLight == (0, 255, 0): # green; player can move
                for player in players:
                    player.clear()
            else:
                # find out maximum movement the past 20 frames, if it exceeds TOLERANCE, detect it as a movement
                maxMovement = 0
                for j in range(len(players[i]) - 1):
                    for k in range(j + 1, len(players[i])):
                        dist = np.sqrt(abs(players[i][j][0] - players[i][k][0])** 2 + abs(players[i][j][1] - players[i][k][1]) ** 2)
                        maxMovement = max(maxMovement, dist)

                if maxMovement > TOLERANCE: # TOLERANCE should be tuned
                    text = "Player " + str(i + 1) + " is out."
                    text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1.5, 2)

                    # bottom center position for player cancel notification
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] - 30

                    cv.putText(frame, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)  

        text = countdown.currTime
        text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
        
        padding = 20
        # top left position for time display
        text_x = padding
        text_y = text_size[1] + padding

        cv.putText(frame, countdown.currTime, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # positioning of light circle
        lightRadius = 20
        center_x = frame.shape[1] - text_size[0] - padding // 2
        center_y = text_size[1] + padding 

        cv.circle(frame, (center_x, center_y), lightRadius, light.currLight, -1)

        cv.imshow('Game Window', frame)

        if winFlag: 
            audio.gameOn = False
            light.gameOn = False
            showResults()
            break

        if (cv.waitKey(1) & 0xFF == ord('q')) or countdown.timeOver:
            audio.gameOn = False
            light.gameOn = False
            showGameOver()
            break

    capture.release()
    cv.destroyAllWindows()

def showResults():
    # sizing results window
    cv.namedWindow('Results Window', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Results Window', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    resultsImage = np.uint8(np.full((800, 1600, 3), 255))

    winnerStr = "Winner: Player " + winner

    text = winnerStr

    text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 2)

    # center position for menu prompt
    text_x = (resultsImage.shape[1] - text_size[0]) // 2
    text_y = (resultsImage.shape[0] + text_size[1]) // 2

    cv.putText(resultsImage, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv.imshow('Results Window', resultsImage)

    cv.waitKey(0)

def showGameOver():

    # sizing game over window
    cv.namedWindow('Game Over Window', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Game Over Window', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    gameOverImage = np.uint8(np.full((800, 1600, 3), 255))

    text = "GAME OVER: Everyone lost!"

    text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)

    # center position for menu prompt
    text_x = (gameOverImage.shape[1] - text_size[0]) // 2
    text_y = (gameOverImage.shape[0] + text_size[1]) // 2

    cv.putText(gameOverImage, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    cv.imshow('Game Over Window', gameOverImage)

    cv.waitKey(0)

def showmenu():
    # sizing menu window 
    cv.namedWindow('Menu Window', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Menu Window', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    menuImage = np.uint8(np.full((800, 1600, 3), 255))

    text = "Press Enter to Start Game"

    text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 2)

    # center position for menu prompt
    text_x = (menuImage.shape[1] - text_size[0]) // 2
    text_y = (menuImage.shape[0] + text_size[1]) // 2

    cv.putText(menuImage, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv.imshow('Menu Window', menuImage)

    key = cv.waitKey(0)

    if key == 13:
        time.sleep(0.2)
        cv.destroyAllWindows()
        return

if __name__ == "__main__":

    showmenu()

    videoThread = threading.Thread(target=startGame, args=())
    videoThread.start()

    audio.gameOn = True
    audioThread = threading.Thread(target=audio.loopAudio, args=())
    audioThread.start()

    light.gameOn = True
    lightThread = threading.Thread(target=light.loopLight, args=())
    lightThread.start()

    audioThread.join()