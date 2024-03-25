import pygame

def playAudio(audioFilePath='./squid.mp3'):
    pygame.init()

    pygame.mixer.music.load(audioFilePath)

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)