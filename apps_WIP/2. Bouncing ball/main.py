import os
import pygame
from pygame.locals import QUIT


pygame.init()
# Resolution is ignored on Android
surface = pygame.display.set_mode((1920 * 2 // 3, 1080 * 2 // 3))

# Obtenir le chemin absolu du dossier du script courant
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin complet vers le fichier image
image_path = os.path.join(script_dir, "did.png")

ball = pygame.image.load(image_path)
ballrect = ball.get_rect()
clock = pygame.time.Clock()

width = surface.get_width()
height = surface.get_height()

speed = [4, 4]
while True:
    for ev in pygame.event.get():
        if ev.type == QUIT:
            pygame.quit()
    clock.tick(60)
    surface.fill((0, 0, 0))
    ballrect = ballrect.move(speed)
    if ballrect.left < 0 or ballrect.right > width:
        speed[0] = -speed[0]
    if ballrect.top < 0 or ballrect.bottom > height:
        speed[1] = -speed[1]
    surface.blit(ball, ballrect)
    pygame.display.flip()
