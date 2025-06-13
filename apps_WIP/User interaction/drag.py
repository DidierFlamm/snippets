import sys
import pygame
from pygame.locals import *

pygame.init()
# Resolution is ignored on Android
surface = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()
surfrect = surface.get_rect()
rect = pygame.Rect((0, 0), (128, 128))
rect.center = (surfrect.w / 2, surfrect.h / 2)
touched = False
while True:
    for ev in pygame.event.get():
        if ev.type == QUIT:
            pygame.quit()
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if rect.collidepoint(ev.pos):
                touched = True
                # This is the starting point
                pygame.mouse.get_rel()
        elif ev.type == pygame.MOUSEBUTTONUP:
            touched = False
    clock.tick(60)
    surface.fill((0, 0, 0))
    if touched:
        rect.move_ip(pygame.mouse.get_rel())
        rect.clamp_ip(surfrect)
    surface.fill((255, 255, 255), rect)
    pygame.display.flip()
