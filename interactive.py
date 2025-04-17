from neuron import Neuron
import pygame
import numpy as np

pygame.init()

W, H = 1280, 720
screen = pygame.display.set_mode((W,H))

clock = pygame.time.Clock()
BACKGROUND = pygame.Color("#000014")
SHADOW = pygame.Color("#001428")
CONTOUR = pygame.Color("#2f4e6e")

track_height = 250
track_margin = 15
track_rect = (track_margin, H-track_height-track_margin, W-2*track_margin, track_height)

control_center = (int(W/4), int((H-track_height-track_margin)/2))
control_radius = 200
cursor_pos = (0, 0)
cursor_radius = 60
cursor_dist = 120
coord_center = (control_center[0] + W/2, control_center[1])

while True:
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # Do logical updates here.
    # ...
    mx, my = pygame.mouse.get_pos()
    mx, my = (mx-control_center[0], my-control_center[1])
    vect_norm = np.sqrt(mx**2 + my**2)
    control = True if vect_norm <= control_radius else False
    if control:
        cursor_pos = (control_center[0] + int(mx / vect_norm * cursor_dist), control_center[1] + int(my / vect_norm * cursor_dist))
    else:
        cursor_pos = control_center

    screen.fill(BACKGROUND)  # Fill the display with a solid color

    # Render the graphics here.
    # ...
    pygame.draw.rect(screen, SHADOW, track_rect)
    pygame.draw.rect(screen, CONTOUR, track_rect, 3)
    pygame.draw.circle(screen, SHADOW, control_center, control_radius)
    pygame.draw.circle(screen, CONTOUR, control_center, control_radius, width=3)
    pygame.draw.circle(screen, CONTOUR, cursor_pos, cursor_radius)

    pygame.draw.circle(screen, SHADOW, coord_center, control_radius)
    pygame.draw.line(screen, CONTOUR, (coord_center[0], coord_center[1]-control_radius), (coord_center[0], control_center[1]+control_radius), width=2)
    pygame.draw.line(screen, CONTOUR, (coord_center[0]-control_radius, coord_center[1]), (coord_center[0]+control_radius, control_center[1]), width=2)

    pygame.display.flip()  # Refresh on-screen display
    clock.tick(60) 