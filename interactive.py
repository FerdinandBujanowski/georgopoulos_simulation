from neuron import Neuron
import pygame
import numpy as np
import time

neurons = []
for i in range(8):
    neurons.append(Neuron(i*int(360/8)))

pygame.init()
DT = 1/60

W, H = 1280, 720
screen = pygame.display.set_mode((W,H))

clock = pygame.time.Clock()
BACKGROUND = pygame.Color("#000014")
SHADOW = pygame.Color("#001428")
CONTOUR = pygame.Color("#2f4e6e")

track_height = 240
track_margin = 15
track_rect = (track_margin, H-track_height-track_margin, W-2*track_margin, track_height)

control_center = (int(W/4), int((H-track_height-track_margin)/2))
control_radius = 200
cursor_pos = (0, 0)
cursor_radius = 60
cursor_dist = 120
coord_center = (control_center[0] + W/2, control_center[1])

# setting up neuron related stuff
TICK_WIDTH = 2 # pixels per tick
TICKS_CONSIDERED = 50
display_space = int(track_rect[2] / TICK_WIDTH)
print(display_space)
activities = np.zeros(shape=(8, display_space))
neuron_height = track_height / 8
n_ticks = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # logical updates 
    angle = None
    decoded = None

    mx, my = pygame.mouse.get_pos()
    mx, my = (mx-control_center[0], my-control_center[1])
    vect_norm = np.sqrt(mx**2 + my**2)
    control = vect_norm <= control_radius

    current_time = n_ticks * DT
    if control:
        unit_v = [mx/vect_norm, my/vect_norm]
        cursor_pos = (control_center[0] + int(unit_v[0] * cursor_dist), control_center[1] + int(unit_v[1] * cursor_dist))
        decoded = []
        
        angle = np.degrees(np.arccos(unit_v[0]))
        if unit_v[1] > 0:
            angle = 360 - angle

        for i, neuron in enumerate(neurons):
            current_freq = neuron.get_freq(angle)
            p_dist = neuron.poisson_dist(current_freq, DT)
            n_pa = np.random.choice([0, 1, 2, 3], p=p_dist)

            # calculate current neuron potential
            pot = neuron.V(current_time, n_pa)

            # check if PA happened
            current_tick = n_ticks % display_space
            activities[i, current_tick] = 1. if neuron.PA else 0.

            # get recent firing frequency
            current_act = list(activities[i])
            act_conc = np.array(list(current_act[len(current_act)-TICKS_CONSIDERED:]) + current_act)
            cons_region = act_conc[current_tick:current_tick+TICKS_CONSIDERED]

            # decode direction
            decoded_freq = np.sum(cons_region) / TICKS_CONSIDERED / DT
            neuron_rad = np.radians(neuron.angle)
            x, y = np.cos(neuron_rad), np.sin(neuron_rad)
            decoded_vect = [decoded_freq*x, decoded_freq*y]

            decoded.append(decoded_vect)

    else:
        cursor_pos = control_center
        for i, neuron in enumerate(neurons):
            activities[i, n_ticks % display_space] = 0.
            neuron.reset(current_time)

    screen.fill(BACKGROUND)

    # graphics
    pygame.draw.rect(screen, SHADOW, track_rect)
    pygame.draw.rect(screen, CONTOUR, track_rect, 3)
    pygame.draw.circle(screen, SHADOW, control_center, control_radius)
    pygame.draw.circle(screen, CONTOUR, control_center, control_radius, width=3)
    pygame.draw.circle(screen, CONTOUR, cursor_pos, cursor_radius)

    pygame.draw.circle(screen, SHADOW, coord_center, control_radius)
    pygame.draw.line(screen, BACKGROUND, (coord_center[0], coord_center[1]-control_radius), (coord_center[0], control_center[1]+control_radius), width=2)
    pygame.draw.line(screen, BACKGROUND, (coord_center[0]-control_radius, coord_center[1]), (coord_center[0]+control_radius, control_center[1]), width=2)

    for i in range(8):
        for j in range(display_space):
            act = activities[i, j]
            pygame.draw.rect(screen, SHADOW.lerp('white', 0.6*act), (track_margin + j*TICK_WIDTH, track_rect[1] + i*neuron_height, TICK_WIDTH, neuron_height))
    pygame.draw.rect(screen, CONTOUR, (track_margin + (n_ticks % display_space)*TICK_WIDTH, track_rect[1], TICK_WIDTH, track_rect[3]))

    if decoded is not None:
        max_norm = 0
        total_v = [0, 0]
        for v in decoded:
            total_v[0] += v[0]
            total_v[1] += v[1]

            norm = np.sqrt(v[0]**2 + v[1]**2)
            if norm > max_norm:
                max_norm = norm
        
        if max_norm != 0:
            coord_scalar = control_radius/max_norm
            for v in decoded:
                pygame.draw.line(screen, CONTOUR, coord_center, (coord_center[0]+v[0]*coord_scalar, coord_center[1]-v[1]*coord_scalar))
            pygame.draw.line(screen, 'red', coord_center, (coord_center[0]+total_v[0]*coord_scalar, coord_center[1]-total_v[1]*coord_scalar), width=3)

        

    
    pygame.display.flip()
    n_ticks += 1
    clock.tick(60) 