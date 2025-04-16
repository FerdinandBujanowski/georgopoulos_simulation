import math
import matplotlib.pyplot as plt
import numpy as np

# constants
TAU = 10 * (10**-3) # 10 ms
RM = 4 * (10**6) # 4 MOhm
I_AMP = 0.1 * (10**-6) # 0.1 µA

# neuron class
class Neuron():

    def __init__(self, angle, threshold=20*(10**-3), max_freq=100):
        self.angle = angle
        self.v0 = 0
        self.offset_t = 0
        self.threshold = threshold
        self.max_freq = max_freq
        self.PA = False

    def V(self, t, n_pa):
        a = -1/TAU
        b = n_pa * I_AMP * RM / TAU

        current_v = -b/a + ((self.v0 + (b/a)) * math.exp(a*(t-self.offset_t)))
        if n_pa > 0:
            self.v0 = current_v
            self.offset_t = t
        if current_v >= self.threshold:
            self.PA = True
            self.v0 = 0
            self.offset_t = t
        else:
            self.PA = False

        return current_v
    
    def poisson_dist(self, freq, dt):
        lam = freq * dt
        dist = []
        for k in range(3):
            dist.append(((lam**k)/math.factorial(k))*math.exp(-lam))
        dist.append(1-np.sum(dist))
        return dist
    
    def get_freq(self, angle):
        return self.max_freq * (7/12 + (5/12 * math.cos(np.radians(angle - self.angle))))
    
    def reset(self):
        self.offset_t = 0
        self.v0 = 0