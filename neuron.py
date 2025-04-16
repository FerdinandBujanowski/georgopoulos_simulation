import math
import matplotlib.pyplot as plt
import numpy as np

# constants
TAU = 10 * (10**-3) # 10 ms
RM = 4 * (10**6) # 4 MOhm
I_AMP = 0.1 * (10**-6) # 0.1 µA

print("Tau:", TAU)
print("Rm", RM)
print("Iamp:", I_AMP)

# neuron class
class Neuron():

    def __init__(self, angle, threshold=20*(10**-3)):
        self.angle = angle
        self.v0 = 0
        self.offset_t = 0
        self.threshold = threshold
        print("Seuil:", self.threshold)

    def V(self, t, n_pa):
        a = -1/TAU
        b = n_pa * I_AMP * RM / TAU

        current_v = -b/a + ((self.v0 + (b/a)) * math.exp(a*(t-self.offset_t)))
        if n_pa > 0:
            self.v0 = current_v
            self.offset_t = t
        if current_v >= self.threshold:
            self.v0 = 0
            self.offset_t = t

        return current_v
    
    def poisson_dist(self, freq, dt):
        lam = freq * dt
        dist = []
        for k in range(4):
            dist.append(((lam**k)/math.factorial(k))*math.exp(-lam))
        dist.append(1-np.sum(dist))
        return dist
    
# simulating for constant incoming PAs

test_neuron = Neuron(180)
T = np.linspace(0, 0.5, 5000)
freq = 10
dt = 0.1 * 10**-3
dist = test_neuron.poisson_dist(freq, dt)
n_ap = 0
activities = []
for i, t in enumerate(T):
    # ap = 1 if i % (1000/freq) == 0 else 0
    ap = np.random.choice(list(range(5)), p=dist)
    if ap > 0:
        n_ap += ap

    v = test_neuron.V(t, ap)
    activities.append(v)

print(n_ap)

plt.plot(T, np.array(activities)*1000)
plt.xlabel("Time (s)")
plt.ylabel("mV")
plt.show()