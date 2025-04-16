import math
import matplotlib.pyplot as plt
import numpy as np

# constants
TAU = 10 * (10**-3) # 10 ms
RM = 4 * (10**6) # 4 MOhm
I_AMP = 0.1 * (10**-6) # 0.1 µA
DT = 0.1 * 10**-3

print("Tau:", TAU)
print("Rm", RM)
print("Iamp:", I_AMP)

# neuron class
class Neuron():

    def __init__(self, angle, threshold=20*(10**-3), max_freq=100):
        self.angle = angle
        self.v0 = 0
        self.offset_t = 0
        self.threshold = threshold
        self.max_freq = max_freq
        self.PA = False
        print("Seuil:", self.threshold)

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
    

# PARTIE I
ANGLE = 180
neuron = Neuron(ANGLE)

MAX_T = 0.5
TRIES = 5
DIRECTIONS = 8
ANGLES = range(0, 360, int(360/DIRECTIONS))
LT = int(MAX_T/DT)
T = np.linspace(0, MAX_T, LT)

exp_data = {}

# loop over angles
for angle in ANGLES:
    spikes = np.zeros(shape=(TRIES, LT))

    # get EPSP frequency
    epsp_freq = neuron.get_freq(angle)
    # get Poisson distribution
    p_dist = neuron.poisson_dist(epsp_freq, DT)

    # loop over tries (5 tries per angle)
    for n_try in range(TRIES):

        # reset neuron's V0 and t_offset
        neuron.reset()

        # loop over time samples
        for i, t in enumerate(T):
            
            # get number of incoming EPSPs
            n_pa = np.random.choice([0, 1, 2, 3], p=p_dist)

            # calculate current neuron potential
            pot = neuron.V(t, n_pa)

            # check if PA happened
            if neuron.PA:
                spikes[n_try, i] = 1.
                if i > 0:
                    spikes[n_try, i-1] = 1.
                if i < len(T)-1:
                    spikes[n_try, i+1] = 1.

    # adding data to dict for later analysis
    exp_data[angle] = spikes

    # plot
    plt.imshow(spikes,aspect='auto', cmap='Greys', interpolation='nearest')
    og_xticks = plt.xticks()[0]
    plt.xticks(og_xticks, [str(tick*10**-4) for tick in og_xticks])
    plt.xlim(0, LT)
    plt.xlabel('Time (s)')
    plt.ylabel('Tries')
    plt.title(f'Spikes over time for angle = {angle} degrees')
    plt.savefig(f'./plots/spikes_{angle}.png')

# courbe d'accord
averages = []
std_devs = []

for angle in ANGLES:
    data = exp_data[angle]
    freqs = np.sum(data, axis=1) / MAX_T
    averages.append(np.average(freqs))
    std_devs.append(np.std(freqs))

plt.clf()
plt.errorbar(ANGLES, averages, std_devs)
plt.xlabel('Angle (degrees)')
plt.ylabel('Spike rate (Hz)')
plt.title(f'Average spike rate for {ANGLE}°-neuron')
plt.show()