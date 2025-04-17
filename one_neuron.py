import matplotlib.pyplot as plt
import numpy as np
from neuron import Neuron

ANGLE = 180
neuron = Neuron(ANGLE)


DIRECTIONS = 8
ANGLES = range(0, 360, int(360/DIRECTIONS))
TRIES = 5
MAX_T = 0.5
DT = 0.1 * 10**-3
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
                spikes[n_try, i] = 1

    # adding data to dict for later analysis
    exp_data[angle] = spikes

    # plot
    # TODO thicken bars
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