from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt

def rand_angle():
    return np.random.random() * 360

M = 30
neurons = []
for i in range(M):
    neurons.append(Neuron(i*13))

RANDOM_ANGLE = 0 #rand_angle()
print(RANDOM_ANGLE)

TRIES = 5
MAX_T = 0.5
DT = 0.1 * 10**-3
LT = int(MAX_T/DT)
T = np.linspace(0, MAX_T, LT)

vectors = []

for n, neuron in enumerate(neurons):
    print("Neuron No.", n+1)
    spikes = np.zeros(shape=(TRIES, LT))

    # get EPSP frequency
    epsp_freq = neuron.get_freq(RANDOM_ANGLE)
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

    # vector decomposition
    avg_freq = np.average(np.sum(spikes, axis=1) / MAX_T) # norm = spike frequency
    neuron_rad = np.radians(neuron.angle)
    x, y = np.cos(neuron_rad), np.sin(neuron_rad)
    vectors.append([avg_freq*x, avg_freq*y]) # final vector = each coordinate times norm

vectors = np.array(vectors)
vector_sum = np.sum(vectors, axis=0)

plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.scatter(vectors[:,0], vectors[:,1], c='white')
plt.scatter(vector_sum[0], vector_sum[1], c='white')
for vector in vectors:
    plt.annotate("", (vector[0], vector[1]), (0, 0), arrowprops={"arrowstyle":'->', "color":'blue', "linewidth":2, "alpha":0.5})
plt.annotate("", (vector_sum[0], vector_sum[1]), (0, 0), arrowprops={"arrowstyle":'->', "color":'red', "linewidth":3})
plt.grid()
plt.show()