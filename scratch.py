import numpy as np
import matplotlib.pyplot as plt
from evo import Evolution, MP
from protocols import *
from plotter import plot_protocol
from fitness import mp_fitness_creator, reference_mp_fitness_creator

fitness = mp_fitness_creator(lhv_size=15, input_size=150)

evo = Evolution(
    pool_size=120, fitness=fitness, individual_class=MP, n_offsprings=60,
    pair_params={'alpha': 0.5},
    mutate_params={'std': 0.5, 'dim': 2},
    init_params={'std': 1, 'dim': 2},
    param_adjustment=lambda mutate_params, epoch: {'std': mutate_params['std'] * 150/(150+epoch), 'dim': mutate_params['dim']} 
)
n_epochs = 50

fitnesses = []
for i in range(n_epochs):
    evo.step()
    print('mutation :', evo.mutate_params['std'])
    print(evo.pool.individuals[-1].value)
    score = evo.pool.fitness(evo.pool.individuals[-1])
    print('score :', score)
    fitnesses.append(score)


Global_M = evo.pool.individuals[-1].value
    
def custom_protocol(vec_alice, lhv):
    prob = 1/2 + 1/2 * np.sign(mdot(vec_alice, lhv[0], Global_M[0])) * np.sign(mdot(vec_alice, lhv[1], Global_M[1]))
    return prob

print(Global_M)
plot_protocol(custom_protocol, (np.array([1,0,0]), np.array([0,0,1])))
plot_protocol(comm_protocol, (np.array([1,0,0]), np.array([0,0,1])))
plt.plot(fitnesses)
plt.show()