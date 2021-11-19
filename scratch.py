import numpy as np
from numpy.random.mtrand import random
from evo import Evolution, MP
from functions import random_vector
from protocols import *
from plotter import plot_protocol
import matplotlib.pyplot as plt

def mp_fitness_creator(lhv_size=20, input_size=200):
    eps = np.finfo(np.float64).eps

    def fitness(M):
        lhvs = []
        for i in range(lhv_size):
            lhvs.append((random_vector(3),random_vector(3)))
    
        inputs = []
        for i in range(input_size):
            inputs.append(random_vector(3))
        
        res = 0
        for j in range(lhv_size):
            lhv = lhvs[j]
            for i in range(input_size):
                input = inputs[i]
                
                prob = comm_protocol(input, lhv)
                true = [prob, 1-prob]
                prob = 1/2 + 1/4 * np.sign(mdot(input, lhv[0], M.value[0])) * np.sign(mdot(input, lhv[1], M.value[0])) \
                        + 1/4 * np.sign(mdot(input, lhv[0], M.value[1])) * np.sign(mdot(input, lhv[1], M.value[1]))
                pred = [prob, 1-prob]
                
                pred = np.clip(pred, eps, 1)
                true = np.clip(true, eps, 1)
                res += np.sum(true*np.log(true/pred))/(lhv_size*input_size)
        return -res

    return fitness

fitness = mp_fitness_creator()

evo = Evolution(
    pool_size=120, fitness=fitness, individual_class=MP, n_offsprings=60,
    pair_params={'alpha': 0.5},
    mutate_params={'std': 0.5, 'dim': 2},
    init_params={'std': 1, 'dim': 2},
    param_adjustment=lambda mutate_params, epoch: {'std': mutate_params['std'] * 200/(200+epoch), 'dim': mutate_params['dim']} 
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