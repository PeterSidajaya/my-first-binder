import matplotlib.pyplot as plt
from numpy import ndarray
from functions import random_vector
from probabilities import *
from protocols import *

def generate_protocol(protocol, lhv):
    data = np.ndarray((3, 10000))
    for i in range(10000):
        vector = random_vector(3)
        data[0,i] , data[1,i] = spherical(vector)
        res = protocol(vector, lhv)
        if isinstance(res, ndarray):
            data[2,i] = res[0]
        else:
            data[2,i] = res
    return data
        
        
def plot_protocol(protocol, lhv):
    data = generate_protocol(protocol, lhv)
    plt.scatter(data[0], data[1], c=data[2], vmin=0, vmax=1)
    
    phi, theta = spherical(lhv[0])
    plt.scatter(phi, theta, c='red')
    
    phi, theta = spherical(lhv[1])
    plt.scatter(phi, theta, c='blue')
    
    plt.show()

def spherical(vector):
    return (np.arctan2(vector[1],vector[0]), np.arccos(vector[2]))

# print(nme_predict(np.pi/4))
# plot_protocol(comm_protocol, (np.array([1,0,0]), np.array([0,0,1])))