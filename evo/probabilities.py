from protocols import *

def probability_single_round(vec_alice, vec_bob, lhv):
    """Calculate the probability distribution from one round of LHV."""
    bit = comm_protocol(vec_alice, lhv)
    return bit * outer_product(alice_protocol_1(vec_alice, lhv), bob_protocol_1(vec_bob, lhv)) + \
        (1 - bit) * outer_product(alice_protocol_2(vec_alice, lhv), bob_protocol_2(vec_bob, lhv))

def probability(vec_alice, vec_bob, lhv_type="double-vector", n=2500, verbose=False):
    """Calculate the probability distribution from multiple rounds of LHV."""
    prob = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(n):
        if lhv_type == "single-vector":
            lhv = random_vector(3)
        elif lhv_type == "double-vector":
            lhv = (random_vector(3), random_vector(3))
        prob += probability_single_round(vec_alice, vec_bob, lhv)
        if verbose:
            print(prob)
    return prob / n

def nme_probability(vec_alice, vec_bob, theta):
    a = [+1, -1]
    ax, ay, az = vec_alice
    bx, by, bz = vec_bob
    lst = []
    for A in a:
        for B in a:
            lst.append(1/4*(1 + A*np.cos(2*theta)*az - B*np.cos(2*theta)*bz + A*B*(-az*bz-np.sin(2*theta)*(ax*bx+ay*by))))
    return np.array(lst)

def nme_predict(theta):
    diff = 0
    eps = np.finfo(np.float64).eps
    for _ in range(100):
        vec_alice, vec_bob = random_vector(3), random_vector(3)
        pred = probability(vec_alice, vec_bob)
        true = nme_probability(vec_alice, vec_bob, theta)
        pred = np.clip(pred, eps, 1)
        true = np.clip(true, eps, 1)
        diff += np.sum(true*np.log(true/pred))/100
    return diff
