import numpy as np


def Rn_coeffcient(c1, c2, c3, c4, alpha):
    return c1 + c2*np.tanh(c3*alpha - c4)

def Rn_probability(c_list = [0.9423, 0.9434, 2.342, 3.026]):
    rn_angle = np.arange(0, np.pi/2, 0.01)
    # xnew = np.array([])
    rn_prob = [Rn_coeffcient(c_list[0], c_list[1], c_list[2], c_list[3], i) for i in rn_angle]
    rn_prob /= rn_prob[-1]
    rn_prob = 1-rn_prob
    return rn_prob