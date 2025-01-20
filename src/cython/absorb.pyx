import numpy as np
cimport cython
from cython.parallel import prange
cimport numpy as cnp
from libc.math cimport fabs, acos

react_table_equation = np.array([[[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, 0]],
                                 [[-1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0,-1]]], dtype=np.int32)

react_type_table = np.array([[1, 1, 1, 4, 0], # 1: chlorination  # 0: no reaction  # 4: Themal etch
                             [2, 2, 2, 2, 2], # 2 for physics and chemical sputtering
                             [3, 3, 3, 3, 3]]) # 3 for redepo

react_prob_chemical = np.array([0.50, 0.50, 0.50, 0.9, 0.0])
# react_prob_chemical = np.array([0.90, 0.90, 0.90, 0.9, 0.0])

# react_yield_p0 = np.array([0.10, 0.10, 0.10, 0.10, 0.10])
react_yield_p0 = np.array([0.30, 0.30, 0.30, 0.30, 0.30])

cdef extern from "particle.h":
    ctypedef struct Particle:
        double[3] pos
        double[3] vel
        double E
        long long id

cdef extern from "film.h":
    ctypedef struct Cell:
        int id
        int[3] index
        double[3] normal

def Rn_coeffcient(double c1, 
                  double c2, 
                  double c3, 
                  double c4, 
                  double alpha):
    cdef double rn_prob
    rn_prob = c1 + c2*np.tanh(c3*alpha - c4)
    return rn_prob

def Rn_probability(double[:] c_list):

    rn_angle = np.arange(0, np.pi/2, 0.01, dtype=np.double)
    rn_angle_view: cython.double[:] = rn_angle

    i_max: cython.Py_ssize_t = rn_angle.shape[0]
    i: cython.Py_ssize_t

    rn_prob = np.zeros(i_max, dtype=np.double)
    rn_prob_view: cython.double[:] = rn_prob

    for i in range(i_max):
        rn_prob_view[i] = Rn_coeffcient(c_list[0], c_list[1], c_list[2], c_list[3], rn_angle_view[i])
    # rn_prob = [Rn_coeffcient(c_list[0], c_list[1], c_list[2], c_list[3], i) for i in rn_angle]
    for i in range(i_max):
        rn_prob_view[i] /= rn_prob_view[i_max-1]
    for i in range(i_max):
        rn_prob_view[i] = 1-rn_prob_view[i]
    return rn_prob

# cdef cnp.ndarray[cnp.float64_t, ndim=2] rn_coeffcients 

# cdef cnp.ndarray[cnp.float64_t, ndim=2] rn_matrix 

rn_coeffcients = np.array([[0.9423, 0.9434, 2.742, 3.026], # 100
                 [0.9620, 0.9608, 2.542, 3.720],  # 1000
                 [0.9458, 0.9445, 2.551, 3.735],  # 1050
                 [1.046, 1.046, 2.686, 4.301]])     # 10000

rn_matrix = np.array([Rn_probability(i) for i in rn_coeffcients])
rn_energy = np.array([100, 1000, 1050, 10000])
rn_angle = np.arange(0, np.pi/2, 0.01)
theta_angle = np.arange(0, np.pi/2, 0.01)
react_redepo_sticking = np.array([1.0, 1.0, 1.0, 1.0, 1.0])


def sticking_probability_structed(Particle particle, 
                                    int[:] film, 
                                    double angle_rad):
    film_layer: cython.Py_ssize_t = film.shape[0]
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    choice = np.random.rand(film_layer)
    choice_view: cython.double[:] = choice
    rn_matrix_view: cython.double[:,::1] = rn_matrix

    sticking_acceptList = np.zeros(film_layer, dtype=np.bool_)

    cdef int e, energy_range
    cdef double sticking_rate
    

    if particle.id >= 2:
        particle.id = 2

    for i in range(film_layer):
        if film[i] <= 0:
            choice_view[i] = 1.0

    for j in range(film_layer):
        if particle.id == 1:  
            energy_range = 0
            for e in range(rn_energy.shape[0]):
                if particle.E < rn_energy[e]:
                    energy_range = e
                    break
            sticking_rate = np.interp(angle_rad, rn_angle, rn_matrix_view[energy_range])
        elif particle.id == 0:  # 化学反应
            sticking_rate = react_prob_chemical[j]
        elif particle.id >= 2:  # redepo
            sticking_rate = react_redepo_sticking[particle.id - 2]

        if sticking_rate > choice_view[j]:
            sticking_acceptList[j] = True

    return sticking_acceptList, particle.id