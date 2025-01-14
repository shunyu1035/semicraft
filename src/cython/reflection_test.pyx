cimport reflection
import cython
import numpy as np
cimport numpy as cnp

cdef extern from "particle.h":
    ctypedef struct Particle:
        double px
        double py
        double pz
        double vx
        double vy
        double vz
        double E
        long long id

cdef extern from "film.h":
    ctypedef struct Cell:
        int id
        int i
        int j
        int k
        double nx
        double ny
        double nz



def specular(Particle particle, Cell cell):
    # vel = np.zeros(3, dtype=np.float64)
    reflection.SpecularReflect(particle, cell)
    # print('veloscity',vel)
    # return vel
