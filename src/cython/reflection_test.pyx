cimport reflection
import cython
import numpy as np
cimport numpy as cnp

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



def specular(Particle particle, Cell cell):
    # vel = np.zeros(3, dtype=np.float64)
    reflection.SpecularReflect(particle, cell)
    # print('veloscity',vel)
    # return vel

def DiffusionReflect(Particle particle, Cell cell):

    reflection.DiffusionReflect(particle, cell)
