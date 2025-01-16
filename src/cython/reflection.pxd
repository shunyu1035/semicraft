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


cdef public double[:] SpecularReflect(Particle particle, Cell cell)
cdef public double[:] DiffusionReflect(Particle particle, Cell cell)