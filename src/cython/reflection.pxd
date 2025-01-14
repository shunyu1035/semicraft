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


cdef public double[:] SpecularReflect(Particle particle, Cell cell)