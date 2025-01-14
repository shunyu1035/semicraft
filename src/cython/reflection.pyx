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


# def SpecularReflect(Particle particle, Cell cell):
#     cdef double dot_product_2 = 0
#     cdef double[3] refect_vel = np.zeros(3, dtype=np.float64)
#     dot_product_2 = (
#         particle.vx * cell.nx +
#         particle.vy * cell.ny +
#         particle.vz * cell.nz
#     )*2

#     refect_vel[0] = particle.vx - dot_product_2 * cell.nx
#     refect_vel[1] = particle.vy - dot_product_2 * cell.ny
#     refect_vel[2] = particle.vz - dot_product_2 * cell.nz
#     return refect_vel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[:] SpecularReflect(Particle particle, Cell cell):
    # 声明返回的反射速度数组
    cdef double[3] refect_vel
    # 计算速度和法向量的点积
    cdef double dot_product_2 = 2 * (particle.vx * cell.nx + particle.vy * cell.ny + particle.vz * cell.nz)
    
    # 计算反射速度
    refect_vel[0] = particle.vx - dot_product_2 * cell.nx
    refect_vel[1] = particle.vy - dot_product_2 * cell.ny
    refect_vel[2] = particle.vz - dot_product_2 * cell.nz
    # print('velosity______1', refect_vel)
    return refect_vel



# def DiffusionReflect(double[:] vel, double[:] normal):
#     cdef double[:] Ut = np.zeros(3, dtype=np.float64)
#     cdef double[:] tw1 = np.zeros(3, dtype=np.float64)
#     cdef double[:] tw2 = np.zeros(3, dtype=np.float64)
#     cdef double pm
#     cdef double[:] U = np.zeros(3, dtype=np.float64)
#     cdef double[:] UN = np.zeros(3, dtype=np.float64)
#     cdef double vel_dot_normal

#     # Compute Ut = vel - vel @ normal * normal
#     vel_dot_normal = np.dot(vel, normal)
#     for i in range(3):
#         Ut[i] = vel[i] - vel_dot_normal * normal[i]

#     # Normalize Ut to get tw1
#     tw1 = Ut / np.linalg.norm(Ut)

#     # Compute tw2 as the cross product of tw1 and normal
#     tw2 = np.cross(tw1, normal)

#     # Determine if vel is in the opposite direction of normal
#     if np.dot(vel, normal) > 0:
#         pm = -1
#     else:
#         pm = 1

#     # Generate U based on random Gaussian distributions
#     for i in range(3):
#         U[i] = np.random.randn() * tw1[i] + np.random.randn() * tw2[i] + pm * np.sqrt(-2 * np.log(1 - np.random.rand())) * normal[i]

#     # Normalize U to get UN
#     UN = U / np.linalg.norm(U)

#     return UN

