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
def cython_cross(double[:] a, double[:] b):
    """
    Compute the cross product of two 3D vectors using Cython.
    
    Parameters:
        a (double[:]): The first vector (1D array with 3 elements).
        b (double[:]): The second vector (1D array with 3 elements).

    Returns:
        np.ndarray: The cross product of a and b as a 1D numpy array.
    """
    cdef double[3] result
    cdef int i

    # Cross product formula
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]

    # Convert to numpy array
    return np.array([result[0], result[1], result[2]], dtype=np.float64)

cdef dot3(double[3] a, double[3] b):
    cdef double result
    cdef int i
    for i in range(3):
        result += a[i]*b[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[:] SpecularReflect(Particle particle, Cell cell):
    # 声明返回的反射速度数组
    cdef double[3] refect_vel
    # 计算速度和法向量的点积
    cdef double dot_product_2 = dot3(particle.vel, cell.normal)*2
    cdef int i
    # 计算反射速度
    for i in range(3):
        refect_vel[i] = particle.vel[i] - dot_product_2 * cell.normal[i]
    # print('velosity______1', refect_vel)
    return refect_vel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[:] DiffusionReflect(Particle particle, Cell cell):

    cdef double[3] Ut
    cdef double[3] UN
    cdef double[3] U
    cdef double[3] tw1
    cdef double[3] tw2
    cdef double pm
    cdef int i
    cdef int j
    cdef double vel_dot_normal = dot3(particle.vel, cell.normal)*2
    random1 = np.random.randn()
    random2 = np.random.randn()
    random3 = np.random.randn()

    # Compute Ut = vel - vel @ normal * normal
    for i in range(3):
        Ut[i] = particle.vel[i] - vel_dot_normal * cell.normal[i]

    # Normalize Ut to get tw1
    tw1 = Ut / np.linalg.norm(Ut)

    # Compute tw2 as the cross product of tw1 and normal
    # tw2 = np.cross(tw1, normal)
    tw2[0] = tw1[1] * cell.normal[2] - tw1[2] * cell.normal[1]
    tw2[1] = tw1[2] * cell.normal[0] - tw1[0] * cell.normal[2]
    tw2[2] = tw1[0] * cell.normal[1] - tw1[1] * cell.normal[0]
    # Determine if vel is in the opposite direction of normal

    # print('tw2:Diffuse', tw2)
    if vel_dot_normal > 0:
        pm = -1
    else:
        pm = 1

    # Generate U based on random Gaussian distributions
    for j in range(3):
        U[j] = random1 * tw1[j] + random2 * tw2[j] + pm * np.sqrt(-2 * np.log(1 - random3)) * cell.normal[j]
    # Normalize U to get UN
    # print('U:Diffuse', U)
    UN = U / np.linalg.norm(U)
    # print('UN:Diffuse', UN)
    return UN

