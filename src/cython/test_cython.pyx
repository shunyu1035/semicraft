import numpy as np
cimport cython
from cython.parallel import prange
cimport numpy as cnp
from libc.math cimport acos
from libc.math cimport fabs


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



# 计算法线点积
cdef double vel_normal_dot(Particle particle, Cell cell):
    return (particle.vx * cell.nx +
            particle.vy * cell.ny +
            particle.vz * cell.nz)

# 主函数：计算角度
def compute_particle_angle(Particle[:] particles, Cell[:] cells):
    cdef int i, j
    cdef double dot_product, angle_rad
    for i in range(particles.shape[0]):
        for j in range(cells.shape[0]):
            # 计算点积
            dot_product = vel_normal_dot(particles[i], cells[j])
            # 取绝对值
            dot_product = fabs(dot_product)
            # 计算角度（弧度）
            angle_rad = acos(dot_product)
            print(f"Particle {i}, Cell {j}, Angle (rad): {angle_rad}")



def test_npdot(Particle particle, Cell cell):
    cdef double dot_product  # 明确声明变量类型
    cdef double[3] vel
    cdef double[3] theta

    # 初始化 vel 数组
    vel[0] = particle.vx
    vel[1] = particle.vy
    vel[2] = particle.vz

    # 初始化 theta 数组
    theta[0] = cell.nx
    theta[1] = cell.ny
    theta[2] = cell.nz

    # 计算点积
    dot_product = vel[0] * theta[0] + vel[1] * theta[1] + vel[2] * theta[2]

    return dot_product