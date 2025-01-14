import numpy as np
cimport cython
from cython.parallel import prange
cimport numpy as cnp
from libc.math cimport fabs, acos
import absorb
cimport sputter_yield


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


def vel_normal_dot(Particle particle, Cell cell):
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


def reaction_rate_parallel_all(
    int[:,:,:,:] filmMatrix, 
    int elements,
    Particle[:] particles, 
    Cell[:,:,:] film_label_index_normal, 
    int[:] cellSizeXYZ):

    cdef int particle_count = particles.shape[0]
    cdef int i, celli, cellj, cellk
    cdef double dot_product, angle_rad
    cdef Particle[:] particle_view = particles
    cdef Cell[:,:,:] cell_view = film_label_index_normal
    cdef int[:,:,:] film_matrix_view = filmMatrix
    cdef int[:] cell_size_xyz_view = cellSizeXYZ

    # Create numpy arrays for intermediate results
    cdef cnp.ndarray[cnp.int_t, ndim=3] update_film_etch = np.zeros(
        (cell_size_xyz_view[0], cell_size_xyz_view[1], cell_size_xyz_view[2]), dtype=np.bool_
    )
    cdef cnp.ndarray[cnp.int_t, ndim=3] update_film_depo = np.zeros(
        (cell_size_xyz_view[0], cell_size_xyz_view[1], cell_size_xyz_view[2]), dtype=np.bool_
    )
    cdef cnp.ndarray[cnp.int_t, ndim=1] reactList = np.full(particle_count, -1, dtype=np.int_)
    cdef cnp.ndarray[cnp.bool_t, ndim=1] indice_1 = np.ones(particle_count, dtype=np.bool_)
    cdef cnp.ndarray[cnp.double_t, ndim=1] depo_particle = np.zeros(particle_count, dtype=np.double)

    # Loop over all particles
    for i in prange(particle_count, nogil=True):
        celli = <int> particle_view[i].px
        cellj = <int> particle_view[i].py
        cellk = <int> particle_view[i].pz

        if cell_view[celli, cellj, cellk].id == 1:
            # Retrieve the film at the current cell
            cdef int[:] film = film_matrix_view[celli, cellj, cellk]

            # Compute dot product between particle velocity and cell normal
            dot_product = (
                particle_view[i].vx * cell_view[celli, cellj, cellk].nx +
                particle_view[i].vy * cell_view[celli, cellj, cellk].ny +
                particle_view[i].vz * cell_view[celli, cellj, cellk].nz
            )
            dot_product = fabs(dot_product)
            angle_rad = acos(dot_product)

            # Calculate sticking probability
            sticking_acceptList, particle_view[i] = absorb.sticking_probability_structed(
                particle_view[i], film, angle_rad
            )

            react_choice_indices = np.where(sticking_acceptList)[0]
            if react_choice_indices.size > 0:
                # Randomly select a reaction type
                react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
                reactList[i] = react_choice
                indice_1[i] = False

                # Determine reaction type and update particle or film
                react_type = absorb.react_type_table[particle_view[i], react_choice]

                if react_type == 1:  # Chemical transfer
                    depo_particle[i] = 1
                elif react_type == 2:  # Physical sputtering
                    depo_particle[i] = 2
                elif react_type == 3:  # Redeposition
                    depo_particle[i] = 3
                elif react_type == 4:  # Thermal etching
                    depo_particle[i] = 4
                elif react_type == 0:  # No reaction
                    depo_particle[i] = 0

                # Update film based on reaction type
                if int(particle_view[i].id) <= 1:  # Gas Cl Ion
                    react_add = absorb.react_table_equation[int(particle_view[i].id), react_choice, :]
                else:  # Redeposited solid
                    react_add = np.zeros(elements, dtype=np.int32)
                    particle_index = int(particle_view[i].id) - 2
                    react_add[int(particle_index)] = 1

                if depo_particle[i] == 1 or depo_particle[i] == 4:  # Chemical transfer or remove
                    film += react_add
                    if np.sum(film) == 0:
                        update_film_etch[celli, cellj, cellk] = True
                elif depo_particle[i] == 2:  # Physical sputtering
                    react_yield = sputter_yield.sputter_yield(
                        absorb.react_yield_p0[0], angle_rad, particle_view[i].vz, 10
                    )
                    if react_yield > np.random.rand():
                        film += react_add
                        if np.sum(film) == 0:
                            update_film_etch[celli, cellj, cellk] = True

            # Save film back to memory
            film_matrix_view[celli, cellj, cellk] = film
            
            # filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
            if reactList[i] == -1:
                # particle[i, 3:6] = reflect.SpecularReflect(particle[i, 3:6], get_theta[i])
                particle[i, 3:6] = reflect.DiffusionReflect(particle[i, 3:6], get_theta)

        particle[i, :3] += particle[i, 3:6]
        if indice_1[i] == True:
            if particle[i, 0] >= cellSizeXYZ[0]:
                particle[i, 0] -= cellSizeXYZ[0]
            elif particle[i, 0] < 0:
                particle[i, 0] += cellSizeXYZ[0]
            if particle[i, 1] >= cellSizeXYZ[1]:
                particle[i, 1] -= cellSizeXYZ[1]
            elif particle[i, 1] < 0:
                particle[i, 1] += cellSizeXYZ[1]
            if (particle[i,0] > cellSizeXYZ[0] or particle[i,0] < 0 or
                particle[i,1] > cellSizeXYZ[1] or particle[i,1] < 0 or
                particle[i,2] > cellSizeXYZ[2] or particle[i,2] < 0):
                indice_1[i] = False
        # particle[i], indice_1[i] = boundary(particle[i], indice_1[i], cellSizeXYZ)

    particle = particle[indice_1]

    return filmMatrix, particle, update_film_etch, update_film_depo, depo_particle



# # # reaction_rate_parallel_all 函数
# def reaction_rate_parallel_all( int [:,:,:,:] filmMatrix, 
#                                 int elements,
#                                 Particle [:] particle, 
#                                 Cell [:,:,:] film_label_index_normal, 
#                                 int [:] cellSizeXYZ):

#     cdef int particle_count = particle.shape[0]
#     particle_view: Particle[:] = particle
#     i: cython.Py_ssize_t
#     celli: cython.int
#     cellj: cython.int
#     cellk: cython.int
#     get_theta: double[3]
#     film: int[elements]
#     react_add: int[elements]
#     update_film_etch = np.zeros((cellSizeXYZ[0], cellSizeXYZ[1], cellSizeXYZ[2]), dtype=np.bool_)
#     update_film_depo = np.zeros((cellSizeXYZ[0], cellSizeXYZ[1], cellSizeXYZ[2]), dtype=np.bool_)
#     reactList = np.ones(particle.shape[0], dtype=np.int_) * -1
#     indice_1 = np.ones(particle.shape[0], dtype=np.bool_)
#     depo_particle = np.zeros(particle.shape[0])

#     for i in prange(particle_count):
#         celli = particle_view[i].px.astype(cython.int)
#         cellj = particle_view[i].py.astype(cython.int)
#         cellk = particle_view[i].pz.astype(cython.int)
#         if film_label_index_normal[celli, cellj, cellk].id == 1:
#             film = filmMatrix[celli, cellj, cellk]
#             dot_product = vel_normal_dot(particle[i], film_label_index_normal[celli, cellj, cellk])
#             dot_product = np.abs(dot_product)
#             angle_rad = np.arccos(dot_product)

#             sticking_acceptList, particle = sticking_probability_structed(particle[i], film, angle_rad)

#             react_choice_indices = np.where(sticking_acceptList)[0]
#             if react_choice_indices.size > 0:
#                 react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
#                 reactList[i] = react_choice
#                 indice_1[i] = False
#                 react_type = react_type_table[particle, react_choice]

#                 if react_type == 1: # chemical transfer || p0 reaction type
#                     depo_particle[i] = 1
#                 elif react_type == 2: # physics sputter || p0 (E**2 - Eth**2) f(theta)
#                     depo_particle[i] = 2
#                 elif react_type == 3: # redepo
#                     depo_particle[i] = 3
#                 elif react_type == 4: # Themal etch || p0 reaction type
#                     depo_particle[i] = 4
#                 elif react_type == 0: # no reaction
#                     depo_particle[i] = 0

#             if int(particle[i, -1]) <= 1: # gas Cl Ion
#                 react_add = react_table_equation[int(particle[i, -1]), react_choice, :]
#             else: # redepo solid
#                 react_add = np.zeros(elements, dtype=np.int32)
#                 particle_index = int(particle[i, -1])-2
#                 react_add[int(particle_index)] = 1

#             if depo_particle[i] == 1: # chemical transfer
#                 film += react_add
#             if depo_particle[i] == 4: # chemical remove
#                 film += react_add
#                 if np.sum(film) == 0:
#                     # if film[i, 3] == 0:
#                     update_film_etch[ijk[0], ijk[1], ijk[2]] =  True
#             if depo_particle[i] == 2: # physics sputter
#                 react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad, particle[i,-2], 10) # physics sputter || p0 (E**2 - Eth**2) f(theta)
#                 # react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], particle[i,-2], film_Eth[int(reactList[i])])
#                 if react_yield > np.random.rand():
#                     film += react_add
#                     if np.sum(film) == 0:
#                         update_film_etch[ijk[0], ijk[1], ijk[2]] =  True
        
#             # if depo_particle[i] == 3: # depo
#             #     film_add_all = np.sum(react_add + film[i, :])
#             #     if film_add_all > film_density:
#             #         film_vaccum[i, :] += react_add
#             #         update_film_etch[int(particle[i, 6]), int(particle[i, 7]), int(particle[i, 8])] = True  
#             #     else:
#             #         film[i, :] += react_add
#             #         if np.sum(film[i, :]) == film_density:
#             #             update_film_depo[int(particle[i, 6]), int(particle[i, 7]), int(particle[i, 8])] = True
#             filmMatrix[ijk[0], ijk[1], ijk[2]] = film
#             # filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
#             if reactList[i] == -1:
#                 # particle[i, 3:6] = reflect.SpecularReflect(particle[i, 3:6], get_theta[i])
#                 particle[i, 3:6] = reflect.DiffusionReflect(particle[i, 3:6], get_theta)

#         particle[i, :3] += particle[i, 3:6]
#         if indice_1[i] == True:
#             if particle[i, 0] >= cellSizeXYZ[0]:
#                 particle[i, 0] -= cellSizeXYZ[0]
#             elif particle[i, 0] < 0:
#                 particle[i, 0] += cellSizeXYZ[0]
#             if particle[i, 1] >= cellSizeXYZ[1]:
#                 particle[i, 1] -= cellSizeXYZ[1]
#             elif particle[i, 1] < 0:
#                 particle[i, 1] += cellSizeXYZ[1]
#             if (particle[i,0] > cellSizeXYZ[0] or particle[i,0] < 0 or
#                 particle[i,1] > cellSizeXYZ[1] or particle[i,1] < 0 or
#                 particle[i,2] > cellSizeXYZ[2] or particle[i,2] < 0):
#                 indice_1[i] = False
#         # particle[i], indice_1[i] = boundary(particle[i], indice_1[i], cellSizeXYZ)

#     particle = particle[indice_1]

#     return filmMatrix, particle, update_film_etch, update_film_depo, depo_particle


