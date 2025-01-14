import numpy as np
cimport cython
from cython.parallel import prange
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

cdef extern from "film.h":
    ctypedef struct Film:
        int id
        int i
        int j
        int k
        double nx
        double ny
        double nz

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
react_prob_chemical = np.array([0.50, 0.50, 0.50, 0.9, 0.0])
react_redepo_sticking = np.array([1.0, 1.0, 1.0, 1.0, 1.0])


# sticking_probability 函数
def sticking_probability(double[:] parcel, 
                         int[:] film, 
                         double angle_rad):
    film_layer: cython.Py_ssize_t = film.shape[0]
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    choice = np.random.rand(film_layer)
    choice_view: cython.double[:] = choice
    rn_matrix_view: cython.double[:,::1] = rn_matrix

    sticking_acceptList = np.zeros(film_layer, dtype=np.bool_)
    cdef int particle = int(parcel[-1])
    cdef int e, energy_range
    cdef double sticking_rate
    
    if particle >= 2:
        particle = 2

    for i in range(film_layer):
        if film[i] <= 0:
            choice_view[i] = 1.0

    for j in range(film_layer):
        if particle == 1:  
            energy_range = 0
            for e in range(rn_energy.shape[0]):
                if parcel[-2] < rn_energy[e]:
                    energy_range = e
                    break
            sticking_rate = np.interp(angle_rad, rn_angle, rn_matrix_view[energy_range])
        elif particle == 0:  # 化学反应
            sticking_rate = react_prob_chemical[j]
        elif particle >= 2:  # redepo
            sticking_rate = react_redepo_sticking[particle - 2]

        if sticking_rate > choice_view[j]:
            sticking_acceptList[j] = True

    return sticking_acceptList, particle

# sticking_probability 函数
def sticking_probability_structed(Particle parcel, 
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
    

    if parcel.id >= 2:
        parcel.id = 2

    for i in range(film_layer):
        if film[i] <= 0:
            choice_view[i] = 1.0

    for j in range(film_layer):
        if parcel.id == 1:  
            energy_range = 0
            for e in range(rn_energy.shape[0]):
                if parcel.E < rn_energy[e]:
                    energy_range = e
                    break
            sticking_rate = np.interp(angle_rad, rn_angle, rn_matrix_view[energy_range])
        elif parcel.id == 0:  # 化学反应
            sticking_rate = react_prob_chemical[j]
        elif parcel.id >= 2:  # redepo
            sticking_rate = react_redepo_sticking[parcel.id - 2]

        if sticking_rate > choice_view[j]:
            sticking_acceptList[j] = True

    return sticking_acceptList, parcel.id


def film_test(Film [:,:,:] film):
    cdef int i,j,k
    cdef int xmax = film.shape[0]
    cdef int ymax = film.shape[1]
    cdef int zmax = film.shape[2]
    cdef double sum
    # film_view: Film [:,:,:] = film

    
    for i in prange(xmax, nogil=True):
        for j in range(ymax):  
            for k in range(zmax):
                sum += film[i,j,k].nx
    return sum




# # # reaction_rate_parallel_all 函数
def reaction_rate_parallel_all(Film [:,:,:] filmMatrix, 
parcel, 
Cell [:,:,:] film_label_index_normal, 
cellSizeXYZ):
    elements = filmMatrix.shape[-1]
    shape = filmMatrix.shape
    update_film_etch = np.zeros((shape[0], shape[1], shape[2]), dtype=np.bool_)
    update_film_depo = np.zeros((shape[0], shape[1], shape[2]), dtype=np.bool_)
    reactList = np.ones(parcel.shape[0], dtype=np.int_) * -1
    indice_1 = np.ones(parcel.shape[0], dtype=np.bool_)
    depo_parcel = np.zeros(parcel.shape[0])

    for i in prange(parcel.shape[0]):
        # ijk = parcel[i, 6:9].astype(np.int32)
        # ijk = np.rint(parcel[i, :3]).astype(np.int32)
        ijk = parcel[i, :3].astype(np.int32)
        if film_label_index_normal[ijk[0], ijk[1], ijk[2], 0] == 1:
            react_add = np.zeros(elements)
            film = filmMatrix[ijk[0], ijk[1], ijk[2]]
            get_theta = film_label_index_normal[ijk[0], ijk[1], ijk[2], -3:]
            # film_vaccum = filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]]
            dot_product = np.dot(parcel[i, 3:6], get_theta)
            dot_product = np.abs(dot_product)
            angle_rad = np.arccos(dot_product)

            sticking_acceptList, particle = sticking_probability_structed(parcel[i], film, angle_rad)

            react_choice_indices = np.where(sticking_acceptList)[0]
            if react_choice_indices.size > 0:
                react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
                reactList[i] = react_choice
                indice_1[i] = False
                react_type = react_type_table[particle, react_choice]

                if react_type == 1: # chemical transfer || p0 reaction type
                    depo_parcel[i] = 1
                elif react_type == 2: # physics sputter || p0 (E**2 - Eth**2) f(theta)
                    depo_parcel[i] = 2
                elif react_type == 3: # redepo
                    depo_parcel[i] = 3
                elif react_type == 4: # Themal etch || p0 reaction type
                    depo_parcel[i] = 4
                elif react_type == 0: # no reaction
                    depo_parcel[i] = 0

            if int(parcel[i, -1]) <= 1: # gas Cl Ion
                react_add = react_table_equation[int(parcel[i, -1]), react_choice, :]
            else: # redepo solid
                react_add = np.zeros(elements, dtype=np.int32)
                parcel_index = int(parcel[i, -1])-2
                react_add[int(parcel_index)] = 1

            if depo_parcel[i] == 1: # chemical transfer
                film += react_add
            if depo_parcel[i] == 4: # chemical remove
                film += react_add
                if np.sum(film) == 0:
                    # if film[i, 3] == 0:
                    update_film_etch[ijk[0], ijk[1], ijk[2]] =  True
            if depo_parcel[i] == 2: # physics sputter
                react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad, parcel[i,-2], 10) # physics sputter || p0 (E**2 - Eth**2) f(theta)
                # react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], parcel[i,-2], film_Eth[int(reactList[i])])
                if react_yield > np.random.rand():
                    film += react_add
                    if np.sum(film) == 0:
                        update_film_etch[ijk[0], ijk[1], ijk[2]] =  True
        
            # if depo_parcel[i] == 3: # depo
            #     film_add_all = np.sum(react_add + film[i, :])
            #     if film_add_all > film_density:
            #         film_vaccum[i, :] += react_add
            #         update_film_etch[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True  
            #     else:
            #         film[i, :] += react_add
            #         if np.sum(film[i, :]) == film_density:
            #             update_film_depo[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True
            filmMatrix[ijk[0], ijk[1], ijk[2]] = film
            # filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
            if reactList[i] == -1:
                # parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], get_theta[i])
                parcel[i, 3:6] = reflect.DiffusionReflect(parcel[i, 3:6], get_theta)

        parcel[i, :3] += parcel[i, 3:6]
        if indice_1[i] == True:
            if parcel[i, 0] >= cellSizeXYZ[0]:
                parcel[i, 0] -= cellSizeXYZ[0]
            elif parcel[i, 0] < 0:
                parcel[i, 0] += cellSizeXYZ[0]
            if parcel[i, 1] >= cellSizeXYZ[1]:
                parcel[i, 1] -= cellSizeXYZ[1]
            elif parcel[i, 1] < 0:
                parcel[i, 1] += cellSizeXYZ[1]
            if (parcel[i,0] > cellSizeXYZ[0] or parcel[i,0] < 0 or
                parcel[i,1] > cellSizeXYZ[1] or parcel[i,1] < 0 or
                parcel[i,2] > cellSizeXYZ[2] or parcel[i,2] < 0):
                indice_1[i] = False
        # parcel[i], indice_1[i] = boundary(parcel[i], indice_1[i], cellSizeXYZ)

    parcel = parcel[indice_1]

    return filmMatrix, parcel, update_film_etch, update_film_depo, depo_parcel


