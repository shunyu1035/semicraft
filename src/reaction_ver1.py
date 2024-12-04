from numba import jit, prange
import numpy as np
import src.reflection as reflect
import src.Rn_coeffcient as Rn_coeffcient
from src.config import react_table, react_type_table

# react_table = np.array([[[0.1, -1, 0, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

# react_table = np.array([[[1.0, -1, 0, 0], [0.0, 0,  0, 0], [1.0, 0, 0, 0]],
#                         [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

# react_type_table = np.array([[2, 0, 0],
#                            [1, 0, 0],
#                            [4, 3, 1]])

rn_angle = np.arange(0, np.pi/2, 0.01)
# rn_prob = Rn_coeffcient.Rn_probability()

rn_coeffcients = [[0.9423, 0.9434, 2.342, 3.026],
                  [0.9423, 0.9434, 2.342, 3.026],
                  [0.9423, 0.9434, 2.342, 3.026]]
# each array correspond to an element
rn_matrix = np.array([Rn_coeffcient.Rn_probability(i) for i in rn_coeffcients])

#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, SiO SiO2, SiOF, SiOF2, SiO2F, SiO2F2]
#react_t g[F, O, ion] s  [1,          2,           3,          4,       5 ,   6,    7,    8,   9,  10]
#react_t g[F, O, ion] s  [Si,       SiF1,       SiF2,       SiF3,      SiO, SiO2, SiOF, SiOF2, SiO2F,SiO2F2]

# F, O, ion, redepo[>3]
# maxwell, maxwell，updown
# chemical. chemical, sputter
# surface_react_type: 0 for chemical, 1 for physics
surface_react_type = np.array([0, 0, 1])

variale_react_type = np.array([2])



@jit(nopython=True, parallel=True)
def reaction_rate(parcel, film, normal):
    num_parcels = parcel.shape[0]
    num_reactions = react_table.shape[1]
    choice = np.random.rand(num_parcels, num_reactions)
    reactList = np.ones(num_parcels, dtype=np.int_) * -1

    # 手动循环替代布尔索引
    for i in prange(film.shape[0]):
        for j in prange(film.shape[1]):
            if film[i, j] <= 0:
                choice[i, j] = 1

    depo_parcel = np.zeros(parcel.shape[0])

    for i in prange(parcel.shape[0]):
        particle = int(parcel[i, -1])

        # particle > 3 for redepo
        if particle >= 3:
            particle = 3
        acceptList = np.zeros(num_reactions, dtype=np.bool_)
        for j in prange(film.shape[1]):
            if particle in variale_react_type:
                dot_product = np.dot(parcel[i, 3:6], normal[i])
                dot_product = np.abs(dot_product)
                angle_rad = np.arccos(dot_product)
                rn_prob = rn_matrix[np.argwhere(variale_react_type == particle)[0][0]]
                react_rate = np.interp(angle_rad, rn_angle, rn_prob)
            else:
                react_rate = react_table[particle, j, 0]
            # react_rate = react_table[particle, j, 0]
            if react_rate > choice[i, j]:
                acceptList[j] = True

        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
            react_type = react_type_table[particle, react_choice]

            if react_type == 1: # physics sputter
                depo_parcel[i] = 1
            elif react_type == 0: # chemical etching
                depo_parcel[i] = 0


    for i in prange(parcel.shape[0]):
        # if depo_parcel[i] == 1:
        #     film[i, :] += react_table[particle, int(reactList[i]), 1:]

        if reactList[i] == -1:
            parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], normal[i])
            # print('reflect')
            # parcel[i, 3:6] = reemission(parcel[i, 3:6], theta[i])

    return film, parcel, reactList, depo_parcel