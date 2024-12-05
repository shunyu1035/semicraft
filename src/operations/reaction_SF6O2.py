from numba import jit, prange
import numpy as np
import src.operations.reflection as reflect


#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, SiO SiO2, SiOF, SiOF2, SiO2F, SiO2F2]
#react_t g[F, O, ion] s  [1,          2,           3,          4,       5 ,   6,    7,    8,   9,  10]
#react_t g[F, O, ion] s  [Si,       SiF1,       SiF2,       SiF3,      SiO, SiO2, SiOF, SiOF2, SiO2F,SiO2F2]

react_table3 = np.array([[[0.9, 2], [0.9, 3], [0.9, 4], [0.9, -4], [0.5, 7], [0.0, 0], [0.5, 8], [0.0, 0], [0.6, 10], [0.0, 0]],
                        [[0.5, 5], [0.0, 0], [0.0, 0], [0.0, 0], [0.5, 6], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
                        [[0.27, -1], [0.27, -2], [0.27, -3], [0.27, -4], [0.27, -5], [0.27, -6], [0.27, -7], [0.27, -8], [0.27, -9], [0.27, -10]]])

# react_table3 = np.array([[[0.1, 2], [0.1, 3], [0.1, 4], [0.1, -4], [0.5, 7], [0.0, 0], [0.5, 8], [0.0, 0], [0.6, 10], [0.0, 0]],
#                         [[0.5, 5], [0.0, 0], [0.0, 0], [0.0, 0], [0.5, 6], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
#                         [[0.27, -1], [0.27, -2], [0.27, -3], [0.27, -4], [0.27, -5], [0.27, -6], [0.27, -7], [0.27, -8], [0.27, -9], [0.27, -10]]])


# print(react_table3.shape)

react_table = np.zeros((3, 10, 11))

for i in range(react_table3.shape[0]):
    for j in range(react_table3.shape[1]):
        for k in range(react_table3.shape[2]):
            react_table[i, j, 0] = react_table3[i, j, 0]
            react_table[i, j, j+1] = -1
            react_chem =  int(np.abs(react_table3[i, j, 1]))
            if react_table3[i, j, 1] > 0:
                react_plus_min = 1
            elif react_table3[i, j, 1] < 0:
                react_plus_min = -1
            elif react_table3[i, j, 1] == 0:
                react_plus_min = 0
            react_table[i, j, react_chem] = react_plus_min


# etching act on film, depo need output
@jit(nopython=True)
def reaction_yield(parcel, film, film_vaccum, theta, update_film):

    num_parcels = parcel.shape[0]
    num_reactions = react_table.shape[1]
    choice = np.random.rand(parcel.shape[0], react_table.shape[1])
    reactList = np.ones(parcel.shape[0])*-1
    for i in range(num_parcels):
        for j in range(num_reactions):
            if film[i, j] <= 0:
                choice[i, j] = 1

    depo_parcel = np.zeros(parcel.shape[0])
    for i in range(parcel.shape[0]):
        acceptList = np.zeros(react_table.shape[1], dtype=np.bool_)
        for j in range(film.shape[1]):
            react_rate = react_table[int(parcel[i, -1]), j, 0]
            if react_rate > choice[i, j]:
                acceptList[j] = True
        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = np.random.choice(react_choice_indices)
            reactList[i] = react_choice
            if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) > 0:
                # print('deposition')
                depo_parcel[i] = 1
            if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) < 0:
                depo_parcel[i] = -1
            if np.sum(react_table[int(parcel[i, -1]), react_choice, 1:]) == 0:
                depo_parcel[i] = -2
    for i in range(parcel.shape[0]):
        react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
        if depo_parcel[i] == -2: # chemical transfer
            film[i, :] += react_add

        if depo_parcel[i] == -1: # etching
            film[i, :] += react_add
            if np.all(film[i, :]) == 0:
                update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True

        if depo_parcel[i] == 1: # depo
            if np.sum(react_add + film[i, :]) > 10:
                film_vaccum[i, :] += react_add
                update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True  

            else:
                film[i, :] += react_add
                if np.sum(film[i, :]) == 10:
                        update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True                

        if reactList[i] == -1:
            parcel[i,3:6] = reflect.SpecularReflect(parcel[i,3:6], theta[i])
            # print('reflection')
            # parcel[i,3:6] = DiffusionReflect(parcel[i,3:6], theta[i])

    return film, film_vaccum, parcel, update_film, reactList, depo_parcel