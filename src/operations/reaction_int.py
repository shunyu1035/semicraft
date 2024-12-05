from numba import jit, prange
import numpy as np
import src.operations.reflection as reflect


react_table = np.array([[[0.3, -1, 0, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])


# react_table = np.array([[[0.3, 1, 0, 0], [1.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[0.8, -1, 0, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
#                         [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

# react_table[0, 3, 4] = -2
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