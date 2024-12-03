from numba import jit, prange
import numpy as np
import src.reflection as reflect


react_table = np.array([[[0.7, -1, 0, 0], [0.0, 0,  0, 0], [1.0, 0, 0, 0]],
                        [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

react_type_table = np.array([[2, 0],
                             [3, 0]])


def Rn_coeffcient(c1, c2, c3, c4, alpha):
    return c1 + c2*np.tanh(c3*alpha - c4)

rn_angle = np.arange(0, np.pi/2, 0.1)
# xnew = np.array([])
rn_prob = [Rn_coeffcient(0.9423, 0.9434, 2.342, 3.026, i) for i in rn_angle]
rn_prob /= rn_prob[-1]


@jit(nopython=True, parallel=True)
def reaction_yield(parcel, film, normal):
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
        acceptList = np.zeros(num_reactions, dtype=np.bool_)
        for j in prange(film.shape[1]):
            if int(parcel[i, -1]) == 1:
                dot_product = np.dot(parcel[i, 3:6], normal[i])
                dot_product = np.abs(dot_product)
                angle_rad = np.arccos(dot_product)
                react_rate = np.interp(angle_rad, rn_angle, rn_prob)
            else:
                react_rate = react_table[int(parcel[i, -1]), j, 0]
            if react_rate < choice[i, j]:
                acceptList[j] = True

        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
            react_type = react_type_table[int(parcel[i, -1]), react_choice]

            if react_type == 2: # kdtree Si-SF
                depo_parcel[i] = 2
            elif react_type == 3: # kdtree Ar-c4f8
                depo_parcel[i] = 3
            elif react_type == 1: # +
                depo_parcel[i] = 1
            elif react_type == 4: # Ar - Si
                depo_parcel[i] = 4
            elif react_type == 0:  # no reaction
                depo_parcel[i] = 0

    for i in prange(parcel.shape[0]):
        if depo_parcel[i] == 1:
            film[i, :] += react_table[int(parcel[i, -1]), int(reactList[i]), 1:]

        if reactList[i] == -1:
            parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], normal[i])
            # parcel[i, 3:6] = reflect.DiffusionReflect(parcel[i, 3:6], normal[i])
    return film, parcel, reactList, depo_parcel