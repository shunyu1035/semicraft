from numba import jit, prange
import numpy as np
import src.operations.reflection as reflect
import src.operations.Rn_coeffcient as Rn_coeffcient
from src.config_SF6O2 import react_weight, react_table, film_Eth
import src.operations.sputterYield as sputterYield

rn_angle = np.arange(0, np.pi/2, 0.01)
theta_angle = np.arange(0, np.pi/2, 0.01)

c_list_100 = [0.9423, 0.9434, 2.742, 3.026]
c_list_1000 = [0.9620, 0.9608, 2.542, 3.720]
c_list_1050 = [0.9458, 0.9445, 2.551, 3.735]
c_list_10000 = [1.046, 1.046, 2.686, 4.301]

rn_coeffcients = [[0.9423, 0.9434, 2.742, 3.026], # 100
                 [0.9620, 0.9608, 2.542, 3.720],  # 1000
                 [0.9458, 0.9445, 2.551, 3.735],  # 1050
                 [1.046, 1.046, 2.686, 4.301]]     # 10000

rn_energy = np.array([100, 1000, 1050, 10000])
# each array correspond to an element
rn_matrix = np.array([Rn_coeffcient.Rn_probability(i) for i in rn_coeffcients])

#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, SiO SiO2, SiOF, SiOF2, SiO2F, SiO2F2, mask]
#react_t g[F, O, ion] s  [1,          2,           3,          4,       5 ,   6,    7,    8,   9,  10]
#react_t g[F, O, ion] s  [Si,       SiF1,       SiF2,       SiF3,      SiO, SiO2, SiOF, SiOF2, SiO2F,SiO2F2]

react_type_table = np.ones((4, 11), dtype=np.int32)
react_type_table[2, :] = 2
react_type_table[3, :] = 3

# F, O, ion, redepo[>3]
# maxwell, maxwell，updown
# chemical. chemical, sputter
# surface_react_type: 0 for chemical, 1 for physics
surface_react_type = np.array([0, 0, 1])

variale_react_type = np.array([2])


@jit(nopython=True, parallel=True)
def chose_rn_coeffcient(rn_energy, ene):
    for i in range(rn_energy.shape[0]):
        if ene < rn_energy[i]:
            ene_list = i
            return ene_list

@jit(nopython=True, parallel=True)
def sticking_probability(parcel, film, angle_rad):
    # num_reactions = react_table.shape[1]
    film_layer = film.shape[0]
    choice = np.random.rand(film_layer)
    for j in prange(film_layer):
        if film[j] <= 0:
            choice[j] = 1

    particle = int(parcel[-1])
    # particle > 3 for redepo
    if particle >= 3:
        particle = 3

    acceptList = np.zeros(film_layer, dtype=np.bool_)
    for j in prange(film_layer):
        if particle in variale_react_type:
            # dot_product = np.dot(parcel[3:6], normal)
            # dot_product = np.abs(dot_product)
            # angle_rad = np.arccos(dot_product)
            rn_prob = rn_matrix[chose_rn_coeffcient(rn_energy, parcel[-1])]
            # rn_prob = rn_matrix[np.argwhere(variale_react_type == particle)[0][0]]
            react_rate = np.interp(angle_rad, rn_angle, rn_prob)
        else:
            react_rate = react_table[particle, j, 0]
        # react_rate = react_table[particle, j, 0]
        if react_rate > choice[j]:
            acceptList[j] = True
    return acceptList, particle

@jit(nopython=True, parallel=True)
def reaction_rate(parcel, film, normal):
    # num_parcels = parcel.shape[0]
    # num_reactions = react_table.shape[1]
    # choice = np.random.rand(num_parcels, num_reactions)
    reactList = np.ones(parcel.shape[0], dtype=np.int_) * -1

    # 手动循环替代布尔索引
    # for i in prange(film.shape[0]):
    #     for j in prange(film.shape[1]):
    #         if film[i, j] <= 0:
    #             choice[i, j] = 1

    depo_parcel = np.zeros(parcel.shape[0])
    angle_rad = np.zeros(parcel.shape[0])
    for i in prange(parcel.shape[0]):
    #     particle = int(parcel[i, -1])
        dot_product = np.dot(parcel[i, 3:6], normal[i])
        dot_product = np.abs(dot_product)
        angle_rad[i] = np.arccos(dot_product)
    #     # particle > 3 for redepo
    #     if particle >= 3:
    #         particle = 3
    #     acceptList = np.zeros(num_reactions, dtype=np.bool_)
    #     for j in prange(film.shape[1]):
    #         if particle in variale_react_type:
    #             dot_product = np.dot(parcel[i, 3:6], normal[i])
    #             dot_product = np.abs(dot_product)
    #             angle_rad = np.arccos(dot_product)
    #             rn_prob = rn_matrix[np.argwhere(variale_react_type == particle)[0][0]]
    #             react_rate = np.interp(angle_rad, rn_angle, rn_prob)
    #         else:
    #             react_rate = react_table[particle, j, 0]
    #         # react_rate = react_table[particle, j, 0]
    #         if react_rate > choice[i, j]:
    #             acceptList[j] = True
        acceptList, particle = sticking_probability(parcel[i], film[i], angle_rad[i])

        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
            react_type = react_type_table[particle, react_choice]

            if react_type == 1: # chemical etching
                depo_parcel[i] = 1
            elif react_type == 2: # physics sputter
                depo_parcel[i] = 2
            elif react_type == 3: # redepo
                depo_parcel[i] = 3

    for i in prange(parcel.shape[0]):
        if depo_parcel[i] == 1: # chemical etching
            film[i, :] += react_table[int(parcel[i, -1]), int(reactList[i]), 1:] * react_weight
        if depo_parcel[i] == 2: # physics sputter
            # react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
            film[i, :] += react_table[int(parcel[i, -1]), int(reactList[i]), 1:] * int(sputterYield.sputter_yield(angle_rad[i], parcel[i,-2], film_Eth[int(reactList[i])]) * react_weight)

        if reactList[i] == -1:
            parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], normal[i])
            # print('reflect')
            # parcel[i, 3:6] = reemission(parcel[i, 3:6], theta[i])
            # react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
    return film, parcel, reactList, depo_parcel