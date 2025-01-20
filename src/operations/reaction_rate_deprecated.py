from numba import jit, prange
import numpy as np
import src.operations.reflection as reflect
import src.operations.Rn_coeffcient as Rn_coeffcient
from src.config_SF6O2 import film_density, react_table, film_Eth
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

print(rn_matrix)
#solid = film[i, j, k, 10][Si, SiF1, SiF2, SiF3, SiO SiO2, SiOF, SiOF2, SiO2F, SiO2F2, mask]
#react_t g[F, O, ion] s  [1,          2,           3,          4,       5 ,   6,    7,    8,   9,  10]
#react_t g[F, O, ion] s  [Si,       SiF1,       SiF2,       SiF3,      SiO, SiO2, SiOF, SiOF2, SiO2F,SiO2F2]

react_type_table = np.ones((4, 11), dtype=np.int32) # 1 for chemical transfer 
react_type_table[2, :] = 2 # 2 for physics
react_type_table[3, :] = 3 # 3 for redepo
react_type_table[0, 3] = 4 # 1 for chemical remove
react_type_table[2, -1] = 0 # 2 for no reaction for mask in test

# F, O, ion, redepo[>3]
# maxwell, maxwell，updown
# chemical. chemical, sputter
# surface_react_type: 0 for chemical, 1 for physics
surface_react_type = np.array([0, 0, 1])

variale_react_type = np.array([2])


@jit(nopython=True)
def chose_rn_coeffcient(rn_energy, ene):
    for i in range(rn_energy.shape[0]):
        if ene < rn_energy[i]:
            return i

# @jit(nopython=True, parallel=True)
@jit(nopython=True)
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
    for j in range(film_layer):
        if particle in variale_react_type:
            # dot_product = np.dot(parcel[3:6], normal)
            # dot_product = np.abs(dot_product)
            # angle_rad = np.arccos(dot_product)
            # energy_range = chose_rn_coeffcient(rn_energy, parcel[-2])
            for e in range(rn_energy.shape[0]):
                if parcel[-2] < rn_energy[e]:
                    energy_range = e
            rn_prob = rn_matrix[energy_range]
            # rn_prob = rn_matrix[np.argwhere(variale_react_type == particle)[0][0]]
            react_rate = np.interp(angle_rad, rn_angle, rn_prob)
        else:
            react_rate = react_table[particle, j, 0]
        # react_rate = react_table[particle, j, 0]
        if react_rate > choice[j]:
            acceptList[j] = True
    return acceptList, particle

@jit(nopython=True, parallel=True)
# def reaction_yield(parcel, film, film_vaccum, theta, update_film):
# def reaction_rate(parcel, film, normal):
def reaction_rate(parcel, film, film_vaccum, normal, update_film):
    reactList = np.ones(parcel.shape[0], dtype=np.int_) * -1
    react_add = np.zeros(film.shape[-1])
    depo_parcel = np.zeros(parcel.shape[0])
    angle_rad = np.zeros(parcel.shape[0])
    for i in prange(parcel.shape[0]):
    #     particle = int(parcel[i, -1])
        dot_product = np.dot(parcel[i, 3:6], normal[i])
        dot_product = np.abs(dot_product)
        angle_rad[i] = np.arccos(dot_product)

        acceptList, particle = sticking_probability(parcel[i], film[i], angle_rad[i])

        react_choice_indices = np.where(acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
            react_type = react_type_table[particle, react_choice]

            if react_type == 1: # chemical transfer
                depo_parcel[i] = 1
            elif react_type == 2: # physics sputter
                depo_parcel[i] = 2
            elif react_type == 3: # redepo
                depo_parcel[i] = 3
            elif react_type == 4: # chemical etching
                depo_parcel[i] = 4
            elif react_type == 0: # no reaction
                depo_parcel[i] = 0

    for i in prange(parcel.shape[0]):
        if int(parcel[i, -1]) <= 2: # gas F O Ion
            react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
        else: # redepo solid
            react_add = np.zeros(film.shape[-1])
            parcel_index = int(parcel[i, -1])-3
            react_add[int(parcel_index)] = 1

        if depo_parcel[i] == 1: # chemical transfer
            film[i, :] += react_add * parcel[i, -3] # react_weight parcel[i, -3]
        if depo_parcel[i] == 4: # chemical remove
            film[i, :] += react_add * parcel[i, -3]
            if np.sum(film[i, :]) <= 0:
                update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True
        if depo_parcel[i] == 2: # physics sputter
            film[i, :] += react_add * int(sputterYield.sputter_yield(angle_rad[i], parcel[i,-2], film_Eth[int(reactList[i])]) * parcel[i, -3])
            if np.sum(film[i, :]) <= 0:
                update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True
        if depo_parcel[i] == 3: # depo
            film_add_all = np.sum(react_add * parcel[i, -3] + film[i, :])
            if film_add_all >= film_density:
                film_split_vaccum = film_add_all - film_density
                film_split_insitu = parcel[i, -3] - film_split_vaccum
                film_vaccum[i, :] += react_add * film_split_vaccum
                film[i, :] += react_add * film_split_insitu
                update_film[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True  
            else:
                film[i, :] += react_add * parcel[i, -3]
 
        if reactList[i] == -1:
            parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], normal[i])
            # print('reflect')
            # parcel[i, 3:6] = reemission(parcel[i, 3:6], theta[i])
            # react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
    return film, film_vaccum, parcel, update_film, reactList, depo_parcel