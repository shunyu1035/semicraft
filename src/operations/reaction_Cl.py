from numba import jit, prange
import numpy as np
import src.operations.reflection as reflect
import src.operations.Rn_coeffcient as Rn_coeffcient
from src.config_Cl import film_density, react_prob_chemical, react_yield_p0, film_Eth, react_table_equation, react_type_table, react_redepo_sticking
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

variale_react_type = np.array([1])


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
    for c in prange(film_layer):
        if film[c] <= 0:
            choice[c] = 1

    particle = int(parcel[-1])
    # particle > 3 for redepo
    if particle >= 2:
        particle = 2

    sticking_acceptList = np.zeros(film_layer, dtype=np.bool_)
    for j in range(film_layer):
        if particle == 1: # ion sputter
            for e in range(rn_energy.shape[0]):
                if parcel[-2] < rn_energy[e]:
                    energy_range = e
            rn_prob = rn_matrix[energy_range]
            sticking_rate = np.interp(angle_rad, rn_angle, rn_prob)

        elif particle == 0: # chemical p0 reaction type
            sticking_rate = react_prob_chemical[j]
        elif particle >= 2:
            sticking_rate = react_redepo_sticking[int(parcel[-1]) - 2]

        if sticking_rate > choice[j]:
            sticking_acceptList[j] = True
    return sticking_acceptList, particle

@jit(nopython=True, parallel=True)
def reaction_rate_parallel(filmMatrix, parcel, get_plane, get_plane_vaccum, get_theta):
    reactList = np.ones(parcel.shape[0], dtype=np.int_) * -1
    elements = filmMatrix.shape[-1]
    depo_parcel = np.zeros(parcel.shape[0])
    angle_rad = np.zeros(parcel.shape[0])
    shape = filmMatrix.shape
    # print(shape)
    update_film_etch = np.zeros((shape[0], shape[1], shape[2]), dtype=np.bool_)
    update_film_depo = np.zeros((shape[0], shape[1], shape[2]), dtype=np.bool_)
    # count_etch = 0
    # update_film_depo = np.zeros(parcel.shape[0], dtype=np.int64)
    # count_depo = 0

    for i in prange(parcel.shape[0]):
        react_add = np.zeros(elements)
        film = filmMatrix[get_plane[i,0], get_plane[i,1],get_plane[i,2]]
        film_vaccum = filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]]
    #     particle = int(parcel[i, -1])
        dot_product = np.dot(parcel[i, 3:6], get_theta[i])
        dot_product = np.abs(dot_product)
        angle_rad[i] = np.arccos(dot_product)

        sticking_acceptList, particle = sticking_probability(parcel[i], film, angle_rad[i])

        react_choice_indices = np.where(sticking_acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
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
            react_add = react_table_equation[int(parcel[i, -1]), int(reactList[i]), :]
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
                update_film_etch[get_plane[i,0], get_plane[i,1],get_plane[i,2]] =  True
        if depo_parcel[i] == 2: # physics sputter
            react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], parcel[i,-2], 10) # physics sputter || p0 (E**2 - Eth**2) f(theta)
            # react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], parcel[i,-2], film_Eth[int(reactList[i])])
            if react_yield > np.random.rand():
                film += react_add
                if np.sum(film) == 0:
                    update_film_etch[get_plane[i,0], get_plane[i,1],get_plane[i,2]] =  True
    
        # if depo_parcel[i] == 3: # depo
        #     film_add_all = np.sum(react_add + film[i, :])
        #     if film_add_all > film_density:
        #         film_vaccum[i, :] += react_add
        #         update_film_etch[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True  
        #     else:
        #         film[i, :] += react_add
        #         if np.sum(film[i, :]) == film_density:
        #             update_film_depo[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True
        filmMatrix[get_plane[i,0], get_plane[i,1],get_plane[i,2]] = film
        filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
        if reactList[i] == -1:
            # parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], get_theta[i])
            parcel[i, 3:6] = reflect.DiffusionReflect(parcel[i, 3:6], get_theta[i])
            # print('reflect')
            # parcel[i, 3:6] = reemission(parcel[i, 3:6], theta[i])
            # react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
    return filmMatrix, parcel, update_film_etch, update_film_depo, reactList, depo_parcel


@jit(nopython=True, parallel=True)
def reaction_rate_parallel_indice(filmMatrix, parcel, indice, get_plane, get_plane_vaccum, get_theta):
    reactList = np.ones(parcel.shape[0], dtype=np.int_) * -1
    elements = filmMatrix.shape[-1]
    depo_parcel = np.zeros(parcel.shape[0])
    angle_rad = np.zeros(parcel.shape[0])
    shape = filmMatrix.shape
    # print(shape)
    update_film_etch = np.zeros((shape[0], shape[1], shape[2]), dtype=np.bool_)
    update_film_depo = np.zeros((shape[0], shape[1], shape[2]), dtype=np.bool_)
    # count_etch = 0
    # update_film_depo = np.zeros(parcel.shape[0], dtype=np.int64)
    # count_depo = 0

    for i in prange(parcel.shape[0]):
        react_add = np.zeros(elements)
        film = filmMatrix[get_plane[i,0], get_plane[i,1],get_plane[i,2]]
        film_vaccum = filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]]
    #     particle = int(parcel[i, -1])
        dot_product = np.dot(parcel[i, 3:6], get_theta[i])
        dot_product = np.abs(dot_product)
        angle_rad[i] = np.arccos(dot_product)

        sticking_acceptList, particle = sticking_probability(parcel[i], film, angle_rad[i])

        react_choice_indices = np.where(sticking_acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
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
            react_add = react_table_equation[int(parcel[i, -1]), int(reactList[i]), :]
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
                update_film_etch[get_plane[i,0], get_plane[i,1],get_plane[i,2]] =  True
        if depo_parcel[i] == 2: # physics sputter
            react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], parcel[i,-2], 10) # physics sputter || p0 (E**2 - Eth**2) f(theta)
            # react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], parcel[i,-2], film_Eth[int(reactList[i])])
            if react_yield > np.random.rand():
                film += react_add
                if np.sum(film) == 0:
                    update_film_etch[get_plane[i,0], get_plane[i,1],get_plane[i,2]] =  True
    
        # if depo_parcel[i] == 3: # depo
        #     film_add_all = np.sum(react_add + film[i, :])
        #     if film_add_all > film_density:
        #         film_vaccum[i, :] += react_add
        #         update_film_etch[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True  
        #     else:
        #         film[i, :] += react_add
        #         if np.sum(film[i, :]) == film_density:
        #             update_film_depo[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True
        filmMatrix[get_plane[i,0], get_plane[i,1],get_plane[i,2]] = film
        filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
        if reactList[i] == -1:
            # parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], get_theta[i])
            parcel[i, 3:6] = reflect.DiffusionReflect(parcel[i, 3:6], get_theta[i])
            # print('reflect')
            # parcel[i, 3:6] = reemission(parcel[i, 3:6], theta[i])
            # react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
    return filmMatrix, parcel, update_film_etch, update_film_depo, reactList, depo_parcel

@jit(nopython=True, parallel=True)
def reaction_rate_parallel_all(filmMatrix, parcel, film_label_index_normal, cellSizeXYZ):
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
            # dot_product = np.dot(parcel[i, 3:6], get_theta)
            dot_product = np.dot(np.ascontiguousarray(parcel[i, 3:6]), np.ascontiguousarray(get_theta))
            dot_product = np.abs(dot_product)
            angle_rad = np.arccos(dot_product)

            sticking_acceptList, particle = sticking_probability(parcel[i], film, angle_rad)

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
                # parcel[i, 3:6] = reflect.DiffusionReflect(parcel[i, 3:6], get_theta)
                parcel[i, 3:6] = reflect.DiffusionReflect(np.ascontiguousarray(parcel[i, 3:6]), np.ascontiguousarray(get_theta))

        parcel[i, :3] += parcel[i, 3:6] #comment for timeit
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

    # parcel = parcel[indice_1] #comment for timeit
    # return filmMatrix, parcel, update_film_etch, update_film_depo, depo_parcel
    return update_film_etch, update_film_depo, depo_parcel, indice_1

@jit(nopython=True)
def boundary(parcel, indice_1, cellSizeXYZ):
    indice = True
    if indice_1 == True:
        # indice = True
        for i in range(2):
            if parcel[i] >= cellSizeXYZ[i]:
                parcel[i] -= cellSizeXYZ[i]
            elif parcel[i] < 0:
                parcel[i] += cellSizeXYZ[i]
        if (parcel[0] > cellSizeXYZ[0] or parcel[0] < 0 or
            parcel[1] > cellSizeXYZ[1] or parcel[1] < 0 or
            parcel[2] > cellSizeXYZ[2] or parcel[2] < 0):
            indice = False
    return parcel, indice



# @jit(nopython=True, parallel=True)
def reaction_rate(parcel, film, film_vaccum, normal):
    reactList = np.ones(parcel.shape[0], dtype=np.int_) * -1
    elements = film.shape[-1]
    react_add = np.zeros(elements)
    depo_parcel = np.zeros(parcel.shape[0])
    angle_rad = np.zeros(parcel.shape[0])

    update_film_etch = np.zeros(parcel.shape[0], dtype=np.int64)
    count_etch = 0
    update_film_depo = np.zeros(parcel.shape[0], dtype=np.int64)
    count_depo = 0

    for i in range(parcel.shape[0]):
    #     particle = int(parcel[i, -1])
        dot_product = np.dot(parcel[i, 3:6], normal[i])
        dot_product = np.abs(dot_product)
        angle_rad[i] = np.arccos(dot_product)

        sticking_acceptList, particle = sticking_probability(parcel[i], film[i], angle_rad[i])

        react_choice_indices = np.where(sticking_acceptList)[0]
        if react_choice_indices.size > 0:
            react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
            reactList[i] = react_choice
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
            react_add = react_table_equation[int(parcel[i, -1]), int(reactList[i]), :]
        else: # redepo solid
            react_add = np.zeros(elements, dtype=np.int32)
            parcel_index = int(parcel[i, -1])-2
            react_add[int(parcel_index)] = 1

        if depo_parcel[i] == 1: # chemical transfer
            film[i, :] += react_add
        if depo_parcel[i] == 4: # chemical remove
            film[i, :] += react_add
            if np.sum(film[i, :]) == 0:
                # if film[i, 3] == 0:
                update_film_etch[count_etch] =  i
                count_etch += 1
        if depo_parcel[i] == 2: # physics sputter
            react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], parcel[i,-2], 10) # physics sputter || p0 (E**2 - Eth**2) f(theta)
            # react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], parcel[i,-2], film_Eth[int(reactList[i])])
            if react_yield > np.random.rand():
                film[i, :] += react_add
                if np.sum(film[i, :]) == 0:
                    update_film_etch[count_etch] =  i
                    count_etch += 1       
        # if depo_parcel[i] == 3: # depo
        #     film_add_all = np.sum(react_add + film[i, :])
        #     if film_add_all > film_density:
        #         film_vaccum[i, :] += react_add
        #         update_film_etch[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True  
        #     else:
        #         film[i, :] += react_add
        #         if np.sum(film[i, :]) == film_density:
        #             update_film_depo[int(parcel[i, 6]), int(parcel[i, 7]), int(parcel[i, 8])] = True
 
        if reactList[i] == -1:
            parcel[i, 3:6] = reflect.SpecularReflect(parcel[i, 3:6], normal[i])
            # print('reflect')
            # parcel[i, 3:6] = reemission(parcel[i, 3:6], theta[i])
            # react_add = react_table[int(parcel[i, -1]), int(reactList[i]), 1:]
    return film, film_vaccum, parcel, update_film_etch[:count_etch], update_film_depo[:count_depo], reactList, depo_parcel