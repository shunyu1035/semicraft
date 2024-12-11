import numpy as np


#solid = film[i, j, k, 10][Si, SiCl1, SiCl2, SiCl3, mask]
#react_t g[Cl,   ion] s   [1,      2,     3,     4,    5]

# black CD
react_prob_chemical = np.array([0.50, 0.50, 0.50, 0.9, 0.0])

react_yield_p0 = np.array([0.10, 0.10, 0.10, 0.10, 0.10])

react_redepo_sticking = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

react_table_equation = np.array([[[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, 0]],
                                 [[-1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0,-1]]], dtype=np.int32)

react_type_table = np.array([[1, 1, 1, 4, 0], # 1: chlorination  # 0: no reaction  # 4: Themal etch
                             [2, 2, 2, 2, 2], # 2 for physics and chemical sputtering
                             [3, 3, 3, 3, 3]]) # 3 for redepo

sputter_yield_coefficient = [0.3, 0.001, np.pi/4]

react_weight = 10
film_density = 20

film_Eth = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])