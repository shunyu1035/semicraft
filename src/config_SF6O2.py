import numpy as np

react_table3 = np.array([[[0.9, 2], [0.9, 3], [0.9, 4], [0.9, -4], [0.5, 7], [0.0, 0], [0.5, 8], [0.0, 0], [0.6, 10], [0.0, 0], [0.0, 0]],
                        [[0.5, 5], [0.0, 0], [0.0, 0], [0.0, 0], [0.5, 6], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
                        [[1, -1], [1, -2], [1, -3], [1, -4], [1, -5], [1, -6], [1, -7], [1, -8], [1, -9], [1, -10], [0.0, 0]]])

react_table = np.zeros((react_table3.shape[0], react_table3.shape[1], react_table3.shape[1] + 1))

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

sputter_yield_coefficient = [0.3, 0.001, np.pi/4]

react_weight = 10


film_Eth = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])