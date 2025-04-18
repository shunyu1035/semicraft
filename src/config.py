import numpy as np

react_table = np.array([[[0.15, -1, 0, 0], [0.0, 0,  0, 0], [1.0, 0, 0, 0]],
                        [[0.8, -1, 1, 0], [0.0, 0,  0, 0], [0.0, 0, 0, 0]],
                        [[1.0,  0, 0, 0], [1.0, 0, -2, 0], [1.0, 0, 0, 0]]])

react_type_table = np.array([[2, 0, 0],
                           [1, 0, 0],
                           [4, 3, 1]])

sputter_yield = [0.3, 0.001, np.pi/5]

react_weight = 10