import numpy as np

def max_velocity_u(random1, random2):
    return np.sqrt(-np.log(random1))*(np.cos(2*np.pi*random2))

def max_velocity_w(random1, random2):
    return np.sqrt(-np.log(random1))*(np.sin(2*np.pi*random2))

def max_velocity_v(random3):
    return -np.sqrt(-np.log(random3))

# cellDimension[cellSizeX, cellSizeY, cellSizeZ, celllength]
def posGenerator(IN, thickness, emptyZ, cellDimension):
    position_matrix = np.array([np.random.rand(IN)*cellDimension[0], \
                                np.random.rand(IN)*cellDimension[1], \
                                np.random.uniform(0, emptyZ, IN)+ thickness + emptyZ]).T
    position_matrix *= cellDimension[3]
    return position_matrix

def posGenerator_full(IN, thickness, emptyZ, cellDimension):
    position_matrix = np.array([np.random.rand(IN)*cellDimension[0], \
                                np.random.rand(IN)*cellDimension[1], \
                                np.random.uniform(0, cellDimension[2]-thickness-emptyZ, IN)+ thickness + emptyZ]).T
    position_matrix *= cellDimension[3]
    return position_matrix

def posGenerator_top(IN, thickness, emptyZ, cellDimension):
    position_matrix = np.array([np.random.rand(IN)*cellDimension[0], \
                                np.random.rand(IN)*cellDimension[1], \
                                np.random.uniform(0, emptyZ, IN) + cellDimension[2] - emptyZ]).T
    position_matrix *= cellDimension[3]
    return position_matrix
    
def posGenerator_benchmark(IN, thickness, emptyZ, cellDimension):
    position_matrix = np.array([np.random.rand(IN)*20 - 10 + cellDimension[0]/2 - 0.5, \
                                np.random.rand(IN)*20 - 10 + cellDimension[1]/2 - 0.5, \
                                np.ones(IN)*cellDimension[2]/2]).T
    position_matrix *= cellDimension[3]
    return position_matrix

def velGenerator_maxwell_normal(IN):
    Random1 = np.random.rand(IN)
    Random2 = np.random.rand(IN)
    Random3 = np.random.rand(IN)
    velosity_matrix = np.array([max_velocity_u(Random1, Random2), \
                                max_velocity_w(Random1, Random2), \
                                    max_velocity_v(Random3)]).T

    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)

    return velosity_matrix

def velGenerator_updown_normal(IN):
    velosity_matrix = np.zeros((IN, 3))
    velosity_matrix[:, 0] = np.random.randn(IN)*0.001 - 0.0005 
    velosity_matrix[:, 1] = np.random.randn(IN)*0.001 - 0.0005
    velosity_matrix[:, 2] = -1 
    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
    return velosity_matrix

def velGenerator_benchmark_normal(IN):
    velosity_matrix = np.zeros((IN, 3))
    velosity_matrix[:, 0] = np.random.randn(IN)*0.01 - 0.005
    velosity_matrix[:, 1] = -np.sqrt(2)/2
    velosity_matrix[:, 2] = -np.sqrt(2)/2
    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
    return velosity_matrix

def velGenerator_input_normal(IN):

    velosity_matrix = np.random.default_rng().choice(vel_matrix, IN)

    return velosity_matrix

# particle array
particle_ratio_list = np.array([])


# O2
def vel_generator():
    N = int(1e7)
    velosity_matrix = np.zeros((N, 3))

    Random1 = np.random.rand(N)
    Random2 = np.random.rand(N)
    Random3 = np.random.rand(N)
    velosity_matrix = np.array([max_velocity_u(Random1, Random2), \
                                max_velocity_w(Random1, Random2), \
                                    max_velocity_v(Random3)]).T

    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)

    typeID = np.zeros(N)
    FO_ratio = int(N/4)
    typeID[-FO_ratio:] = 1

    ion_ration = int(N/8)
    typeID[-ion_ration:] = 2
    velosity_matrix[-ion_ration:, 0] = np.random.rand(ion_ration)*0.001
    velosity_matrix[-ion_ration:, 1] = np.random.rand(ion_ration)*0.001
    velosity_matrix[-ion_ration:, 2] = -1 

    vel_type_shuffle = np.zeros((N, 4))
    vel_type_shuffle[:, :3] = velosity_matrix
    vel_type_shuffle[:, -1] = typeID

    np.random.shuffle(vel_type_shuffle)
    
    return vel_type_shuffle