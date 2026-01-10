import numpy as np

# def max_velocity_u(random1, random2):
#     return np.sqrt(-np.log(random1))*(np.cos(2*np.pi*random2))

# def max_velocity_w(random1, random2):
#     return np.sqrt(-np.log(random1))*(np.sin(2*np.pi*random2))

# def max_velocity_v(random3):
#     return -np.sqrt(-np.log(random3))

def max_velocity_u( random1, random2, Cm):
    return Cm*np.sqrt(-np.log(random1))*(np.cos(2*np.pi*random2))

def max_velocity_w( random1, random2, Cm):
    return Cm*np.sqrt(-np.log(random1))*(np.sin(2*np.pi*random2))

def max_velocity_v( random3, Cm):
    return -Cm*np.sqrt(-np.log(random3))

def cosn_velocity_u(random1, random2, n):
    return np.sqrt(-np.log(random1))*(np.abs(np.cos(2 * np.pi * random2))) ** n

def cosn_velocity_w(random1, random2, n):
    return np.sqrt(-np.log(random1))*(np.abs(np.sin(2 * np.pi * random2))) ** n

def cosn_velocity_v(random3):
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

def velGenerator_maxwell_normal(IN, T, atomMass):
    Cm = (2*1.380649e-23*T/(atomMass*1.66e-27) )**0.5 # (2kT/m)**0.5 27 for the Al
    Random1 = np.random.rand(IN)
    Random2 = np.random.rand(IN)
    Random3 = np.random.rand(IN)
    velosity_matrix = np.array([max_velocity_u(Random1, Random2, Cm), \
                                max_velocity_w(Random1, Random2, Cm), \
                                max_velocity_v(Random3, Cm)]).T

    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)

    ev = 1.60217662e-19
    energy_ev = energy**2*atomMass*1.66e-27*0.5/ev*1000
    return velosity_matrix, energy_ev

def velGenerator_cosn_normal(IN, n):
    Random1 = np.random.rand(IN)
    Random2 = np.random.rand(IN)
    Random3 = np.random.rand(IN)
    velosity_matrix = np.array([(2 * np.random.randint(0, 2, size=IN) - 1)*cosn_velocity_u(Random1, Random2, n), \
                                (2 * np.random.randint(0, 2, size=IN) - 1)*cosn_velocity_w(Random1, Random2, n), \
                                cosn_velocity_v(Random3)]).T

    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)

    return velosity_matrix

def velGenerator_maxwell_normal_nolength(IN):
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
    velosity_matrix[:, 0] = np.random.randn(IN)*0.001
    velosity_matrix[:, 1] = np.random.randn(IN)*0.001
    velosity_matrix[:, 2] = -1 
    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
    return velosity_matrix

def velGenerator_benchmark_normal(IN):
    velosity_matrix = np.zeros((IN, 3))
    velosity_matrix[:, 0] = np.random.randn(IN)*0.01
    velosity_matrix[:, 1] = -np.sqrt(2)/2
    velosity_matrix[:, 2] = -np.sqrt(2)/2
    energy = np.linalg.norm(velosity_matrix, axis=1)
    velosity_matrix[:,0] = np.divide(velosity_matrix[:,0], energy)
    velosity_matrix[:,1] = np.divide(velosity_matrix[:,1], energy)
    velosity_matrix[:,2] = np.divide(velosity_matrix[:,2], energy)
    return velosity_matrix


def velGenerator_guass_normalized(IN, sigma):

    Random2 = np.random.rand(IN)
    angle = np.random.normal(0, sigma, IN)
    velosity_matrix = np.array([np.sin(angle)*(np.cos(2*np.pi*Random2)), 
                                np.sin(angle)*(np.sin(2*np.pi*Random2)),
                                -np.cos(angle)]).T

    return velosity_matrix
# particle array
# particle_ratio_list = np.array([])

# generator type 
# [[counts, typeID, distribution, Energy]]
# particle_list = [[int(1e3), 1, 'maxwell', 50], [int(1e3), 2, 'undown', 50]]
def vel_generator(particle_list):
    vel_type_shuffle = np.zeros((1, 5))
    for i in particle_list:
        particle_matrix = np.zeros((i[0], 5))
        # print('generator particle counts:{}, type:{}, distribution:{}, energy:{}, T or cosn:{}, atomMass:{}'.format(i[0], i[1], i[2], i[3], i[4], i[5]))
        if i[2] == 'maxwell_energy':
            print('generator particle counts:{}, type:{}, distribution:{}, energy:{}, T:{}, atomMass:{}'.format(i[0], i[1], i[2], i[3], i[4], i[5]))
            velosity_matrix, energy = velGenerator_maxwell_normal(i[0], i[4], i[5])
            particle_matrix[:, 3] = energy
        elif i[2] == 'maxwell':
            print('generator particle counts:{}, type:{}, distribution:{}, energy set:{}, T:{}, atomMass:{}'.format(i[0], i[1], i[2], i[3], i[4], i[5]))
            velosity_matrix, energy = velGenerator_maxwell_normal(i[0], i[4], i[5])
            particle_matrix[:, 3] = i[3]
        elif i[2] == 'updown':
            print('generator particle counts:{}, type:{}, distribution:{}, energy:{}, cosn:{}, atomMass:{}'.format(i[0], i[1], i[2], i[3], i[4]))
            velosity_matrix = velGenerator_updown_normal(i[0])
            particle_matrix[:, 3] = i[3]
        elif i[2] == 'cosn':
            print('generator particle counts:{}, type:{}, distribution:{}, energy:{}, cosn:{}'.format(i[0], i[1], i[2], i[3], i[4]))
            velosity_matrix = velGenerator_cosn_normal(i[0], i[4])
            particle_matrix[:, 3] = i[3]

        elif i[2] == 'guass':
            print('generator particle counts:{}, type:{}, distribution:{}, energy:{}, sigma:{}'.format(i[0], i[1], i[2], i[3], i[4]))
            velosity_matrix = velGenerator_guass_normalized(i[0], i[4])
            particle_matrix[:, 3] = i[3]

        else:
            print('type error')
            return 0
        particle_matrix[:, :3] = velosity_matrix
        # particle_matrix[:, 3] = i[3]
        particle_matrix[:, -1] = i[1]
        vel_type_shuffle = np.vstack((vel_type_shuffle, particle_matrix))
    vel_type_shuffle = vel_type_shuffle[1:,:]
    np.random.shuffle(vel_type_shuffle)

    return vel_type_shuffle

def vel_generator_nolength(particle_list):
    vel_type_shuffle = np.zeros((1, 5))
    for i in particle_list:
        particle_matrix = np.zeros((i[0], 5))
        print('generator particle counts:{}, type:{}, distribution:{}, energy:{}'.format(i[0], i[1], i[2], i[3]))
        if i[2] == 'maxwell':
            velosity_matrix = velGenerator_maxwell_normal(i[0])
        elif i[2] == 'updown':
            velosity_matrix = velGenerator_updown_normal(i[0])
        else:
            print('type error')
            return 0
        particle_matrix[:, :3] = velosity_matrix
        particle_matrix[:, 3] = i[3]
        particle_matrix[:, -1] = i[1]
        vel_type_shuffle = np.vstack((vel_type_shuffle, particle_matrix))
    vel_type_shuffle = vel_type_shuffle[1:,:]
    np.random.shuffle(vel_type_shuffle)

    return vel_type_shuffle