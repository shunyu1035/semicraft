from numba import jit
import numpy as np

@jit(nopython=True)
def emission_angle(theta, phi, theta1, Eth, E):
    gamma = (3*np.sin(theta1)**2 - 1)/np.sin(theta1)**2 + \
            (np.cos(theta1)**2) * (3*np.sin(theta1)**2 + 1)/(2*np.sin(theta1)**3)*np.log((1+ np.sin(theta1))/(1-np.sin(theta1)))

    return np.cos(theta)*(1 - (0.25*(Eth/E)**0.5)*(np.cos(theta) * gamma + 1.5*np.pi*np.sin(theta)*np.sin(theta1)*np.cos(phi) )) 

# # no numba
# def phi_theta_dist(resolu, theta1, Eth, E):
#     theta, phi = np.linspace(-np.pi/2, np.pi/2, resolu), np.linspace(-np.pi/2, np.pi/2, resolu)
#     THETA, PHI = np.meshgrid(theta, phi)

#     R = emission_angle( THETA, PHI, theta1, Eth, E)
#     return R, theta, phi

@jit(nopython=True)
def manual_meshgrid(x, y):
    nx, ny = len(x), len(y)
    X = np.empty((ny, nx))
    Y = np.empty((ny, nx))
    for i in range(ny):
        for j in range(nx):
            X[i, j] = x[j]
            Y[i, j] = y[i]
    return X, Y

@jit(nopython=True)
def phi_theta_dist(resolu, theta1, Eth, E):
    theta = np.linspace(-np.pi / 2, np.pi / 2, resolu)
    phi = np.linspace(-np.pi / 2, np.pi / 2, resolu)
    
    # 手动生成网格
    THETA, PHI = manual_meshgrid(theta, phi)

    # 调用用户定义函数 emission_angle
    R = emission_angle(THETA, PHI, theta1, Eth, E)
    return R, theta, phi

@jit(nopython=True)
def accept_reject_phi_theta(R, theta, phi):
    reject_bound_random = np.random.uniform(0, R.max())
    accept_region = np.argwhere(R > reject_bound_random)
    get_phi_theta = accept_region[np.random.choice(accept_region.shape[0])]
    get_theta = theta[get_phi_theta[1]]
    get_phi = phi[get_phi_theta[0]]
    return get_theta, get_phi



if __name__ == "__main__":
    N = 1e5
    Eth = 26.8
    resolu = 100
    for i in range(int(N)):
        # print(i)
        E = np.random.uniform(40, 60)
        theta1 = np.random.uniform(np.pi/100, np.pi/2)
        R, theta, phi = phi_theta_dist(resolu, theta1, Eth, E)
        phitheta = accept_reject_phi_theta(R, theta, phi)
        