import numpy as np

def sputterYield_Func(gamma0, gammaMax, thetaMax):
    f = -np.log(gammaMax/gamma0)/(np.log(np.cos(gammaMax)) + 1 - np.cos(thetaMax))
    s = f*np.cos(thetaMax)
    theta = np.arange(0, np.pi/2, 0.01)
    sputterYield = gamma0*np.cos(theta)**(-f)*np.exp(-s*(1/np.cos(theta) - 1))
    sputterYield[-1] = 0
    theta[-1] = np.pi/2
    yield_hist = np.zeros((2, theta.shape[0]))

    yield_hist[0, :] = sputterYield
    yield_hist[1, :] = theta
    return yield_hist