import numpy as np
from src.config_SF6O2 import sputter_yield_coefficient
from numba import jit, prange

def sputter_yield_angle(gamma0, gammaMax, thetaMax):
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

@jit(nopython=True)
def sputter_yield_energy(E, Eth):
    return E**0.5 - Eth**0.5


# sputterYield_ion = sputter_yield_angle(0.3, 0.001, np.pi/4)

@jit(nopython=True)
def sputter_yield(p0, theta, energy, Eth):
    return p0*np.interp(theta, sputterYield_ion[1], sputterYield_ion[0])*sputter_yield_energy(energy, Eth)

sputterYield_ion = sputter_yield_angle(sputter_yield_coefficient[0], sputter_yield_coefficient[1], sputter_yield_coefficient[2])