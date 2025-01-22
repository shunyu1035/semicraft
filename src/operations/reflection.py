from numba import jit
import numpy as np


@jit(nopython=True)
def SpecularReflect(vel, normal):
    return vel - 2*vel@normal*normal

@jit(nopython=True)
def DiffusionReflect(vel, normal):
    Ut = vel - vel@normal*normal
    tw1 = Ut/np.linalg.norm(Ut)
    tw2 = np.cross(tw1, normal)
    # U = np.sqrt(kB*T/particleMass[i])*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
    if np.dot(vel, normal) > 0:
        pm = -1
    else:
        pm = 1
    U = np.random.randn()*tw1 + np.random.randn()*tw2  + pm*np.sqrt(-2*np.log((1-np.random.rand())))*normal
    UN = U / np.linalg.norm(U)
        # UN[i] = U
    return UN
