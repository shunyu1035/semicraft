import numpy as np

def Parcelgen(parcel, celllength, pos, vel, weight, energy, typeID):

    # i = np.floor((pos[:, 0]/celllength) + 0.5).astype(int)
    # j = np.floor((pos[:, 1]/celllength) + 0.5).astype(int)
    # k = np.floor((pos[:, 2]/celllength) + 0.5).astype(int)
    i = np.floor((pos[:, 0]/celllength)).astype(int)
    j = np.floor((pos[:, 1]/celllength)).astype(int)
    k = np.floor((pos[:, 2]/celllength)).astype(int)
    # parcelIn = np.zeros((pos.shape[0], 10), order='F')
    parcelIn = np.zeros((pos.shape[0], 12))
    parcelIn[:, :3] = pos
    parcelIn[:, 3:6] = vel
    parcelIn[:, 6] = i
    parcelIn[:, 7] = j
    parcelIn[:, 8] = k
    parcelIn[:, 9] = weight
    parcelIn[:, 10] = energy
    parcelIn[:, 11] = typeID

    parcel = np.concatenate((parcel, parcelIn))

    return parcel