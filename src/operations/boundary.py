from numba import jit, prange
import numpy as np
import numba as nb
@jit(nopython=True)
def boundaryNumba(parcel, cellSizeX, cellSizeY, cellSizeZ, celllength):
    # Adjust X dimension
    indiceXMax = parcel[:, 6] >= cellSizeX
    indiceXMin = parcel[:, 6] < 0

    parcel[indiceXMax, 6] -= cellSizeX
    parcel[indiceXMax, 0] -= celllength * cellSizeX

    parcel[indiceXMin, 6] += cellSizeX
    parcel[indiceXMin, 0] += celllength * cellSizeX

    # Adjust Y dimension
    indiceYMax = parcel[:, 7] >= cellSizeY
    indiceYMin = parcel[:, 7] < 0

    parcel[indiceYMax, 7] -= cellSizeY
    parcel[indiceYMax, 1] -= celllength * cellSizeY

    parcel[indiceYMin, 7] += cellSizeY
    parcel[indiceYMin, 1] += celllength * cellSizeY

    # Check if any particles are outside bounds in any direction
    indices = (parcel[:, 6] >= cellSizeX) | (parcel[:, 6] < 0) | \
              (parcel[:, 7] >= cellSizeY) | (parcel[:, 7] < 0) | \
              (parcel[:, 8] >= cellSizeZ) | (parcel[:, 8] < 0)

    # Remove particles outside the boundary
    return parcel[~indices]


# @jit(nopython=True)
def boundaryNumba_nolength(parcel, cellSizeX, cellSizeY, cellSizeZ):
    # Adjust X dimension
    indiceXMax = parcel[:, 6] >= cellSizeX
    indiceXMin = parcel[:, 6] < 0

    parcel[indiceXMax, 6] -= cellSizeX
    parcel[indiceXMax, 0] -= cellSizeX

    parcel[indiceXMin, 6] += cellSizeX
    parcel[indiceXMin, 0] += cellSizeX

    # Adjust Y dimension
    indiceYMax = parcel[:, 7] >= cellSizeY
    indiceYMin = parcel[:, 7] < 0

    parcel[indiceYMax, 7] -= cellSizeY
    parcel[indiceYMax, 1] -= cellSizeY

    parcel[indiceYMin, 7] += cellSizeY
    parcel[indiceYMin, 1] += cellSizeY

    # Check if any particles are outside bounds in any direction
    indices = (parcel[:, 6] >= cellSizeX) | (parcel[:, 6] < 0) | \
              (parcel[:, 7] >= cellSizeY) | (parcel[:, 7] < 0) | \
              (parcel[:, 8] >= cellSizeZ) | (parcel[:, 8] < 0)

    # Remove particles outside the boundary
    return parcel[~indices]

@jit(nopython=True, parallel=True)
def boundaryNumba_nolength_parallel(parcel, cellSizeX, cellSizeY, cellSizeZ):
    n = parcel.shape[0]
    mask = np.ones(n, dtype=nb.boolean)

    for i in prange(n):
        # X boundary
        if parcel[i, 6] >= cellSizeX:
            parcel[i, 6] -= cellSizeX
            parcel[i, 0] -= cellSizeX
        elif parcel[i, 6] < 0:
            parcel[i, 6] += cellSizeX
            parcel[i, 0] += cellSizeX

        # Y boundary
        if parcel[i, 7] >= cellSizeY:
            parcel[i, 7] -= cellSizeY
            parcel[i, 1] -= cellSizeY
        elif parcel[i, 7] < 0:
            parcel[i, 7] += cellSizeY
            parcel[i, 1] += cellSizeY

        # Z boundary
        if (parcel[i, 6] >= cellSizeX or parcel[i, 6] < 0 or
            parcel[i, 7] >= cellSizeY or parcel[i, 7] < 0 or
            parcel[i, 8] >= cellSizeZ or parcel[i, 8] < 0):
            mask[i] = False

    return parcel[mask]

# @jit(nopython=True)
def boundaryNumba_nolength_posvel(parcel, cellSizeX, cellSizeY, cellSizeZ):
    # Adjust X dimension
    indiceXMax = parcel[:, 0] >= cellSizeX
    indiceXMin = parcel[:, 0] < 0

    parcel[indiceXMax, 0] -= cellSizeX
    parcel[indiceXMin, 0] += cellSizeX

    # Adjust Y dimension
    indiceYMax = parcel[:, 1] >= cellSizeY
    indiceYMin = parcel[:, 1] < 0

    parcel[indiceYMax, 1] -= cellSizeY
    parcel[indiceYMin, 1] += cellSizeY

    # Check if any particles are outside bounds in any direction
    indices = (parcel[:, 0] >= cellSizeX) | (parcel[:, 0] < 0) | \
              (parcel[:, 1] >= cellSizeY) | (parcel[:, 1] < 0) | \
              (parcel[:, 2] >= cellSizeZ) | (parcel[:, 2] < 0)

    # Remove particles outside the boundary
    return parcel[~indices]