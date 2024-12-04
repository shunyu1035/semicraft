from numba import jit
import numpy as np

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