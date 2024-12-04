from src.configuration import configuration
import numpy as np

class mirror(configuration):
    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, typeID])
    def boundary(self):

        # print('bf bound',self.parcel.flags.f_contiguous)
        # if self.symmetry == True:
        indiceXMax = self.parcel[:, 6] >= self.cellSizeX
        indiceXMin = self.parcel[:, 6] < 0

        # 使用布尔索引进行调整
        self.parcel[indiceXMax, 6] -= self.cellSizeX
        self.parcel[indiceXMax, 0] -= self.celllength * self.cellSizeX

        self.parcel[indiceXMin, 6] += self.cellSizeX
        self.parcel[indiceXMin, 0] += self.celllength * self.cellSizeX

        # 检查并调整 j_cp 和对应的 pos_cp
        indiceYMax = self.parcel[:, 7] >= self.cellSizeY
        indiceYMin = self.parcel[:, 7] < 0

        # 使用布尔索引进行调整
        self.parcel[indiceYMax, 7] -= self.cellSizeY
        self.parcel[indiceYMax, 1] -= self.celllength * self.cellSizeY

        self.parcel[indiceYMin, 7] += self.cellSizeY
        self.parcel[indiceYMin, 1] += self.celllength * self.cellSizeY
        
        indices = np.logical_or(self.parcel[:, 6] >= self.cellSizeX, self.parcel[:, 6] < 0)
        indices |= np.logical_or(self.parcel[:, 7] >= self.cellSizeY, self.parcel[:, 7] < 0)
        indices |= np.logical_or(self.parcel[:, 8] >= self.cellSizeZ, self.parcel[:, 8] < 0)
        # print('af bound',self.parcel.flags.f_contiguous)
        if np.any(indices):
            self.parcel = self.parcel[~indices]

    def update_surface_mirror_noetching(self, surface_etching):
        self.surface_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching
        self.surface_mirror[:self.mirrorGap, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[-self.mirrorGap:, :, :]
        self.surface_mirror[-self.mirrorGap:, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[:self.mirrorGap, :, :]
        self.surface_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, :self.mirrorGap, :] = surface_etching[:, -self.mirrorGap:, :]
        self.surface_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX:, -self.mirrorGap:, :] = surface_etching[:, :self.mirrorGap, :]
        self.surface_mirror[:self.mirrorGap, :self.mirrorGap, :] = surface_etching[-self.mirrorGap:, -self.mirrorGap:, :]
        self.surface_mirror[:self.mirrorGap, -self.mirrorGap:, :] = surface_etching[-self.mirrorGap:, :self.mirrorGap, :]
        self.surface_mirror[-self.mirrorGap:, :self.mirrorGap, :] = surface_etching[:self.mirrorGap, -self.mirrorGap:, :]
        self.surface_mirror[-self.mirrorGap:, -self.mirrorGap:, :] = surface_etching[:self.mirrorGap, :self.mirrorGap, :]

    def update_surface_mirror(self, surface_etching):
        self.surface_etching_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching
        self.surface_etching_mirror[:self.mirrorGap, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[-self.mirrorGap:, :, :]
        self.surface_etching_mirror[-self.mirrorGap:, self.mirrorGap:self.mirrorGap+self.cellSizeY, :] = surface_etching[:self.mirrorGap, :, :]
        self.surface_etching_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX, :self.mirrorGap, :] = surface_etching[:, -self.mirrorGap:, :]
        self.surface_etching_mirror[self.mirrorGap:self.mirrorGap+self.cellSizeX:, -self.mirrorGap:, :] = surface_etching[:, :self.mirrorGap, :]
        self.surface_etching_mirror[:self.mirrorGap, :self.mirrorGap, :] = surface_etching[-self.mirrorGap:, -self.mirrorGap:, :]
        self.surface_etching_mirror[:self.mirrorGap, -self.mirrorGap:, :] = surface_etching[-self.mirrorGap:, :self.mirrorGap, :]
        self.surface_etching_mirror[-self.mirrorGap:, :self.mirrorGap, :] = surface_etching[:self.mirrorGap, -self.mirrorGap:, :]
        self.surface_etching_mirror[-self.mirrorGap:, -self.mirrorGap:, :] = surface_etching[:self.mirrorGap, :self.mirrorGap, :]
  