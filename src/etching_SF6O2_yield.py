import numpy as np

from src.operations.surface_int import surface_normal
from src.configuration import configuration
import src.operations.reaction_rate as reaction


class etching(configuration, surface_normal):
    def __init__(self, etchingPoint,depoPoint,density, 
                 center_with_direction, range3D, InOrOut, yield_hist,
                 maskTop, maskBottom, maskStep, maskCenter, backup,#surface_normal
                 mirrorGap, offset_distence, # mirror
                 reaction_type,  #reaction 
                 celllength, kdtreeN,filmKDTree,weightDepo,weightEtching,
                 tstep, substrateTop, logname):
        
        configuration.__init__(self, etchingPoint,depoPoint,density, 
                 mirrorGap, # mirror
                 reaction_type,  #reaction 
                 celllength, kdtreeN,filmKDTree,weightDepo,weightEtching,
                 tstep, substrateTop, logname)
        
            # def __init__(self, center_with_direction, range3D, InOrOut, celllength, yield_hist = None):
        surface_normal.__init__(self, center_with_direction, range3D, InOrOut, celllength, yield_hist)


    def get_indices(self):
        # 直接将切片操作和数据类型转换合并
        return self.parcel[:, 6].astype(int), self.parcel[:, 7].astype(int), self.parcel[:, 8].astype(int)

    def clear_minus(self):
        minus_indice = self.sumFilm < 0
        self.film[minus_indice, :] = 0

    def etching_film(self):

        i_etch, j_etch, k_etch  = self.get_indices()

        # indice_inject_depo = np.array(self.sumFilm[i_depo, j_depo, k_depo] >= 10) # depo
        indice_inject = np.array(self.sumFilm[i_etch, j_etch, k_etch] > 0 ) # ethicng

        reactListAll = np.ones(indice_inject.shape[0])*-2

        pos_1 = self.parcel[indice_inject, :3]
        vel_1 = self.parcel[indice_inject, 3:6]
        ijk_1 = self.parcel[indice_inject, 6:9]

        if np.any(indice_inject):
            # self.planes = self.get_pointcloud(sumFilm)
            self.indice_inject = indice_inject
            get_plane, get_theta, get_plane_vaccum = self.get_inject_normal(self.planes, self.planes_vaccum, pos_1, vel_1)

            self.film[get_plane[:,0], get_plane[:,1],get_plane[:,2]],\
            self.film[get_plane_vaccum[:,0], get_plane_vaccum[:,1], get_plane_vaccum[:,2]], \
            self.parcel[indice_inject,:], self.update_film,\
            reactList, depo_parcel = \
            reaction.reaction_rate(self.parcel[indice_inject], \
                           self.film[get_plane[:,0], get_plane[:,1],get_plane[:,2]], \
                           self.film[get_plane_vaccum[:,0], get_plane_vaccum[:,1], get_plane_vaccum[:,2]], \
                           get_theta, self.update_film)
            if np.any(self.update_film):
                # self.planes = self.update_pointcloud(self.planes, self.film, self.update_film)
                self.sumFilm = np.sum(self.film, axis=-1)
                self.clear_minus()
                self.planes, self.planes_vaccum = self.get_pointcloud(self.sumFilm)
            # self.reactList_debug = reactList
            reactListAll[indice_inject] = reactList
            if np.any(reactListAll != -1):
                indice_inject[np.where(reactListAll == -1)] = False
                self.parcel = self.parcel[~indice_inject]

            return np.sum(depo_parcel == self.depo_count_type), 0, 0, 0, 0
        else:
            return 0, 0, 0, 0, 0