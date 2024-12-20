import numpy as np

from src.operations.surface_fast import surface_normal
from src.configuration import configuration
import src.operations.reaction_Cl as reaction
import src.operations.mirror as mirror
# from numba import jit, prange

# @jit(nopython=True)
# def sumFilm_numba(film):
#     return np.sum(film, axis=-1)

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
        ijk_1 = self.parcel[indice_inject, 6:9].astype(np.int32)

        if np.any(indice_inject):
            # self.planes = self.get_pointcloud(sumFilm)
            self.indice_inject = indice_inject

            # get_theta = self.normal_matrix[ijk_1]
            # get_plane = self.film[ijk_1]

            # 可以把kdtree分散方法写在这里用作判断反应发生位置
            plane_bool = self.film_label_index_normal[:, :, :, 0] == 1
            # print(f'self.film_label_index_normas{self.film_label_index_normal[plane_bool]}')
            vacuum_bool = self.film_label_index_normal[:, :, :, 0] == -1
            get_plane, get_theta, get_plane_vaccum = self.get_inject_normal_kdtree(self.film_label_index_normal[plane_bool], self.film_label_index_normal[vacuum_bool], pos_1)

            # reaction
            self.film[get_plane[:,0], get_plane[:,1],get_plane[:,2]],\
            self.film[get_plane_vaccum[:,0], get_plane_vaccum[:,1], get_plane_vaccum[:,2]], \
            self.parcel[indice_inject,:], update_film_etch, update_film_depo, \
            reactList, depo_parcel = \
            reaction.reaction_rate(self.parcel[indice_inject], \
                           self.film[get_plane[:,0], get_plane[:,1],get_plane[:,2]], \
                           self.film[get_plane_vaccum[:,0], get_plane_vaccum[:,1], get_plane_vaccum[:,2]], \
                           get_theta)
            
            # update film_label_index_normal
            etch_bool =  update_film_etch.shape[0] > 0
            depo_bool =  update_film_depo.shape[0] > 0
            if etch_bool:
                point_etch_add_depo = np.zeros((update_film_etch.shape[0] + update_film_depo.shape[0], 3))
                point_etch_add_depo[:update_film_etch.shape[0], :] = update_film_etch
                self.film_label_index_normal, self.film_label_index_normal_mirror = self.update_film_label_index_normal_etch(self.film_label_index_normal_mirror, self.mirrorGap, update_film_etch.astype(np.int64))
            if depo_bool:
                point_etch_add_depo[update_film_etch.shape[0]:, :] = update_film_depo
                self.film_label_index_normal, self.film_label_index_normal_mirror = self.update_film_label_index_normal_depo(self.film_label_index_normal_mirror, self.mirrorGap, update_film_depo.astype(np.int64))
            if etch_bool | depo_bool:
                self.film_label_index_normal_mirror = mirror.update_surface_mirror(self.film_label_index_normal, self.film_label_index_normal_mirror, self.mirrorGap, self.cellSizeX, self.cellSizeY)
                self.film_label_index_normal = self.update_normal_in_matrix(self.film_label_index_normal_mirror, self.film_label_index_normal, self.mirrorGap, point_etch_add_depo.astype(np.int64))
                self.log.info('refreshFilm')
                self.sumFilm = np.sum(self.film, axis=-1)

            # 去除反应的粒子
            reactListAll[indice_inject] = reactList
            if np.any(reactListAll != -1):
                indice_inject[np.where(reactListAll == -1)] = False
                self.parcel = self.parcel[~indice_inject]

            return np.sum(depo_parcel == self.depo_count_type), 0, 0, 0, 0
        else:
            return 0, 0, 0, 0, 0