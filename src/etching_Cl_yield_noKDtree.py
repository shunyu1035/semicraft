import numpy as np

from src.operations.surface_fast_noKDtree import surface_normal, update_normal_in_matrix_numba
from src.configuration import configuration
import src.operations.reaction_Cl as reaction
import src.operations.mirror as mirror
from numba import jit, prange

from src.cython.plane_index_fast_cython import plane_index_fast_cython

@jit(nopython=True, parallel=True)
def plane_index_numba(film_label_index_normal):
    plane_data = []
    vacuum_data = []
    shape = film_label_index_normal.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                value = film_label_index_normal[i, j, k, 0]
                if value == 1:
                    plane_data.append(film_label_index_normal[i, j, k])
                elif value == -1:
                    vacuum_data.append(film_label_index_normal[i, j, k])
    plane_data_array = np.array(plane_data)
    vacuum_data_array = np.array(vacuum_data)
    return plane_data_array, vacuum_data_array


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

    def inject(self, i_etch, j_etch, k_etch):
        indice_inject = np.array(self.sumFilm[i_etch, j_etch, k_etch] > 0 ) # ethicng

        reactListAll = np.ones(indice_inject.shape[0])*-2

        pos_1 = self.parcel[indice_inject, :3] 
        return indice_inject, reactListAll, pos_1

    def plane_index(self):
        labels = self.film_label_index_normal[:, :, :, 0]
        plane_bool = labels == 1
        vacuum_bool = labels == -1
        # plane_bool = self.film_label_index_normal[:, :, :, 0] == 1
        # vacuum_bool = self.film_label_index_normal[:, :, :, 0] == -1
        # plane_indices = np.argwhere(plane_bool)
        # vacuum_indices = np.argwhere(vacuum_bool)
        return plane_bool, vacuum_bool
    
    def plane_index_fast(self):
        labels = self.film_label_index_normal[:, :, :, 0]
        plane_bool = labels == 1
        vacuum_bool = labels == -1
        plane_indices = np.argwhere(plane_bool)
        vacuum_indices = np.argwhere(vacuum_bool)
        plane_data = self.film_label_index_normal[plane_indices[:, 0], plane_indices[:, 1], plane_indices[:, 2]]
        vacuum_data = self.film_label_index_normal[vacuum_indices[:, 0], vacuum_indices[:, 1], vacuum_indices[:, 2]]
        return plane_data, vacuum_data

    def reaction_numba(self, film, parcel, get_plane, get_plane_vaccum, get_theta):
        return reaction.reaction_rate_parallel(film, parcel, get_plane, get_plane_vaccum, get_theta)

    def etching_film(self):

        i_etch, j_etch, k_etch  = self.get_indices()

        # indice_inject_depo = np.array(self.sumFilm[i_depo, j_depo, k_depo] >= 10) # depo
        # indice_inject = np.array(self.sumFilm[i_etch, j_etch, k_etch] > 0 ) # ethicng

        # reactListAll = np.ones(indice_inject.shape[0])*-2

        # pos_1 = self.parcel[indice_inject, :3]
        indice_inject, reactListAll, pos_1 = self.inject(i_etch, j_etch, k_etch)

        if np.any(indice_inject):

            # 可以把kdtree分散方法写在这里用作判断反应发生位置
            # plane_bool = self.film_label_index_normal[:, :, :, 0] == 1
            # # print(f'self.film_label_index_normas{self.film_label_index_normal[plane_bool]}')
            # vacuum_bool = self.film_label_index_normal[:, :, :, 0] == -1
            plane_bool, vacuum_bool = self.plane_index()
            # print(self.film_label_index_normal.shape)
            # plane_data, vacuum_data = plane_index_fast_cython(self.film_label_index_normal)
            # plane_data, vacuum_data = plane_index_numba(self.film_label_index_normal)
            # get_plane, get_theta, get_plane_vaccum = self.get_inject_normal_kdtree(plane_data, vacuum_data, pos_1)
            # get_plane, get_theta, get_plane_vaccum = self.get_inject_normal_kdtree(self.film_label_index_normal[plane_indices[:, 0], plane_indices[:, 1], plane_indices[:, 2]], \
            #                                         self.film_label_index_normal[vacuum_indices[:, 0], vacuum_indices[:, 1], vacuum_indices[:, 2]], pos_1)

            get_plane, get_theta, get_plane_vaccum = self.get_inject_normal_kdtree(self.film_label_index_normal[plane_bool], self.film_label_index_normal[vacuum_bool], pos_1)

            # reaction parallel
            # self.film, self.parcel[indice_inject], update_film_etch, update_film_depo, reactList, depo_parcel = \
            # reaction.reaction_rate_parallel(self.film, self.parcel[indice_inject], get_plane, get_plane_vaccum, get_theta)

            self.film, self.parcel[indice_inject], update_film_etch, update_film_depo, reactList, depo_parcel = \
            self.reaction_numba(self.film, self.parcel[indice_inject], get_plane, get_plane_vaccum, get_theta)

            point_etch = self.film_label_index_normal[update_film_etch, 1:4].astype(np.int64)
            point_depo = self.film_label_index_normal[update_film_depo, 1:4].astype(np.int64)
            point_etch_add_depo = np.zeros((point_etch.shape[0] + point_depo.shape[0], 3), dtype=np.int64)
            if update_film_etch.any():
                point_etch_add_depo[:point_etch.shape[0], :] = point_etch
                self.film_label_index_normal, self.film_label_index_normal_mirror = \
                    self.update_film_label_index_normal_etch(self.film_label_index_normal_mirror, self.mirrorGap, point_etch)
            if update_film_depo.any():
                point_etch_add_depo[point_etch.shape[0]:, :] = point_depo
                self.film_label_index_normal, self.film_label_index_normal_mirror = \
                    self.update_film_label_index_normal_depo(self.film_label_index_normal_mirror, self.mirrorGap, point_depo)
            if update_film_etch.any() or update_film_depo.any():
                self.film_label_index_normal_mirror = mirror.update_surface_mirror(self.film_label_index_normal, self.film_label_index_normal_mirror, self.mirrorGap, self.cellSizeX, self.cellSizeY)
                self.film_label_index_normal = self.update_normal_in_matrix(self.film_label_index_normal_mirror, self.film_label_index_normal, self.mirrorGap, point_etch_add_depo)
                # self.film_label_index_normal = update_normal_in_matrix_numba(self.film_label_index_normal_mirror, self.film_label_index_normal, self.mirrorGap, point_etch_add_depo)
                # self.log.info('refreshFilm')
                # self.sumFilm = np.sum(self.film, axis=-1)
                self.sumFilm[point_etch_add_depo[:, 0], point_etch_add_depo[:, 1], point_etch_add_depo[:, 2]] = np.sum(self.film[point_etch_add_depo[:, 0], point_etch_add_depo[:, 1], point_etch_add_depo[:, 2]], axis=-1)

            # 去除反应的粒子
            reactListAll[indice_inject] = reactList
            if np.any(reactListAll != -1):
                indice_inject[np.where(reactListAll == -1)] = False
                self.parcel = self.parcel[~indice_inject]

            return np.sum(depo_parcel == self.depo_count_type), 0, 0, 0, 0
        else:
            return 0, 0, 0, 0, 0