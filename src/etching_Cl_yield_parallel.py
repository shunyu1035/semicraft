import numpy as np

from src.operations.surface_fast_noKDtree import surface_normal, update_normal_in_matrix_numba
from src.configuration import configuration
import src.operations.reaction_Cl as reaction
import src.operations.mirror as mirror
from numba import jit, prange

# from src.cython.plane_index_fast_cython import plane_index_fast_cython
import src.cython.get_plane_vaccum as get_plane_vaccum_cython
# from src.cython.get_plane_vaccum import get_plane_vaccum

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

    def get_indices_ijk(self):
        return self.parcel[:, 6:9].astype(int)
    
    def get_indices_posvel(self):
        return (self.parcel[:, 0] + 0.5).astype(int), (self.parcel[:, 1] + 0.5).astype(int), (self.parcel[:, 2] + 0.5).astype(int)

    def clear_minus(self):
        minus_indice = self.sumFilm < 0
        self.film[minus_indice, :] = 0

    def inject(self, i_etch, j_etch, k_etch):
        # indice_1 = np.array(self.sumFilm[i_etch, j_etch, k_etch] > 0 ) # ethicng
        indice_1 = self.film_label_index_normal[i_etch, j_etch, k_etch, 0] == 1 # surface

        reactListAll = np.ones(indice_1.shape[0])*-2

        # pos_1 = self.parcel[indice_inject, :3] 
        return indice_1, reactListAll
    
    def inject_ijk(self, ijk):
        # indice_1 = np.array(self.sumFilm[i_etch, j_etch, k_etch] > 0 ) # ethicng
        indice_1 = self.film_label_index_normal[ijk[:, 0], ijk[:, 1], ijk[:, 2], 0] == 1 # surface

        reactListAll = np.ones(indice_1.shape[0])*-2

        # pos_1 = self.parcel[indice_inject, :3] 
        return indice_1, reactListAll
    
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

    def get_inject_normal_direct(self, indice_1):
        # indice_1 = np.array(self.film_label_index_normal[i_etch, j_etch, k_etch, 0] == 1 ) # surface
        get_plane = self.parcel[indice_1, 6:9].astype(int)
        get_theta = self.film_label_index_normal[get_plane[:, 0], get_plane[:, 1], get_plane[:, 2], -3:]
        # get_plane_vaccum = np.zeros_like(get_plane)
        # grid_cross =  np.array([[1, 0, 0],
        #                         [-1,0, 0],
        #                         [0, 1, 0],
        #                         [0,-1, 0],
        #                         [0, 0, 1],
        #                         [0, 0,-1]])
        
        # for i in range(get_plane.shape[0]):
        #     point_1 = get_plane[i]
        #     point_nn = point_1 + grid_cross
        #     point_nn[:, 0]  += self.mirrorGap
        #     point_nn[:, 1]  += self.mirrorGap
        #     # 2. 筛选邻居点对应的 film_label 值为 0 的点
        #     mask = self.film_label_index_normal_mirror[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == -1
        #     vaccum_candidate = self.film_label_index_normal_mirror[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 1:4]
        #     vaccum_candidate_indice = np.arange(vaccum_candidate.shape[0])
        #     candi = np.random.choice(vaccum_candidate_indice)
        #     get_plane_vaccum[i] = vaccum_candidate[candi]
        return get_plane, get_theta

    def reaction_numba(self, film, parcel, film_label_index_normal):
        return reaction.reaction_rate_parallel_all(film, parcel, film_label_index_normal)

    def remove_noReact(self, reactListAll, reactList, indice_1):
        reactListAll[indice_1] = reactList
        if np.any(reactListAll != -1):
            indice_1[np.where(reactListAll == -1)] = False
            self.parcel = self.parcel[~indice_1]

    def etching_film(self):

        self.film, self.parcel, update_film_etch, update_film_depo, reactList, depo_parcel = \
        self.reaction_numba(self.film, self.parcel, self.film_label_index_normal)

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

        self.remove_noReact(reactListAll, reactList, indice_1)
        return np.sum(depo_parcel == self.depo_count_type), 0, 0, 0, 0
