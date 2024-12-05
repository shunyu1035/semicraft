import numpy as np
from pykdtree.kdtree import KDTree

from src.operations.surface_int import surface_normal
from src.configuration import configuration
import src.operations.reaction_int as reaction
import src.operations.mirror as mirror
import src.operations.sputter_angle_dist as sp_angle
from src.operations.parcel import Parcelgen



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
    # def etching_film(self):
    #     i, j, k = self.get_indices()
    #     # self.sumFilm = np.sum(self.film, axis=-1)
    #     # indice_inject = np.array(sumFilm[i, j, k] != 0) # etching
    #     indice_inject = np.array(self.sumFilm[i, j, k] >= 1) # depo
    #     reactListAll = np.ones(indice_inject.shape[0])*-2
    #     oscilationList = np.zeros_like(indice_inject, dtype=np.bool_)

    #     if np.any(indice_inject):
    #         pos_1, vel_1, weight_1 = self.get_positions_velocities_weight(indice_inject)
    #         get_plane, etch_yield, get_theta, ddshape, maxdd, ddi, dl1, oscilation_indice = self.calculate_injection(pos_1, vel_1)

    #         film_update_results = self.update_film(get_plane, get_theta, indice_inject, ddi, dl1, ddshape, maxdd)

    #         self.handle_surface_depo(film_update_results, etch_yield, get_theta, pos_1, vel_1, weight_1, indice_inject, reactListAll, oscilationList, film_update_results['reactList'], oscilation_indice)

    #         return film_update_results['depo_count'], ddshape, maxdd, ddi, dl1
    #     else:
    #         return 0, 0, 0, 0, 0

    def get_indices(self):
        # 直接将切片操作和数据类型转换合并
        return self.parcel[:, 6].astype(int), self.parcel[:, 7].astype(int), self.parcel[:, 8].astype(int)

    # def get_positions_velocities_weight(self, indice_inject):
    #     # 直接返回切片
    #     return self.parcel[indice_inject, :3], self.parcel[indice_inject, 3:6], self.parcel[indice_inject, 9]

    # def calculate_injection(self, pos_1, vel_1):
    #     # self.planes = self.get_pointcloud(sumFilm)
    #     get_plane, etch_yield, get_theta, ddshape, maxdd, ddi, dl1, pos1e4, vel1e4, oscilation_indice = self.get_inject_normal(self.planes, pos_1, vel_1)
    #     return get_plane, etch_yield, get_theta, ddshape, maxdd, ddi, dl1, oscilation_indice

    # def update_film(self, get_plane, get_theta, indice_inject, ddi, dl1, ddshape, maxdd):
    #     self.film[get_plane[:,0], get_plane[:,1], get_plane[:,2]], self.parcel[indice_inject, :], reactList, depo_parcel = \
    #         reaction.reaction_yield(self.parcel[indice_inject], self.film[get_plane[:,0], get_plane[:,1], get_plane[:,2]], get_theta)

    #     results = {
    #         'reactList': reactList,
    #         'depo_parcel': depo_parcel,
    #         'depo_count': np.sum(depo_parcel == self.depo_count_type),
    #         'ddshape': ddshape,
    #         'maxdd': maxdd,
    #         'ddi': ddi,
    #         'dl1': dl1
    #     }
    #     return results

    # def toKDtree(self):
    #     inKDtree = np.argwhere(self.surface_etching_mirror == True) * self.celllength
    #     return KDTree(inKDtree)

    # def handle_surface_depo(self, film_update_results, etch_yield, get_theta, pos_1, vel_1, weight_1, indice_inject, reactListAll, oscilationList, reactList, oscilation_indice):
    #     depo_parcel = film_update_results['depo_parcel']

    #     reactListAll[indice_inject] = reactList
    #     oscilationList[indice_inject] = oscilation_indice

    #     if np.any(reactListAll != -1):
    #         indice_inject[np.where(reactListAll == -1)] = False
    #         indice_inject[oscilationList == True] = False
    #         self.parcel = self.parcel[~indice_inject]

    #     for type in self.filmKDTree:
    #         # Check if the depo_parcel matches the current type
    #         react_classify = depo_parcel == type[0]
    #         # to_depo = np.where(depo_parcel == type[0])[0]
    #         if np.any(react_classify):
                
    #             # Process surface deposition
    #             self.process_surface_depo(type)

    #             # Build surface KDTree
    #             # surface_tree = cKDTree(np.argwhere(self.surface_etching_mirror == True) * self.celllength)
    #             surface_tree = self.toKDtree()
                
    #             # Query the KDTree for neighbors
    #             ii, dd, surface_indice = self.query_surface_tree(surface_tree, pos_1, react_classify)

    #             # Distribute deposition
    #             self.distribute_depo(surface_indice, ii, dd, type, etch_yield[react_classify], pos_1[react_classify], vel_1[react_classify], weight_1[react_classify], get_theta[react_classify])

    #             # Handle deposition or etching
    #             self.handle_deposition_or_etching(type)

    #     small_weight = self.parcel[:, 9] < 0.1
    #     self.parcel = self.parcel[~small_weight]
    #     # reactListAll[indice_inject] = reactList
    #     # oscilationList[indice_inject] = oscilation_indice

    #     # if np.any(reactListAll != -1):
    #     #     indice_inject[np.where(reactListAll == -1)] = False
    #     #     indice_inject[oscilationList == True] = False
    #     #     self.parcel = self.parcel[~indice_inject]

    # def process_surface_depo(self, type):
    #     # Generate surface deposition mask
    #     if type[2] == -1:
    #         # surface_etching = np.array(self.film[:, :, :, type[1]] > 1) # etching
    #         surface_etching = np.array(self.film[:, :, :, type[1]] > 0) # etching
    #     elif type[2] == 1:
    #         surface_etching = np.logical_or(self.film[:, :, :, type[1]] == 0, self.film[:, :, :, type[1]] != self.density) #depo
    #     # self.update_surface_mirror(surface_etching)
    #     self.surface_etching_mirror = mirror.update_surface_mirror(surface_etching,self.surface_etching_mirror, self.mirrorGap, self.cellSizeX, self.cellSizeY)
        
    #     # return surface_etching

    # def query_surface_tree(self, surface_tree, pos_1, to_depo):
    #     # Adjust positions for mirror and query nearest neighbors
    #     # to_depo = np.where(depo_parcel == type[0])[0]
    #     pos_mirror = np.copy(pos_1)
    #     pos_mirror[:, 0] += self.mirrorGap * self.celllength
    #     pos_mirror[:, 1] += self.mirrorGap * self.celllength
    #     dd, ii = surface_tree.query(pos_mirror[to_depo], k=self.kdtreeN)
    #     surface_indice = np.argwhere(self.surface_etching_mirror == True)
    #     return ii, dd, surface_indice

    # def handle_deposition_or_etching(self, type):
    #     surface_film_depo = np.logical_and(self.film[:, :, :, type[1]] > 1, self.film[:,:,:,type[1]] < 2)
    #     self.film[surface_film_depo, type[1]] = self.density

    #     surface_film_etching = np.logical_and(self.film[:,:,:,type[1]] < 9, self.film[:,:,:,type[1]] > 8)
    #     self.film[surface_film_etching, type[1]] = 0

    #     if np.any(surface_film_depo) or np.any(surface_film_etching):
    #         self.sumFilm = np.sum(self.film, axis=-1)
    #         # self.update_surface_mirror_noetching(self.sumFilm)
    #         self.surface_mirror = mirror.update_surface_mirror(self.sumFilm,self.surface_mirror, self.mirrorGap, self.cellSizeX, self.cellSizeY)
    #         self.planes = self.get_pointcloud(self.surface_mirror)

    # def distribute_depo(self, surface_indice, ii, dd, type, etch_yield, pos, vel, weight, normal):
    #     ddsum = np.sum(dd, axis=1)

    #     for kdi in range(self.kdtreeN):
    #         i1 = surface_indice[ii][:, kdi, 0]
    #         j1 = surface_indice[ii][:, kdi, 1]
    #         k1 = surface_indice[ii][:, kdi, 2]
            
    #         # Apply mirror gap correction
    #         i1 -= self.mirrorGap
    #         j1 -= self.mirrorGap
    #         indiceXMax = i1 >= self.cellSizeX
    #         indiceXMin = i1 < 0
    #         i1[indiceXMax] -= self.cellSizeX
    #         i1[indiceXMin] += self.cellSizeX

    #         indiceYMax = j1 >= self.cellSizeY
    #         indiceYMin = j1 < 0
    #         j1[indiceYMax] -= self.cellSizeY
    #         j1[indiceYMin] += self.cellSizeY

    #         if type[2] == 1:
    #             self.film[i1, j1, k1, type[1]] += weight * dd[:, kdi] / ddsum # depo
    #         elif type[2] == -1:
    #             # self.film[i1, j1, k1, type[1]] -= 0 * etch_yield * dd[:, kdi] / ddsum  # etching
    #             # self.film[i1, j1, k1, type[1]] -= weight * etch_yield * dd[:, kdi] / ddsum  # etching
    #             self.film[i1, j1, k1, type[1]] -= weight * dd[:, kdi] / ddsum  # etching
    #             # energy = 50
    #             # self.redepo_Generator(pos, vel, normal, weight * etch_yield, energy)

    # def redepo_Generator(self, pos, vel, normal, weight, energy):
    #     # pos[:,0] -= 2*self.mirrorGap*self.celllength
    #     # pos[:,1] -= 2*self.mirrorGap*self.celllength
    #     vels = sp_angle.reemission_multi(vel, normal)
    #     typeID = np.zeros(vel.shape[0]) # 0 for Al depo
    #     pos += vels*self.timeStep*5
    #     self.parcel = Parcelgen(self.parcel, self.celllength, pos, vels, weight, energy, typeID)

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
            reaction.reaction_yield(self.parcel[indice_inject], \
                           self.film[get_plane[:,0], get_plane[:,1],get_plane[:,2]], \
                           self.film[get_plane_vaccum[:,0], get_plane_vaccum[:,1], get_plane_vaccum[:,2]], \
                           get_theta, self.update_film)
            if np.any(self.update_film):
                # self.planes = self.update_pointcloud(self.planes, self.film, self.update_film)
                self.sumFilm = np.sum(self.film, axis=-1)
                self.planes, self.planes_vaccum = self.get_pointcloud(self.sumFilm)
            # self.reactList_debug = reactList
            reactListAll[indice_inject] = reactList
            if np.any(reactListAll != -1):
                indice_inject[np.where(reactListAll == -1)] = False
                self.parcel = self.parcel[~indice_inject]

        #     return np.sum(depo_parcel == self.depo_count_type) #, film_max, np.sum(surface_film)
        # else:
        #     return 0

            return np.sum(depo_parcel == self.depo_count_type), 0, 0, 0, 0
        else:
            return 0, 0, 0, 0, 0