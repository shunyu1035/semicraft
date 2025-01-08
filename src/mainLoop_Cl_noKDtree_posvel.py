import numpy as np
import time as Time
from tqdm import tqdm

import src.operations.mirror as mirror
from src.operations.boundary import boundaryNumba, boundaryNumba_nolength, boundaryNumba_nolength_parallel, boundaryNumba_nolength_posvel
from src.operations.update_parcel import update_parcel, update_parcel_nolength
from src.operations.parcel import Parcelgen, Parcelgen_nolength, posvel

from src.etching_Cl_yield_parallel import etching
from numba import jit, prange

@jit(nopython=True)
def get_filmThickness_numba(sumfilm):
    for thick in range(sumfilm.shape[2]):
        if np.any(sumfilm[:, :, thick,]== 0):
            filmThickness = thick
            break  
    return filmThickness  


class mainLoop(etching):

    def initial(self):
        start_time = Time.time()
        t = 0
        count_reaction = 0
        inputAll = 0
        filmThickness = self.substrateTop
        self.sumFilm = np.sum(self.film, axis=-1)

        self.surface_mirror= mirror.update_surface_mirror(self.sumFilm,self.surface_mirror, self.mirrorGap, self.cellSizeX, self.cellSizeY)

        self.film_label_index_normal = self.build_film_label_index_normal(self.sumFilm, self.mirrorGap)
        self.film_label_index_normal_mirror = np.zeros((self.cellSizeX+int(self.mirrorGap*2), self.cellSizeY+int(self.mirrorGap*2), self.cellSizeZ, 7))
        self.film_label_index_normal_mirror = mirror.update_surface_mirror(self.film_label_index_normal, self.film_label_index_normal_mirror, self.mirrorGap, self.cellSizeX, self.cellSizeY)

        return start_time, t, count_reaction, inputAll, filmThickness
    
    def count_time(self, start_time, count_reaction,inputAll):
        end_time = Time.time()

        # 计算运行时间并转换为分钟和秒
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        # 输出运行时间
        self.log.info(f"run time: {minutes} min {seconds} sec")
        self.log.info('DataFind---count_reaction_all:{},inputAll:{}'.format(count_reaction,inputAll))

    def posvelGenerator(self):
        self.posGenerator = self.posGenerator_top
        self.velGenerator = self.velGenerator_input_normal  

    def posvelGenerator_nolength(self):
        self.posGenerator = self.posGenerator_top_nolength
        self.velGenerator = self.velGenerator_input_normal  

    def posGenerator_top(self, IN):
        emptyZ = 5
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.rand(IN)*self.cellSizeY, \
                                    np.random.uniform(0, emptyZ, IN) + self.cellSizeZ - emptyZ]).T
        position_matrix *= self.celllength
        return position_matrix
     
    def posGenerator_top_nolength(self, IN):
        emptyZ = 5
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.rand(IN)*self.cellSizeY, \
                                    np.random.uniform(0, emptyZ, IN) + self.cellSizeZ - emptyZ]).T
        return position_matrix

    def velGenerator_input_normal(self, IN):

        velosity_matrix = np.random.default_rng().choice(self.vel_matrix, IN)

        return velosity_matrix

    def particleIn(self, inputCount):
        p1 = self.posGenerator(inputCount)
        vel_matrix = self.velGenerator(inputCount)
        v1 = vel_matrix[:, :3]
        typeID = vel_matrix[:, 4]
        energy = vel_matrix[:, 3]
        self.parcel = Parcelgen(self.parcel, self.celllength, p1, v1, self.weightEtching, energy, typeID)  

    def particleIn_nolength(self, inputCount):
        p1 = self.posGenerator(inputCount)
        vel_matrix = self.velGenerator(inputCount)
        v1 = vel_matrix[:, :3]
        typeID = vel_matrix[:, 4]
        energy = vel_matrix[:, 3]
        # self.parcel = Parcelgen_nolength(self.parcel, p1, v1, self.weightEtching, energy, typeID)  
        self.parcel = posvel(self.parcel, p1, v1, self.weightEtching, energy, typeID)  

    def toboundary(self):
        self.parcel = boundaryNumba(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)

    def toupdate_parcel(self, tStep):
        self.parcel = update_parcel(self.parcel, self.celllength, tStep)

    def toboundary_nolength(self):
        # self.parcel = boundaryNumba_nolength(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ)
        self.parcel = boundaryNumba_nolength_posvel(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ)
    def toupdate_parcel_nolength(self):
        self.parcel = update_parcel_nolength(self.parcel)

    def getAcc_depo(self):

        self.toboundary_nolength()

        depo_count, ddshape, maxdd, ddi, dl1 = self.etching_film()

        # self.toupdate_parcel_nolength()

        return depo_count, ddshape, maxdd, ddi, dl1 #, film_max, surface_true
          
    def update_logs(self, previous_percentage, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, ddi, dl1, ddshape, maxdd, gen_redepo, weightMax, weightMin):
        self.log.info('particleIn:{}, depo_count_step:{}, count_reaction_all:{},inputAll:{},vzMax:{:.3f},vzMin:{:.3f}, filmThickness:{}, input_count:{}, ddi:{}, dl1:{}, ddshape:{:.3f}, maxdd:{:.3f}, gen_redepo:{}, weightMax:{}, weightMin:{}'\
                    .format(previous_percentage, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, self.parcel.shape[0], ddi, dl1, ddshape, maxdd, gen_redepo, weightMax, weightMin))
        

    # def get_filmThickness(self):
    def get_filmThickness(self):
        for thick in range(self.sumFilm.shape[2]):
            if np.any((self.sumFilm[:, :, thick])) == 0:
                filmThickness = thick
                break  
        return filmThickness  



    def runEtch(self, inputCount, runningCount, max_react_count):

        start_time, t, count_reaction, inputAll, filmThickness = self.initial()
        self.posvelGenerator_nolength()
        self.particleIn_nolength(inputCount)

        ti = 0
        with tqdm(total=100, desc='particle input', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            previous_percentage = 0  # 记录上一次的百分比
            while True:
                # self.log.info('particleIn:{},'.format(self.parcel.shape[0]))
                ti += 1
                depo_count, ddshape, maxdd, ddi, dl1 = self.getAcc_depo()
                count_reaction += depo_count
                t += self.timeStep
                if count_reaction > max_react_count:
                    self.count_time(start_time, count_reaction,inputAll)
                    self.log.info('1:{},'.format(self.parcel.shape[0]))
                    break
                
                if self.depo_or_etching == 'depo' and self.depoPoint[2] <= filmThickness and depo_count < 1 and self.parcel.shape[0] < 2000:
                    self.count_time(start_time, count_reaction,inputAll)
                    self.log.info('2:{},'.format(self.parcel.shape[0]))
                    break

                vzMax = np.max(self.parcel[:,5])
                vzMin = np.min(self.parcel[:,5])
                # weightMin = np.min(self.parcel[:,9])
                # weightMax = np.max(self.parcel[:,9])
                # if self.inputMethod == 'bunch' and inputAll < max_react_count:
                if self.depo_or_etching == 'depo':
                    if self.parcel.shape[0] < runningCount and self.depoPoint[2] >= filmThickness and ti%3 == 0:
                        inputAll += inputCount
                        self.particleIn_nolength(inputCount)
                elif self.parcel.shape[0] < runningCount and self.depo_or_etching == 'etching':
                    inputAll += inputCount
                    self.particleIn_nolength(inputCount)

                current_percentage = int(count_reaction / max_react_count * 100)  # 当前百分比
                if current_percentage > previous_percentage:
                    update_value = current_percentage - previous_percentage  # 计算进度差值
                    pbar.update(update_value)
                    previous_percentage = current_percentage  # 更新上一次的百分比

                gen_redepo = np.sum(self.parcel[:, -1] == 0)

                filmThickness = get_filmThickness_numba(self.sumFilm)
                self.update_logs(previous_percentage, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, ddi, dl1, ddshape, maxdd, gen_redepo, 0, 0)
                
                # if ti%10 == 0:
                #     self.removeFloat()
                #     self.cleanMinusFilm()
        print('--------end----------')
        return self.film, filmThickness, self.parcel