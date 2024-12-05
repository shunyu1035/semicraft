import numpy as np
import time as Time
from tqdm import tqdm

import src.operations.mirror as mirror
from src.operations.boundary import boundaryNumba
from src.operations.update_parcel import update_parcel
from src.operations.parcel import Parcelgen

from src.etching import etching


class mainLoop(etching):

    def initial(self):
        start_time = Time.time()
        t = 0
        count_reaction = 0
        inputAll = 0
        filmThickness = self.substrateTop
        self.sumFilm = np.sum(self.film, axis=-1)
        # self.update_surface_mirror_noetching(self.sumFilm)
        self.surface_mirror= mirror.update_surface_mirror(self.sumFilm,self.surface_mirror, self.mirrorGap, self.cellSizeX, self.cellSizeY)
        self.planes = self.get_pointcloud(self.surface_mirror)

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

    def posGenerator_top(self, IN):
        emptyZ = 5
        position_matrix = np.array([np.random.rand(IN)*self.cellSizeX, \
                                    np.random.rand(IN)*self.cellSizeY, \
                                    np.random.uniform(0, emptyZ, IN) + self.cellSizeZ - emptyZ]).T
        position_matrix *= self.celllength
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
        # typeIDIn = np.zeros(inputCount)
        # typeIDIn[:] = typeID
        self.parcel = Parcelgen(self.parcel, self.celllength, p1, v1, self.weightEtching, energy, typeID)  

    # particle data struction np.array([posX, posY, posZ, velX, velY, velZ, i, j, k, weight, energy, typeID])
    # def Parcelgen(self, pos, vel, weight, energy, typeID):

    #     # i = np.floor((pos[:, 0]/self.celllength) + 0.5).astype(int)
    #     # j = np.floor((pos[:, 1]/self.celllength) + 0.5).astype(int)
    #     # k = np.floor((pos[:, 2]/self.celllength) + 0.5).astype(int)
    #     i = np.floor((pos[:, 0]/self.celllength)).astype(int)
    #     j = np.floor((pos[:, 1]/self.celllength)).astype(int)
    #     k = np.floor((pos[:, 2]/self.celllength)).astype(int)
    #     # parcelIn = np.zeros((pos.shape[0], 10), order='F')
    #     parcelIn = np.zeros((pos.shape[0], 12))
    #     parcelIn[:, :3] = pos
    #     parcelIn[:, 3:6] = vel
    #     parcelIn[:, 6] = i
    #     parcelIn[:, 7] = j
    #     parcelIn[:, 8] = k
    #     parcelIn[:, 9] = weight
    #     parcelIn[:, 10] = energy
    #     parcelIn[:, 11] = typeID

    #     # print(self.parcel.flags.f_contiguous)
    #     self.parcel = np.concatenate((self.parcel, parcelIn))
    #     # print(self.parcel.flags)

    def toboundary(self):
        # self.parcel = boundary(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)
        self.parcel = boundaryNumba(self.parcel, self.cellSizeX, self.cellSizeY, self.cellSizeZ, self.celllength)

    def toupdate_parcel(self, tStep):
        self.parcel = update_parcel(self.parcel, self.celllength, tStep)

    def getAcc_depo(self, tStep):

        self.toboundary()

        depo_count, ddshape, maxdd, ddi, dl1 = self.etching_film()

        self.toupdate_parcel(tStep)

        return depo_count, ddshape, maxdd, ddi, dl1 #, film_max, surface_true
          
    def update_logs(self, previous_percentage, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, ddi, dl1, ddshape, maxdd, gen_redepo, weightMax, weightMin):
        self.log.info('particleIn:{}, depo_count_step:{}, count_reaction_all:{},inputAll:{},vzMax:{:.3f},vzMin:{:.3f}, filmThickness:{}, input_count:{}, ddi:{}, dl1:{}, ddshape:{:.3f}, maxdd:{:.3f}, gen_redepo:{}, weightMax:{}, weightMin:{}'\
                    .format(previous_percentage, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, self.parcel.shape[0], ddi, dl1, ddshape, maxdd, gen_redepo, weightMax, weightMin))
        

    def get_filmThickness(self):
        for thick in range(self.film.shape[2]):
            if np.sum(self.film[int(self.cellSizeX/2),int(self.cellSizeY/2), thick, :]) == 0:
                filmThickness = thick
                break  
        return filmThickness   



    def runEtch(self, inputCount, runningCount, max_react_count):

        start_time, t, count_reaction, inputAll, filmThickness = self.initial()
        self.posvelGenerator()
        self.particleIn(inputCount)

        ti = 0
        with tqdm(total=100, desc='particle input', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
            previous_percentage = 0  # 记录上一次的百分比
            while self.parcel.shape[0] > 500:
                ti += 1
                depo_count, ddshape, maxdd, ddi, dl1 = self.getAcc_depo(self.timeStep)
                count_reaction += depo_count
                t += self.timeStep
                if count_reaction > max_react_count:
                    self.count_time(start_time, count_reaction,inputAll)
                    break
                
                if self.depo_or_etching == 'depo' and self.depoPoint[2] <= filmThickness and depo_count < 1 and self.parcel.shape[0] < 2000:
                    self.count_time(start_time, count_reaction,inputAll)
                    break

                vzMax = np.max(self.parcel[:,5])
                vzMin = np.min(self.parcel[:,5])
                weightMin = np.min(self.parcel[:,9])
                weightMax = np.max(self.parcel[:,9])
                # if self.inputMethod == 'bunch' and inputAll < max_react_count:
                if self.depo_or_etching == 'depo':
                    if self.parcel.shape[0] < runningCount and self.depoPoint[2] >= filmThickness and ti%3 == 0:
                        inputAll += inputCount
                        self.particleIn(inputCount)
                elif self.parcel.shape[0] < runningCount and self.depo_or_etching == 'etching':
                    inputAll += inputCount
                    self.particleIn(inputCount)

                current_percentage = int(count_reaction / max_react_count * 100)  # 当前百分比
                if current_percentage > previous_percentage:
                    update_value = current_percentage - previous_percentage  # 计算进度差值
                    pbar.update(update_value)
                    previous_percentage = current_percentage  # 更新上一次的百分比

                gen_redepo = np.sum(self.parcel[:, -1] == 0)

                self.update_logs(previous_percentage, depo_count, count_reaction, inputAll,  vzMax, vzMin,  filmThickness, ddi, dl1, ddshape, maxdd, gen_redepo, weightMax, weightMin)
                filmThickness = self.get_filmThickness()
                
                # if ti%10 == 0:
                #     self.removeFloat()
                #     self.cleanMinusFilm()
        return self.film, filmThickness, self.parcel