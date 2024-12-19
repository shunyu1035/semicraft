import numpy as np
import logging
# from src.operations.surface import surface_normal

# class configuration(surface_normal):
class configuration():
    def __init__(self, etchingPoint,depoPoint,density, 
                 mirrorGap, # mirror
                 reaction_type,  #reaction 
                 celllength, kdtreeN,filmKDTree,weightDepo,weightEtching,
                 tstep, substrateTop, logname):
        self.kdtreeN = kdtreeN
        self.celllength = celllength
        self.timeStep = tstep

        self.depoPoint = depoPoint
        self.etchingPoint = etchingPoint
        self.density = density
        self.filmKDTree = filmKDTree
        self.weightDepo = weightDepo
        self.weightEtching = weightEtching
        # filmKDTree=np.array([[2, 0], [3, 1]])
        #       KDTree    [depo_parcel,  film]
        self.mirrorGap = mirrorGap
        self.reaction_type = reaction_type
        self.substrateTop = substrateTop
        self.indepoThick = substrateTop
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        # self.fh = logging.FileHandler(filename='./logfiles/{}.log'.format(logname), mode='w')
        self.fh = logging.FileHandler(filename='{}.log'.format(logname), mode='w')
        self.fh.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
                    fmt='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
        self.fh.setFormatter(self.formatter)
        self.log.addHandler(self.fh)
        self.log.info('-------Start--------')
        
        self.film = None
        self.sumFilm = None
        self.normal_matrix = None
        # self.film_label = None
        # self.film_label_mirror = None
        self.film_label_index_normal = None
        self.film_label_index_normal_mirror = None


    def input(self, film, parcel, depo_or_etching, vel_matrix, depo_count_type):
        self.depo_count_type = depo_count_type
        self.depo_or_etching = depo_or_etching
        self.film = film
        self.vel_matrix = vel_matrix
        self.parcel = parcel
        self.cellSizeX = self.film.shape[0]
        self.cellSizeY = self.film.shape[1]
        self.cellSizeZ = self.film.shape[2]
        self.surface_etching_mirror = np.zeros((self.cellSizeX+int(self.mirrorGap*2), self.cellSizeY+int(self.mirrorGap*2), self.cellSizeZ))
        self.surface_mirror = np.zeros((self.cellSizeX+int(self.mirrorGap*2), self.cellSizeY+int(self.mirrorGap*2), self.cellSizeZ))


