import os
import pickle
import copy
import galsim
from galsim.config import RegisterInputType, InputLoader


def get_gs_bounds(bbox):
    return galsim.BoundsI(xmin=bbox.getMinX() + 1, xmax=bbox.getMaxX(),
                          ymin=bbox.getMinY() + 1, ymax=bbox.getMaxY())


class Amp:
    def __init__(self):
        self.bounds = None
        self.raw_flip_x = None
        self.raw_flip_y = None
        self.gain = None
        self.raw_bounds = None
        self.raw_data_bounds = None
        self.read_noise = None
        self.bias_level = None

    @staticmethod
    def make_amp_from_lsst(lsst_amp, bias_level=1000.):
        my_amp = Amp()
        my_amp.bounds = get_gs_bounds(lsst_amp.getBBox())
        my_amp.raw_flip_x = lsst_amp.getRawFlipX()
        my_amp.raw_flip_y = lsst_amp.getRawFlipY()
        my_amp.gain = lsst_amp.getGain()
        my_amp.raw_bounds = get_gs_bounds(lsst_amp.getRawBBox())
        my_amp.raw_data_bounds = get_gs_bounds(lsst_amp.getRawDataBBox())
        my_amp.read_noise = lsst_amp.getReadNoise()
        my_amp.bias_level = bias_level
        return my_amp


class CCD(dict):
    def __init__(self):
        super().__init__()
        self.bounds = None

    @staticmethod
    def make_ccd_from_lsst(lsst_ccd):
        my_ccd = CCD()
        my_ccd.bounds = get_gs_bounds(lsst_ccd.getBBox())
        for lsst_amp in lsst_ccd:
            my_ccd[lsst_amp.getName()] = Amp.make_amp_from_lsst(lsst_amp)
        return my_ccd


class Camera(dict):
    _req_params = {'file_name' : str}

    def __init__(self, file_name, logger=None):
        super().__init__()
        if file_name is not None:
            self.update(self.read_pickle(file_name))

    def to_pickle(self, pickle_file):
        with open(pickle_file, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def read_pickle(pickle_file):
        with open(pickle_file, 'rb') as fd:
            return pickle.load(fd)


def make_camera_from_lsst(lsst_camera):
    my_camera = Camera(None)
    for lsst_ccd in lsst_camera:
        my_camera[lsst_ccd.getName()] = CCD.make_ccd_from_lsst(lsst_ccd)
    return my_camera


RegisterInputType('camera_geometry', InputLoader(Camera, takes_logger=True))
