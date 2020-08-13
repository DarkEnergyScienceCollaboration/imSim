import os
import pickle
import copy
import numpy as np
import galsim
from galsim.config import RegisterInputType, InputLoader


# Crosstalk cofficients from lsst.obs.lsst.imsim.ImsimMapper.  These
# values are from TS3 measurements of an ITL sensor.  The current
# LsstCamMapper doesn't include crosstalk coefficients.
_ts3_xtalk = np.array(
      [[ 0.000e+00,  1.853e-04, -1.828e-05, -1.662e-05, -8.816e-06,
        -3.135e-05, -1.051e-05, -1.446e-05, -4.855e-06, -1.047e-06,
        -6.288e-06, -1.155e-05, -3.673e-06, -1.293e-07,  1.368e-06,
         1.320e-05],
       [ 1.471e-04,  0.000e+00,  2.140e-04,  3.205e-05,  1.636e-05,
        -8.482e-06, -1.115e-05, -1.293e-06,  1.499e-06, -4.990e-06,
        -1.553e-06, -7.008e-06, -9.064e-06, -8.080e-06, -1.246e-06,
        -8.548e-06],
       [ 4.833e-06,  2.371e-04,  0.000e+00,  2.590e-04,  7.124e-05,
         2.774e-05, -1.623e-06, -1.262e-06, -3.879e-06, -2.969e-06,
        -8.837e-06, -3.708e-06, -2.174e-06, -1.011e-06, -1.078e-06,
        -4.047e-06],
       [ 1.403e-05,  3.873e-05,  1.986e-04,  0.000e+00,  1.741e-04,
         4.324e-05, -2.511e-06, -4.667e-06, -5.438e-06,  8.597e-06,
        -4.245e-06, -3.300e-06, -1.918e-06, -4.402e-06, -3.976e-06,
        -1.954e-06],
       [ 9.693e-06,  3.970e-05,  6.096e-05,  2.760e-04,  0.000e+00,
         2.880e-04,  1.409e-06, -1.682e-05, -4.058e-06, -4.968e-06,
        -1.654e-06, -3.237e-07,  1.592e-05, -2.075e-06, -3.157e-06,
         2.038e-06],
       [ 1.226e-05,  2.966e-05,  2.939e-05,  5.116e-05,  2.278e-04,
         0.000e+00,  1.513e-04, -4.866e-06, -7.089e-06, -6.077e-06,
        -8.232e-07, -3.834e-06,  4.158e-06,  2.208e-06, -8.436e-06,
        -3.659e-06],
       [ 2.049e-05, -1.874e-07, -3.758e-06,  3.142e-05,  8.992e-06,
         2.512e-04,  0.000e+00,  2.256e-04, -2.744e-06, -2.332e-06,
         9.297e-07, -2.087e-06, -1.822e-05, -8.894e-07,  5.322e-06,
        -4.110e-07],
       [-4.268e-06, -1.040e-05,  5.205e-06, -6.137e-06, -1.294e-05,
        -2.772e-05,  1.524e-04,  0.000e+00,  4.567e-06, -1.460e-06,
        -4.831e-06, -3.221e-06, -7.115e-06, -1.233e-05, -5.403e-06,
        -4.244e-06],
       [-5.172e-06, -7.756e-06, -8.336e-06, -1.408e-05, -4.458e-06,
        -1.380e-05, -1.060e-05,  1.659e-05,  0.000e+00,  1.216e-04,
        -4.814e-05, -1.986e-05, -2.384e-05, -9.500e-06, -2.324e-05,
        -1.113e-05],
       [-9.830e-07, -1.272e-05, -7.881e-07, -6.432e-06, -1.385e-05,
        -1.300e-05, -2.105e-06, -8.165e-06,  2.307e-04,  0.000e+00,
         2.355e-04, -1.962e-05,  5.389e-06, -2.902e-05, -1.873e-05,
         6.009e-06],
       [-5.927e-07, -6.295e-06, -1.401e-05, -1.124e-05,  3.554e-05,
         7.674e-06, -9.341e-06, -2.883e-06, -1.006e-05,  1.408e-04,
         0.000e+00,  2.078e-04,  3.319e-05,  1.429e-05,  1.495e-05,
         1.808e-06],
       [-5.798e-06, -1.441e-06, -1.121e-06, -4.816e-06, -8.629e-06,
        -1.091e-05, -1.031e-06, -3.452e-06, -1.108e-05,  1.801e-06,
         1.865e-04,  0.000e+00,  2.463e-04,  5.126e-05,  2.549e-05,
         1.471e-06],
       [-2.223e-06, -1.362e-05, -5.876e-06,  7.127e-06, -8.328e-06,
        -7.463e-06, -6.203e-06, -7.061e-06,  3.693e-06,  1.259e-05,
         4.356e-05,  1.919e-04,  0.000e+00,  2.341e-04,  3.252e-05,
         8.089e-06],
       [-5.490e-06, -7.426e-06, -2.481e-06, -3.336e-06, -4.742e-06,
        -6.149e-06, -2.120e-06, -5.738e-06,  5.064e-06,  8.367e-06,
         3.078e-05,  8.073e-05,  2.339e-04,  0.000e+00,  2.353e-04,
         1.012e-05],
       [ 3.341e-05,  2.142e-06, -9.158e-06, -2.997e-06, -6.738e-06,
        -4.613e-06, -5.319e-06,  1.471e-06,  7.978e-06,  6.713e-07,
         2.615e-06,  3.148e-05,  4.254e-05,  2.332e-04,  0.000e+00,
         1.551e-04],
       [-1.393e-06, -9.933e-06,  3.235e-06, -2.560e-06, -1.852e-06,
        -3.721e-06, -5.246e-06, -1.690e-06, -9.347e-06, -3.114e-06,
        -2.440e-05,  5.865e-06, -9.615e-06,  7.554e-06,  1.770e-04,
         0.000e+00]], dtype=np.float32)


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

    def update(self, other):
        self.__dict__.update(other.__dict__)

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
        self.xtalk = _ts3_xtalk

    def update(self, other):
        self.__dict__.update(other.__dict__)
        for key, value in other.items():
            if not key in self:
                self[key] = Amp()
            self[key].update(value)

    @staticmethod
    def make_ccd_from_lsst(lsst_ccd):
        my_ccd = CCD()
        my_ccd.bounds = get_gs_bounds(lsst_ccd.getBBox())
        for lsst_amp in lsst_ccd:
            my_ccd[lsst_amp.getName()] = Amp.make_amp_from_lsst(lsst_amp)
        if lsst_ccd.hasCrosstalk():
            my_ccd.xtalk = lsst_ccd.getCrosstalk()
        return my_ccd


class Camera(dict):
    _req_params = {'file_name' : str}

    def __init__(self, file_name, logger=None):
        super().__init__()
        if file_name is not None:
            self.update(self.read_pickle(file_name))

    def update(self, other):
        self.__dict__.update(other.__dict__)
        for key, value in other.items():
            if not key in self:
                self[key] = CCD()
            self[key].update(value)

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
