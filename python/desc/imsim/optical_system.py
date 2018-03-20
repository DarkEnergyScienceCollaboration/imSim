"""
Code to determine deviations in the Zernike coefficients determined by the
LSST AOS closed loop control system. Results do not include contributions from
the open loop lookup table.
"""

import os

from astropy.io import fits
import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import leastsq
from timeit import timeit

from polar_zernikes import gen_superposition

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sim_data/sensitivity_matrix.txt')
AOS_PATH = os.path.join(FILE_DIR, 'sim_data/aos_sim_results.txt')
NOMINAL_PATH = os.path.join(FILE_DIR, 'sim_data/annular_nominal_coeff.txt')


def _calc_fit_error(p, r_arr, t_arr, z_arr):
    """
    Calculates the residuals of a superposition of zernike polynomials

    Generates a function representing a superposition of 22 zernike polynomials
    using given coefficients and returns the residuals.

    @param [in] p is an array of 22 polynomial coefficients for the superposition

    @param [in] r_arr is an array of rho coordinates

    @param [in] t_arr is an array of theta coordinates

    @param [in] z_val is an array of expected or measured values

    @param [out] An array of the residuals
    """

    return gen_superposition(p)(r_arr, t_arr) - z_arr


def cartesian_coords():
    """
    Return 35 cartesian sampling coordinates in the exit pupil

    @param [out] an array of 35 x coordinates

    @param [out] an array of 35 y coordinates
    """

    # Initialize with central point
    x_list = [0.]
    y_list = [0.]

    # Loop over points on spines
    radii = [0.379, 0.841, 1.237, 1.535, 1.708]
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    for radius in radii:
        for angle in angles:
            x_list.append(radius * np.cos(angle))
            y_list.append(radius * np.sin(angle))

    # Add Corner raft points by hand
    x_list.extend([1.185, -1.185, -1.185, 1.185])
    y_list.extend([1.185, 1.185, -1.185, -1.185])

    return np.array(x_list), np.array(y_list)


def polar_coords():
    """
    Return 35 polar sampling coordinates in the exit pupil.

    Angular values are returned in radians

    @param [out] an array of 35 r coordinates

    @param [out] an array of 35 theta coordinates
    """

    # Initialize with central point
    r_list = [0.]
    theta_list = [0.]

    # Loop over points on spines
    radii = [0.379, 0.841, 1.237, 1.535, 1.708]
    angles = [0, 60, 120, 180, 240, 300]
    for radius in radii:
        for angle in angles:
            r_list.append(radius)
            theta_list.append(np.deg2rad(angle))

    # Add Corner raft points
    x_raft_coords = [1.185, -1.185, -1.185, 1.185]
    y_raft_coords = [1.185, 1.185, -1.185, -1.185]
    for x, y in zip(x_raft_coords, y_raft_coords):
        theta_list.append(np.arctan2(y, x))
        r_list.append(np.sqrt(x * x + y * y))

    return np.array(r_list), np.array(theta_list)


def _interp_nominal_coeff(zemax_est, fp_x, fp_y):
    """
    Interpolates the nominal annular Zernike coefficients in the exit pupil

    @param [in] fp_x is an x coordinate in the LSST exit pupil

    @param [in] fp_y is an x coordinate in the LSST exit pupil

    @param [out] An array of 19 zernike coefficients for z=4 through z=22
    """

    # Determine x and y coordinates of zemax_est
    n_samples = 32  # grid size
    fov = [-2.0, 2.0, -2.0, 2.0]  # [x_min, x_max, y_min, y_max]
    x_sampling = np.arange(fov[0], fov[1], (fov[1] - fov[0]) / n_samples)
    y_sampling = np.arange(fov[2], fov[3], (fov[3] - fov[2]) / n_samples)

    max_fov = 1.75
    if abs(fp_x) > max_fov or abs(fp_y) > max_fov:
        raise ValueError('Given coordinates are outside the field of view.')

    num_zernike_coeff = zemax_est.shape[2]
    out_arr = np.zeros(num_zernike_coeff)
    for i in range(num_zernike_coeff):
        interp_func = interp2d(x_sampling, y_sampling,
                               zemax_est[:, :, i],
                               kind='linear')

        out_arr[i] = interp_func(fp_x, fp_y)[0]

    return out_arr[3:] # Remove first four Zernike coefficients


def gen_nominal_coeff(zemax_path):
    """
    Use zemax estimates to determine the nominal coeff at the sampling coords

    Results are written to NOMINAL_PATH

    @param [in] zemax_path is the path of a fits files containing zemax
        estimates for the nominal Zernike coefficients
    """

    # Nominal coefficients of annular Zernikes generated by zemax
    zemax_est = fits.open(zemax_path)[0].data
    assert zemax_est.shape == (32, 32, 22)

    x_coords, y_coords = cartesian_coords()
    num_cords = len(x_coords)

    out_array = np.zeros((num_cords, 19))
    for i in range(num_cords):
        out_array[i] = _interp_nominal_coeff(zemax_est, x_coords[i], y_coords[i])

    # Required output is (19, 35)
    out_array_t = out_array.transpose()
    np.savetxt(NOMINAL_PATH, out_array_t)


def moc_deviations():
    """
    Returns an array of random mock optical deviations as a (35, 50) array.

    For each optical degree of freedom in LSST, deviations are chosen randomly
    at 35 positions in the exit pupil using a normal distribution. Parameters
    for each distribution are hardcoded based on Angeli et al. 2014.

    @param [out] A (35, 50) array representing mock optical distortions
    """

    aos_sim_results = np.genfromtxt(AOS_PATH)
    assert aos_sim_results.shape[0] == 50
    avg = np.average(aos_sim_results, axis=1)
    std = np.std(aos_sim_results, axis=1)

    distortion = np.zeros((35, 50))
    for i in range(35):
        distortion[i] = np.random.normal(avg, std)

    return distortion


def test_runtime(n_runs, n_coords, verbose=False):
    """
    Determines average runtimes to both instantiate the OpticalZernikes class
    and to evaluate the cartesian_coeff method.

    @param [in] n_runs is the total number of runs to average runtimes over

    @param [in] n_coords is the total number of cartesian coordinates
        to average runtimes over

    @param [in] verbose is a boolean specifying whether to print results
        (default = false)

    @param [out] The average initialization time in seconds

    @param [out] The average evaluation time of cartesian_coeff in seconds
    """

    init_time = timeit('OpticalZernikes()', globals=globals(), number=n_runs)

    optical_state = OpticalZernikes()
    x_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))
    y_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))
    runtime = timeit('optical_state.cartesian_coeff(x_coords, y_coords)',
                     globals=locals(), number=n_runs)

    if verbose:
        print('Averages over {} runs:'.format(n_runs))
        print('Init time (s):', init_time / n_runs)
        print('Run time for (s)', n_coords, 'cartesian coords:',
              runtime / n_runs)

    return


class OpticalZernikes:
    """
    This class provides fit functions for the zernike coefficients returned by
    the LSST AOS closed loop control system. It includes 19 functions that map
    a cartesian position in the exit pupil to the coefficient of zernike 4
    through zernike 22 (in the NOLL indexing scheme)
    """

    sensitivity = np.genfromtxt(MATRIX_PATH).reshape((35, 19, 50))
    nominal_coeff = np.genfromtxt(NOMINAL_PATH)
    polar_coords = polar_coords()
    _cartesian_cords = None

    def __init__(self, deviations=None):
        """
        @param [in] deviations is a (35, 50) array representing deviations in
        each of LSST's optical degrees of freedom at 35 sampling coordinates
        """

        if deviations is None:
            deviations = moc_deviations()

        self.deviations = self._calc_sampling_coeff(deviations)
        self.sampling_coeff = np.add(self.deviations, self.nominal_coeff)
        self._fit_functions = self._optimize_fits()

    def _calc_sampling_coeff(self, deviations):
        """
        Calculates 19 zernike coefficients at 35 positions in the exit pupil

        @param [in] deviations is a (35, 50) array representing deviations in
        each optical degree of freedom at the 35 sampling coordinates

        @param [out] a (19, 35) array of zernike coefficients
        """

        num_sampling_coords = self.sensitivity.shape[0]
        num_zernike_coeff = self.sensitivity.shape[1]

        coefficients = np.zeros((num_sampling_coords, num_zernike_coeff))
        for i in range(num_sampling_coords):
            coefficients[i] = self.sensitivity[i].dot(deviations[i])

        return coefficients.transpose()

    @property
    def cartesian_coords(self):
        """
        Lazy loads 35 cartesian sampling coordinates in the exit pupil

        @param [out] an array of 35 x coordinates

        @param [out] an array of 35 y coordinates
        """

        if self._cartesian_cords is None:
            self._cartesian_cords = cartesian_coords()

        return self._cartesian_cords

    def _optimize_fits(self):
        """
        Generate a separate fit function for each zernike coefficient

        @param [out] A list of 19 functions
        """

        out = []
        r, t = self.polar_coords
        for coefficient in self.sampling_coeff:
            optimal = leastsq(_calc_fit_error, np.ones((22,)),
                              args=(r, t, coefficient))

            fit_coeff = optimal[0]
            fit_func = gen_superposition(fit_coeff)
            out.append(fit_func)

        return out

    def _interp_deviations(self, fp_x, fp_y, kind='cubic'):
        """
        Determine the zernike coefficients at given coordinates by interpolating

        @param [in] fp_x is the desired exit pupil x coordinate

        @param [in] fp_y is the desired exit pupil y coordinate

        @param [in] kind is the type of interpolation to perform. (eg. "linear")

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        x, y = self.cartesian_coords
        num_zernike_coeff = self.sampling_coeff.shape[0]
        out_arr = np.zeros(num_zernike_coeff)

        for i, coeff in enumerate(self.sampling_coeff):
            interp_func = interp2d(x, y, coeff, kind=kind)
            out_arr[i] = interp_func(fp_x, fp_y)[0]

        return out_arr

    def polar_coeff(self, fp_r, fp_t):
        """
        Determine the zernike coefficients using a fit of zernike polynomials

        @param [in] fp_r is the desired exit pupil radial coordinate in rads

        @param [in] fp_t is the desired exit pupil angular coordinate

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        return np.array([f(fp_r, fp_t) for f in self._fit_functions])

    def cartesian_coeff(self, fp_x, fp_y):
        """
        Determine the zernike coefficients using a fit of zernike polynomials

        @param [in] fp_x is the desired exit pupil x coordinate

        @param [in] fp_y is the desired exit pupil y coordinate

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        fp_r = np.sqrt(fp_x ** 2 + fp_y ** 2)
        fp_t = np.arctan2(fp_y, fp_x)
        return self.polar_coeff(fp_r, fp_t)


if __name__ == '__main__':
    # Check run times for OpticalZernikes
    test_runtime(100, 1000, verbose=True)
