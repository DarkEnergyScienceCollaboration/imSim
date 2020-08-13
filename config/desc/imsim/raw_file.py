import copy
import numpy as np
import scipy.special
import galsim
from galsim.config import RegisterOutputType, RegisterInputType, InputLoader
from .ccd import LSST_CCDBuilder
from .camera import Camera


def cte_matrix(npix, cti, ntransfers=20, nexact=30):
    """
    Compute the CTE matrix so that the apparent charge q_i in the i-th
    pixel is given by
    q_i = Sum_j cte_matrix_ij q0_j
    where q0_j is the initial charge in j-th pixel.  The corresponding
    python code would be
    >>> cte = cte_matrix(npix, cti)
    >>> qout = numpy.dot(cte, qin)
    Parameters
    ----------
    npix : int
        Total number of pixels in either the serial or parallel
        directions.
    cti : float
        The charge transfer inefficiency.
    ntransfers : int, optional
        Maximum number of transfers to consider as contributing to
        a target pixel.
    nexact : int, optional
        Number of transfers to use exact the binomial distribution
        expression, otherwise use Poisson's approximation.
    Returns
    -------
    numpy.array
        The npix x npix numpy array containing the CTE matrix.
    Notes
    -----
    This implementation is based on
    Janesick, J. R., 2001, "Scientific Charge-Coupled Devices", Chapter 5,
    eqs. 5.2a,b.
    """
    ntransfers = min(npix, ntransfers)
    nexact = min(nexact, ntransfers)
    my_matrix = np.zeros((npix, npix), dtype=np.float)
    for i in range(1, npix):
        jvals = np.concatenate((np.arange(1, i+1), np.zeros(npix-i)))
        index = np.where(i - nexact < jvals)
        j = jvals[index]
        my_matrix[i-1, :][index] \
            = scipy.special.binom(i, j)*(1 - cti)**i*cti**(i - j)
        if nexact < ntransfers:
            index = np.where((i - nexact >= jvals) & (i - ntransfers < jvals))
            j = jvals[index]
            my_matrix[i-1, :][index] \
                = (j*cti)**(i-j)*np.exp(-j*cti)/scipy.special.factorial(i-j)
    return my_matrix


class LSST_RawFileBuilder(LSST_CCDBuilder):
    """Produce LSST raw image files that contain one segment per hdu with
    the pixels written in readout order.
    """
    def setup(self, config, base, file_num, logger):
        """Do the set up, as the name implies.

        Parameters:
            config:     The configuration dict for the output type.
            base:       The base configuration dict.
            file_num:   The current file_num.
            logger:     If given, a logger object to log progress.
        """
        super(LSST_RawFileBuilder, self).setup(config, base, file_num, logger)
        self.camera = galsim.config.GetInputObj('camera_geometry', config, base,
                                                'Camera')
        self.params = galsim.config.ParseValue(config, 'readout', base, dict)[0]
        self.make_cte_matrices()
        seed = galsim.config.SetupConfigRNG(base, logger=logger)
        self.rng = galsim.BaseDeviate(seed)

    def make_cte_matrices(self):
        scti = self.params['scti']
        pcti = self.params['pcti']
        self.scte_matrix = dict()
        self.pcte_matrix = dict()
        for ccd in self.camera.values():
            if ccd.bounds in self.scte_matrix:
                continue
            amp = list(ccd.values())[0]
            self.scte_matrix[ccd.bounds] \
                = None if scti == 0 else cte_matrix(amp.raw_bounds.xmax, scti)
            self.pcte_matrix[ccd.bounds] \
                = None if pcti == 0 else cte_matrix(amp.raw_bounds.ymax, pcti)
        assert(len(self.scte_matrix) == 3)

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore,
                    logger):
        det_name = base['det_name'].replace('-', '_')
        ccd = self.camera[det_name]
        base['image']['xsize'] = ccd.bounds.xmax + 1
        base['image']['ysize'] = ccd.bounds.ymax + 1
        eimage = LSST_CCDBuilder.buildImages(self, config, base, file_num,
                                             image_num, obj_num, ignore,
                                             logger)[0]
        amp_arrays = []
        for amp in ccd.values():
            amp_data = copy.deepcopy(eimage[amp.bounds].array)/amp.gain
            if amp.raw_flip_x:
                amp_data = amp_data[:, ::-1]
            if amp.raw_flip_y:
                amp_data = amp_data[::-1, :]
            amp_arrays.append(amp_data)
        amp_arrays = self.apply_crosstalk(amp_arrays, ccd.xtalk)

        # Create full readout segments, including prescan and overscan regions.
        amp_images = []
        for amp_data, amp in zip(amp_arrays, ccd.values()):
            full_segment = galsim.Image(amp.raw_bounds)
            full_segment[amp.raw_data_bounds].array[:] += amp_data
            amp_images.append(full_segment)
        amp_images = self.apply_cte(amp_images, ccd.bounds)

        # Add readout bias and read noise:
        bias_level = self.params['bias_level']
        #bias_level = amp.bias_level
        for full_segment in amp_images:
            full_segment += bias_level
            read_noise = galsim.CCDNoise(self.rng, gain=amp.gain,
                                         read_noise=amp.read_noise)
            full_segment.addNoise(read_noise)

        return amp_images

    def apply_cte(self, amp_images, ccd_bounds):
        pcte_matrix = self.pcte_matrix[ccd_bounds]
        scte_matrix = self.scte_matrix[ccd_bounds]
        for full_segment in amp_images:
            full_arr = full_segment.array
            if pcte_matrix is not None:
                for col in range(full_arr.shape[1]):
                    full_arr[:, col] = pcte_matrix @ full_arr[:, col]
            if scte_matrix is not None:
                for row in range(full_arr.shape[0]):
                    full_arr[row, :] = scte_matrix @ full_arr[row, :]
        return amp_images

    def apply_crosstalk(self, amp_arrays, xtalk_coeffs):
        if xtalk_coeffs is None:
            return amp_arrays
        output = []
        for amp_index, xtalk_row in enumerate(xtalk_coeffs):
            output.append(amp_arrays[amp_index] +
                          sum([x*y for x, y in zip(amp_arrays, xtalk_row)]))
        return output

RegisterOutputType('LSST_RAW_FILE', LSST_RawFileBuilder())
