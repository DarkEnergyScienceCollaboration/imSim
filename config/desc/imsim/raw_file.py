import copy
import galsim
from galsim.config import RegisterOutputType, RegisterInputType, InputLoader
from .ccd import LSST_CCDBuilder
from .camera import Camera


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
        self.camera = base['input']['camera_geometry']['current'][0]
        seed = galsim.config.SetupConfigRNG(base, logger=logger)
        self.rng = galsim.BaseDeviate(seed)

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore,
                    logger):
        eimage = LSST_CCDBuilder.buildImages(self, config, base, file_num,
                                             image_num, obj_num, ignore,
                                             logger)[0]
        det_name = base['det_name'].replace('-', '_')
        ccd = self.camera[det_name]
        amp_images = []
        for amp in ccd.values():
            amp_data = copy.deepcopy(eimage[amp.bounds].array)*amp.gain
            if amp.raw_flip_x:
                amp_data = amp_data[:, ::-1]
            if amp.raw_flip_y:
                amp_data = amp_data[::-1, :]

            full_segment = galsim.Image(amp.raw_bounds)# + amp.bias_level
            read_noise = galsim.CCDNoise(self.rng, gain=amp.gain,
                                         read_noise=amp.read_noise)
            full_segment.addNoise(read_noise)
            full_segment[amp.raw_data_bounds].array[:] += amp_data
            amp_images.append(full_segment)
        return amp_images

RegisterOutputType('LSST_RAW_FILE', LSST_RawFileBuilder())
