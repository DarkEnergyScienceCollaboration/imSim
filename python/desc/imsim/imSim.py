"""
Base module for the imSim package.
"""
import os
import sys
import warnings
from collections import namedtuple, defaultdict
import pickle
import logging
import traceback
import gc
import copy
import psutil
import galsim
import eups
import configparser

import numpy as np
import lsst.log as lsstLog
import lsst.obs.lsstSim as obs_lsstSim
import lsst.utils as lsstUtils
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.photUtils import LSSTdefaults, PhotometricParameters
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec
from lsst.sims.utils import applyProperMotion, ModifiedJulianDate
from lsst.sims.coordUtils import getCornerPixels
from lsst.sims.coordUtils import pixelCoordsFromPupilCoords
from lsst.sims.catUtils.mixins import PhoSimAstrometryBase
from lsst.sims.utils import _pupilCoordsFromObserved
from lsst.sims.utils import _observedFromAppGeo
from lsst.sims.utils import radiansFromArcsec
from lsst.sims.GalSimInterface import GalSimCelestialObject, SNRdocumentPSF,\
    Kolmogorov_and_Gaussian_PSF
from lsst.sims.photUtils import BandpassDict, Sed, getImsimFluxNorm
from lsst.sims.utils import defaultSpecMap
from .tree_rings import TreeRings
from .cosmic_rays import CosmicRays
from .fopen import fopen
from .trim import InstCatTrimmer
from .sed_wrapper import SedWrapper
from .atmPSF import AtmosphericPSF
from .version import __version__ as imsim_version

_POINT_SOURCE = 1
_SERSIC_2D = 2
_RANDOM_WALK = 3
_FITS_IMAGE = 4

__all__ = ['PhosimInstanceCatalogParseError',
           'photometricParameters', 'phosim_obs_metadata',
           'metadata_from_file',
           'read_config', 'get_config', 'get_logger', 'get_image_dirs',
           'get_obs_lsstSim_camera',
           'add_cosmic_rays',
           '_POINT_SOURCE', '_SERSIC_2D', '_RANDOM_WALK', '_FITS_IMAGE',
           'parsePhoSimInstanceFile',
           'add_treering_info', 'airmass', 'FWHMeff', 'FWHMgeom', 'make_psf',
           'save_psf', 'load_psf', 'TracebackDecorator', 'GsObjectList',
           'get_stack_products', 'get_version_keywords']


class PhosimInstanceCatalogParseError(RuntimeError):
    "Exception class for instance catalog parser."

PhoSimInstanceCatalogContents = namedtuple('PhoSimInstanceCatalogContents',
                                            ('obs_metadata',
                                             'phot_params',
                                             'sources'))

_required_commands = set("""rightascension
declination
mjd
altitude
azimuth
filter
rotskypos
rottelpos
dist2moon
moonalt
moondec
moonphase
moonra
nsnap
obshistid
seed
seeing
sunalt
vistime""".split())


def get_obs_lsstSim_camera(log_level=lsstLog.WARN):
    """
    Get the obs_lsstSim CameraMapper object, setting the default
    log-level at WARN in order to silence the INFO message about
    "Loading Posix exposure registry from .". Note that this only
    affects the 'CameraMapper' logging level.  The logging level set
    by any calling code (e.g., imsim.py) will still apply to other log
    messages made by imSim code.
    """
    lsstLog.setLevel('CameraMapper', log_level)
    return lsst_camera()


def metadata_from_file(file_name):
    """
    Read in the InstanceCatalog specified by file_name.
    Return a dict of the header values from that
    InstanceCatalog.
    """
    input_params = {}
    with fopen(file_name, mode='rt') as in_file:
        for line in in_file:
            if line[0] == '#':
                continue

            params = line.strip().split()

            if params[0] == 'object':
                break

            float_val = float(params[1])
            int_val = int(float_val)
            if np.abs(float_val-int_val)>1.0e-10:
                val = float_val
            else:
                val = int_val
            input_params[params[0]] = val

    command_set = set(input_params.keys())
    missing_commands = _required_commands - command_set
    if missing_commands:
        message = "\nRequired commands that are missing from the instance catalog %s:\n   " \
            % file_name + "\n   ".join(missing_commands)
        raise PhosimInstanceCatalogParseError(message)

    # Report on commands that are not part of the required set.
    extra_commands = command_set - _required_commands
    if extra_commands:
        message = "\nExtra commands in the instance catalog %s that are not in the required set:\n   " \
            % file_name + "\n   ".join(extra_commands)
        warnings.warn(message)

    commands = dict(((key, value) for key, value in input_params.items()))
    # Add bandpass for convenience
    commands['bandpass'] = 'ugrizy'[commands['filter']]

    return commands


def uss_mem():
    """Return unique set size memory of current process in GB."""
    my_process = psutil.Process(os.getpid())
    return my_process.memory_full_info().uss/1024.**3


def _chip_downselect(mag_norm, x_pupil, y_pupil, logger, target_chips=None):
    """
    Down-select objects based on focalplane location relative to chip
    boundaries.  If target_chips is None, then the down-selection will
    be made for all of the science sensors in the focalplane.

    Returns
    -------
    dict: Dictionary of np.where indexes keyed by chip name
    """
    if target_chips is None:
        target_chips = [det.getName() for det in lsst_camera()]

    # how close to the edge of the detector a source has
    # to be before we will just simulate it anyway
    pix_tol = 50.0

    # any source brighter than this will be considered
    # so bright that it should be simulated for all
    # detectors, just in case light scatters onto them.
    max_mag = 16.0

    # Down-select by object location in focalplane relative to chip
    # boundaries.
    logger.debug('down-selecting by chip, %s GB', uss_mem())
    on_chip_dict = {}
    for chip_name in target_chips:
        pixel_corners = getCornerPixels(chip_name, lsst_camera())
        x_min = pixel_corners[0][0]
        x_max = pixel_corners[2][0]
        y_min = pixel_corners[0][1]
        y_max = pixel_corners[3][1]
        xpix, ypix = pixelCoordsFromPupilCoords(x_pupil, y_pupil,
                                                chipName=chip_name,
                                                camera=lsst_camera())

        on_chip = np.where(np.logical_or(mag_norm<max_mag,
                           np.logical_and(xpix>x_min-pix_tol,
                           np.logical_and(xpix<x_max+pix_tol,
                           np.logical_and(ypix>y_min-pix_tol,
                                          ypix<y_max+pix_tol)))))

        on_chip_dict[chip_name] = on_chip
    return on_chip_dict

def get_image_dirs():
    """
    Return a list of possible directories for FITS images, making sure
    the list of image_dirs contains '.'.
    """
    image_dirs = os.environ.get('IMSIM_IMAGE_PATH', '.').split(':')
    if '.' not in image_dirs:
        # Follow the usual convention of searching '.' first.
        image_dirs.insert(0, '.')
    return image_dirs

def sed_dirs(instcat_file):
    """
    Return a list of SED directories to check for SED folders. This
    includes $SIMS_SED_LIBRARY_DIR and the directory containing
    the instance catalog.
    """
    return [lsstUtils.getPackageDir('sims_sed_library'),
            os.path.dirname(os.path.abspath(instcat_file))]

def find_file_path(file_name, path_directories):
    """
    Search the path_directories for the specific filename
    and return the first file found.
    """
    for path_dir in path_directories:
        my_path = os.path.join(path_dir, file_name)
        if os.path.isfile(my_path):
            return my_path
    return file_name

def photometricParameters(phosim_commands):
    """
    Factory function to create a PhotometricParameters object based on
    the instance catalog commands.

    Parameters
    ----------
    dict
        The phosim commands provided by parsePhoSimInstanceFile.

    Returns
    -------
    lsst.sims.photUtils.PhotometricParameters
        The object containing the photometric parameters.

    Notes
    -----
    The gain is set to unity so that the resulting eimage has units of
    electrons/pixel.  Read noise and dark current are set to zero.
    The effects from all three of those will be added by the
    electronics chain readout code.
    """
    config = get_config()
    nsnap = phosim_commands['nsnap']
    vistime = phosim_commands['vistime']
    readout_time = config['electronics_readout']['readout_time']
    exptime = (vistime - (nsnap-1)*readout_time)/float(nsnap)
    return PhotometricParameters(exptime=exptime,
                                 nexp=nsnap,
                                 gain=1,
                                 readnoise=0,
                                 darkcurrent=0,
                                 bandpass=phosim_commands['bandpass'])


def phosim_obs_metadata(phosim_commands):
    """
    Factory function to create an ObservationMetaData object based
    on the PhoSim commands extracted from an instance catalog.

    Parameters
    ----------
    phosim_commands : dict
        Dictionary of PhoSim physics commands.

    Returns
    -------
    lsst.sims.utils.ObservationMetaData
    """
    bandpass = phosim_commands['bandpass']
    fwhm_eff = FWHMeff(phosim_commands['seeing'], bandpass,
                       phosim_commands['altitude'])
    fwhm_geom = FWHMgeom(phosim_commands['seeing'], bandpass,
                         phosim_commands['altitude'])
    obs_md = ObservationMetaData(pointingRA=phosim_commands['rightascension'],
                                 pointingDec=phosim_commands['declination'],
                                 mjd=phosim_commands['mjd'],
                                 rotSkyPos=phosim_commands['rotskypos'],
                                 bandpassName=bandpass,
                                 m5=LSSTdefaults().m5(bandpass),
                                 seeing=fwhm_eff)
    # Set the OpsimMetaData attribute with the obshistID info.
    obs_md.OpsimMetaData = {'obshistID': phosim_commands['obshistid']}
    obs_md.OpsimMetaData['FWHMgeom'] = fwhm_geom
    obs_md.OpsimMetaData['FWHMeff'] = fwhm_eff
    obs_md.OpsimMetaData['rawSeeing'] = phosim_commands['seeing']
    obs_md.OpsimMetaData['altitude'] = phosim_commands['altitude']
    obs_md.OpsimMetaData['airmass'] = airmass(phosim_commands['altitude'])
    obs_md.OpsimMetaData['seed'] = phosim_commands['seed']
    return obs_md


def parsePhoSimInstanceFile(fileName, sensor_list, numRows=None,
                            checkpoint_files=None, log_level='INFO'):
    """
    Read a PhoSim instance catalog into a Pandas dataFrame. Then use
    the information that was read-in to build and return a command
    dictionary and object dataFrame.

    Parameters
    ----------
    fileName: str
        The instance catalog filename.
    sensor_list: list
        List of sensors for which to extract object lists.
    numRows: int [None]
        The number of rows to read from the instance catalog.
        If None, then all of the rows will be read in.
    checkpoint_files: dict [None]
        Checkpoint files keyed by sensor name, e.g., "R:2,2 S:1,1".
        The instance catalog lines corresponding to drawn_objects in
        the checkpoint files will be skipped on ingest.
    log_level: str ['INFO']
        Logging level.

    Returns
    -------
    namedtuple
        This contains the PhoSim commands, the objects, and the
        original DataFrames containing the header lines and object
        lines
    """
    logger = get_logger(log_level, 'parsePhoSimInstanceFile')
    config = get_config()
    commands = metadata_from_file(fileName)
    obs_metadata = phosim_obs_metadata(commands)
    phot_params = photometricParameters(commands)
    logger.debug('creating InstCatTrimmer object')
    sort_magnorm = config['objects']['sort_magnorm']
    instcats = InstCatTrimmer(fileName, sensor_list, numRows=numRows,
                              checkpoint_files=checkpoint_files,
                              log_level=log_level, sort_magnorm=sort_magnorm)
    gs_object_dict = {detname: GsObjectList(instcats[detname], instcats.obs_md,
                                            phot_params, instcats.instcat_file,
                                            detname, log_level=log_level)
                      for detname in sensor_list}

    return PhoSimInstanceCatalogContents(obs_metadata,
                                         phot_params,
                                         ([], gs_object_dict))

class GsObjectList:
    def __init__(self, object_lines, obs_md, phot_params, instcat_file,
                 chip_name, log_level='INFO'):
        self.obs_md = obs_md
        self.logger = get_logger(log_level, name=chip_name)
        self._find_objects_on_chip(object_lines, chip_name)
        self.phot_params = phot_params
        self.sed_dirs = sed_dirs(instcat_file)
        config = get_config()
        self.gamma2_sign = config['wl_params']['gamma2_sign']
        self.nxy = config['objects']['nxy']

    def _find_objects_on_chip(self, object_lines, chip_name):
        num_lines = len(object_lines)
        ra_phosim = np.zeros(num_lines, dtype=float)
        dec_phosim = np.zeros(num_lines, dtype=float)
        mag_norm = 55.*np.ones(num_lines, dtype=float)
        for i, line in enumerate(object_lines):
            if not line.startswith('object'):
                raise RuntimeError('Trying to process non-object entry from '
                                   'the instance catalog.')
            tokens = line.split()
            ra_phosim[i] = float(tokens[2])
            dec_phosim[i] = float(tokens[3])
            mag_norm[i] = float(tokens[4])
        ra_appGeo, dec_appGeo \
            = PhoSimAstrometryBase._appGeoFromPhoSim(np.radians(ra_phosim),
                                                     np.radians(dec_phosim),
                                                     self.obs_md)
        ra_obs_rad, dec_obs_rad \
            = _observedFromAppGeo(ra_appGeo, dec_appGeo,
                                  obs_metadata=self.obs_md,
                                  includeRefraction=True)
        x_pupil, y_pupil = _pupilCoordsFromObserved(ra_obs_rad, dec_obs_rad,
                                                    self.obs_md)
        on_chip_dict = _chip_downselect(mag_norm, x_pupil, y_pupil,
                                        self.logger, [chip_name])
        index = on_chip_dict[chip_name]

        self.object_lines = []
        for i in index[0]:
            self.object_lines.append(object_lines[i])
        self.x_pupil = list(x_pupil[index])
        self.y_pupil = list(y_pupil[index])
        self.bp_dict = BandpassDict.loadTotalBandpassesFromFiles()

    def __len__(self):
        return len(self.object_lines)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while True:
                gs_obj = self._make_gs_object(self.object_lines.pop(0),
                                              self.x_pupil.pop(0),
                                              self.y_pupil.pop(0))
                if gs_obj is not None:
                    return gs_obj
        except IndexError:
            raise StopIteration()

    def _make_gs_object(self, object_line, x_pupil, y_pupil):
        params = object_line.strip().split()
        unique_id = params[1]
        ra_phosim = float(params[2])
        dec_phosim = float(params[3])
        mag_norm = float(params[4])
        sed_name = params[5]
        redshift = float(params[6])
        gamma1 = float(params[7])
        gamma2 = self.gamma2_sign*float(params[8])
        kappa = float(params[9])
        internal_av = 0
        internal_rv = 0
        galactic_av = 0
        galactic_rv = 0
        fits_file = None
        pixel_scale = 0
        rotation_angle = 0
        npoints = 0
        semi_major_arcsec = 0
        semi_minor_arcsec = 0
        position_angle_degrees = 0
        sersic_index = 0
        if params[12].lower() == 'point':
            gs_type = 'pointSource'
            i_gal_dust_model = 14
            if params[13].lower() != 'none':
                i_gal_dust_model = 16
                internal_av = float(params[14])
                internal_rv =float(params[15])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av = float(params[i_gal_dust_model+1])
                galactic_rv = float(params[i_gal_dust_model+2])
        elif params[12].lower() == 'sersic2d':
            gs_type = 'sersic'
            semi_major_arcsec = float(params[13])
            semi_minor_arcsec = float(params[14])
            position_angle_degrees = float(params[15])
            sersic_index = float(params[16])
            i_gal_dust_model = 18
            if params[17].lower() != 'none':
                i_gal_dust_model = 20
                internal_av = float(params[18])
                internal_rv = float(params[19])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av = float(params[i_gal_dust_model+1])
                galactic_rv = float(params[i_gal_dust_model+2])
        elif params[12].lower() == 'knots':
            gs_type = 'RandomWalk'
            semi_major_arcsec = float(params[13])
            semi_minor_arcsec = float(params[14])
            position_angle_degrees = float(params[15])
            npoints = int(params[16])
            i_gal_dust_model = 18
            if params[17].lower() != 'none':
                i_gal_dust_model = 20
                internal_av = float(params[18])
                internal_rv = float(params[19])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av = float(params[i_gal_dust_model+1])
                galactic_rv = float(params[i_gal_dust_model+2])
        elif (params[12].endswith('.fits') or params[12].endswith('.fits.gz')):
            gs_type = 'FitsImage'
            fits_file = find_file_path(params[12], get_image_dirs())
            pixel_scale = float(params[13])
            rotation_angle = float(params[14])
            i_gal_dust_model = 16
            if params[15].lower() != 'none':
                i_gal_dust_model = 18
                internal_av = float(params[16])
                internal_rv = float(params[17])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av = float(params[i_gal_dust_model+1])
                galactic_rv = float(params[i_gal_dust_model+2])
        else:
            raise RuntimeError("Do not know how to handle "
                               "object type: %s" % params[12])

        object_is_valid = (mag_norm < 50.0 and
                           (galactic_av != 0 or galactic_rv != 0) and
                           not (gs_type == 'sersic' and
                                semi_major_arcsec < semi_minor_arcsec) and
                           not (gs_type == 'RandomWalk' and npoints <=0))
        if not object_is_valid:
            return None
        sed_obj = SedWrapper(find_file_path(sed_name, self.sed_dirs),
                             mag_norm, redshift, internal_av, internal_rv,
                             galactic_av, galactic_rv, self.bp_dict)
        position_angle_radians = np.radians(360 - position_angle_degrees)
        gs_object = GalSimCelestialObject(gs_type, x_pupil, y_pupil,
                                          radiansFromArcsec(semi_major_arcsec),
                                          radiansFromArcsec(semi_minor_arcsec),
                                          radiansFromArcsec(semi_major_arcsec),
                                          position_angle_radians,
                                          sersic_index,
                                          sed_obj,
                                          self.bp_dict,
                                          self.phot_params,
                                          npoints,
                                          fits_file,
                                          pixel_scale*self.nxy,
                                          rotation_angle,
                                          gamma1=gamma1,
                                          gamma2=gamma2,
                                          kappa=kappa,
                                          uniqueId=unique_id,
                                          nx=self.nxy,
                                          ny=self.nxy)
        return gs_object


class ImSimConfiguration(object):
    """
    Configuration parameters for the simulation.  All parameters are
    set in a class-level dictionary to ensure that they are the same
    across all class instances.

    Individual parameter access is via section name:

    >>> config = get_config()
    >>> config['electronics_readout']['readout_time']
    3.
    """
    imsim_sections = defaultdict(dict)

    def __getitem__(self, section_name):
        return self.imsim_sections[section_name]

    def set_from_config(self, section_name, key, value):
        "Set the parameter value with the cast from a string applied."
        self[section_name][key] = self.cast(value)

    @staticmethod
    def cast(value):
        """
        Try to do sensible default casting of string representations
        of the parameters that are read from the config file.

        Parameters
        ----------
        value : str
            The string value returned, e.g., by ConfigParser.items(...).

        Returns
        -------
        None, int, float, bool, str
            Depending on the first workable cast, in that order.
        """
        # Remove any inline comments after a '#' delimiter.
        value = value.split('#')[0].strip()

        if value == 'None':
            return None
        try:
            if value.find('.') == -1 and value.find('e') == -1:
                return int(value)
            else:
                return float(value)
        except ValueError:
            # Check if it can be cast as a boolean.
            if value in 'True False'.split():
                return eval(value)
            # Return as the original string.
            return value


def get_config():
    """
    Get an ImSimConfiguration object with the current configuration.

    Returns
    -------
    ImSimConfiguration object
    """
    return ImSimConfiguration()


def _disable_iers_download(config):
    """
    Disable the remote IERS file download by setting the download url
    to point at a local file specified in the config.  Also, set the
    maximum age to None, so that astropy doesn't try to update from
    the remote mirror.
    """
    from astropy.utils import iers
    if not config['iers_download']['disable']:
        return
    iers.conf.auto_max_age = None

    local_iers_file = config['iers_download']['iers_file']
    if not os.path.isfile(local_iers_file):
        # Assume the file must be in the `data` folder in this repo.
        local_iers_file = os.path.join(os.environ['IMSIM_DIR'], 'data',
                                       local_iers_file)
    iers.conf.iers_auto_url = 'file:' + os.path.abspath(local_iers_file)


def read_config(config_file=None):
    """
    Read the configuration parameters for the simulation that are not
    given in the instance catalogs.

    Parameters
    ----------
    config_file : str, optional
        The file containing the configuration parameters.  If None
        (the default), then read the default parameters from
        data/default_imsim_configs.

    Returns
    -------
    dict ImSimConfiguration object
        An instance of ImSimConfiguration filled with the parameters from
        config_file.
    """
    my_config = ImSimConfiguration()
    cp = configparser.ConfigParser(allow_no_value=True)
    cp.optionxform = str
    if config_file is None:
        config_file = os.path.join(lsstUtils.getPackageDir('imsim'),
                                   'data', 'default_imsim_configs')

    if not cp.read(config_file):
        raise FileNotFoundError("Config file {} not found".format(config_file))

    for section in cp.sections():
        for key, value in cp.items(section):
            my_config.set_from_config(section, key, value)

    # Disable remote downloading of the IERS data if so specified in
    # the config file.
    _disable_iers_download(my_config)

    return my_config

read_config()


def get_logger(log_level, name=None):
    """
    Set up standard logging module and set lsst.log to the same log
    level.

    Parameters
    ----------
    log_level: str
        This is converted to logging.<log_level> and set in the logging
        config.
    name: str [None]
        The name to preprend to the log message to identify different
        logging contexts.  If None, then the root context is used.
    """
    # Setup logging output.
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(eval('logging.' + log_level))

#    # Set similar logging level for Stack code.
#    if log_level == "CRITICAL":
#        log_level = "FATAL"
#    lsstLog.setLevel(lsstLog.getDefaultLoggerName(),
#                     eval('lsstLog.%s' % log_level))

    return logger

def add_cosmic_rays(gs_interpreter, phot_params):
    """
    Add cosmic rays draw from a catalog of CRs extracted from single
    sensor darks.

    Parameters
    ----------
    gs_interpreter: lsst.sims.GalSimInterface.GalSimInterpreter
        The object that is actually drawing the images

    phot_params: lsst.sims.photUtils.PhotometricParameters
        An object containing the physical parameters characterizing
        the photometric properties of the telescope/camera system.

    Returns
    -------
    None
        Will act on gs_interpreter, adding cosmic rays to its images.
    """
    config = get_config()
    ccd_rate = config['cosmic_rays']['ccd_rate']
    if ccd_rate == 0:
        return
    catalog = config['cosmic_rays']['catalog']
    if catalog == 'default':
        catalog = os.path.join(lsstUtils.getPackageDir('imsim'),
                               'data', 'cosmic_ray_catalog.fits.gz')
    crs = CosmicRays.read_catalog(catalog, ccd_rate=ccd_rate)

    # Retrieve the visit number for the random seeds.
    visit = gs_interpreter.obs_metadata.OpsimMetaData['obshistID']
    band = gs_interpreter.obs_metadata.bandpass

    exptime = phot_params.nexp*phot_params.exptime
    for name, image in gs_interpreter.detectorImages.items():
        imarr = copy.deepcopy(image.array)
        # Set the random number seed for painting the CRs.
        cr_seed = CosmicRays.generate_seed(visit, name)
        crs.set_seed(cr_seed)
        gs_interpreter.detectorImages[name] = \
            galsim.Image(crs.paint(imarr, exptime=exptime), wcs=image.wcs)
        image.wcs.fitsHeader.set('CR_SEED', str(cr_seed))


def add_treering_info(detectors, tr_filename=None):
    """
    Adds tree ring info based on a model derived from measured sensors.

    Parameters
    ----------
    detectors: list (or other iterable)
        A list of GalSimDetector objects.
    tr_filename: str
        Filename of tree rings parameter file.

    Returns
    -------
    None
        Will add tree ring information to each of the detectors.
    """
    if tr_filename is None:
        tr_filename = os.path.join(lsstUtils.getPackageDir('imsim'),
                                   'data', 'tree_ring_data',
                                   'tree_ring_parameters_2018-04-26.txt')
    TR = TreeRings(tr_filename)
    for detector in detectors:
        [Rx, Ry, Sx, Sy] = [int(s) for s in list(detector.name) if s.isdigit()]
        (tr_center, tr_function) = TR.Read_DC2_Tree_Ring_Model(Rx, Ry, Sx, Sy)
        new_center = galsim.PositionD(tr_center.x + detector._xCenterPix, tr_center.y + detector._yCenterPix)
        detector.tree_rings = (new_center, tr_function)
    return None


def airmass(altitude):
    """
    Function to compute the airmass from altitude using equation 3
    of Krisciunas and Schaefer 1991.

    Parameters
    ----------
    altitude: float
        Altitude of pointing direction in degrees.

    Returns
    -------
    float: the airmass in units of sea-level airmass at the zenith.
    """
    altRad = np.radians(altitude)
    return 1.0/np.sqrt(1.0 - 0.96*(np.sin(0.5*np.pi - altRad))**2)


def FWHMeff(rawSeeing, band, altitude):
    """
    Compute the effective FWHM for a single Gaussian describing the PSF.

    Parameters
    ----------
    rawSeeing: float
        The "ideal" seeing in arcsec at zenith and at 500 nm.
        reference: LSST Document-20160
    band: str
        The LSST ugrizy band.
    altitude: float
        The altitude in degrees of the pointing.

    Returns
    -------
    float: Effective FWHM in arcsec.
    """
    X = airmass(altitude)

    # Find the effective wavelength for the band.
    wl = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]

    # Compute the atmospheric contribution.
    FWHMatm = rawSeeing*(wl/500)**(-0.3)*X**(0.6)

    # The worst case instrument contribution (see LSE-30).
    FWHMsys = 0.4*X**(0.6)

    # From LSST Document-20160, p. 8.
    return 1.16*np.sqrt(FWHMsys**2 + 1.04*FWHMatm**2)


def FWHMgeom(rawSeeing, band, altitude):
    """
    FWHM of the "combined PSF".  This is FWHMtot from
    LSST Document-20160, p. 8.

    Parameters
    ----------
    rawSeeing: float
        The "ideal" seeing in arcsec at zenith and at 500 nm.
        reference: LSST Document-20160
    band: str
        The LSST ugrizy band.
    altitude: float
        The altitude in degrees of the pointing.

    Returns
    -------
    float: FWHM of the combined PSF in arcsec.
    """
    return 0.822*FWHMeff(rawSeeing, band, altitude) + 0.052


def make_psf(psf_name, obs_md, log_level='WARN', rng=None, **kwds):
    """
    Make the requested PSF object.

    Parameters
    ----------
    psf_name: str
        Either "DoubleGaussian", "Kolmogorov", or "Atmospheric".
        The name is case-insensitive.
    obs_md: lsst.sims.utils.ObservationMetaData
        Metadata associated with the visit, e.g., pointing direction,
        observation time, seeing, etc..
    log_level: str ['WARN']
        Logging level ('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL').
    rng: galsim.BaseDeviate
        Instance of the galsim.baseDeviate random number generator.
    **kwds: **dict
        Additional keyword arguments to pass to the AtmosphericPSF,
        i.e., screen_size(=819.2) and screen_scale(=0.1).

    Returns
    -------
    lsst.sims.GalSimInterface.PSFbase: Instance of a subclass of PSFbase.
    """
    if psf_name.lower() == 'doublegaussian':
        return SNRdocumentPSF(obs_md.OpsimMetaData['FWHMgeom'])

    rawSeeing = obs_md.OpsimMetaData['rawSeeing']

    my_airmass = airmass(obs_md.OpsimMetaData['altitude'])

    if psf_name.lower() == 'kolmogorov':
        psf = Kolmogorov_and_Gaussian_PSF(my_airmass,
                                          rawSeeing=rawSeeing,
                                          band=obs_md.bandpass)
    elif psf_name.lower() == 'atmospheric':
        if rng is None:
            # Use the 'seed' value from the instance catalog for the rng
            # used by the atmospheric PSF.
            rng = galsim.UniformDeviate(obs_md.OpsimMetaData['seed'])
        if 'gaussianFWHM' not in kwds:
            # Retrieve the additional instrumental PSF FWHM from the
            # imSim config file.
            config = get_config()
            kwds['gaussianFWHM'] = config['psf']['gaussianFWHM']
        logger = get_logger(log_level, 'psf')
        psf = AtmosphericPSF(airmass=my_airmass,
                             rawSeeing=rawSeeing,
                             band=obs_md.bandpass,
                             rng=rng,
                             logger=logger, **kwds)
    return psf

def save_psf(psf, outfile):
    """
    Save the psf as a pickle file.
    """
    # Set any logger attribute to None since loggers cannot be persisted.
    if hasattr(psf, 'logger'):
        psf.logger = None
    with open(outfile, 'wb') as output:
        pickle.dump(psf, output)

def load_psf(psf_file, log_level='INFO'):
    """
    Load a psf from a pickle file.
    """
    with open(psf_file, 'rb') as fd:
        psf = pickle.load(fd)

    # Since save_psf sets any logger attribute to None, restore
    # it here.
    if hasattr(psf, 'logger'):
        psf.logger = get_logger(log_level, 'psf')

    return psf

class TracebackDecorator:
    """
    Decorator class for printing exception traceback messages from
    call-back functions executed in a multiprocessing pool subprocess.
    """
    def __init__(self, func):
        """
        Parameters
        ----------
        func: function
            The call-back function to decorate.
        """
        self.func = func

    def __call__(self, *args, **kwds):
        """
        Enclose the underlying function call in a try/except block,
        and print the exception info via `traceback.print_exc()`,
        re-raising the exception.
        """
        try:
            return self.func(*args, **kwds)
        except Exception as eobj:
            traceback.print_exc()
            raise eobj

def get_stack_products(product_names=None):
    """Get the LSST Stack products corresponding to a set of product
    names.

    Parameters
    ----------
    product_names: dict [None]
        A dict with LSST Stack package names as keys and either
        'metapackage' or None as values. The setup eups.Product
        corresponding to each of these packages will be returned in a
        dictionary.  If None, then use the products listed in the
        config file.

    Returns
    -------
    dict of eups.Products keyed by package name.

    """
    config = get_config()
    stack_packages = config['stack_packages'] if product_names is None \
                     else product_names
    eupsenv = eups.Eups()
    products = dict()
    for product_name, product_type in stack_packages.items():
        products[product_name] = eupsenv.getSetupProducts(product_name)[0]
        products[product_name].type = product_type
    return products

def get_version_keywords():
    """
    Return a dictionary of header keywords containing eups tagging and
    version information.
    """
    keywords = {'IMSIMVER': imsim_version}
    products = get_stack_products()
    for iprod, (product_name, product) in enumerate(products.items()):
        keywords[f'PKG{iprod:05d}'] = product_name
        if product.type == 'metapackage':
            tag = [_ for _ in product.tags if _ != 'current'][0]
            keywords[f'TAG{iprod:05d}'] = tag
        else:
            keywords[f'VER{iprod:05d}'] = product.version
    return keywords
