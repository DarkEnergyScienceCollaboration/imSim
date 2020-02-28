"""
Code to convert an eimage to individual sensor segments, applying
electronics readout effects.

 * Retrieve pixel geometry for each segment
 * Copy imaging segment pixels from eimage
 * Add dark current
 * Add defects (bright defects, dark defects, traps)
 * Apply CTE
 * Apply gain
 * Apply crosstalk
 * Add read noise and bias offset
 * Write FITS file for each amplifier
"""
from __future__ import print_function, absolute_import, division
import os
import warnings
from collections import namedtuple, OrderedDict
import tempfile
import sqlite3
import numpy as np
import scipy
from astropy.io import fits
import astropy.time
import galsim
import lsst.geom as lsst_geom
import lsst.afw.image as afwImage
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
    from lsst.sims.utils import \
        getRotSkyPos, ObservationMetaData, altAzPaFromRaDec
    from lsst.sims.GalSimInterface import LsstObservatory
from .camera_info import CameraInfo, getHourAngle
from .imSim import get_logger, get_config, airmass, get_version_keywords
from .cosmic_rays import CosmicRays

__all__ = ['ImageSource', 'set_itl_bboxes', 'set_e2v_bboxes',
           'set_phosim_bboxes', 'set_noao_keywords', 'cte_matrix']

class ImageSource(object):
    '''
    Class to create single segment images based on the pixel geometry
    described by a Camera object from lsst.afw.cameraGeom.

    Attributes
    ----------
    eimage: astropy.io.fits.HDUList
        The input eimage data.  This is used as a container for both
        the pixel data and image metadata.
    eimage_data: np.array
        The data attribute of the eimage PrimaryHDU.
    exptime: float
        The exposure time of the image in seconds.
    sensor_id: str
        The raft and sensor identifier, e.g., 'R22_S11'.
    amp_images: OrderedDict
        Dictionary of amplifier images.
    camera_info: CameraInfo object
        Object containing the readout properties of the sensors in the
        focal plane, provided by lsst.obs.lsst.imsim.ImsimMapper().camera.
    '''
    def __init__(self, image_array, exptime, sensor_id, visit=42, logger=None):
        """
        Class constructor.

        Parameters
        ----------
        image_array: np.array
            A numpy array containing the pixel data for an eimage.
        exptime: float
            The exposure time of the image in seconds.
        sensor_id: str
            The raft and sensor identifier, e.g., 'R22_S11'.
        visit: int [42]
            Visit number to be used with the sensor_id for generating
            the random seed for the dark current and read noise.
        logger: logging.Logger [None]
            logging.Logger object to use. If None, then a logger with level
            INFO will be used.
        """
        if logger is None:
            self.logger = get_logger('INFO')
        else:
            self.logger = logger

        self.config = get_config()

        self.eimage = fits.HDUList()
        self.eimage.append(fits.PrimaryHDU(image_array))
        self.eimage_data = self.eimage[0].data.transpose()

        self.exptime = exptime
        self.sensor_id = sensor_id
        self.visit = visit

        self._rng = None
        self.seed = None

        self.camera_info = CameraInfo()

        self._make_amp_images()

        self.ratel = 0
        self.dectel = 0
        self.rotangle = 0

    def __del__(self):
        self.eimage.close()

    @staticmethod
    def create_from_galsim_image(gs_image, logger=None):
        """
        Create an ImageSource object from a galsim Image object
        produced by a GalSimInterpreter.
        """
        # Extract the exptime and sensor_id from the
        # sims_GalSimInterface.GalSim_afw_TanSimWCS object.
        wcs = gs_image.wcs
        exptime = wcs.photParams.exptime*wcs.photParams.nexp
        sensor_id = 'R{}{}_S{}{}'\
                    .format(*[x for x in wcs.detectorName if x.isdigit()])

        raw_image = ImageSource(gs_image.array, exptime, sensor_id,
                                visit=wcs.fitsHeader.getScalar('OBSID'))
        raw_image.eimage_data = gs_image.array

        # Add the keywords from wcs.fitsHeader to the raw_image.eimage
        # attribute so that they are propagated to the raw file.
        for name in wcs.fitsHeader.names():
            raw_image.eimage[0].header[name] = wcs.fitsHeader.getScalar(name)
        raw_image._read_pointing_info(None)
        return raw_image

    @staticmethod
    def create_from_eimage(eimage_file, sensor_id=None, opsim_db=None,
                           logger=None):
        """
        Create an ImageSource object from a PhoSim eimage file.

        Parameters
        ----------
        eimage_file: str
           Filename of the eimage FITS file from which the amplifier
           images will be extracted.
        sensor_id: str [None]
            The raft and sensor identifier, e.g., 'R22_S11'.  If None,
            then extract the CHIPID keyword in the primarey HDU.
        opsim_db: str [None]
            OpSim db file to use to find pointing information for each
            visit.  This is needed for older imSim eimage files that
            do not have the RATEL, DECTEL, and ROTANGLE keywords
            set in the FITS header.   If None and if an eimage file without
            those keywords is provided, then a RuntimeError will be raised.
        logger: logging.Logger [None]
            logging.Logger object to use. If None, then a logger with level
            INFO will be used.

        Returns
        -------
        ImageSource object
            An ImageSource object with the pixel data and metadata from
            the eimage file.
        """
        eimage = fits.open(eimage_file)
        exptime = eimage[0].header['EXPTIME']
        if sensor_id is None:
            sensor_id = eimage[0].header['CHIPID']

        image_source = ImageSource(eimage[0].data, exptime, sensor_id,
                                   visit=eimage[0].header['OBSID'],
                                   logger=logger)
        image_source.eimage = eimage
        image_source.eimage_data = eimage[0].data
        image_source._read_pointing_info(opsim_db)
        return image_source

    def _read_pointing_info(self, opsim_db):
        try:
            self.ratel = self.eimage[0].header['RATEL']
            self.dectel = self.eimage[0].header['DECTEL']
            self.rotangle = self.eimage[0].header['ROTANGLE']
            return
        except KeyError:
            if opsim_db is None:
                raise RuntimeError("eimage file does not have pointing info. "
                                   "Need an opsim db file.")
        # Read from the opsim db.
        # We need an ObservationMetaData object to use the getRotSkyPos
        # function.
        obs_gen = ObservationMetaDataGenerator(database=opsim_db,
                                               driver="sqlite")
        obs_md = obs_gen.getObservationMetaData(obsHistID=self.visit,
                                                boundType='circle',
                                                boundLength=0)[0]
        # Extract pointing info from opsim db for desired visit.
        conn = sqlite3.connect(opsim_db)
        query = """select descDitheredRA, descDitheredDec,
        descDitheredRotTelPos from summary where
        obshistid={}""".format(self.visit)
        curs = conn.execute(query)
        ra, dec, rottelpos = [np.degrees(x) for x in curs][0]
        conn.close()
        self.ratel, self.dectel = ra, dec
        obs_md.pointingRA = ra
        obs_md.pointingDec = dec
        self.rotangle = getRotSkyPos(ra, dec, obs_md, rottelpos)

    def get_amp_image(self, amp_info_record, imageFactory=afwImage.ImageI):
        """
        Return an amplifier afwImage.Image object with electronics
        readout effects applied.  This method is only provided so that
        the pixel data and geometry can be displayed using
        lsst.afw.cameraGeom.utils.showAmp.

        Parameters
        ----------
        amp_info_record : lsst.afw.table.tableLib.AmpInfoRecord
            Data structure used by cameraGeom to contain the amplifier
            information such as pixel geometry, gain, noise, etc..
        imageFactory : lsst.afw.image.Image[DFIU], optional
            Image factory to be used for creating the return value.

        Returns
        -------
        lsst.afw.Image[DFIU]
            The image object containing the pixel data.
        """
        amp_name = self.amp_name(amp_info_record)
        float_image = self.amp_images[amp_name]
        if imageFactory == afwImage.ImageF:
            return float_image
        # Return image as the type given by imageFactory.
        output_image = imageFactory(amp_info_record.getRawBBox())
        output_image.getArray()[:] = float_image.getArray()
        return output_image

    def amp_name(self, amp_info):
        """
        The ampifier name derived from a
        lsst.afw.table.ampInfo.ampInfo.AmpInfoRecord.

        Parameters
        ----------
        amp_info: lsst.afw.table.ampInfo.ampInfo.AmpInfoRecord.

        Returns
        -------
        str
             The amplifier name, e.g., "R22_S22_C00".
        """
        return '_'.join((self.sensor_id, amp_info.getName()))

    def _make_amp_images(self):
        """
        Make the amplifier images for all the amps in the sensor.
        """
        config = self.config['electronics_readout']
        self.amp_images = OrderedDict()
        amp_names = self.camera_info.get_amp_names(self.sensor_id)
        for amp_name in amp_names:
            self._make_amp_image(amp_name)
        try:
            disable_crosstalk = config['disable_crosstalk']
        except KeyError:
            disable_crosstalk = False
        if not disable_crosstalk:
            self._apply_crosstalk()
        if config['disable_readnoise_bias_darkcurrent']:
            return
        for amp_name in amp_names:
            self._add_read_noise_and_bias(amp_name)

    def _make_amp_image(self, amp_name):
        """
        Create the segment image for the amplier geometry specified in amp.

        Parameters
        ----------
        amp_name : str
            The amplifier name, e.g., "R22_S11_C00".
        """
        amp_info = self.camera_info.get_amp_info(amp_name)
        bbox = self.camera_info.mosaic_section(amp_info)
        full_segment = afwImage.ImageF(amp_info.getRawBBox())

        # Get the imaging segment (i.e., excluding prescan and
        # overscan regions), and fill with data from the eimage.
        imaging_segment \
            = full_segment.Factory(full_segment, amp_info.getRawDataBBox())
        data = self.eimage_data[bbox.getMinY():bbox.getMaxY()+1,
                                bbox.getMinX():bbox.getMaxX()+1].copy()

        # Apply flips in x and y relative to assembled eimage in order
        # to have the pixels in readout order.
        if amp_info.getRawFlipX():
            data = data[:, ::-1]
        if amp_info.getRawFlipY():
            data = data[::-1, :]
        imaging_segment.getArray()[:] = data

        # Add dark current.
        if (self.exptime > 0 and not self.config['electronics_readout']
            ['disable_readnoise_bias_darkcurrent']):
            dark_current = self.config['electronics_readout']['dark_current']
            imaging_arr = imaging_segment.getArray()
            rng = galsim.PoissonDeviate(seed=self.rng,
                                        mean=dark_current*self.exptime)
            dc_data = np.zeros(np.prod(imaging_arr.shape))
            rng.generate(dc_data)
            imaging_arr += dc_data.reshape(imaging_arr.shape)

        # Add defects.

        # Apply CTE.
        full_arr = full_segment.getArray()
        pcti = self.config['electronics_readout']['pcti']
        pcte_matrix = cte_matrix(full_arr.shape[0], pcti)
        for col in range(0, full_arr.shape[1]):
            full_arr[:, col] = np.dot(pcte_matrix, full_arr[:, col])

        scti = self.config['electronics_readout']['scti']
        scte_matrix = cte_matrix(full_arr.shape[1], scti)
        for row in range(0, full_arr.shape[0]):
            full_arr[row, :] = np.dot(scte_matrix, full_arr[row, :])

        # Convert to ADU.
        full_arr /= amp_info.getGain()

        self.amp_images[amp_name] = full_segment

    def _add_read_noise_and_bias(self, amp_name):
        """
        Add read noise and bias.  This should be done as the final
        step before returning the processed image.

        Parameters
        ----------
        amp_name : str
            The amplifier name, e.g., "R22_S11_C00".
        """
        amp_info = self.camera_info.get_amp_info(amp_name)
        full_arr = self.amp_images[amp_name].getArray()
        rng = galsim.GaussianDeviate(seed=self.rng,
                                     sigma=amp_info.getReadNoise())
        rn_data = np.zeros(np.prod(full_arr.shape))
        rng.generate(rn_data)
        full_arr += rn_data.reshape(full_arr.shape)
        full_arr += self.config['electronics_readout']['bias_level']

    @property
    def rng(self):
        """
        galsim.BaseDeviate seeded with an integer derived from a hash
        of the visit and sensor id.  This is used as the underlying
        random number generator for both the read noise and dark
        current calculations.
        """
        if self._rng is None:
            self.seed = CosmicRays.generate_seed(self.visit, self.sensor_id)
            self._rng = galsim.BaseDeviate(self.seed)
        return self._rng

    def _apply_crosstalk(self):
        """
        Apply intra-CCD crosstalk using the cross-talk matrix
        from obs_lsst.  This should be run only once and
        only after ._make_amp_image has been run for each amplifier.
        """
        if not self.camera_info.det_catalog[self.sensor_id].hasCrosstalk():
            return
        xtalk = self.camera_info.det_catalog[self.sensor_id].getCrosstalk()
        amp_names = self.camera_info.get_amp_names(self.sensor_id)
        imarrs = np.array([self.amp_images[amp_name].getArray()
                           for amp_name in amp_names])
        for amp_index, amp_name, xtalk_row in zip(range(len(amp_names)),
                                                  amp_names, xtalk):
            self.amp_images[amp_name].getArray()[:, :] \
                = (imarrs[amp_index] +
                   sum([x*y for x, y in zip(imarrs, xtalk_row)]))

    def get_amplifier_hdu(self, amp_name, compress=True):
        """
        Get an astropy.io.fits.HDU for the specified amplifier.

        Parameters
        ----------
        amp_name: str
            The amplifier name, e.g., "R22_S11_C00".
        compress: bool [True]
            Use RICE_1 compression.

        Returns
        -------
        astropy.io.fits.ImageHDU
            Image HDU with the pixel data and header keywords
            appropriate for the requested sensor segment.
        """
        data = self.amp_images[amp_name].getArray().astype(np.int32)
        if compress:
            hdu = fits.CompImageHDU(data=data, compression_type='RICE_1')
        else:
            hdu = fits.ImageHDU(data=data)
        hdr = hdu.header
        amp_info = self.camera_info.get_amp_info(amp_name)
        # Copy keywords from eimage primary header.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for key in self.eimage[0].header.keys():
                if key in ('BITPIX', 'NAXIS'):
                    continue
                try:
                    hdr[key] = self.eimage[0].header[key]
                except ValueError:
                    # eimages produced by phosim contain non-ASCII or
                    # non-printable characters resulting in a ValueError.
                    self.logger.warn("ValueError raised while attempting to "
                                     "read %s from eimage header", key)

        # Transpose the WCS matrix elements to account for the use of the
        # Camera Coordinate System in the eimage.  These changes
        # neglect any implied changes in the SIP coefficients.
        channels = '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()
        amp_nums = dict(kv_pair for kv_pair in zip(channels, range(16)))
        amp_num = amp_nums[amp_name[-2:]]
        # These keywords seem to give approximately correct per amp
        # WCS's when viewed with ds9.
        x_pos = (list(range(1, 9)) + list(range(8, 0, -1)))[amp_num]
        hdr['CRPIX1'], hdr['CRPIX2'] \
            = (hdr['CRPIX2'] - amp_info.getRawDataBBox().getWidth()*(8 - x_pos),
               hdr['CRPIX1'])
        if amp_num < 8:
            hdr['CD1_1'], hdr['CD1_2'] = -hdr['CD1_2'], hdr['CD1_1']
            hdr['CD2_1'], hdr['CD2_2'] = -hdr['CD2_2'], hdr['CD2_1']
        else:
            hdr['CD1_1'], hdr['CD1_2'] = -hdr['CD1_2'], -hdr['CD1_1']
            hdr['CD2_1'], hdr['CD2_2'] = -hdr['CD2_2'], -hdr['CD2_1']

        # Set NOAO geometry keywords.
        hdr['DATASEC'] = self._noao_section_keyword(amp_info.getRawDataBBox())
        hdr['DETSEC'] = \
            self._noao_section_keyword(self.camera_info.mosaic_section(amp_info),
                                       flipx=amp_info.getRawFlipX(),
                                       flipy=amp_info.getRawFlipY())
        hdr['GAIN'] = amp_info.getGain()

        return hdu

    def write_fits_file(self, outfile, overwrite=True, run_number=None,
                        lsst_num='LCA-11021_RTM-000', compress=True,
                        image_type='SKYEXP', added_keywords=None):
        """
        Write the processed eimage data as a multi-extension FITS file.

        Parameters
        ----------
        outfile: str
            Name of the output FITS file.
        overwrite: bool [True]
            Flag whether to overwrite an existing output file.
        run_number: int [None]
            Run number.  If None, then the visit number is used.
        compress: bool [True]
            Use RICE_1 compression for each image HDU.
        image_type: str ['SKYEXP']
            Image type to write to the `OBSTYPE` and `IMGTYPE` keywords.
        added_keywords: dict [None]
            Python dict of key/value pairs to add to the primary HDU.
            These will be set just before writing the FITS file, so
            they will override any existing keywords.
        """
        output = fits.HDUList(fits.PrimaryHDU())
        output[0].header = self.eimage[0].header
        # Since `.header` acts like an unordered dict when set like
        # this, we need to re-insert the WCSAXES keyword to precede
        # any other WCS keywords as stipulated in the FITS Standard,
        # version 4.0, section 8.2.
        wcsaxes = output[0].header['WCSAXES']
        del output[0].header['WCSAXES']
        output[0].header.insert(5, ('WCSAXES', wcsaxes, ''))
        if run_number is None:
            run_number = self.visit
        output[0].header['RUNNUM'] = str(run_number)
        output[0].header['DARKTIME'] = output[0].header['EXPTIME']
        output[0].header['TIMESYS'] = 'TAI'
        output[0].header['LSST_NUM'] = lsst_num
        output[0].header['TESTTYPE'] = 'IMSIM'
        output[0].header['IMGTYPE'] = image_type
        output[0].header['OBSTYPE'] = output[0].header['IMGTYPE']
        output[0].header['MONOWL'] = -1
        raft, ccd = output[0].header['CHIPID'].split('_')
        output[0].header['RAFTNAME'] = raft
        output[0].header['SENSNAME'] = ccd
        output[0].header['OUTFILE'] = os.path.basename(outfile)
        # Add boresight pointing angles and rotskypos (angle of sky
        # relative to Camera coordinates) from which obs_lsst can
        # infer the CCD-wide WCS.
        output[0].header['RATEL'] = self.ratel
        output[0].header['DECTEL'] = self.dectel
        output[0].header['ROTANGLE'] = self.rotangle

        # Add various keywords needed for jointcal, including  hour angle
        # and airmass values at the start and end of the observation.
        mjd_obs = output[0].header['MJD-OBS']
        mjd_end = astropy.time.Time(output[0].header['DATE-END'], format='isot',
                                    scale='tai').mjd
        observatory = LsstObservatory()
        output[0].header['HASTART'] \
            = getHourAngle(observatory, mjd_obs, self.ratel)
        output[0].header['HAEND'] \
            = getHourAngle(observatory, mjd_end, self.ratel)

        # Set the airmass from the start of the observation using the
        # opsim db value and compute the airmass from the altitude at
        # the end of the observation.
        output[0].header['AMSTART'] = output[0].header['AIRMASS']
        obs_md = ObservationMetaData(mjd=mjd_end, pointingRA=self.ratel,
                                     pointingDec=self.dectel,
                                     rotSkyPos=self.rotangle)
        alt_end, _, _ = altAzPaFromRaDec(self.ratel, self.dectel, obs_md)
        output[0].header['AMEND'] = airmass(alt_end)

        # Write the added_keywords.
        if added_keywords is not None:
            for key, value in added_keywords.items():
                output[0].header[key] = value

        # Write the seed used by the read noise and dark current.
        output[0].header['RN_SEED'] = self.seed
        # Use seg_ids to write the image extensions in the order
        # specified by LCA-10140.
        seg_ids = '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()
        for seg_id in seg_ids:
            amp_name = '_C'.join((self.sensor_id, seg_id))
            output.append(self.get_amplifier_hdu(amp_name, compress=compress))
            output[-1].header['EXTNAME'] = 'Segment%s' % seg_id

        # Set the imSim version and LSST Stack product versions and
        # tags in the primary HDU.
        output[0].header.update(get_version_keywords())
        self.fits_atomic_write(output, outfile, overwrite=overwrite)

    @staticmethod
    def fits_atomic_write(hdulist, outfile, overwrite=True):
        """
        Perform an atomic write of a FITS file using astropy.io.fits by
        writing to a temporary file then renaming to the final filename.

        Parameters
        ----------
        hdulist: astropy.io.HDUList
            HDU list object to write as a FITS file.
        outfile: str
            Destination filename.
        overwrite: bool [True]
            Flag to overwrite an existing output file.  If False and the
            file already exisits, this function will raise a RuntimeError.
        """
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir='.') as tmp:
            hdulist.writeto(tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.chmod(tmp.name, 0o660)
        if not overwrite and os.path.isfile(outfile):
            raise RuntimeError(
                'File {} exists already. Cannot overwrite.'.format(outfile))
        os.rename(tmp.name, outfile)

    @staticmethod
    def _noao_section_keyword(bbox, flipx=False, flipy=False):
        """
        Convert bounding boxes into NOAO section keywords.

        Parameters
        ----------
        bbox : lsst.geom.Box2I
            Bounding box.
        flipx : bool
            Flag to indicate that data should be flipped in the x-direction.
        flipy : bool
            Flag to indicate that data should be flipped in the y-direction.
        """
        xmin, xmax = bbox.getMinX()+1, bbox.getMaxX()+1
        ymin, ymax = bbox.getMinY()+1, bbox.getMaxY()+1
        if flipx:
            xmin, xmax = xmax, xmin
        if flipy:
            ymin, ymax = ymax, ymin
        return '[%i:%i,%i:%i]' % (xmin, xmax, ymin, ymax)


def set_itl_bboxes(amp):
    """
    Function to apply realistic pixel geometry for ITL sensors.

    Parameters
    ----------
    amp : lsst.afw.table.tableLib.AmpInfoRecord
        Data structure containing the amplifier information such as
        pixel geometry, gain, noise, etc..

    Returns
    -------
    lsst.afw.table.tableLib.AmpInfoRecord
        The updated AmpInfoRecord.
    """
    amp.setRawBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 0),
                                   lsst_geom.Extent2I(544, 2048)))
    amp.setRawDataBBox(lsst_geom.Box2I(lsst_geom.Point2I(3, 0),
                                       lsst_geom.Extent2I(509, 2000)))
    amp.setRawHorizontalOverscanBBox(lsst_geom.Box2I(lsst_geom.Point2I(512, 0),
                                                     lsst_geom.Extent2I(48, 2000)))
    amp.setRawVerticalOverscanBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 2000),
                                                   lsst_geom.Extent2I(544, 48)))
    amp.setRawPrescanBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 0),
                                          lsst_geom.Extent2I(3, 2000)))
    return amp


def set_e2v_bboxes(amp):
    """
    Function to apply realistic pixel geometry for e2v sensors.

    Parameters
    ----------
    amp : lsst.afw.table.tableLib.AmpInfoRecord
        Data structure containing the amplifier information such as
        pixel geometry, gain, noise, etc..

    Returns
    -------
    lsst.afw.table.tableLib.AmpInfoRecord
        The updated AmpInfoRecord.
    """
    amp.setRawBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 0),
                                   lsst_geom.Extent2I(542, 2022)))
    amp.setRawDataBBox(lsst_geom.Box2I(lsst_geom.Point2I(10, 0),
                                       lsst_geom.Extent2I(522, 2002)))
    amp.setRawHorizontalOverscanBBox(lsst_geom.Box2I(lsst_geom.Point2I(522, 0),
                                                     lsst_geom.Extent2I(20, 2002)))
    amp.setRawVerticalOverscanBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 2002),
                                                   lsst_geom.Extent2I(542, 20)))
    amp.setRawPrescanBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 0),
                                          lsst_geom.Extent2I(10, 2002)))
    return amp


def set_phosim_bboxes(amp):
    """
    Function to apply the segmentation.txt geometry.

    Parameters
    ----------
    amp : lsst.afw.table.tableLib.AmpInfoRecord
        Data structure containing the amplifier information such as
        pixel geometry, gain, noise, etc..

    Returns
    -------
    lsst.afw.table.tableLib.AmpInfoRecord
        The updated AmpInfoRecord.
    """
    amp.setRawBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 0),
                                   lsst_geom.Extent2I(519, 2001)))
    amp.setRawDataBBox(lsst_geom.Box2I(lsst_geom.Point2I(4, 1),
                                       lsst_geom.Extent2I(509, 2000)))
    amp.setRawHorizontalOverscanBBox(lsst_geom.Box2I(lsst_geom.Point2I(513, 1),
                                                     lsst_geom.Extent2I(6, 2000)))
    amp.setRawVerticalOverscanBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 2001),
                                                   lsst_geom.Extent2I(519, 0)))
    amp.setRawPrescanBBox(lsst_geom.Box2I(lsst_geom.Point2I(0, 1),
                                          lsst_geom.Extent2I(4, 2000)))
    return amp

PixelParameters = namedtuple('PixelParameters',
                             ('''dimv dimh ccdax ccday ccdpx ccdpy gap_inx
                                 gap_iny gap_outx gap_outy preh'''.split()))
pixel_parameters \
    = {'E2V': PixelParameters(2002, 512, 4004, 4096, 4197, 4200, 28,
                              25, 26.5, 25, 10),
       'ITL': PixelParameters(2000, 509, 4000, 4072, 4198, 4198, 27,
                              27, 26.0, 26, 3)}

# The image extension headers do not include the CCD vendor so we
# infer that from the values of the DETSIZE keyword.
vendors = {'[1:4096,1:4004]': 'E2V',
           '[1:4072,1:4000]': 'ITL'}

def set_noao_keywords(hdu, slot_name):
    """
    Update the image header for one of the readout segments.  Adds raft-level
    coordinates (one set in Camera coordinates and one set rotated so the CCD
    orientation has serial direction horizontal).  Adds amplifier and rotated
    CCD coordinates as well.  See LCA-13501.

    Parameters
    ----------
    hdu : fits.ImageHDU
        FITS image whose header is being updated
    slot_name : str
        Name of the slot within the raft

    Returns
    -------
    fits.ImageHDU : The modified hdu.
    """
    hdu.header['SLOT'] = slot_name

    # Infer the CCD vendor from the DETSIZE keyword.
    try:
        vendor = vendors[hdu.header['DETSIZE']]
    except KeyError:
        raise RuntimeError("DETSIZE not recognized.")

    # Pixel geometries differ between the two CCD vendors
    pixel_pars = pixel_parameters[vendor]

    # Get the segment 'coordinates' from the extension name, e.g., 'Segment00'.
    extname = hdu.header['EXTNAME']
    sx = int(extname[-2])
    sy = int(extname[-1])

    # For convenience of notation in LCA-13501 these are also defined
    # as 'serial' and 'parallel' indices, with Segment = Sp*10 + Ss.
    sp = sx
    ss = sy

    # Extract the x and y location indexes in the raft from the slot
    # name.  Also define the serial and parallel versions.
    cx = int(slot_name[-2])
    cy = int(slot_name[-1])
    cp = cx
    cs = cy

    # Define the WCS and Mosaic keywords.
    hdu.header['WCSNAMEA'] = 'AMPLIFIER'
    hdu.header['CTYPE1A'] = 'Seg_X   '
    hdu.header['CTYPE2A'] = 'Seg_Y   '

    hdu.header['WCSNAMEC'] = 'CCD     '
    hdu.header['CTYPE1C'] = 'CCD_X   '
    hdu.header['CTYPE2C'] = 'CCD_Y   '

    hdu.header['WCSNAMER'] = 'RAFT    '
    hdu.header['CTYPE1R'] = 'RAFT_X  '
    hdu.header['CTYPE2R'] = 'RAFT_Y  '

    hdu.header['WCSNAMEF'] = 'FOCAL_PLANE'

    hdu.header['WCSNAMEB'] = 'CCD_SERPAR'
    hdu.header['CTYPE1B'] = 'CCD_S   '
    hdu.header['CTYPE2B'] = 'CCD_P   '

    hdu.header['WCSNAMEQ'] = 'RAFT_SERPAR'
    hdu.header['CTYPE1Q'] = 'RAFT_S  '
    hdu.header['CTYPE2Q'] = 'RAFT_P  '

    # Keyword values that are common betweem E2V and ITL CCDs.
    hdu.header['PC1_1A'] = 0
    hdu.header['PC1_2A'] = 1 - 2*sx
    hdu.header['PC2_2A'] = 0
    hdu.header['CDELT1A'] = 1
    hdu.header['CDELT2A'] = 1
    hdu.header['CRPIX1A'] = 0
    hdu.header['CRPIX2A'] = 0
    hdu.header['CRVAL1A'] = sx*(pixel_pars.dimv + 1)

    hdu.header['PC1_1C'] = 0
    hdu.header['PC1_2C'] = 1 - 2*sx
    hdu.header['PC2_2C'] = 0
    hdu.header['CDELT1C'] = 1
    hdu.header['CDELT2C'] = 1
    hdu.header['CRPIX1C'] = 0
    hdu.header['CRPIX2C'] = 0
    hdu.header['CRVAL1C'] = sx*(2*pixel_pars.dimv + 1)

    hdu.header['PC1_1R'] = 0
    hdu.header['PC1_2R'] = 1 - 2*sx
    hdu.header['PC2_2R'] = 0
    hdu.header['CDELT1R'] = 1
    hdu.header['CDELT2R'] = 1
    hdu.header['CRPIX1R'] = 0
    hdu.header['CRPIX2R'] = 0
    hdu.header['CRVAL1R'] = (sx*(2*pixel_pars.dimv + 1) + pixel_pars.gap_outx
                             + (pixel_pars.ccdpx - pixel_pars.ccdax)/2.
                             + cx*(2*pixel_pars.dimv + pixel_pars.gap_inx
                                   + pixel_pars.ccdpx - pixel_pars.ccdax))

    hdu.header['PC1_1B'] = 0
    hdu.header['PC1_2B'] = 0
    hdu.header['PC2_2B'] = 1 - 2*sp
    hdu.header['CDELT1B'] = 1
    hdu.header['CDELT2B'] = 1
    hdu.header['CRPIX1B'] = 0
    hdu.header['CRPIX2B'] = 0

    hdu.header['PC1_1Q'] = 0
    hdu.header['PC1_2Q'] = 0
    hdu.header['PC2_2Q'] = 1 - 2*sp
    hdu.header['CDELT1Q'] = 1
    hdu.header['CDELT2Q'] = 1
    hdu.header['CRPIX1Q'] = 0
    hdu.header['CRPIX2Q'] = 0
    hdu.header['CRVAL2Q'] = (sp*(2*pixel_pars.dimv + 1) + pixel_pars.gap_outx
                             + (pixel_pars.ccdpx - pixel_pars.ccdax)/2.
                             + cp*(2*pixel_pars.dimv + pixel_pars.gap_inx
                                   + pixel_pars.ccdpx - pixel_pars.ccdax))

    if vendor == 'ITL':
        hdu.header['PC2_1A'] = -1
        hdu.header['CRVAL2A'] = pixel_pars.dimh + 1 - pixel_pars.preh

        hdu.header['PC2_1C'] = -1
        hdu.header['CRVAL2C'] \
            = pixel_pars.dimh + 1 + sy*pixel_pars.dimh - pixel_pars.preh

        hdu.header['PC2_1R'] = -1
        hdu.header['CRVAL2R'] = (pixel_pars.dimh + 1 + sy*pixel_pars.dimh
                                 + pixel_pars.gap_outy
                                 + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
                                 + cy*(8*pixel_pars.dimh + pixel_pars.gap_iny
                                       + pixel_pars.ccdpy - pixel_pars.ccday)
                                 - pixel_pars.preh)

        hdu.header['PC1_1B'] = -1
        hdu.header['CRVAL1B'] = (ss + 1)*pixel_pars.dimh + 1 - pixel_pars.preh
        hdu.header['CRVAL2B'] = sp*(2*pixel_pars.dimv + 1)

        hdu.header['PC1_1Q'] = -1
        hdu.header['CRVAL1Q'] \
            = (pixel_pars.gap_outy + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
               + cs*(8*pixel_pars.dimh + pixel_pars.gap_iny
                     + pixel_pars.ccdpy - pixel_pars.ccday)
               + (ss + 1)*pixel_pars.dimh + 1 - pixel_pars.preh)

        hdu.header['DTM1_1'] = -1
        hdu.header['DTV1'] \
            = (pixel_pars.dimh + 1) + sy*pixel_pars.dimh + pixel_pars.preh
        hdu.header['DTV2'] = (2*pixel_pars.dimv + 1)*(1 - sx)
    else:
        # vendor == 'E2V'
        hdu.header['PC2_1A'] = 1 - 2*sx
        hdu.header['CRVAL2A'] \
            = sx*(pixel_pars.dimh + 1) + (2*sx - 1)*pixel_pars.preh

        hdu.header['PC2_1C'] = 1 - 2*sx
        hdu.header['CRVAL2C'] = (sx*(pixel_pars.dimh+1) + sy*pixel_pars.dimh
                                 + (2*sx - 1)*pixel_pars.preh)

        hdu.header['PC2_1R'] = 1 - 2*sx
        hdu.header['CRVAL2R'] \
            = (sx*(pixel_pars.dimh + 1) + sy*pixel_pars.dimh
               + pixel_pars.gap_outy
               + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
               + cy*(8*pixel_pars.dimh + pixel_pars.gap_iny + pixel_pars.ccdpy
                     - pixel_pars.ccday) + (2*sx - 1)*pixel_pars.preh)
        hdu.header['PC1_1B'] = 1 - 2*sp
        hdu.header['CRVAL1B'] = (sp*(pixel_pars.dimh + 1) + ss*pixel_pars.dimh
                                 + (2*sp - 1)*pixel_pars.preh)
        hdu.header['CRVAL2B'] = sp*(2*pixel_pars.dimv + 1)

        hdu.header['PC1_1Q'] = 1 - 2*sp
        hdu.header['CRVAL1Q'] \
            = (pixel_pars.gap_outy + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
               + cs*(8*pixel_pars.dimh + pixel_pars.gap_iny
                     + pixel_pars.ccdpy - pixel_pars.ccday)
               + sp*(pixel_pars.dimh + 1) + ss*pixel_pars.dimh
               + (2*sp - 1)*pixel_pars.preh)

        hdu.header['DTM1_1'] = 1 - 2*sx
        hdu.header['DTV1'] = ((pixel_pars.dimh + 1 + 2*pixel_pars.preh)*sx
                              + sy*pixel_pars.dimh - pixel_pars.preh)
        hdu.header['DTV2'] = (2*pixel_pars.dimv + 1)*(1 - sx)

    return hdu


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
