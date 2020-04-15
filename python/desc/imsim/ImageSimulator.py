"""
Code to manage the parallel simulation of sensors using the
multiprocessing module.
"""
import os
import re
import multiprocessing
import warnings
import gzip
import shutil
import tempfile
import sqlite3
import numpy as np
from astropy._erfa import ErfaWarning
import galsim
from lsst.afw.cameraGeom import DetectorType
from lsst.sims.photUtils import BandpassDict
from lsst.sims.GalSimInterface import make_galsim_detector
from lsst.sims.GalSimInterface import make_gs_interpreter
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from .imSim import read_config, parsePhoSimInstanceFile, add_cosmic_rays,\
    add_treering_info, get_logger, TracebackDecorator, get_version_keywords
from .bleed_trails import apply_channel_bleeding
from .skyModel import make_sky_model
from .process_monitor import process_monitor
from .camera_readout import ImageSource
from .atmPSF import AtmosphericPSF

__all__ = ['ImageSimulator', 'compress_files']

# This is global variable is needed since instances of ImageSimulator
# contain references to unpickleable objects in the LSST Stack (e.g.,
# various cameraGeom objects), and the multiprocessing module can only
# execute pickleable functions. The SimulateSensor functor class below
# uses the global IMAGE_SIMULATOR variable to access its
# gs_interpreter objects, and the ImageSimulator.run method sets
# IMAGE_SIMULATOR to self so that it is available in the
# SimulateSensor functors.
IMAGE_SIMULATOR = None

# This global variable is needed since the CheckpointSummary class, of
# which CHECKPOINT_SUMMARY would be an instance, uses an unpickleable
# sqlite db connection to persist the checkpoint info.
CHECKPOINT_SUMMARY = None


class ImageSimulator:
    """
    Class to manage the parallel simulation of sensors using the
    multiprocessing module.
    """
    def __init__(self, instcat, psf, numRows=None, config=None, seed=267,
                 outdir='fits', sensor_list=None, apply_sensor_model=True,
                 create_centroid_file=False, file_id=None, log_level='WARN',
                 ckpt_archive_dir=None):
        """
        Parameters
        ----------
        instcat: str
            The instance catalog for the desired visit.
        psf: lsst.sims.GalSimInterface.PSFbase subclass
            PSF to use for drawing objects.  A single instance is used
            by all of the GalSimInterpreters so that memory for any
            atmospheric screen data can be shared among the processes.
        numRows: int [None]
            The number of rows to read in from the instance catalog.
            If None, then all rows will be read in.
        config: str [None]
            Filename of config file to use.  If None, then the default
            config will be used.
        seed: int [267]
            Random number seed to pass to the GalSimInterpreter objects.
        outdir: str ['fits']
            Output directory to write the FITS images.
        sensor_list: tuple or other container [None]
            The names of sensors (e.g., "R:2,2 S:1,1") to simulate.
            If None, then all sensors in the camera will be
            considered.
        apply_sensor_model: bool [True]
            Flag to apply galsim.SiliconSensor model.
        file_id: str [None]
            string to use for the output files like the checkpoint file.
            If None, then no checkpoint file will be used
        log_level: str ['WARN']
            Logging level ('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL').
        ckpt_archive_dir: str [None]
            If this is not None, then after a sensor-visit has written
            its output FITS file(s), the associated checkpoint file will
            be moved to this directory instead of deleted.  If set to
            `None`, then delete the checkpoint file (assuming the
            `checkpointing.cleanup` config parameter is True).
        """
        self.config = read_config(config)
        self.log_level = log_level
        self.logger = get_logger(self.log_level, name='ImageSimulator')
        self.create_centroid_file = create_centroid_file
        self.psf = psf
        self.outdir = outdir
        self.camera_wrapper = LSSTCameraWrapper()
        if sensor_list is None:
            sensor_list = self._get_all_sensors()
        self.logger.debug("parsing instance catalog for %d sensor(s)",
                          len(sensor_list))
        checkpoint_files = self._gather_checkpoint_files(sensor_list, file_id)
        self.obs_md, self.phot_params, sources \
            = parsePhoSimInstanceFile(instcat, sensor_list, numRows=numRows,
                                      checkpoint_files=checkpoint_files,
                                      log_level=log_level)
        self.gs_obj_dict = sources[1]
        self.apply_sensor_model = apply_sensor_model
        self.file_id = file_id
        self._make_gs_interpreters(seed, sensor_list, file_id)
        self.log_level = log_level
        self.logger = get_logger(self.log_level, name='ImageSimulator')
        self.ckpt_archive_dir = ckpt_archive_dir
        if (self.ckpt_archive_dir is not None and
            not os.path.isdir(self.ckpt_archive_dir)):
            os.makedirs(self.ckpt_archive_dir)

    def _gather_checkpoint_files(self, sensor_list, file_id=None):
        """
        Gather any checkpoint files that have been created for the
        desired sensors and return a dictionary with the discovered
        filenames.
        """
        if file_id is None:
            return None
        checkpoint_files = dict()
        for det_name in sensor_list:
            filename = self.checkpoint_file(file_id, det_name)
            if os.path.isfile(filename):
                checkpoint_files[det_name] = filename
        return checkpoint_files

    def _make_gs_interpreters(self, seed, sensor_list, file_id):
        """
        Create a separate GalSimInterpreter for each sensor so that
        they can be run in parallel and maintain separate checkpoint
        files.

        Also extract GsObjectLists from gs_obj_dict for only the
        sensors in sensor_list so that the memory in the underlying
        InstCatTrimmer object in gs_obj_dict can be recovered.
        """
        bp_dict = BandpassDict.loadTotalBandpassesFromFiles(bandpassNames=self.obs_md.bandpass)
        disable_sky_model = self.config['sky_model']['disable_sky_model']
        bf_strength = self.config['ccd']['bf_strength']
        ckpt_threshold = self.config['checkpointing']['fits_image_thresh']
        noise_and_background \
            = make_sky_model(self.obs_md, self.phot_params, seed=seed,
                             apply_sensor_model=self.apply_sensor_model,
                             disable_sky_model=disable_sky_model)
        self.gs_interpreters = dict()
        for det in self.camera_wrapper.camera:
            det_type = det.getType()
            det_name = det.getName()
            if sensor_list is not None and det_name not in sensor_list:
                continue
            if det_type in (DetectorType.WAVEFRONT, DetectorType.GUIDER):
                continue
            gs_det = make_galsim_detector(self.camera_wrapper, det_name,
                                          self.phot_params, self.obs_md)
            self.gs_interpreters[det_name] \
                = make_gs_interpreter(self.obs_md, [gs_det], bp_dict,
                                      noise_and_background,
                                      epoch=2000.0, seed=seed,
                                      apply_sensor_model=self.apply_sensor_model,
                                      bf_strength=bf_strength,
                                      fits_image_ckpt_threshold=ckpt_threshold)

            self.gs_interpreters[det_name].sky_bg_per_pixel \
                = noise_and_background.sky_counts(det_name)
            self.gs_interpreters[det_name].setPSF(PSF=self.psf)

            if self.apply_sensor_model:
                add_treering_info(self.gs_interpreters[det_name].detectors)

            if file_id is not None:
                self.gs_interpreters[det_name].checkpoint_file \
                    = self.checkpoint_file(file_id, det_name)
                self.gs_interpreters[det_name].nobj_checkpoint \
                    = self.config['checkpointing']['nobj']
                self.gs_interpreters[det_name]\
                    .restore_checkpoint(self.camera_wrapper,
                                        self.phot_params,
                                        self.obs_md)

            if self.create_centroid_file:
                self.gs_interpreters[det_name].centroid_base_name = \
                    os.path.join(self.outdir,
                                 self.config['persistence']['centroid_prefix'])

    def _get_all_sensors(self):
        """Get a list of all of the science sensors."""
        return [det.getName() for det in self.camera_wrapper.camera
                if det.getType() not in (DetectorType.WAVEFRONT, DetectorType.GUIDER)]

    @staticmethod
    def checkpoint_file(file_id, det_name):
        """
        Function to create a checkpoint filename of the form
           checkpoint_<file_id>_Rxx_Syy.ckpt
        from a file_id and the detector name.

        Parameters
        ----------
        file_id: str
            User-supplied ID string to insert in the filename.
        det_name: str
            Detector slot name following DM conventions, e.g., 'R:2,2 S:1,1'.

        Returns
        -------
        str: The checkpoint file name.
        """
        my_detname = "R{}{}_S{}{}".format(*[_ for _ in det_name if _.isdigit()])
        return '-'.join(('checkpoint', file_id, my_detname)) + '.ckpt'

    def output_file(self, det_name, raw=True):
        """
        Generate the path of the output FITS file for either raw or
        eimage files.

        Parameters
        ----------
        det_name: str
            Detector slot name following DM conventions, e.g., 'R:2,2 S:1,1'.
        raw: bool [True]
            Generate a raw filename.

        Returns
        -------
        str: The output file path.
        """
        prefix_key = 'raw_file_prefix' if raw else 'eimage_prefix'
        detector = self.gs_interpreters[det_name].detectors[0]
        prefix = self.config['persistence'][prefix_key]
        visit = str(self.obs_md.OpsimMetaData['obshistID'])
        return os.path.join(self.outdir, prefix + '_'.join(
            (visit, detector.fileName, self.obs_md.bandpass + '.fits')))

    def run(self, processes=1, wait_time=None, node_id=0):
        """
        Use multiprocessing module to simulate sensors in parallel.
        """
        # Set the IMAGE_SIMULATOR variable so that the SimulateSensor
        # instance can use it to access the GalSimInterpreter
        # instances to draw objects.
        global IMAGE_SIMULATOR
        if IMAGE_SIMULATOR is None:
            IMAGE_SIMULATOR = self
        elif id(IMAGE_SIMULATOR) != id(self):
            raise RuntimeError("Attempt to use more than one instance of "
                               "ImageSimulator in the same python interpreter")

        # Set the CHECKPOINT_SUMMARY variable so that the summary info
        # from the subprocesses can be persisted.
        global CHECKPOINT_SUMMARY
        do_checkpoint_summary = self.config['checkpointing']['do_summary']

        if (self.file_id is not None and processes > 1 and
            CHECKPOINT_SUMMARY is None and do_checkpoint_summary):
            db_file = 'ckpt_{}_{}.sqlite3'.format(self.file_id, node_id)
            CHECKPOINT_SUMMARY = CheckpointSummary(db_file=db_file)

        results = []
        if processes == 1:
            # Don't need multiprocessing, so just run serially.
            for det_name in self.gs_interpreters:
                if self._outfiles_exist(det_name):
                    continue
                simulate_sensor = SimulateSensor(det_name, self.log_level)
                results.append(simulate_sensor(self.gs_obj_dict[det_name]))
            return results

        # Use multiprocessing.
        pool = multiprocessing.Pool(processes=processes)
        receivers = []
        if wait_time is not None:
            results.append(pool.apply_async(TracebackDecorator(process_monitor),
                                            (), dict(wait_time=wait_time)))
        for det_name in self.gs_interpreters:
            gs_objects = self.gs_obj_dict[det_name]
            if self._outfiles_exist(det_name) or not gs_objects:
                continue

            # If we are checkpointing, create the connections between
            # the checkpoint_aggregator and the SimulateSensor
            # functors, insert a record into the summary db for the
            # current detector, and add the receiver to the list to
            # pass to the checkpoint_aggregator.
            sender = None
            if CHECKPOINT_SUMMARY is not None:
                receiver, sender = multiprocessing.Pipe(duplex=False)
                CHECKPOINT_SUMMARY.insert_record(det_name, len(gs_objects))
                receivers.append(receiver)

            # Create the function that renders the night sky on
            # the sensor.
            simulate_sensor = SimulateSensor(det_name, self.log_level, sender)

            # Add it to the processing pool.
            results.append(pool.apply_async(TracebackDecorator(simulate_sensor),
                                            (gs_objects,)))
        pool.close()

        if CHECKPOINT_SUMMARY is not None:
            # Create a separate processing pool for the
            # checkpoint_aggregator.
            agg_pool = multiprocessing.Pool(processes=1)
            aggregator \
                = agg_pool.apply_async(TracebackDecorator(checkpoint_aggregator),
                                       (receivers,))
            agg_pool.close()
            agg_pool.join()
            aggregator.get()

        # The simulate_sensor pool must be joined after the aggregator
        # pool so that the checkpoint_aggregator is running when the
        # summary info is sent by the simulate_sensor workers.
        pool.join()
        return [res.get() for res in results]

    def _outfiles_exist(self, det_name):
        """
        Check if requested output files (raw or eimage) exist.  If
        either do not, then return False.  Otherwise, return True.

        However, if the overwrite flag in the config is True, then
        return False so that any existing files are overwritten.
        """
        persist = self.config['persistence']

        if persist['overwrite']:
            return False

        if persist['make_eimage']:
            eimage_file = self.output_file(det_name, raw=False)
            if persist['eimage_compress']:
                eimage_file += '.gz'
            if not os.path.isfile(eimage_file):
                return False

        if persist['make_raw_file']:
            raw_file = self.output_file(det_name, raw=True)
            if not os.path.isfile(raw_file):
                return False

        self.logger.info("%s output files already exists, skipping.", det_name)
        return True


class SimulateSensor:
    """
    Functor class for simulating sensors in parallel using the
    multiprocessing module.  Note that the IMAGE_SIMULATOR variable is
    defined in the global scope.
    """
    def __init__(self, sensor_name, log_level='WARN', sender=None):
        """
        Parameters
        ----------
        sensor_name: str
            The name of the sensor to be simulated, e.g., "R:2,2 S:1,1".
        log_level: str ['WARN']
            Logging level ('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL').
        sender: multiprocessing.connection.Connection
            Sender to the checkpoint_aggregator.
        """
        self.sensor_name = sensor_name
        self.log_level = log_level
        self.sender = sender

    def __call__(self, gs_objects):
        """
        Draw objects using the corresponding GalSimInterpreter.

        Parameters
        ----------
        gs_objects: list of GalSimCelestialObjects
            The list of objects to draw.  This should be restricted to
            the objects for the corresponding sensor.
        """
        if not gs_objects:
            return

        logger = get_logger(self.log_level, name=self.sensor_name)

        # IMAGE_SIMULATOR must be a variable declared in the
        # outer scope and set to an ImageSimulator instance.
        max_flux_simple = IMAGE_SIMULATOR.config['ccd']['max_flux_simple']
        sensor_limit = IMAGE_SIMULATOR.config['ccd']['sensor_limit']
        fft_sb_thresh = IMAGE_SIMULATOR.config['ccd'].get('fft_sb_thresh',None)
        gs_interpreter = IMAGE_SIMULATOR.gs_interpreters[self.sensor_name]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Automatic n_photons',
                                    UserWarning)
            warnings.filterwarnings('ignore', 'ERFA function', ErfaWarning)
            nan_fluxes = 0
            starting_for_loop = True
            for gs_obj in gs_objects:
                if starting_for_loop:
                    logger.info("drawing %d objects", len(gs_objects))
                    starting_for_loop = False
                if gs_obj.uniqueId in gs_interpreter.drawn_objects:
                    continue
                flux = gs_obj.flux(IMAGE_SIMULATOR.obs_md.bandpass)
                if not np.isnan(flux):
                    logger.debug("%s  %s  %s", gs_obj.uniqueId, flux,
                                 gs_obj.galSimType)
                    gs_interpreter.drawObject(gs_obj,
                                              max_flux_simple=max_flux_simple,
                                              sensor_limit=sensor_limit,
                                              fft_sb_thresh=fft_sb_thresh)
                    # Ensure the object's id is added to the drawn
                    # object set.
                    gs_interpreter.drawn_objects.add(gs_obj.uniqueId)
                    self.update_checkpoint_summary(gs_interpreter,
                                                   len(gs_objects))
                else:
                    nan_fluxes += 1
                gs_obj.sed.delete_sed_obj()
            if nan_fluxes > 0:
                logger.info("%s objects had nan fluxes", nan_fluxes)

        add_cosmic_rays(gs_interpreter, IMAGE_SIMULATOR.phot_params)
        full_well = int(IMAGE_SIMULATOR.config['ccd']['full_well'])
        apply_channel_bleeding(gs_interpreter, full_well)

        outdir = IMAGE_SIMULATOR.outdir
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if IMAGE_SIMULATOR.config['persistence']['make_eimage']:
            self.write_eimage_files(gs_interpreter)
        if IMAGE_SIMULATOR.config['persistence']['make_raw_file']:
            self.write_raw_files(gs_interpreter)

        # Write out the centroid files if they were made.
        gs_interpreter.write_centroid_files()

        # The image for the sensor-visit has been drawn, so delete or
        # move to the archive area any existing checkpoint file if the
        # config says perform the cleanup.
        if (gs_interpreter.checkpoint_file is not None
                and os.path.isfile(gs_interpreter.checkpoint_file)
                and IMAGE_SIMULATOR.config['checkpointing']['cleanup']):
            if IMAGE_SIMULATOR.ckpt_archive_dir is None:
                os.remove(gs_interpreter.checkpoint_file)
            else:
                src_file = gs_interpreter.checkpoint_file
                dest_dir = os.path.abspath(IMAGE_SIMULATOR.ckpt_archive_dir)
                dest_file = os.path.join(dest_dir, os.path.basename(src_file))
                with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                                 dir=dest_dir) as tmp:
                    shutil.copy(src_file, tmp.name)
                    os.rename(tmp.name, dest_file)
                os.remove(src_file)
        # Remove reference to gs_interpreter in order to recover the
        # memory associated with that object.
        IMAGE_SIMULATOR.gs_interpreters[self.sensor_name] = None

    def update_checkpoint_summary(self, gs_interpreter, num_objects):
        """
        If the checkpoint file has been updated, send the summary
        information to the checkpoint_aggregator.
        """
        if CHECKPOINT_SUMMARY is None:
            # No checkpointing, so return without sending.
            return
        # Apply the checkpointing criterion used by the gs_interpreter.
        nobjs = len(gs_interpreter.drawn_objects)
        if nobjs % gs_interpreter.nobj_checkpoint == 0:
            self.sender.send((nobjs, self.sensor_name, num_objects,
                              gs_interpreter.nobj_checkpoint))

    def write_raw_files(self, gs_interpreter):
        """
        Write the raw files directly from galsim images.

        Parameters
        ----------
        gs_interpreter: GalSimInterpreter object
        """
        persist = IMAGE_SIMULATOR.config['persistence']
        band = IMAGE_SIMULATOR.obs_md.bandpass
        for detector in gs_interpreter.detectors:
            filename = gs_interpreter._getFileName(detector, band)
            try:
                gs_image = gs_interpreter.detectorImages[filename]
            except KeyError:
                continue
            else:
                raw = ImageSource.create_from_galsim_image(gs_image)
                outfile = IMAGE_SIMULATOR.output_file(detector.name, raw=True)
                added_keywords = dict()
                if isinstance(IMAGE_SIMULATOR.psf, AtmosphericPSF):
                    gaussianFWHM = IMAGE_SIMULATOR.config['psf']['gaussianFWHM']
                    added_keywords['GAUSFWHM'] = gaussianFWHM
                raw.write_fits_file(outfile,
                                    compress=persist['raw_file_compress'],
                                    added_keywords=added_keywords)

    def write_eimage_files(self, gs_interpreter):
        """
        Write the eimage files.

        Parameters
        ----------
        gs_interpreter: GalSimInterpreter object
        """
        # Add version keywords to eimage headers
        version_keywords = get_version_keywords()
        for image in gs_interpreter.detectorImages.values():
            image.header = galsim.FitsHeader(header=version_keywords)

        # Write the eimage files using filenames containing the visit number.
        prefix = IMAGE_SIMULATOR.config['persistence']['eimage_prefix']
        obsHistID = str(IMAGE_SIMULATOR.obs_md.OpsimMetaData['obshistID'])
        nameRoot = os.path.join(IMAGE_SIMULATOR.outdir, prefix) + obsHistID
        outfiles = gs_interpreter.writeImages(nameRoot=nameRoot)
        if IMAGE_SIMULATOR.config['persistence']['eimage_compress']:
            compress_files(outfiles)

def compress_files(file_list, remove_originals=True, compresslevel=1):
    """
    Use gzip to compress a list of files.

    Parameters
    ----------
    file_list: list
        A list of the names of files to compress.
    remove_originals: bool [True]
        Flag to remove original files.
    compresslevel: int [1]
        Compression level for gzip.  1 is fastest, 9 is slowest.

    Notes
    -----
    The compressed files will have a .gz extension added to the
    original filename.
    """
    for infile in file_list:
        outfile = infile + '.gz'
        with open(infile, 'rb') as src, \
             gzip.open(outfile, 'wb', compresslevel) as output:
            shutil.copyfileobj(src, output)
        if remove_originals:
            os.remove(infile)


class CheckpointSummary:
    """
    Class to manage the sqlite3 db file.  Since sqlite3 connection
    objects are not pickleable, this class should live in the global
    namespace.
    """
    def __init__(self, db_file='checkpoint_summary.sqlite',
                 table='summary', overwrite=True):
        """
        Parameters
        ----------
        db_file: str ['checkpoint_summary.sqlite']
            sqlite3 db file to contain the checkpoint summary info.
        table: str ['summary']
            The name of the summary table.
        overwrite: bool [True]
            Flag to overwrite any existing sqlite db file.
        """
        self.db_file = db_file
        self.table = table
        if overwrite and os.path.isfile(db_file):
            os.remove(db_file)
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        sql = """create table if not exists {}
              (detector text primary key, objects_drawn int default 0,
              total_objects int)""".format(self.table)
        self.cursor.execute(sql)
        self.conn.commit()

    def update_record(self, objects_drawn, detector, num_objects):
        """
        Update the number of objects drawn for the given detector.

        Parameters
        ----------
        objects_drawn: int
            The number of objects drawn, i.e., in the checkpoint file.
        detector: str
            The detector name, e.g., "R22_S11".
        num_objects: int
            The expected number of objects to be drawn.
        """
        sql = """update {} set objects_drawn={}, total_objects={}
              where detector='{}'"""\
                  .format(self.table, objects_drawn, num_objects, detector)
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except sqlite3.OperationalError:
            # This will occur if an external process is accessing the
            # db file and a database lock is encountered.
            pass

    def insert_record(self, detector, nobjects):
        """
        Method to insert a new record into the summary table.

        Parameters
        ----------
        detector: str
            The detector name, e.g., "R22_S11".
        nobjects: int
            The number of objects to be drawn.
        """
        sql = """insert into {} (detector, objects_drawn, total_objects)
              values ('{}', 0, {})""".format(self.table, detector, nobjects)
        self.cursor.execute(sql)
        self.conn.commit()


def checkpoint_aggregator(receivers):
    """
    This function receives checkpoint summary info from the
    SimulateSensor class via multiprocessing.Pipe connections and
    writes that info via the global CHECKPOINT_SUMMARY object to
    an sqlite3 db.

    Parameters
    ----------
    receivers: list
        List of receiver connections for each SimulateSensor instance.
    """
    global CHECKPOINT_SUMMARY
    while receivers:
        for receiver in multiprocessing.connection.wait(receivers, timeout=0.1):
            try:
                nobj, det, nmax, nobj_ckpt = receiver.recv()
                if nobj >= nmax - nobj_ckpt:
                    receivers.remove(receiver)
            except EOFError:
                receivers.remove(receiver)
            else:
                CHECKPOINT_SUMMARY.update_record(nobj, det, nmax)
