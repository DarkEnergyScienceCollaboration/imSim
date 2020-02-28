#!/usr/bin/env python
"""
This is the imSim program, used to drive GalSim to simulate the LSST.  Written
for the DESC collaboration and LSST project.  This version of the program can
read phoSim instance files as is.  It leverages the LSST Sims GalSim interface
code found in sims_GalSimInterface.
"""
import os
import argparse
import warnings
from astropy._erfa import ErfaWarning
import desc.imsim

parser = argparse.ArgumentParser()
parser.add_argument('instcat', help="The instance catalog")
parser.add_argument('-n', '--numrows', default=None, type=int,
                    help="Read the first numrows of the file.")
parser.add_argument('--outdir', type=str, default='fits',
                    help='Output directory for eimage file')
parser.add_argument('--sensors', type=str, default=None,
                    help='Sensors to simulate, e.g., '
                    '"R:2,2 S:1,1^R:2,2 S:1,0". '
                    'If None, then simulate all sensors with sources on them')
parser.add_argument('--config_file', type=str, default=None,
                    help="Config file. If None, the default config will be used.")
parser.add_argument('--log_level', type=str,
                    choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
                    default='INFO', help='Logging level. Default: INFO')
parser.add_argument('--psf', type=str, default='Kolmogorov',
                    choices=['DoubleGaussian', 'Kolmogorov', 'Atmospheric'],
                    help="PSF model to use.  Default: Kolmogorov")
parser.add_argument('--disable_sensor_model', default=False,
                    action='store_true',
                    help='disable sensor effects')
parser.add_argument('--file_id', type=str, default=None,
                    help='ID string to use for checkpoint filenames.')
parser.add_argument('--create_centroid_file', default=False, action="store_true",
                    help='Write centroid file(s).')
parser.add_argument('--seed', type=int, default=267,
                    help='integer used to seed random number generator')
parser.add_argument('--processes', type=int, default=1,
                    help='number of processes to use in multiprocessing mode')
parser.add_argument('--psf_file', type=str, default=None,
                    help="Pickle file containing for the persisted PSF. "
                    "If the file exists, the psf will be loaded from that "
                    "file, ignoring the --psf option; "
                    "if not, a PSF will be created and saved to that filename.")
parser.add_argument('--image_path', type=str, default=None,
                    help="search path for FITS postage stamp images."
                    "This will be prepended to any existing IMSIM_IMAGE_PATH "
                    "environment variable, for which $CWD is included by "
                    "default.")
parser.add_argument('--ckpt_archive_dir', type=str, default=None,
                    help="Archive directory for checkpoint files. "
                    "If None, then delete them (if the checkpointing.cleanup "
                    "configuration is True).")

args = parser.parse_args()

# Read in the config
config = desc.imsim.read_config(args.config_file)

# Prepend any additional paths to IMSIM_IMAGE_PATH.
if args.image_path is not None:
    os.environ['IMSIM_IMAGE_PATH']\
        = ':'.join([args.image_path] + desc.imsim.get_image_dirs())

commands = desc.imsim.metadata_from_file(args.instcat)

obs_md = desc.imsim.phosim_obs_metadata(commands)

if args.psf_file is None or not os.path.isfile(args.psf_file):
    try:
        nproc = config['psf']['nproc']
    except KeyError:
        nproc = None
    psf = desc.imsim.make_psf(args.psf, obs_md, log_level=args.log_level,
                              nproc=nproc)
    if args.psf_file is not None:
        desc.imsim.save_psf(psf, args.psf_file)
else:
    psf = desc.imsim.load_psf(args.psf_file, log_level=args.log_level)

sensor_list = args.sensors.split('^') if args.sensors is not None \
    else args.sensors

apply_sensor_model = not args.disable_sensor_model

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'ERFA', ErfaWarning)
    image_simulator \
        = desc.imsim.ImageSimulator(args.instcat, psf,
                                    numRows=args.numrows,
                                    config=args.config_file,
                                    seed=args.seed,
                                    outdir=args.outdir,
                                    sensor_list=sensor_list,
                                    apply_sensor_model=apply_sensor_model,
                                    create_centroid_file=args.create_centroid_file,
                                    file_id=args.file_id,
                                    log_level=args.log_level,
                                    ckpt_archive_dir=args.ckpt_archive_dir)

    image_simulator.run(processes=args.processes)
