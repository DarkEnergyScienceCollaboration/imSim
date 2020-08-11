
import os
import gzip
import numpy as np
import astropy
import astropy.coordinates

from contextlib import contextmanager
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, RegisterObjectType
from galsim.config import RegisterSEDType, RegisterBandpassType
from galsim import CelestialCoord
import galsim

# Some helpers to read in a file that might be gzipped.
@contextmanager
def fopen(filename, **kwds):
    """
    Return a file descriptor-like object that closes the underlying
    file descriptor when used with the with-statement.

    Parameters
    ----------
    filename: str
        Filename of the instance catalog.
    **kwds: dict
        Keyword arguments to pass to the gzip.open or open functions.

    Returns
    -------
    generator: file descriptor-like generator object that can be iterated
        over to return the lines in a file.
    """
    abspath = os.path.split(os.path.abspath(filename))[0]
    if not os.path.isfile(filename):
        raise OSError("File not found: %s"%filename)
    try:
        if filename.endswith('.gz'):
            fd = gzip.open(filename, **kwds)
        else:
            fd = open(filename, **kwds)
        yield fopen_generator(fd, abspath, **kwds)
    finally:
        if fd: fd.close()


def fopen_generator(fd, abspath, **kwds):
    """
    Return a generator for the provided file descriptor that knows how
    to recursively read in instance catalogs specified by the
    includeobj directive.
    """
    with fd as input_:
        for line in input_:
            if not line.startswith('includeobj'):
                yield line
            else:
                filename = os.path.join(abspath, line.strip().split()[-1])
                with fopen(filename, **kwds) as my_input:
                    for line in my_input:
                        yield line

class InstCatalog(object):
    """This just handles the objects part of the instance catalog.

    The other "phosim commands" are handled by OpsimMetaDict.
    """
    _bp500 = galsim.Bandpass(galsim.LookupTable([499,500,501],[0,1,0]),
                             wave_type='nm').withZeropoint('AB')

    # Using area-weighted effective aperture over FOV
    # from https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
    _rubin_area = 0.25 * np.pi * 649**2  # cm^2

    def __init__(self, file_name, wcs, sed_dir=None, edge_pix=100, sort_mag=True, flip_g2=True,
                 logger=None):
        logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.flip_g2 = flip_g2
        self._sed_cache = {}

        if sed_dir is None:
            self.sed_dir = os.environ.get('SIMS_SED_LIBRARY_DIR')
        else:
            self.sed_dir = sed_dir

        # Allow objects to be centered somewhat off the image.
        min_x = 0 - edge_pix
        min_y = 0 - edge_pix
        max_x = 4096 + edge_pix  # The image max_x,max_y isn't actually 4096, but close enough.
        max_y = 4096 + edge_pix

        # Check the min/max ra and dec to faster remove objects that cannot be on image
        ll = galsim.PositionD(min_x,min_y)
        lr = galsim.PositionD(min_x,max_y)
        ul = galsim.PositionD(max_x,min_y)
        ur = galsim.PositionD(max_x,max_y)
        ll = wcs.toWorld(ll)
        lr = wcs.toWorld(lr)
        ul = wcs.toWorld(ul)
        ur = wcs.toWorld(ur)
        min_ra = min([ll.ra.deg, lr.ra.deg, ul.ra.deg, ur.ra.deg])
        max_ra = max([ll.ra.deg, lr.ra.deg, ul.ra.deg, ur.ra.deg])
        min_dec = min([ll.dec.deg, lr.dec.deg, ul.dec.deg, ur.dec.deg])
        max_dec = max([ll.dec.deg, lr.dec.deg, ul.dec.deg, ur.dec.deg])
        logger.debug("RA range for image is %f .. %f", min_ra, max_ra)
        logger.debug("Dec range for image is %f .. %f", min_dec, max_dec)

        # What position do the dust parameters start, based on object type.
        dust_index_dict = {
            'point' : 13,
            'sersic2d' : 17,
            'knots' : 17,
        }
        default_dust_index = 15  # For fits files

        id_list = []
        world_pos_list = []
        magnorm_list = []
        sed_list = []
        lens_list = []
        objinfo_list = []
        dust_list = []
        g2_sign = -1 if flip_g2 else 1
        logger.warning('Reading instance catalog %s', self.file_name)
        nuse = 0
        ntot = 0
        with fopen(self.file_name, mode='rt') as _input:
            for line in _input:
                if line.startswith('object'):
                    ntot += 1
                    if ntot % 10000 == 0:
                        logger.info('Using %d of %d objects so far.',nuse,ntot)
                    # Check if the object is on our image
                    tokens = line.strip().split()
                    ra = float(tokens[2])
                    dec = float(tokens[3])
                    #logger.debug('object at %s,%s',ra,dec)
                    if not (min_ra <= ra <= max_ra and min_dec <= dec <= max_dec):
                        continue
                    world_pos = galsim.CelestialCoord(ra * galsim.degrees, dec * galsim.degrees)
                    #logger.debug('world_pos = %s',world_pos)
                    try:
                        image_pos = wcs.toImage(world_pos)
                    except RuntimeError:
                        # Inverse direction can fail for objects off the image.
                        #logger.debug('failed to determine image_pos')
                        continue
                    #logger.debug('image_pos = %s',image_pos)
                    if not (min_x <= image_pos.x <= max_x and min_y <= image_pos.y <= max_y):
                        continue
                    # OK, keep this object.  Finish parsing it.
                    id_list.append(tokens[1])
                    nuse += 1
                    world_pos_list.append(world_pos)
                    magnorm_list.append(float(tokens[4]))
                    sed_list.append((tokens[5], float(tokens[6])))
                    # gamma1,gamma2,kappa
                    lens_list.append((float(tokens[7]), g2_sign*float(tokens[8]), float(tokens[9])))
                    # what are 10,11?
                    dust_index = dust_index_dict.get(tokens[12], default_dust_index)
                    objinfo_list.append(tokens[12:dust_index])
                    dust_list.append(tokens[dust_index:])

        assert nuse == len(id_list)
        logger.warning("Total objects in file = %d",ntot)
        logger.warning("Found %d objects potentially on image", nuse)

        # Sort the object lists by mag and convert to numpy arrays.
        self.id = np.array(id_list, dtype=str)
        self.world_pos = np.array(world_pos_list, dtype=object)
        self.magnorm = np.array(magnorm_list, dtype=float)
        self.sed = np.array(sed_list, dtype=object)
        self.lens = np.array(lens_list, dtype=object)
        self.objinfo = np.array(objinfo_list, dtype=object)
        self.dust = np.array(dust_list, dtype=object)
        if sort_mag:
            index = np.argsort(self.magnorm)
            self.id = self.id[index]
            self.world_pos = self.world_pos[index]
            self.magnorm = self.magnorm[index]
            self.sed = self.sed[index]
            self.lens = self.lens[index]
            self.objinfo = self.objinfo[index]
            self.dust = self.dust[index]
            logger.warning("Sorted objects by magnitude (brightest first).")

    def getNObjects(self, logger=None):
        # Note: This method name is required by the config parser.
        return len(self.id)

    @property
    def nobjects(self):
        return self.getNObjects()

    # Note: Proxies can call methods, but not access attributes, so it's helpful to have
    #       method versions of things like this, which are maybe more obvious using the
    #       attribute directly.  Since input objects such as this are used via proxy in
    #       multiprocessing contexts, we need to keep the method version around.
    def getID(self, index):
        return self.id[index]

    def getWorldPos(self, index):
        return self.world_pos[index]

    def getMagNorm(self, index):
        return self.magnorm[index]

    def getSED(self, index):
        # These require reading in an input file.  So cache the raw (unredshifted versions)
        # to try to minimize how much I/O we'll need for these.
        name, redshift = self.sed[index]
        if name in self._sed_cache:
            sed = self._sed_cache[name]
        else:
            full_name = os.path.join(self.sed_dir, name)
            sed = galsim.SED(full_name, wave_type='nm', flux_type='flambda')
            self._sed_cache[name] = sed

        # TODO: Handle the dust effects.  internal, then redshift, then galactic.
        #       Also probably need to think about the normalization and how that interacts with
        #       the flux and/or mag.
        return sed.atRedshift(redshift)

    def getLens(self, index):
        # The galsim.lens(...) function wants to be passed reduced
        # shears and magnification, so convert the WL parameters as
        # defined in phosim instance catalogs to these values.  See
        # https://github.com/GalSim-developers/GalSim/blob/releases/1.4/doc/GalSim_Quick_Reference.pdf
        # and Hoekstra, 2013, http://lanl.arxiv.org/abs/1312.5981
        gamma1, gamma2, kappa = self.lens[index]
        g1 = gamma1/(1. - kappa)   # real part of reduced shear
        g2 = gamma2/(1. - kappa)   # imaginary part of reduced shear
        mu = 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)) # magnification
        return g1,g2,mu

    def getDust(self, index):
        params = dust[index]
        if params[0].lower() != 'none':
            internal_av = float(params[1])
            internal_rv = float(params[2])
            params = params[3:]
        else:
            internal_av = 0.
            internal_rv = 0.
            params = params[1:]

        if params[0].lower() != 'none':
            galactic_av = float(params[1])
            galactic_rv = float(params[2])
        else:
            galactic_av = 0.
            galactic_rv = 0.

        return internal_av, internal_rv, galactic_av, galactic_rv

    def getObj(self, index, gsparams=None, rng=None, bandpass=None, chromatic=False, exp_time=30):
        params = self.objinfo[index]

        magnorm = self.getMagNorm(index)
        if magnorm >= 50:
            # Mark of invalid object apparently
            return None

        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)

        # Make the object according to the values in the objinfo

        # Note: params here starts at 12, so all indices are 12 less than in imSim code.
        if params[0].lower() == 'point':
            obj = galsim.DeltaFunction(gsparams=gsparams)

        elif params[0].lower() == 'sersic2d':
            a = float(params[1])
            b = float(params[2])
            if b > a:
                # Invalid, but imSim code just let it pass.
                return None
            pa = float(params[3])
            if self.flip_g2:
                # imSim code first did PA = 360 - params[3]
                # Then beta = 90 + PA
                beta = float(90 - pa) * galsim.degrees
            else:
                beta = float(90 + pa) * galsim.degrees

            n = float(params[4])
            # GalSim can amortize some calculations for Sersics, but only if n is the same
            # as a previous galaxy.  So quantize the n values at 0.05.  There's no way anyone
            # cares about this at higher resolution than that.
            # For now, this is not actually helpful, since n is always either 1 or 4, but if
            # we ever start having more variable n, this will prevent it from redoing Hankel
            # integrals for every galaxy.
            n = round(n * 20.) / 20.

            hlr = (a * b)**0.5  # geometric mean of a and b is close to right.
            # XXX: Note: The original imSim code had hlr = a, which is
            # wrong. (?)  Galaxies were too large.
            #      Especially when they were more elliptical.  Oops.
            # TODO: Maybe not?  Check if this should be a.
            obj = galsim.Sersic(n=n, half_light_radius=hlr, gsparams=gsparams)
            obj = obj.shear(q=b/a, beta=beta)
            g1,g2,mu = self.getLens(index)
            obj = obj.lens(g1, g2, mu)

        elif params[0].lower() == 'knots':
            a = float(params[1])
            b = float(params[2])
            if b > a:
                return None
            pa = float(params[3])
            if self.flip_g2:
                beta = float(90 - pa) * galsim.degrees
            else:
                beta = float(90 + pa) * galsim.degrees
            npoints = int(params[4])
            if npoints <= 0:
                # Again, weird, but imSim let this pass without comment.
                return None
            hlr = (a * b)**0.5
            obj = galsim.RandomKnots(npoints=npoints, half_light_radius=hlr, rng=rng,
                                     gsparams=gsparams)
            obj = obj.shear(q=b/a, beta=beta)
            # TODO: These look bad in space images (cf. Troxel's talks about Roman sims.)
            #       Should convolve this by a smallish Gaussian *here*:
            #       I'd guess 0.3 arcsec is a good choice for the fwhm of this Gaussian.
            # obj = galsim.Convolve(obj, galsim.Gaussian(fwhm=0.3))
            g1,g2,mu = self.getLens(index)
            obj = obj.lens(g1, g2, mu)

        elif (params[0].endswith('.fits') or params[0].endswith('.fits.gz')):
            fits_file = find_file_path(params[0], get_image_dirs())
            pixel_scale = float(params[1])
            theta = float(params[2])
            obj = galsim.InterpolatedImage(fits_file, scale=pixel_scale, gsparams=gsparams)
            if theta != 0.:
                obj = obj.rotate(-theta * galsim.degrees)
            g1,g2,mu = self.getLens(index)
            obj = obj.lens(g1, g2, mu)

        else:
            raise RuntimeError("Do not know how to handle object type: %s" % params[0])

        # magnorm is a monochromatic magnitude at 500 nm.
        # So this step normalizes the SED to have the right magnitude.
        # Then below we can calculate the flux for the current bandpass.
        sed = self.getSED(index).withMagnitude(magnorm, self._bp500)

        # This gives the normalization in photons/cm^2/sec.
        # Multiply by area and exptime to get photons.
        At = self._rubin_area * exp_time

        if chromatic:
            return obj.withFlux(At) * sed
        else:
            flux = sed.calculateFlux(bandpass) * At
            return obj.withFlux(flux)

    def getHourAngle(self, mjd, ra):
        """
        Compute the local hour angle of an object for the specified
        MJD and RA.

        Parameters
        ----------
        mjd: float
            Modified Julian Date of the observation.
        ra: float
            Right Ascension (in degrees) of the object.

        Returns
        -------
        float: hour angle in degrees
        """
        # cf. http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon
        lsst_lat = '-30d 14m 40.68s'
        lsst_long = '-70d 44m 57.90s'
        lsst_elev = '2647m'
        lsst_loc = astropy.coordinates.EarthLocation.from_geodetic(
                        lsst_lat, lsst_long, lsst_elev)

        time = astropy.time.Time(mjd, format='mjd', location=lsst_loc)
        # Get the local apparent sidereal time.
        last = time.sidereal_time('apparent').degree
        ha = last - ra
        return ha

class OpsimMetaDict(object):
    """This just handles the meta information at the start of the instance catalog file.

    The objects are handled by InstCatalog.
    """
    _req_params = { 'file_name' : str, }
    _opt_params = {}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, logger=None):
        logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.meta = {}

        logger.warning('Reading instance catalog %s', self.file_name)
        with fopen(self.file_name, mode='rt') as _input:
            for line in _input:
                if line.startswith('#'):  # comments
                    continue
                if line.startswith('object'):
                    # Assumes objects are all at the end.  Is this necessarily true?
                    break

                key, value = line.split()
                logger.debug('meta value: %s = %s',key,value)
                value = float(value)
                if int(value) == value:
                    self.meta[key] = int(value)
                else:
                    self.meta[key] = float(value)

        logger.warning("Done reading meta information from instance catalog")

        # Add a couple derived quantities to meta values
        self.meta['bandpass'] = 'ugrizy'[self.meta['filter']]
        self.meta['HA'] = self.getHourAngle(self.meta['mjd'], self.meta['rightascension'])
        logger.debug("Bandpass = %s",self.meta['bandpass'])
        logger.debug("HA = %s",self.meta['HA'])

    def get(self, field):
        if field not in self.meta:
            raise ValueError("OpsimMeta field %s not present in instance catalog"%field)
        return self.meta[field]

    def getHourAngle(self, mjd, ra):
        """
        Compute the local hour angle of an object for the specified
        MJD and RA.

        Parameters
        ----------
        mjd: float
            Modified Julian Date of the observation.
        ra: float
            Right Ascension (in degrees) of the object.

        Returns
        -------
        float: hour angle in degrees
        """
        # cf. http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon
        lsst_lat = '-30d 14m 40.68s'
        lsst_long = '-70d 44m 57.90s'
        lsst_elev = '2647m'
        lsst_loc = astropy.coordinates.EarthLocation.from_geodetic(
                        lsst_lat, lsst_long, lsst_elev)

        time = astropy.time.Time(mjd, format='mjd', location=lsst_loc)
        # Get the local apparent sidereal time.
        last = time.sidereal_time('apparent').degree
        ha = last - ra
        return ha

def OpsimMeta(config, base, value_type):
    """Return one of the meta values stored in the instance catalog.
    """
    meta = galsim.config.GetInputObj('opsim_meta_dict', config, base, 'OpsimMeta')

    req = { 'field' : str }
    opt = { 'num' : int }  # num, if present, was used by GetInputObj
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    field = kwargs['field']

    val = value_type(meta.get(field))
    return val, safe

def InstCatObj(config, base, ignore, gsparams, logger):
    """Build an object according to info in instance catalog.
    """
    inst = galsim.config.GetInputObj('instance_catalog', config, base, 'InstCat')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, inst.getNObjects())

    req = { 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs['index']

    rng = galsim.config.GetRNG(config, base, logger, 'InstCatObj')
    bp = base['bandpass']
    exp_time = base.get('exp_time',None)

    obj = inst.getObj(index, gsparams=gsparams, rng=rng, bandpass=bp, exp_time=exp_time)
    return obj, safe

def InstCatWorldPos(config, base, value_type):
    """Return a value from the object part of the instance catalog
    """
    inst = galsim.config.GetInputObj('instance_catalog', config, base, 'InstCatWorldPos')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, inst.getNObjects())

    req = { 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs['index']

    pos = inst.getWorldPos(index)
    return pos, safe

class InstCatSEDBuilder(galsim.config.SEDBuilder):
    """A class for loading an SED from the instance catalog.
    """
    def buildSED(self, config, base, logger):
        """Build the SED based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the SED type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed SED object.
        """
        inst = galsim.config.GetInputObj('instance_catalog', config, base, 'InstCatWorldPos')

        galsim.config.SetDefaultIndex(config, inst.getNObjects())

        req = { 'index' : int }
        opt = { 'num' : int }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        index = kwargs['index']
        sed = inst.getSED(index)
        return sed, safe

class OpsimMetaBandpass(galsim.config.BandpassBuilder):
    """A class for loading a Bandpass for a given instcat
    """
    def buildBandpass(self, config, base, logger):
        """Build the Bandpass object based on the LSST filter name.

        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Bandpass object.
        """
        meta = galsim.config.GetInputObj('opsim_meta_dict', config, base, 'InstCatWorldPos')

        # Note: imSim used the lsst.sims versions of these.  Here we just use the ones in the 
        #       GalSim share directory.  Not sure whether those are current, but probably
        #       good enough for now.
        bp = meta.get('bandpass')
        bandpass = galsim.Bandpass('LSST_%s.dat'%bp, wave_type='nm')
        bandpass = bandpass.withZeropoint('AB')
        logger.debug('bandpass = %s',bandpass)
        return bandpass, False

# The basic InputLoader almost works.  Just need to handle the wcs.
class InstCatalogLoader(InputLoader):
    def getKwargs(self, config, base, logger):
        import galsim
        req = {
                'file_name' : str,
              }
        opt = {
                'sed_dir' : str,
                'edge_pix' : float,
                'sort_mag' : bool,
                'flip_g2' : bool,
              }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['logger'] = galsim.config.GetLoggerProxy(logger)
        return kwargs, safe

RegisterInputType('opsim_meta_dict', InputLoader(OpsimMetaDict, file_scope=True, takes_logger=True))
RegisterValueType('OpsimMeta', OpsimMeta, [float, int, str], input_type='opsim_meta_dict')
RegisterBandpassType('OpsimMetaBandpass', OpsimMetaBandpass(), input_type='opsim_meta_dict')

RegisterInputType('instance_catalog', InstCatalogLoader(InstCatalog, has_nobj=True))
RegisterValueType('InstCatWorldPos', InstCatWorldPos, [CelestialCoord],
                  input_type='instance_catalog')
RegisterObjectType('InstCatObj', InstCatObj, input_type='instance_catalog')
RegisterSEDType('InstCatSED', InstCatSEDBuilder(), input_type='instance_catalog')
