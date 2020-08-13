from astropy.utils import iers
iers.conf.auto_max_age = None
local_iers_file = '/global/homes/j/jchiang8/dev/imSim/data/19-10-30-finals2000A.all'
iers.conf.iers_auto_url = 'file:' + local_iers_file
from .stamp import *
from .instcat import *
from .ccd import *
from .wcs import *
from .treerings import *
from .raw_file import *
from .camera import *
