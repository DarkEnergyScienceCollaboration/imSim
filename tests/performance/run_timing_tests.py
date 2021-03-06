"""
Run the timing tests comparing the sky background with bundled and
unbundled photons and for a single star with and without the sensor
model enabled and write the output to files for plotting.
"""
import sys
import desc.imsim
from timing_tests import sky_bg_timing, StarTimer, instcat

seed = 1001

print("sky bg timing tests")
nxy_min = 10
nxy_max = 300
npts = 8
bundled_skymodel \
    = desc.imsim.ESOSiliconSkyModel(instcat.obs_metadata,
                                    instcat.phot_params,
                                    seed=seed, addNoise=False,
                                    addBackground=True)
unbundled_skymodel \
    = desc.imsim.ESOSiliconSkyModel(instcat.obs_metadata,
                                    instcat.phot_params,
                                    seed=seed, addNoise=False,
                                    addBackground=True, bundles_per_pix=1)

fast_sky_bg_timing, fast_nphot = sky_bg_timing(bundled_skymodel,
                                               nxy_min=nxy_min,
                                               nxy_max=nxy_max, npts=npts)
slow_sky_bg_timing, slow_nphot = sky_bg_timing(unbundled_skymodel,
                                               nxy_min=nxy_min,
                                               nxy_max=nxy_max, npts=npts)

with open('sky_bg_timing.txt', 'w') as output:
    for numpix in fast_sky_bg_timing:
        line = '%i  %s  %s  %s  %s' % \
               (numpix, fast_sky_bg_timing[numpix], slow_sky_bg_timing[numpix],
                fast_nphot[numpix], slow_nphot[numpix])
        output.write(line + '\n')
        print(line)
        sys.stdout.flush()

print("single star timing tests")
star_timer = StarTimer(instcat.obs_metadata, seed)
with open('star_timing.txt', 'w') as output:
    for nrecalc in (1000, 3e3, 1e4, 3e4):
        sensor = star_timer.flux_loop_timing(True, nrecalc=nrecalc)
        no_sensor = star_timer.flux_loop_timing(False, nrecalc=nrecalc)
        for flux in sensor:
            line = '%i  %s  %s  %s' % (nrecalc, flux, no_sensor[flux],
                                       sensor[flux])
            output.write(line + '\n')
            print(line)
            sys.stdout.flush()
