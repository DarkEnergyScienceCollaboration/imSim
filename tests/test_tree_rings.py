"""
Unit tests for tree rings code.  Just tests that the tree ring data was found and could be read.
"""
import os
import unittest
import numpy as np
import lsst.utils as lsstUtils
from lsst.sims.GalSimInterface import make_galsim_detector
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from lsst.sims.GalSimInterface import GalSimInterpreter
import desc.imsim


class TreeRingsTestCase(unittest.TestCase):
    """TestCase class for the tree rings code."""
    def setUp(self):
        self.sensors = ['R22_S11', 'R34_S22']
        self.instcat_file = os.path.join(lsstUtils.getPackageDir('imsim'),
                               'tests', 'tiny_instcat.txt')
        self.rtest = 5280.0 # Just a value to test the radial function at
        self.rvalues = [.0030205, -.0034135] # Expected results
        self.centers = [(-3026.3, -3001.0), (3095.5, -2971.3)] # Input center values

    def test_read_tree_rings(self):
        """Check reading of tree_ring_parameters file"""
        camera_wrapper = LSSTCameraWrapper()
        desc.imsim.read_config()
        needed_stuff = desc.imsim.parsePhoSimInstanceFile(self.instcat_file, ())
        obs_md = needed_stuff.obs_metadata
        phot_params = needed_stuff.phot_params

        detector_list = []
        for sensor in self.sensors:
            detector_list.append(make_galsim_detector(camera_wrapper, sensor, phot_params, obs_md))

        gs_interpreter = GalSimInterpreter(detectors=detector_list)
        tr_filename = os.path.join(lsstUtils.getPackageDir('imsim'),
                                   'data', 'tree_ring_data',
                                   'tree_ring_parameters_19mar18.txt')
        desc.imsim.add_treering_info(gs_interpreter.detectors,
                                     tr_filename=tr_filename)

        for i, detector in enumerate(gs_interpreter.detectors):
            center = detector.tree_rings.center
            shifted_center = (center.x - detector._xCenterPix, center.y - detector._yCenterPix)
            self.assertAlmostEqual(shifted_center, self.centers[i], 1)
            r_value_test = detector.tree_rings.func(self.rtest)
            self.assertAlmostEqual(r_value_test, self.rvalues[i], 6)

if __name__ == '__main__':
    unittest.main()


