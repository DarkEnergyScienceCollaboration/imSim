"""
Unit test code for raw file writing.
"""
import os
import unittest
from astropy.io import fits
import lsst.sims.GalSimInterface as sims_gsi
from lsst.sims.photUtils import PhotometricParameters
from lsst.sims.utils import ObservationMetaData
import desc.imsim


class RawFileOutputTestCase(unittest.TestCase):
    """
    TestCase subclass to test direct writing of `raw` files from a
    galsim Image produced by a GalSimInterpreter object.
    """
    def setUp(self):
        self.outfile = None

    def tearDown(self):
        if os.path.isfile(self.outfile):
            os.remove(self.outfile)

    def test_raw_file_writing(self):
        """
        Test the writing of raw files directly from a galsim.Image.
        This is mostly an operational test that the raw files can be
        written from a galsim image.
        """
        camera_wrapper = sims_gsi.LSSTCameraWrapper()
        phot_params = PhotometricParameters()
        obs_md = ObservationMetaData(pointingRA=31.1133844,
                                     pointingDec=-10.0970060,
                                     rotSkyPos=69.0922930,
                                     mjd=59797.2854090,
                                     bandpassName='r')
        obs_md.OpsimMetaData = {'obshistID': 161899,
                                'airmass': desc.imsim.airmass(43.6990272)}
#        detname = "R:2,2 S:1,1"
#        chipid = 'R{}_S{}'.format(detname[2:5:2], detname[8:11:2])
        detname = 'R22_S11'
        chipid = detname
        detector = sims_gsi.make_galsim_detector(camera_wrapper, detname,
                                                 phot_params, obs_md)
        gs_interpreter = sims_gsi.GalSimInterpreter(detectors=[detector])
        gs_image = gs_interpreter.blankImage(detector)
        raw_image = desc.imsim.ImageSource.create_from_galsim_image(gs_image)
        self.outfile = 'lsst_a_{}_r.fits'.format(chipid)

        # Add keyword values via an optional dict.
        added_keywords = {'GAUSFWHM': 0.4,
                          'FOOBAR': 'hello, world'}

        raw_image.write_fits_file(self.outfile, overwrite=True,
                                  added_keywords=added_keywords)

        # Test some keywords.
        with fits.open(self.outfile) as raw_file:
            hdr = raw_file[0].header
            self.assertAlmostEqual(hdr['RATEL'], obs_md.pointingRA)
            self.assertAlmostEqual(hdr['DECTEL'], obs_md.pointingDec)
            self.assertAlmostEqual(hdr['ROTANGLE'], obs_md.rotSkyPos)
            self.assertEqual(hdr['CHIPID'], chipid)

            # Ensure the following keywords are set.
            self.assertNotEqual(hdr['AMSTART'], hdr['AMEND'])
            self.assertNotEqual(hdr['HASTART'], hdr['HAEND'])
            self.assertEqual(hdr['DATE-OBS'], '2022-08-06T06:50:59.338')
            self.assertEqual(hdr['DATE-END'], '2022-08-06T06:51:29.338')
            self.assertEqual(hdr['TIMESYS'], 'TAI')
            self.assertEqual(hdr['OBSTYPE'], 'SKYEXP')

            # Test the added_keywords.
            for key, value in added_keywords.items():
                self.assertEqual(hdr[key], value)


if __name__ == '__main__':
    unittest.main()
