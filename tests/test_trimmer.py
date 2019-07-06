"""
Unit tests for InstCatTrimmer class.
"""
import os
import unittest
import desc.imsim


class InstCatTrimmerTestCase(unittest.TestCase):
    """
    TestCase class for InstCatTrimmer.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_InstCatTrimmer(self):
        """Unit test for InstCatTrimmer class."""
        instcat = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                               'tiny_instcat.txt')
        sensor = 'R22_S11'

        # Check the application of minsource.
        objs = desc.imsim.InstCatTrimmer(instcat, [sensor], minsource=10)
        self.assertEqual(len(objs[sensor]), 24)

        objs = desc.imsim.InstCatTrimmer(instcat, [sensor], minsource=12)
        self.assertEqual(len(objs[sensor]), 0)

        # Check various values of chunk_size.
        for chunk_size in (5, 10, 100):
            objs = desc.imsim.InstCatTrimmer(instcat, [sensor], minsource=None,
                                             chunk_size=chunk_size)
            self.assertEqual(len(objs[sensor]), 24)

    def test_inf_filter(self):
        """
        Test filtering of the ` inf ` string (i.e., bracked by spaces)
        appearing anywhere in the instance catalog entry to avoid
        underflows, floating point exceptions, etc.. from badly formed
        entries.
        """
        instcat = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                               'bad_instcat.txt')
        sensor = 'R22_S11'
        objs = desc.imsim.InstCatTrimmer(instcat, [sensor], minsource=10)
        self.assertEqual(len(objs[sensor]), 26)


if __name__ == '__main__':
    unittest.main()
