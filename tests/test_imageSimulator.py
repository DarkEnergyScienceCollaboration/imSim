"""
Unit tests for the ImageSimulator module.
"""
import os
import random
import string
import unittest
import gzip
import desc.imsim

class ImageSimulatorTestCase(unittest.TestCase):
    """TestCase class for the ImageSimulator code."""

    def setUp(self):
        self.outdir = 'imageSimulator_dir'
        os.makedirs(self.outdir, exist_ok=True)
        self.files = [os.path.join(self.outdir, 'my_file_{}.txt'.format(i))
                      for i in range(5)]
        nchar = 20
        nlines = 10
        for outfile in self.files:
            with open(outfile, 'w') as output:
                for _ in range(nlines):
                    output.write(
                        ''.join(random.choices(string.ascii_letters, k=nchar)))

    def tearDown(self):
        for item in self.files:
            if os.path.isfile(item):
                os.remove(item)
            gzip_file = item + '.gz'
            if os.path.isfile(gzip_file):
                os.remove(gzip_file)
        if os.path.isdir(self.outdir):
            os.rmdir(self.outdir)

    def test_compress_files(self):
        """Unit test for compress_files function."""
        desc.imsim.compress_files(self.files, remove_originals=False)
        for item in self.files:
            with open(item, 'r') as src1, gzip.open(item+'.gz', 'rb') as src2:
                for line1, line2 in zip(src1, src2):
                    self.assertEqual(line1, line2.decode('utf-8'))

    def test_checkpoint_file_name(self):
        """Unit test for checkpoint filename generation."""
        file_id = 'v123456-i'
        det_name = 'R22_S11'
        fn_expected = 'checkpoint-{}-{}.ckpt'.format(file_id, det_name)

        fn = desc.imsim.ImageSimulator.checkpoint_file(file_id, det_name)
        self.assertEqual(fn, fn_expected)


if __name__ == '__main__':
    unittest.main()
