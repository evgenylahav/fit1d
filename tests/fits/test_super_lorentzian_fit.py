from unittest import TestCase
from fit1d.fits.super_lorentzian_fit import SuperLorentzianFit
from fit1d.common.outlier import OutLierMock


class TestSuperLorentzian(TestCase):
    def setUp(self):
        outlier = OutLierMock
        self.fitter = SuperLorentzianFit(outlier=OutLierMock)

    def test_init(self):
        self.assertIsInstance(self.fitter, SuperLorentzianFit)