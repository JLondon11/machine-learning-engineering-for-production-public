"""Dependency-light reproducibility checks for Chapter 12 drift formula."""
import unittest
import numpy as np
from population_stability_index import population_stability_index as psi
class TestPSI(unittest.TestCase):
    def test_identity_zero(self):
        r=np.random.default_rng(8).normal(size=1500)
        self.assertAlmostEqual(psi(r,r),0.0,places=12)
    def test_shift_positive(self):
        r=np.random.default_rng(8).normal(size=1500)
        self.assertGreater(psi(r,r+2),0.1)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):psi([0,np.nan],[1,2])
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):psi([],[])
if __name__=="__main__":unittest.main()
