import unittest
from abstention_contract import FeatureSnapshot, score_machine

class Feat:
    def __init__(self,snap): self.snap=snap
    def fetch(self,machine): return self.snap
class Model:
    def __init__(self,p): self.p=p
    def predict_probability(self,vec): return self.p

class TestContract(unittest.TestCase):
    def score(self,snap,p=0.9):
        return score_machine("M42","trace42",feature_service=Feat(snap),
                             classifier=Model(p),model_version="baseline")
    def test_stale_abstains_unknown_not_zero(self):
        r=self.score(FeatureSnapshot([1.0],15,"feat-v1"))
        self.assertTrue(r.abstained)
        self.assertIsNone(r.risk_probability)
        self.assertIsNone(r.alert)
    def test_valid_risk(self):
        r=self.score(FeatureSnapshot([1.0],1,"feat-v1"),0.9)
        self.assertFalse(r.abstained)
        self.assertEqual(r.risk_probability,0.9)
        self.assertTrue(r.alert)
    def test_nan_abstains(self):
        r=self.score(FeatureSnapshot([float("nan")],1,"feat-v1"))
        self.assertEqual(r.reasons,("INVALID_FEATURES",))
    def test_invalid_score_abstains(self):
        r=self.score(FeatureSnapshot([1],1,"feat-v1"),float("nan"))
        self.assertEqual(r.reasons,("INVALID_MODEL_OUTPUT",))
    def test_bad_age_abstains(self):
        r=self.score(FeatureSnapshot([1],float("inf"),"feat-v1"))
        self.assertEqual(r.reasons,("INVALID_FEATURE_AGE",))

if __name__=="__main__": unittest.main()
