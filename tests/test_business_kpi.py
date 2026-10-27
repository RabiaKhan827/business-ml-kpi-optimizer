import unittest
import numpy as np
from business_kpi_ml import OperationalKPIOptimizer

class TestBusinessKPI(unittest.TestCase):
    def test_pipeline(self):
        opt = OperationalKPIOptimizer()
        flags = opt.detect_bottlenecks(np.array([4.0, 4.2, 3.9, 4.1, 14.5]))
        self.assertTrue(bool(flags[-1]))
        rep = opt.run_benchmark()
        self.assertAlmostEqual(sum(rep.optimal_allocation_shares.values()), 1.0, places=3)
        self.assertGreater(rep.cost_reduction_pct, 15.0)

if __name__ == "__main__":
    unittest.main()
