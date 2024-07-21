import unittest

from sphere.parameters.functional_params import (ConstantParam,
                                                 StepFunctionParam)


class TestFunctionalParam(unittest.TestCase):
    def test_constant_param(self):
        const_param = ConstantParam(5.0)
        self.assertEqual(const_param.get_current_value(0), 5.0)
        self.assertEqual(const_param.get_current_value(10), 5.0)
        self.assertEqual(const_param.get_current_value(100), 5.0)

    def test_step_beta(self):
        step_beta = StepFunctionParam(values=[0.3, 0.1], period=30)
        self.assertEqual(step_beta.get_current_value(0), 0.3)
        self.assertEqual(step_beta.get_current_value(28), 0.3)
        self.assertEqual(step_beta.get_current_value(29), 0.1)
        self.assertEqual(step_beta.get_current_value(58), 0.1)
        self.assertEqual(step_beta.get_current_value(59), 0.3)

        step_beta = StepFunctionParam(values=[0.1, 0.3], period=30)
        self.assertEqual(step_beta.get_current_value(0), 0.1)
        self.assertEqual(step_beta.get_current_value(28), 0.1)
        self.assertEqual(step_beta.get_current_value(29), 0.3)
        self.assertEqual(step_beta.get_current_value(58), 0.3)
        self.assertEqual(step_beta.get_current_value(59), 0.1)


if __name__ == "__main__":
    unittest.main()
