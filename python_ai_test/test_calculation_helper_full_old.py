
import unittest
import numpy as np
from calculation_helper_full import CalculationHelperBase

class TestCalculationHelperBase(unittest.TestCase):

    def test_calculate_encoder_max_global(self):
        data = [1, 3, 2, 5, 4]
        self.assertEqual(CalculationHelperBase.calculate_encoder_max_global(data), 5)

    def test_find_all_peaks_higher_than_max(self):
        data = [1, 3, 6, 4, 5]
        result = CalculationHelperBase.find_all_peaks_higher_than_max(data, 6)
        self.assertTrue(np.all(result[np.array(data) <= 6 * 0.95] == 0))

    def test_find_local_zero_angle_peaks(self):
        data = [0, 1, 0, 1, 0, 1, 0]
        time = np.linspace(0, 0.01, len(data))
        v_max, t_max = CalculationHelperBase.find_local_zero_angle_peaks(data, time)
        self.assertEqual(len(v_max), len(t_max))

    def test_calculate_delta_t(self):
        t_max = [0.01, 0.02, 0.03]
        self.assertAlmostEqual(CalculationHelperBase.calculate_delta_t(t_max), 0.01)

    def test_calculate_encoder_max(self):
        self.assertEqual(CalculationHelperBase.calculate_encoder_max([1, 2, 3]), 3)

    def test_calculate_time_offset(self):
        t = [0.01, 0.02, 0.03]
        t_max = [0.01]
        result = CalculationHelperBase.calculate_time_offset(t, t_max)
        self.assertTrue(np.allclose(result, [0.0, 0.01, 0.02]))

    def test_calculate_matrix_a(self):
        signal = [1, 2, 3, 4]
        time_offset = np.linspace(0, 1, len(signal))
        matrix_a = CalculationHelperBase.calculate_matrix_a(signal, 1, time_offset)
        self.assertEqual(matrix_a.shape[1], 3)

    def test_calculate_fit(self):
        signal = np.array([1, 2, 3, 4])
        time_offset = np.linspace(0, 1, len(signal))
        matrix_a = CalculationHelperBase.calculate_matrix_a(signal, 1, time_offset)
        fit = CalculationHelperBase.calculate_fit(matrix_a, signal)
        self.assertEqual(len(fit), 3)

    def test_calculate_fit_amp(self):
        self.assertAlmostEqual(CalculationHelperBase.calculate_fit_amp([0, 3, 4]), 5)

    def test_calculate_fit_phase(self):
        result = CalculationHelperBase.calculate_fit_phase([0, 1, 1])
        self.assertTrue(isinstance(result, float))

    def test_calculate_fit_dc(self):
        self.assertEqual(CalculationHelperBase.calculate_fit_dc([10, 2, 3]), 10)

    def test_calculate_fit_freq(self):
        self.assertEqual(CalculationHelperBase.calculate_fit_freq(2), 1)

    def test_calculate_fit_signal(self):
        time_offset = np.linspace(0, 1, 10)
        signal = CalculationHelperBase.calculate_fit_signal(1, 1, 1, time_offset, 0, 0)
        self.assertEqual(len(signal), len(time_offset))

    def test_detect_peaks(self):
        x = [0, 1, 0, 1, 0, 1, 0]
        peaks = CalculationHelperBase.detect_peaks(x, mph=0.5, mpd=1)
        self.assertTrue(np.all(np.array(peaks) >= 0))

if __name__ == "__main__":
    unittest.main()
