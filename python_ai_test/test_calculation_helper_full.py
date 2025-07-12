
import unittest
import numpy as np
from calculation_helper_full import CalculationHelperBase

class TestCalculationHelperBase(unittest.TestCase):

    def test_calculate_encoder_max_global(self):
        self.assertEqual(CalculationHelperBase.calculate_encoder_max_global([1, 2, 3]), 3)
        self.assertEqual(CalculationHelperBase.calculate_encoder_max_global([-1, -5, -3]), -1)

    def test_find_all_peaks_higher_than_max(self):
        original = [1, 2, 3, 10, 2]
        result = CalculationHelperBase.find_all_peaks_higher_than_max(original, 10)
        self.assertEqual(result[3], 10)
        self.assertTrue(np.all(result[np.array(original) <= 9.5] == 0))

    def test_find_local_zero_angle_peaks(self):
        signal = [0, 1, 0, 1, 0, 1, 0]
        time = np.linspace(0, 0.01, len(signal))
        v_max, t_max = CalculationHelperBase.find_local_zero_angle_peaks(signal, time)
        self.assertIsInstance(v_max, list)
        self.assertEqual(len(v_max), len(t_max))

    def test_calculate_delta_t(self):
        self.assertAlmostEqual(CalculationHelperBase.calculate_delta_t([0.1, 0.2, 0.3]), 0.1)
        self.assertTrue(np.isnan(CalculationHelperBase.calculate_delta_t([1])) or CalculationHelperBase.calculate_delta_t([1]) == 0)

    def test_calculate_encoder_max(self):
        self.assertEqual(CalculationHelperBase.calculate_encoder_max([5, 1, 9]), 9)

    def test_calculate_time_offset(self):
        t = [0.3, 0.4, 0.5]
        t_max = [0.3]
        result = CalculationHelperBase.calculate_time_offset(t, t_max)
        self.assertTrue(np.allclose(result, [0, 0.1, 0.2]))

    def test_calculate_matrix_a(self):
        signal = [1, 2, 3, 4]
        time_offset = np.linspace(0, 1, 4)
        a = CalculationHelperBase.calculate_matrix_a(signal, 1, time_offset)
        self.assertEqual(a.shape, (4, 3))

    def test_calculate_fit(self):
        signal = np.array([1, 2, 3, 4])
        time_offset = np.linspace(0, 1, 4)
        matrix_a = CalculationHelperBase.calculate_matrix_a(signal, 1, time_offset)
        fit = CalculationHelperBase.calculate_fit(matrix_a, signal)
        self.assertEqual(len(fit), 3)

    def test_calculate_fit_amp(self):
        self.assertAlmostEqual(CalculationHelperBase.calculate_fit_amp([1, 3, 4]), 5)

    def test_calculate_fit_phase(self):
        phase1 = CalculationHelperBase.calculate_fit_phase([0, 1, 1])
        phase2 = CalculationHelperBase.calculate_fit_phase([0, 1, -1])
        phase3 = CalculationHelperBase.calculate_fit_phase([0, -1, -1])
        self.assertTrue(isinstance(phase1, float))
        self.assertTrue(isinstance(phase2, float))
        self.assertTrue(isinstance(phase3, float))

    def test_calculate_fit_dc(self):
        self.assertEqual(CalculationHelperBase.calculate_fit_dc([9, 2, 3]), 9)

    def test_calculate_fit_freq(self):
        self.assertEqual(CalculationHelperBase.calculate_fit_freq(2), 1)

    def test_calculate_fit_signal(self):
        t = np.linspace(0, 1, 100)
        result = CalculationHelperBase.calculate_fit_signal(1, 1, 2, t, 0.5, 0)
        self.assertEqual(result.shape, t.shape)

    def test_detect_peaks_basic(self):
        x = [0, 1, 0, 2, 0, 3, 0]
        peaks = CalculationHelperBase.detect_peaks(x, mph=1, mpd=1)
        self.assertTrue(all(p >= 0 for p in peaks))

    def test_detect_peaks_with_valley(self):
        x = [0, -1, 0, -2, 0, -3, 0]
        valleys = CalculationHelperBase.detect_peaks(x, mph=-1, mpd=1, valley=True)
        self.assertTrue(len(valleys) > 0)

    def test_detect_peaks_nan_handling(self):
        x = [0, 1, np.nan, 1, 0]
        peaks = CalculationHelperBase.detect_peaks(x, mph=0.5, mpd=1)
        self.assertIsInstance(peaks, np.ndarray)

    def test_detect_peaks_edges(self):
        x = [1, 2, 1, 2, 1]
        rising = CalculationHelperBase.detect_peaks(x, edge='rising')
        falling = CalculationHelperBase.detect_peaks(x, edge='falling')
        both = CalculationHelperBase.detect_peaks(x, edge='both')
        self.assertTrue(len(rising) > 0 or len(falling) > 0 or len(both) > 0)

if __name__ == "__main__":
    unittest.main()
