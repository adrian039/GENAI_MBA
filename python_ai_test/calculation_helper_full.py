
import numpy as np
#from scipy import linalg

motor_delay = 0  # Define as needed in your context
dark_dc = 0      # Define as needed or pass appropriately

class CalculationHelperBase(object):
    @staticmethod
    def calculate_encoder_max_global(rot0angle):
        """Calculates the encoder max"""
        return max(rot0angle)

    @staticmethod
    def find_all_peaks_higher_than_max(rot0angle, encoder_max_global):
        """Find all peaks higher than max"""
        rot0angle = np.array(rot0angle)
        index_to_filter = rot0angle <= encoder_max_global * 0.95  # 5 - 10 mV
        rot0angle[index_to_filter] = 0
        return rot0angle

    @staticmethod
    def find_local_zero_angle_peaks(rot0angle, time_rot):
        """Find the local zero-angle peaks"""
        max_to_use = max(rot0angle) * 0.95
        number_of_indexes_apart = np.argmax(np.array(time_rot) > time_rot[0] + 1e-3)
        index_to_use = CalculationHelperBase.detect_peaks(rot0angle, mpd=number_of_indexes_apart, mph=max_to_use)
        t_max = []
        v_max = []
        for i in index_to_use:
            t_max.append(time_rot[i])
            v_max.append(rot0angle[i])
        return v_max, t_max

    @staticmethod
    def calculate_delta_t(t_max):
        """Calculate the average encoder period between peaks"""
        delta_t = np.mean(np.diff(t_max))
        return delta_t

    @staticmethod
    def calculate_encoder_max(v_max):
        """Calculates the encoder max"""
        return max(v_max)

    @staticmethod
    def calculate_time_offset(time_rot, t_max):
        """Calculate the time offset"""
        return np.array(time_rot) - min(t_max)

    @staticmethod
    def calculate_matrix_a(signal, delta_t, time_offset):
        """Calculate the matrix A"""
        column_one = np.ones(len(signal[motor_delay:]))
        column_two = np.cos(2*np.pi/delta_t**2*time_offset[motor_delay:])
        column_three = np.sin(2*np.pi/delta_t**2*time_offset[motor_delay:])
        matrix_a = np.column_stack((column_one, column_two, column_three))
        return matrix_a

    @staticmethod
    def calculate_fit(matrix_a, signal):
        """Calculate the fit"""
        fit = np.linalg.lstsq(matrix_a, signal[motor_delay:])[0]
        return fit

    @staticmethod
    def calculate_fit_amp(fit):
        """Calculate the fit amp"""
        return np.sqrt(fit[1]**2 + fit[2]**2)

    @staticmethod
    def calculate_fit_phase(fit):
        """Calculate the fit phase"""
        fit_phase = np.arctan(fit[1]/fit[2])
        if fit[2] < 0 and (fit[1] < 0 or fit[1] > 0):
            fit_phase += np.pi
        return fit_phase

    @staticmethod
    def calculate_fit_dc(fit):
        """Calculate the fit dc"""
        return fit[0]

    @staticmethod
    def calculate_fit_freq(delta_t):
        """Calculate fit freq"""
        return 2 / delta_t

    @staticmethod
    def calculate_fit_signal(fit_dc, fit_amp, fit_freq, time_offset, dark_dc, fit_phase):
        """Calculate the fit signal"""
        return fit_dc + fit_amp * np.sin(2 * np.pi * fit_freq * time_offset + fit_phase) - dark_dc

    @staticmethod
    def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):
        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        dx = x[1:] - x[:-1]
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        if ind.size and indnan.size:
            ind = ind[~np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))))]
        if ind.size and (ind[0] == 0):
            ind = ind[1:]
        if ind.size and (ind[-1] == x.size - 1):
            ind = ind[:-1]
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])[::-1]]
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)
                    idel[i] = 0
            ind = np.sort(ind[~idel])
        return ind
