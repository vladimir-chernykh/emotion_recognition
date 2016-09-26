import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from features import stFeatureExtraction

    
def calculate_features(frames, freq, options):
    window_sec = 0.2
    window_n = int(freq * window_sec)
    use_derivatives = False

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)
    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1
        if use_derivatives:
            deriv_st_f = np.zeros((st_f.shape[0]*3, i1 - i0), dtype=float)
        else:
            deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
            if use_derivatives:
                if st_f.shape[1] >= 2:
                    deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, i - i0] = (st_f[:, i_right] - st_f[:, i_left]) / 2.
                    deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, i - i0] = \
                        st_f[:, i] - 0.5*(st_f[:, i_left] + st_f[:, i_right])
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        if use_derivatives:
            deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, 0] = st_f[:, 1] - st_f[:, 0]
            deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, 0] = np.zeros(st_f.shape[0])
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        if use_derivatives:
            deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, 0] = np.zeros(st_f.shape[0])
            deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, 0] = np.zeros(st_f.shape[0])
        return deriv_st_f
