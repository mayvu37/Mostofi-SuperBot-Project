# Functions pertaining to classification/ person identification, walking segmentation, and fall detection

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import hermite, factorial
from scipy.signal import stft, get_window, decimate
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestCentroid
from scipy.stats import binned_statistic
import pandas as pd
import os




def classify_feature_vector(training_vectors, training_labels, feature_vector):

    """
    Given a feature vector for one test point, classify who it is based on 
    training feature vector and labels (person identity)
    """
    # classifier computes centroids
    clf = NearestCentroid()
    clf.fit(training_vectors, training_labels)
    predicted_person = clf.predict(feature_vector)

    return predicted_person


def compute_spectral_variance(f, S, f_min=5.0, f_max=60.0):
    if S.ndim != 2:
        raise ValueError(f"Got shape {S.shape}")

    freq_mask = (f >= f_min) & (f <= f_max)
    f_band = f[freq_mask]
    S_band = np.maximum(S[freq_mask, :], 0.0)

    energy_t = np.sum(S_band, axis=0) + 1e-12
    P = S_band / energy_t[np.newaxis, :]

    mu_f = np.sum(f_band[:, np.newaxis] * P, axis=0)
    second_moment = np.sum((f_band[:, np.newaxis]**2) * P, axis=0)

    V = second_moment - mu_f**2
    return np.maximum(V, 0.0)
                                                       
def find_constant_psi_segment(t, torso_speed, V_t, Tmin=3.0,
                              V_percentile_thresh=80,
                              torso_std_thresh=0.12):
    """
    Find the FIRST constant-psi segment using low spectral variance
    and low torso speed variation.
    """
    dt = np.mean(np.diff(t))
    win_len = int(np.ceil(Tmin / dt))

    if win_len < 2:
        raise ValueError("Window length too small")

    # Threshold for acceptable spectral spread
    Vth = np.percentile(V_t, V_percentile_thresh)

    valid_windows = []

    for start in range(0, len(t) - win_len + 1):
        end = start + win_len

        V_win = V_t[start:end]
        torso_win = torso_speed[start:end]

        # Segment-level checks
        V_ok = np.percentile(V_win, 90) <= Vth
        torso_ok = np.nanstd(torso_win) <= torso_std_thresh

        if V_ok and torso_ok:
            valid_windows.append((start, end))

    if not valid_windows:
        return None, None, np.zeros(len(t), dtype=bool)

    # Merge overlapping valid windows
    merged = []
    cur_start, cur_end = valid_windows[0]

    for s, e in valid_windows[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e

    merged.append((cur_start, cur_end))

    # Pick the first merged segment
    best_start, best_end = merged[0]

    mask_best = np.zeros(len(t), dtype=bool)
    mask_best[best_start:best_end] = True

    return t[best_start], t[best_end - 1], mask_best

def compute_spectral_variance(f, S, f_min=30.0, f_max=60.0):
    """
    Compute spectral variance over a selected frequency band
    """
    if S.ndim != 2:
        raise ValueError(f"S must be 2D. Got shape {S.shape}")

    freq_mask = (f >= f_min) & (f <= f_max)
    f_band = f[freq_mask]
    S_band = np.maximum(S[freq_mask, :], 0.0)

    energy_t = np.sum(S_band, axis=0) + 1e-12
    P = S_band / energy_t[np.newaxis, :]

    mu_f = np.sum(f_band[:, np.newaxis] * P, axis=0)
    second_moment = np.sum((f_band[:, np.newaxis] ** 2) * P, axis=0)

    V = second_moment - mu_f ** 2
    return np.maximum(V, 0.0)



def detect_fall_event(S, t, f):
    """
    opposite of walking sementation. A fall is
        - not periodic
        - high variance
        - has energy spike
        - followed by a sharp energy collapse

    need to tune threshold values. get more data. adjust so theyre a weighted score like walking segmentation

    returns a list of times where a fall was detected
    """
    window_size = 20 #about 2 seconds
    periodicity_before_thresh = 0.4
    periodicity_after_thresh  = 0.25

    energy_spike_thresh = 1.8
    variance_spike_thresh = 2.0
    energy_drop_thresh = 0.6
    periodicity_thresh = 0.3

    E = compute_band_energy_curve(S, f, v_min=0.2, v_max=1.5)
    E_smooth = gaussian_filter1d(E, sigma=1.5)
    V = compute_spectral_variance(f, S)

    dE = np.abs(np.diff(E_smooth))
    dV = np.abs(np.diff(V))

    # pad to match length
    dE = np.concatenate([[0], dE])
    dV = np.concatenate([[0], dV])

    dE_thresh = 1.5 * np.mean(dE)
    dV_thresh = 1.5 * np.mean(dV)

    falls = []
    for i in range(10, len(t) - 10):
        Ew = E_smooth[i - window_size:i + window_size]
        periodicity = normalized_autocorr_peak(Ew)

        pre_energy = np.mean(E_smooth[i-10:i])
        post_energy = np.mean(E_smooth[i:i+10])

        # sudden change in energy band curve
        if dE[i] < dE_thresh and dV[i] < dV_thresh:
            continue

        # high variance
        if V[i] < 1.2 * np.mean(V):
            continue

        # periodicity
        if periodicity > periodicity_thresh:
            continue

        # periodicity before (must be walking before)
        E_before = E_smooth[i - 2*window_size : i - window_size]
        p_before = normalized_autocorr_peak(E_before)

        if p_before < periodicity_before_thresh:
            continue

        # periodicity must break after
        E_after = E_smooth[i : i + window_size]
        p_after = normalized_autocorr_peak(E_after)

        if p_after > periodicity_after_thresh:
            continue

        # #  after event, low motion 
        # pre_energy = np.mean(E_smooth[i-10:i])
        # post_energy = np.mean(E_smooth[i:i+10])
        # if post_energy > 0.9 * pre_energy:
        #     continue

        falls.append(t[i])

    return falls, E_smooth, V

def group_falls(falls, gap=1.0, min_dur=0.3, max_dur=1.5):
    """
    Merge nearby detections into one fall event
    returns tuples of fall's start and end timestamps
     """
    if not falls:
        return []
    grouped = []
    current = [falls[0]]
    for f in falls[1:]:
        if f - current[-1] < gap:
            current.append(f)
        else:
            grouped.append((current[0], current[-1]))
            current = [f]

    grouped.append((current[0], current[-1]))
    filtered = []
    for fs, fe in grouped:
        dur = fe - fs
        if dur < min_dur:
            continue  # too short = noise
        if dur > max_dur:
            continue  # too long = not a fall

        filtered.append((fs, fe))

    return filtered
