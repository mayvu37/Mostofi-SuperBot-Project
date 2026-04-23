# Only for feature extraction functions

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


def calculate_movement_speed(S, t, f, percentile=0.5):
    """
    S: spectrogram magnitudes
    t: time vector
    f: frequency vector
    percentile: threshold. percentile >=50 is torso, percentile >= 95 is limb

    returns a vector of the velocity over time
    """
    n_freq, n_time = S.shape
    freq_speed = np.zeros(n_time)
    cumulative_energy = np.cumsum(S, axis = 0)

    #last row = total energy per time
    total_energy = cumulative_energy[-1, :] + 1e-12

    P = cumulative_energy / total_energy[None, :]

    for t_idx in range(n_time):
        idx = np.where(P[:, t_idx] >= percentile)[0]
        if len(idx) > 0:
            freq_speed[t_idx] = f[idx[0]]
        else:
            freq_speed[t_idx] = 0

    lambda_val = 0.06  # meters for 5 GHz WiFi
    velocity = freq_speed * lambda_val / 2
    return velocity

def calculate_torso_contour(S, f_band, gamma=0.015):
    energy = np.sum(S, axis=0) + 1e-12
    energy_ratio = S / energy[None, :]
    freq_tc = np.zeros(S.shape[1])
    for tt in range(S.shape[1]):
        idx = np.where(energy_ratio[:, tt] > gamma)[0]
        freq_tc[tt] = f_band[idx].max() if idx.size > 0 else 0.0
    return freq_tc
    

def estimate_gate_cycle_time(S, f, lambda_ = 0.05, fs=250):
    from scipy.signal import butter, filtfilt, find_peaks
    torso_contour_freq = calculate_torso_contour(S, f)
    velocity_tc = torso_contour_freq * lambda_ / 2
    b,a = butter(2, 2.0 / (fs/2)) # 2hz cuttoff
    velocity_tc = filtfilt(b,a,velocity_tc)

    vtc_max = np.max(velocity_tc)
    steady_index = velocity_tc > 0.8 * vtc_max
    vtc_steady = velocity_tc[steady_index]


    vtc_centered = vtc_steady - np.mean(vtc_steady)
    autocorrelation = np.correlate(vtc_centered, vtc_centered, mode='full')
    lags = np.arange(-len(vtc_centered) + 1, len(vtc_centered))
    autocorrelation = autocorrelation[lags >= 0]
    lags = lags[lags >= 0]
    tau = lags / fs

    peaks, _ = find_peaks(autocorrelation, distance = fs*0.3)
    tau_half = tau[peaks[0]]
    gait_cycle_time = 2 * tau_half
    
    for p in peaks[:5]:
        print("gait cycle time: ", tau[p])
    return gait_cycle_time

def plot_spectrogram_overlay(S, t, f, 
                            freq_tc=None, 
                            torso_speed=None, 
                            limb_speed=None):
                            
    plt.figure(figsize=(9,4.5))
    extent = [t[0], t[-1], f[0], f[-1]]

    lambda_val = 0.06  # meters for 5 GHz WiFi
    v = f * lambda_val / 2

    # Spectrogram
    plt.pcolormesh(t, v, S, shading='gouraud', cmap='jet')
    plt.colorbar(label='Magnitude (linear)')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.ylim([0.3, 2.25])

    plt.title("Feature Extraction Overlay")

    # Overlay feature extraction
    if freq_tc is not None:
        plt.plot(t, freq_tc, color='red', linewidth=1, label="Torso contour frequency")
    if torso_speed is not None:
            plt.plot(t, torso_speed, color='red', linewidth=2, label="Torso speed (50%)")
    if limb_speed is not None:
            plt.plot(t, limb_speed, color='pink', linewidth=1, label="Limb speed (95%)")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spec_and_features(S,t,f,
                           freq_tc = None,
                           torso_speed=None,
                           limb_speed=None):
    v = f * 0.06 / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0 = axes[0]
    pcm = ax0.pcolormesh(t, v, S, shading='gouraud', cmap='jet')
    fig.colorbar(pcm, ax=ax0, label='Magnitude (linear)')
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Velocity (m/s)")
    ax0.set_title("Spectrogram")
    ax0.set_ylim([0.35, 2])

    # ---------------------------
    # RIGHT: Extracted Features
    # ---------------------------
    ax1 = axes[1]

    if freq_tc is not None:
        freq_tc_speed = freq_tc * 0.06 / 2
        ax1.plot(t, freq_tc_speed, linewidth=1, label="Torso contour")

    if torso_speed is not None:
        ax1.plot(t, torso_speed, linewidth=1, label="Torso speed (50%)")

    if limb_speed is not None:
        ax1.plot(t, limb_speed, linewidth=1, label="Limb speed (95%)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Extracted Gait Features")
    ax1.set_ylim([0.3, 2])
    ax1.legend()

    plt.tight_layout()
    plt.show()


def visualize_gait_cycle(torso_contour_freq, lambda_=0.06, fs=250):
    from scipy.signal import butter, filtfilt, find_peaks

    # Convert to velocity
    velocity_tc = torso_contour_freq * lambda_ / 2

    # Low-pass filter (2 Hz)
    b, a = butter(2, 2.0 / (fs/2))
    velocity_tc = filtfilt(b, a, velocity_tc)

    # Keep steady-state region
    vtc_max = np.max(velocity_tc)
    steady_index = velocity_tc > 0.8 * vtc_max
    vtc_steady = velocity_tc[steady_index]

    # Remove mean
    vtc_centered = vtc_steady - np.mean(vtc_steady)

    # Autocorrelation
    autocorrelation = np.correlate(vtc_centered, vtc_centered, mode='full')
    lags = np.arange(-len(vtc_centered) + 1, len(vtc_centered))
    autocorrelation = autocorrelation[lags >= 0]
    lags = lags[lags >= 0]
    tau = lags / fs

    peaks, _ = find_peaks(autocorrelation, distance=fs*0.3)

    tau_half = tau[peaks[0]]
    gait_cycle_time = 2 * tau_half

    # -------------------------
    # Plotting
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Velocity plot
    axes[0].plot(velocity_tc)
    axes[0].set_title("Filtered Torso Velocity")
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("Velocity (m/s)")

    # Autocorrelation plot
    axes[1].plot(tau, autocorrelation)
    axes[1].plot(tau[peaks], autocorrelation[peaks], "ro")
    axes[1].axvline(tau_half, linestyle="--")
    axes[1].set_title("Autocorrelation")
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.show()

    print("Half-cycle time:", tau_half)
    print("Estimated gait cycle time:", gait_cycle_time)

    return gait_cycle_time


def bin_freq_distribution(S, f):
    """
    bins the frequency distribution vector with 30 bins from 0.5 to 2.5 m/s
    """
    lambda_val = 0.06  # meters for 5 GHz WiFi
    v = f * lambda_val / 2
    v_mask = (v >= 0.5) & (v <= 2.5)
    velocity = v[v_mask]
    freq_dist = np.mean(S, axis=1)
    freq_dist = freq_dist[v_mask]


    avg_energy, bin_edges, bin_number = binned_statistic(velocity, freq_dist, statistic='mean', bins=30)
    bin_centers =  (bin_edges[:-1] + bin_edges[1:]) / 2
    return avg_energy


def freq_distribution_gait_phase(S):
    """
    where S is the spectrogram for one half-cycle
    """
    n_freq, n_times = S.shape
    phase_length = n_times // 4
    
    freq_distr_gait_phase = np.zeros((4,n_freq))
    for phase in range(4):
        start = phase * phase_length
        end = start + phase_length

        end = min(end, n_times)

        freq_distr_gait_phase[phase] = np.mean(S[:, start:end], axis=1)

    return freq_distr_gait_phase


def generate_feature_vector(S, t, f):
    """
    S: spectrogram magnitudes during a constant psi or walking pattern
    """
    torso_velocity = calculate_movement_speed(S, t, f, percentile = 0.5)
    torso_avg = np.mean(torso_velocity)
    torso_range = np.max(torso_velocity) - np.min(torso_velocity)

    cycle_time = estimate_gate_cycle_time(S, f)
    stride_length = torso_avg * cycle_time
    
    freq_dist = np.mean(S, axis=1)
    freq_dist_variance = np.var(freq_dist)
    freq_dist_max = np.max(freq_dist)
    freq_binned = bin_freq_distribution(S, f)


    feature_vector = [torso_avg, torso_range, cycle_time, stride_length,
                      freq_dist_variance, freq_dist_max]
    feature_vector.extend(freq_binned.tolist())
    
    print(feature_vector)
    return feature_vector


def update_feature_file(feature_vector, data_name, filename='training_feature_vectors.xlsx'):
    label = input("Input name (press Enter to skip saving): ").strip()
    if label == "":
        print("No label entered. Feature vector not saved.")
        return
    
    row = [label] + list(feature_vector) + [data_name]

    columns = (["Label", "torso_avg", "torso_range", "cycle_time", "stride_length",
                      "freq_dist_variance", "freq_dist_max"]
                + [f"velocity_bin_{i}" for i in range(30)]
                + ["Data Name"])

    df = pd.DataFrame([row], columns=columns)

    if os.path.exists(filename):
        existing = pd.read_excel(filename)
        df = pd.concat([existing, df], ignore_index=True)
    else:
        print("Filepath does not exist")
        return
    df.to_excel(filename, index=False)
    print(f"Feature vector saved for '{label}'.")
