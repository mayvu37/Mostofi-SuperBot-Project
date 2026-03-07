import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import hermite, factorial
from scipy.signal import stft, get_window, decimate
from scipy.ndimage import gaussian_filter1d


def compute_STFT(signal_in, fs, T_win=0.4):
    """
    Computes STFT for each column of CSI magnitude.

    Parameters:
        signal_in : np.ndarray
            2D CSI data (time x subcarriers) or 1D array (single time series).
        fs : float
            Sampling rate (Hz)
        T_win : float
            Window length (seconds)
        T_overlap : float
            Overlap length (seconds)
    Returns:
        freq : list of arrays
            Frequency arrays for each column
        time : list of arrays
            Time arrays for each column
        mag : list of 2D arrays
            STFT magnitude arrays for each column
    """
    # Determine if the original input was a single time series (1D or 2D with one column)
    is_single_time_series = (signal_in.ndim == 1) or (signal_in.ndim == 2 and signal_in.shape[1] == 1)

    # Ensure signal_in is always (N_samples, N_components) for consistent processing
    if signal_in.ndim == 1:
        signal_in_processed = signal_in.reshape(-1, 1)
    elif signal_in.ndim == 2 and signal_in.shape[0] == 1 and signal_in.shape[1] > 1:
        # If it's a row vector (1, N), transpose to (N, 1) to treat as N_samples, 1_component
        signal_in_processed = signal_in.T
    else:
        signal_in_processed = signal_in

    T_overlap = T_win * 0.99

    nperseg = int(T_win * fs)
    noverlap = int(T_overlap * fs)
    std = nperseg / 8
    window = signal.windows.gaussian(nperseg,std=std)
    # Initialize as empty lists
    freq = []
    time = []
    mag = []

    for i in range(signal_in_processed.shape[1]):
        # Compute STFT for this subcarrier/component
        f, t, Zxx = signal.stft( #check overlap vs how much you step
            signal_in_processed[:, i],
            fs=fs,
            window="hann", #gaussian
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            return_onesided=True,
            nfft=1024 # for better resolution of frequency bins
        )
        # Append to lists instead of indexing
        freq.append(f)
        time.append(t)
        mag.append(np.abs(Zxx))

    # Return based on whether the original input was a single time series
    # f and t are compressed because they stay the same for all inputs as they are fucntions of fs Twin and t_overlap
    if is_single_time_series:
        return np.array(freq[0]), np.array(time[0]), np.array(mag[0])
    else:
        return np.array(freq[0]), np.array(time[0]), np.array(mag)


def plot_spectrogram(f, t, S,
                     v_min=0.3,
                     v_max=2.25,
                     title="Spectrogram",
                     cmap='jet',
                     dB=True,
                     figsize=(14,6)):

    """
    Plots a spectrogram given frequency, time, and magnitude values.

    Parameters:
        f : 1D array
            Frequencies (Hz)
        t : 1D array
            Time stamps (s)
        S : 2D array
            Magnitude or power matrix with shape (len(f), len(t))
        title : str
            Plot title
        f_max : float
            Maximum frequency to show (Hz)
        cmap : str
            Matplotlib colormap
        dB : bool
            If True, convert magnitude to dB
        vmin_percentile : float
            Lower percentile for dynamic range scaling
        vmax_percentile : float
            Upper percentile for dynamic range scaling
        figsize : tuple
            Figure size
    """

    # Convert to dB if needed
    if dB:
        S_plot = 10 * np.log10(S + 1e-12)
    else:
        S_plot = S

    # convert to frequency to velocity
    lambda_val = 0.06  # meters for 5 GHz WiFi
    v = f * lambda_val / 2

    # Dynamic range scaling (prevents overly bright outliers)
    #vmin = np.percentile(S_plot, vmin_percentile)
    #vmax = np.percentile(S_plot, vmax_percentile)

    # Plot
    plt.figure(figsize=figsize)
    plt.pcolormesh(t, v, S_plot,
                   shading='gouraud',
                   cmap=cmap)

    plt.title(title, fontsize=14)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Velocity [m/s]", fontsize=12)

    # Limit frequency axis
    if v_max is not None:
        plt.ylim([v_min, v_max])
        # plt.xlim([0, 10])

    cbar = plt.colorbar()
    cbar.set_label("Power (dB)" if dB else "Magnitude", fontsize=11)

    plt.tight_layout()
    plt.show()

def compute_pca_components(signal_clean, n_components=20, skip_first=1, whiten=False):
    """
    Compute PCA on CSI data and project onto principal components.

    Parameters:
    -----------
    signal_clean : ndarray
        Preprocessed CSI data, shape (T, N) where:
        - T = number of time samples
        - N = number of subcarriers/streams
    n_components : int
        Number of principal components to extract (default: 80)
    skip_first : int
        Number of first PCs to skip (default: 1, skips PC1 which contains noise)
    whiten : bool
        If True, apply whitening to normalize variance across PCs (default: False)
        [TODO]: delete variable entirely
    Returns:
    --------
    X_proj : ndarray
        Projected data onto selected PCs, shape (T, n_components)
        If whiten=True, this is whitened; otherwise raw projections
    eigvals : ndarray
        All eigenvalues sorted descending, shape (N,)
    eigvecs : ndarray
        All eigenvectors sorted by eigenvalue, shape (N, N)
    """

    # Step 1: Compute covariance matrix
    T, N = signal_clean.shape
    Cov = (signal_clean.T @ signal_clean) / (T - 1)

    # Step 2: Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(Cov)

    # Step 3: Sort eigenvalues (descending) and reorder eigenvectors
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]  # columns are eigenvectors

    # Step 4: Select PCs from skip_first to (skip_first + n_components)
    V_k = eigvecs[:, skip_first:skip_first + n_components]  # shape (N, n_components)

    # Step 5: Project data onto selected PCs
    X_proj = signal_clean @ V_k  # shape (T, n_components)

    # Step 6: Optional whitening
    if whiten:
        # Get eigenvalues corresponding to selected PCs
        lambda_k = eigvals[skip_first:skip_first + n_components]  # shape (n_components,)
        whitening_scale = 1.0 / np.sqrt(lambda_k)  # shape (n_components,)

        # Apply whitening (broadcast over time dimension)
        X_proj = X_proj * whitening_scale[np.newaxis, :]

    # Print diagnostic info
    total_variance = np.sum(eigvals)
    selected_variance = np.sum(eigvals[skip_first:skip_first + n_components])
    print(f"PCA Info:")
    print(f"  Input shape: {signal_clean.shape}")
    print(f"  Using PCs {skip_first+1} to {skip_first+n_components}")
    print(f"  Variance explained: {selected_variance/total_variance*100:.2f}%")
    print(f"  Whitening: {'ON' if whiten else 'OFF'}")
    print(f"  Output shape: {X_proj.shape}")

    return X_proj, eigvals, eigvecs

def adaptive_noise_floor_per_pc(S, f_band, f1, f2):
    """
    Adaptive noise floor removal for spectrograms.

    Parameters:
    -----------
    S : ndarray
        Spectrogram(s) with shape:
        - (n_freqs, n_times) for single spectrogram, OR
        - (n_pcs, n_freqs, n_times) for multiple PC spectrograms
    f_band : ndarray
        Frequency values, shape (n_freqs,)
    noise_threshold : float
        Frequency (Hz) above which to estimate noise floor

    Returns:
    --------
    S_clean : ndarray
        Cleaned spectrogram, same shape as input S
    """
    # Handle both 2D and 3D inputs
    if S.ndim == 2:
        # Single spectrogram: (n_freqs, n_times)
        # Add dummy dimension to treat as single PC
        S = S[np.newaxis, :, :]
        squeeze_output = True
    elif S.ndim == 3:
        # Multiple PC spectrograms: (n_pcs, n_freqs, n_times)
        squeeze_output = False
    else:
        raise ValueError(f"S must be 2D or 3D, got shape {S.shape}")

    n_pcs, n_freqs, n_times = S.shape
    S_clean = np.zeros_like(S)

    # Select frequencies in the noise band f1 → f2
    noise_mask = (f_band >= f1) & (f_band <= f2)

    for pc in range(n_pcs):
        # Extract this PC's spectrogram
        S_pc = S[pc]  # shape (n_freqs, n_times)

        # Noise region for this PC
        noise_region = S_pc[noise_mask, :]

        # Compute noise floor per time bin
        noise_floor = np.mean(noise_region, axis=0)  # shape (n_times,)

        # Subtract noise floor
        S_clean_pc = S_pc - noise_floor[np.newaxis, :]

        # Clip negative values
        S_clean_pc = np.maximum(S_clean_pc, 0)

        # Store result
        S_clean[pc] = S_clean_pc

    # Remove dummy dimension if input was 2D
    if squeeze_output:
        S_clean = S_clean[0]

    return S_clean

def normalize_by_sum_per_time(mag_pca):
    """
    Normalize by sum of all frequencies per time step (paper's method).

    Parameters:
        mag_pca: (n_pcs, n_freqs, n_times) array of PC spectrograms

    Returns:
        mag_normalized: Same shape, normalized per time step
    """
    n_pcs, n_freqs, n_times = mag_pca.shape
    mag_normalized = np.zeros_like(mag_pca)

    for pc in range(n_pcs):
        # Sum over all frequencies at each time step
        sum_per_t = np.sum(mag_pca[pc], axis=0, keepdims=True) + 1e-12 # Shape: (1, n_times)

        # Normalize (creates probability distribution over frequencies)
        mag_normalized[pc] = mag_pca[pc] / sum_per_t

    return mag_normalized

def remove_low_freq(csi_data, w=50):
        """
        Remove low frequency components using rectangular window average.
        This removes static reflections and DC offset.
        
        Parameters:
        -----------
        csi_data : ndarray
            CSI data (time x channels)
        w : int
            Window width for moving average
            
        Returns:
        --------
        csi_filtered : ndarray
            High-pass filtered CSI data
        """
        from scipy.ndimage import uniform_filter1d
        
        # Compute moving average for each channel
        csi_smoothed = uniform_filter1d(csi_data, size=w, axis=0, mode='nearest')
        
        # Subtract moving average to remove low frequencies
        csi_filtered = csi_data - csi_smoothed
        
        return csi_filtered


def bandpass_filter(data, fs, low=0.3, high=60, order=4):
    """
    Apply a bandpass filter to raw CSI data in the frequency domain

    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=0)


def hermite_function(n, t):
    """
    Generate the nth Hermite function.

    Parameters:
    -----------
    n : int
        Order (0, 1, 2, ...)
    t : array
        Time values

    Returns:
    --------
    chi_n : array
        Hermite function values
    """
    # Get Hermite polynomial
    H_n = hermite(n)

    # Normalization constant
    norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))

    # Hermite function = polynomial * Gaussian
    chi_n = norm * H_n(t) * np.exp(-t**2 / 2)

    return chi_n


def stft_with_hermite_window(signal_in, fs, T_win=0.4, hermite_order=0):
    """
    Compute STFT using a Hermite function as the window.

    This is just regular STFT but with a Hermite function window
    instead of Hann/Hamming/etc.

    Parameters:
    -----------
    signal_in : array
        1D or 2D array (time) or (time x components)
    fs : float
        Sampling rate (Hz)
    T_win : float
        Window length (seconds)
    hermite_order : int
        Which Hermite function to use (0, 1, 2, ...)
        0 = Gaussian-like
        1 = First Hermite
        2 = Second Hermite, etc.

    Returns:
    --------
    freq : array
        Frequency array
    time : array
        Time array
    mag : array
        Magnitude spectrogram
        2D if input is 1D, 3D if input is 2D
    """
    # Handle input dimensions
    is_single_time_series = (signal_in.ndim == 1) or (signal_in.ndim == 2 and signal_in.shape[1] == 1)

    if signal_in.ndim == 1:
        signal_in_processed = signal_in.reshape(-1, 1)
    elif signal_in.ndim == 2 and signal_in.shape[0] == 1 and signal_in.shape[1] > 1:
        signal_in_processed = signal_in.T
    else:
        signal_in_processed = signal_in

    # Window parameters
    T_overlap = T_win * 0.99
    nperseg = int(T_win * fs)
    noverlap = int(T_overlap * fs)

    # Create Hermite window
    t_window = np.linspace(-T_win/2, T_win/2, nperseg)
    hermite_window = hermite_function(hermite_order, t_window)

    # Initialize output lists
    freq = []
    time = []
    mag = []

    # Process each component
    for i in range(signal_in_processed.shape[1]):
        f, t, Zxx = signal.stft(
            signal_in_processed[:, i],
            fs=fs,
            window=hermite_window,  # Use Hermite window instead of 'hann'
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            return_onesided=True
        )
        freq.append(f)
        time.append(t)
        mag.append(np.abs(Zxx))

    # Return in same format as csi_stft
    if is_single_time_series:
        return np.array(freq[0]), np.array(time[0]), np.array(mag[0])
    else:
        return np.array(freq[0]), np.array(time[0]), np.array(mag)


def per_pc_normalization(
    mag,
    f,
    noise_f1=60,
    noise_f2=80,
    frame_percentile=95,
    smooth_sigma=1.0
):
    """
    Hybrid spectrogram processing.
    mag: (n_pcs, n_freqs, n_times)
    """

    n_pcs, n_freqs, n_times = mag.shape
    mag_pc_norm = np.zeros_like(mag)

    # ---------- 1. Robust per-PC normalization ----------
    for pc in range(n_pcs):
        pc_energy = np.percentile(mag[pc], 90)
        mag_pc_norm[pc] = mag[pc] / (pc_energy + 1e-12)

    # ---------- 2. Average PCs ----------
    S_avg = np.mean(mag_pc_norm, axis=0)   # (n_freqs, n_times)

    # ---------- 3. Adaptive noise floor (per-frame) ----------
    noise_mask = (f >= noise_f1) & (f <= noise_f2)
    noise_floor = np.mean(S_avg[noise_mask, :], axis=0)
    S_nf = np.maximum(S_avg - noise_floor[np.newaxis, :], 0.0)

    # ---------- 4. Per-frame gain equalization ----------
    frame_level = np.percentile(S_nf, frame_percentile, axis=0)
    valid = frame_level > 0
    ref_level = np.median(frame_level[valid]) if np.any(valid) else 1.0

    gains = ref_level / (frame_level + 1e-12)
    gains = np.clip(gains, 0.5, 3.0)
    S_eq = S_nf * gains[np.newaxis, :]

    # ---------- 5. Optional smoothing ----------
    if smooth_sigma > 0:
        S_eq = gaussian_filter1d(S_eq, sigma=smooth_sigma, axis=1)

    return S_eq


def frequency_weighting(mag, f, f_cut=15.0, alpha=2.0):
    """
    Smoothly down-weight low frequencies
    """
    w = np.ones_like(f)
    idx = f < f_cut
    w[idx] = (f[idx] / f_cut) ** alpha
    return mag * w[:, np.newaxis]


def process_stft_results(mag, f):
    """
    Normalized magnitiude data of time-frequency- to be plotted as spectrogram

    mag : ndarray
        Spectrogram magnitude data (n_pcs, n_freqs, n_times)
    f : ndarray
        Frequency values (n_freqs,) 
    """
    f_min = 8 # can change and optimize
    freq_mask = f >= f_min
    f_filtered = f[freq_mask]
    mag_filtered = mag[:, freq_mask, :]

    mag_nf   = adaptive_noise_floor_per_pc(mag_filtered, f_filtered, 80, 100) # TODO 
    mag_norm = normalize_by_sum_per_time(mag_nf)
    # average across PCs
    mag_pc_avg = np.mean(mag_norm, axis=0)
    return mag_pc_avg, f_filtered


def calculate_movement_speed(S, t, f, percentile=0.5):
    """
    S: spectrogram magnitudes
    t: time vector
    f: frequency vector
    percentile: threshold. percentile >=50 is torso, percentile >= 95 is limb
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
    

def estimate_gate_cycle_time(torso_contour_freq, lambda_ = 0.05, fs=250):
    from scipy.signal import butter, filtfilt, find_peaks
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
        print(tau[p])
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

def downsample_csi(data, fs, target_fs):
    """
    data: csi data (time x subcarriers)
    fs: original sampling rate
    target_fs: sampling rate after downsampling, must be integer divisor of fs
    """
    factor = int(fs / target_fs)

    data_downsampled = decimate(data, factor, axis=0, zero_phase=True)

    fs_new = fs / factor
    print(fs_new)
    return data_downsampled, fs_new


def freq_distribution(S):
    freq_distribution = np.mean(S, axis=1)
    return freq_distribution

def freq_distribution_gait_phase(S, t):
    n_freq, n_times = S.shape
    phase_length = n_times // 4
    
    freq_distr_gait_phase = np.zeros((4,n_freq))
    for phase in range(4):
        start = phase * phase_length
        end = start + phase_length

        end = min(end, n_times)

        freq_distr_gait_phase[phase] = np.mean(S[:, start:end], axis=1)

    return freq_distr_gait_phase
