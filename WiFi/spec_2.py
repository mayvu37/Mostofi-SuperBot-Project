# cleaned and updated version of spectrogram.py
import spec_functions as sf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

class SpectrogramGenerator:
    def __init__(self, CSI_data):
        self.fs = 500 # Sampling rate of our data fs=1/detla*t
        self.N_packets, N_streams = CSI_data.shape# from CSI we get two columns, smaples=packets and subcarriers=streams
        self.n_components=8  #number of principal components to use
        self.f_min=8
        self.f_max=100 # was 70 previously
        self.num_skip_pcs=1 #number of pcs to skip
        

    def prep_CSI_data(self, raw_CSI_data):
        """
        Given raw CSI data, process and return PCA-projected data (time series data)

        CSI_data : ndarray
            Raw CSI data (N_packets, N_streams)
        """
        time_total = np.arange(self.N_packets) / self.fs  # Time vector: each data stamp is taken at a frequncy of 500Hz (assumed from the file being labled 500)
        duration = time_total[-1] # gets last index meanign total duration of the signal
        CSI_mag = np.abs(raw_CSI_data[:,:]) 
        signal_clean = CSI_mag - np.mean(CSI_mag, axis=0)# remove DC component (might need changing if we want to add realtime data)
        signal_filtered = sf.bandpass_filter(signal_clean, self.fs, low=self.f_min, high=self.f_max, order=4)
        X_PCA, eigvals, eigvecs = sf.compute_pca_components(signal_filtered,
                                                                n_components=self.n_components,
                                                                skip_first=self.num_skip_pcs,
                                                                whiten=False)
        return X_PCA


# raw_CSI_data must be loaded from .mat file
# path = r"c:/Users/ptv57/Downloads/data_500.mat"
# path = r"c:/Users/ptv57/Downloads/xmodal_data_2000.mat"
path = r"c:/Users/ptv57/Downloads/Jose_500_T1.mat"

filename = os.path.splitext(os.path.basename(path))[0]

mat = sio.loadmat(path)
keys = [k for k in mat.keys() if not k.startswith("__")]
if not keys:
    raise ValueError("No data variables found in the .mat file.")
var_name = keys[0]
X = mat[var_name]

print(f"Loaded '{var_name}' with shape {X.shape} and dtype {X.dtype}")

if X.ndim != 2:
    raise ValueError("Expected a 2-D array (time x streams).")
T, N = X.shape

raw_CSI_data = np.abs(X).astype(np.float64)

spec = SpectrogramGenerator(raw_CSI_data)
csi_post_pca = spec.prep_CSI_data(raw_CSI_data)
f, t, mag = sf.compute_STFT(csi_post_pca, spec.fs, T_win=0.4)
STFT_data, f = sf.process_stft_results(mag, f)

sf.plot_spectrogram(f, t, STFT_data,
                 v_min = 0.3,
                 v_max = 2.25,
                 title="Full Pipeline Spectrogram",
                 cmap='jet',#jet, viridis
                 dB=False,

                 figsize=(14,6))

t_min = 2.4
t_max = 8
time_mask = (t >= t_min) & (t <= t_max)

STFT_data = STFT_data[:, time_mask]
t = t[time_mask]


feature_vector = sf.generate_feature_vector(STFT_data, t, f)
sf.update_feature_file(feature_vector, filename)


file = "training_feature_vectors.xlsx"

df = pd.read_excel(file)

print(df.head())
print(df.columns)

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="torso_avg",
    y="cycle_time",
    hue="Label",
    palette="tab10",
    s=80
)

plt.title("Feature Comparison: Torso Avg vs Stride Length")
plt.xlabel("Torso Avg Velocity")
plt.ylabel("Cycle Time")
plt.legend(title="Person")
plt.grid(True)

plt.show()


velocity_bins = [c for c in df.columns if "velocity_bin" in c]
labels = df["Label"].unique()

# iterate through labels in pairs
for i in range(0, len(labels), 2):

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    for j in range(2):

        if i + j >= len(labels):
            break

        label = labels[i + j]
        ax = axes[j]

        subset = df[df["Label"] == label]

        for _, row in subset.iterrows():

            y = row[velocity_bins].values.astype(float)
            x = np.arange(len(y))

            ax.plot(x, y, alpha=0.6)

        ax.set_title(f"Velocity Distribution: {label}")
        ax.set_xlabel("Velocity Bin")
        ax.set_ylabel("Energy")
        ax.grid(True)

    plt.tight_layout()
    plt.show()
