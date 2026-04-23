import numpy as np
import matplotlib.pyplot as plt
from picoscenes import Picoscenes
import spec_functions as sf
import glob
import os
import pandas as pd
import seaborn as sns

# --- YOUR PIPELINE CLASS ---
class SpectrogramGenerator:
    def __init__(self, CSI_data):
        self.fs = 500 
        self.N_packets, N_streams = CSI_data.shape
        self.n_components = 8 
        self.f_min = 8
        self.f_max = 100 
        self.num_skip_pcs = 1 

    def prep_CSI_data(self, raw_CSI_data):
        CSI_mag = np.abs(raw_CSI_data) 
        signal_clean = CSI_mag - np.mean(CSI_mag, axis=0)
        signal_filtered = sf.bandpass_filter(signal_clean, self.fs, low=self.f_min, high=self.f_max, order=4)
        X_PCA, _, _ = sf.compute_pca_components(signal_filtered,
                                                n_components=self.n_components,
                                                skip_first=self.num_skip_pcs,
                                                whiten=False)
        return X_PCA

# --- AUTOMATIC FILE SELECTION ---
def get_latest_csi():
    list_of_files = glob.glob('*.csi')
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getmtime)

csi_filename = get_latest_csi()

if csi_filename is None:
    print("Error: No .csi files found in this folder!")
    exit()

print(f"--- Processing Most Recent File: {csi_filename} ---")

# --- PICOSCENES PARSING SECTION ---
try:
    frames = Picoscenes(csi_filename)
    all_csi = [f["CSI"]["CSI"] for f in frames.raw if "CSI" in f]
    X = np.array(all_csi)
    
    if X.size == 0:
        raise ValueError(f"The file '{csi_filename}' was parsed but contains no CSI data.")

    print(f"Parsed PicoScenes data. Shape: {X.shape}")

except Exception as e:
    print(f"Error: {e}")
    exit()

raw_CSI_data = np.abs(X).astype(np.float64)
T, N = raw_CSI_data.shape

spec = SpectrogramGenerator(raw_CSI_data)
csi_post_pca = spec.prep_CSI_data(raw_CSI_data)
f, t, mag = sf.compute_STFT(csi_post_pca, spec.fs, T_win=0.4)
STFT_data, f = sf.process_stft_results(mag, f)

sf.plot_spectrogram(f, t, STFT_data,
                  v_min = 0.3,
                  v_max = 2.25,
                  title=f"Spectrogram: {csi_filename}",
                  cmap='jet',
                  dB=False,
                  figsize=(14,6))
                  
  
t_min = 4
t_max = 7.5


time_mask = (t >= t_min) & (t <= t_max)

STFT_data = STFT_data[:, time_mask]
t = t[time_mask]                
                  
           
                 
feature_vector = sf.generate_feature_vector(STFT_data, t, f)
sf.update_feature_file(feature_vector, csi_filename, "features.csv")                 
                  
# 1. Load from CSV instead of Excel
file = "features.csv"
df = pd.read_csv(file)

# Optional: Remove any completely empty rows/columns that CSVs sometimes pick up
df = df.dropna(how='all')

print(df.head())
print(df.columns)

plt.figure(figsize=(8,6))

# 2. Scatterplot (Logic remains the same as long as column names match)
sns.scatterplot(
    data=df,
    x="torso_avg",
    y="cycle_time",
    hue="Label",
    palette="tab10",
    s=80
)

plt.title("Feature Comparison: Torso Avg vs Cycle Time")
plt.xlabel("Torso Avg Velocity")
plt.ylabel("Cycle Time")
plt.legend(title="Person")
plt.grid(True)

plt.show()

predicted_person = sf.classify_latest_row_csv('features.csv')

"""
V_t = sf.compute_spectral_variance(f=f, S=STFT_data, f_min=30.0, f_max=60.0)


t_start_best, t_end_best, mask_best_segment = sf.find_constant_psi_segment(
    t=t,
    torso_speed=torso_avg,
    V_t=V_t,
    Tmin=3.0,
    V_percentile_thresh=80,
    torso_std_thresh=0.2
)

if t_start_best is None:
    print("No constant-psi segment found")
else:
    print("Constant-psi segment:", t_start_best, "to", t_end_best)

S = STFT_data
f_plot = STFT_freq
v = f_plot * 0.06 / 2

plt.figure(figsize=(14, 6))
plt.pcolormesh(t, v, S, shading='gouraud', cmap='viridis')
plt.ylim([0, 2.5])
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Constant-ψ Overlay")
plt.colorbar(label="Magnitude")

if t_start_best is not None:
    plt.axvline(t_start_best, color='w', linestyle='--', linewidth=2, label='Constant-ψ start')
    plt.axvline(t_end_best, color='r', linestyle='--', linewidth=2, label='Constant-ψ end')
    plt.legend()

plt.tight_layout()
plt.show()
           
"""                 
