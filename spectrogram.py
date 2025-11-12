import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# Config
path = r"/content/data_500.mat"  
fs = 2000                        # packets/s (from Xmodal assuming it's close)
static_avg_secs = 4.0            # long-term averaging window to estimate static offset
chunk_secs = 1.0                 # length of analysis window in seconds
discard_first_pc_when_reconstructing = True  # recommended in reference paper


#Load CSI (.mat) and find the array
mat = sio.loadmat(path)
keys = [k for k in mat.keys() if not k.startswith("__")]
if not keys:
    raise ValueError("No data variables found in the .mat file.")
var_name = keys[0]
X = mat[var_name]  # expected shape (T, N_streams), dtype complex128
print(f"Loaded '{var_name}' with shape {X.shape} and dtype {X.dtype}")

# Sanity checks
if X.ndim != 2:
    raise ValueError("Expected a 2-D array (time x streams).")

T, N = X.shape

#Use magnitudes to kill CFO sensitivity (ignore phase)
# s_b(t) -> |s_b(t)| per stream/subcarrier
A = np.abs(X).astype(np.float64)   # shape (T, N)


# Remove static component (long-term mean over ~4 s per stream)
L_static = int(round(static_avg_secs * fs))
if L_static < T:
    static_offset = A[:L_static].mean(axis=0)  # mean for each stream over first 4 s
else:
    static_offset = A.mean(axis=0)             # fallback if record < 4 s

A_dc = A - static_offset  # remove static paths per stream

#Chop into 1-s non-overlapping chunks to form H matrices
L = int(round(chunk_secs * fs))  # samples per chunk
n_chunks = T // L                 # drop tail if not full second
if n_chunks == 0:
    raise ValueError("Recording too short for the chosen chunk length.")

# Pre-allocate per-chunk explained variance (eigenvalue ratios)
explained_ratios = np.zeros((n_chunks, N), dtype=np.float64)

def eigen_explained_variance(H):
    """
    Given H (L x N), column-centered, compute R = H^T H,
    eigendecompose, and return sorted eigenvalue ratios (descending).
    """
    # Correlation matrix (N x N)
    R = H.T @ H
    # Symmetric => use eigh; returns ascending eigenvalues
    w, v = np.linalg.eigh(R)
    # Sort descending
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    # Explained variance ratio = eigenvalue / sum(eigenvalues)
    w_sum = w.sum()
    # Handle degenerate case (all zeros)
    if w_sum <= 0:
        evr = np.zeros_like(w)
    else:
        evr = w / w_sum
    return evr, v


# Per-chunk PCA via eigenanalysis of H^T H

for k in range(n_chunks):
    # Take 1-s slice
    H = A_dc[k*L:(k+1)*L, :]              # shape (L, N)
    # Column-center H (mean 0 per stream within the chunk)
    H = H - H.mean(axis=0, keepdims=True)

    # Eigen on R = H^T H -> explained variance ratios
    evr, eigvecs = eigen_explained_variance(H)
    explained_ratios[k, :] = evr

    # PCs needed to hit 90% / 95%
    cume = np.cumsum(evr)
    pcs_90[k] = int(np.searchsorted(cume, 0.90, side="left") + 1)
    pcs_95[k] = int(np.searchsorted(cume, 0.95, side="left") + 1)
