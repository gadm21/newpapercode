"""Generate rolling variance CSV from real CSI data for LaTeX pgfplots figure."""
import numpy as np
import pandas as pd
import os, json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats_results')
PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'newpaper2026')

# Load a short segment of Office HAR "work" activity
csv_path = os.path.join(DATA_DIR, 'office_har_data', 'work.csv')
df = pd.read_csv(csv_path, nrows=5000)

# CSI data is in the 'data' column as a JSON-like list string
def parse_csi_row(data_str):
    vals = json.loads(data_str)
    return np.array(vals, dtype=np.float32)

csi_matrix = np.stack(df['data'].apply(parse_csi_row).values)
print(f"CSI matrix shape: {csi_matrix.shape}")  # (N, 128)

# Parse complex CSI: pairs of (imag, real) -> amplitude
n_sub = 64
imag = csi_matrix[:, 0::2][:, :n_sub]
real = csi_matrix[:, 1::2][:, :n_sub]
amp = np.sqrt(real**2 + imag**2)

# Apply subcarrier mask (keep indices 6-31, 33-58)
mask = np.zeros(64, dtype=bool)
mask[6:32] = True   # negative freq
mask[32] = False     # DC
mask[33:59] = True   # positive freq
amp_masked = amp[:, mask]  # shape: (N, 52)
print(f"Amplitude matrix shape after masking: {amp_masked.shape}")

def rolling_variance_2d(mag, var_window):
    """Compute rolling variance over a sliding window per subcarrier.
    Same implementation as in utils.py _rolling_variance method."""
    if var_window <= 1:
        return np.zeros_like(mag)
    n = mag.shape[0]
    cs = np.cumsum(mag, axis=0)
    cs2 = np.cumsum(mag ** 2, axis=0)
    cs = np.vstack([np.zeros((1, mag.shape[1])), cs])
    cs2 = np.vstack([np.zeros((1, mag.shape[1])), cs2])
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - var_window, 0, None)
    counts = (hi - lo).reshape(-1, 1)
    means = (cs[hi] - cs[lo]) / counts
    mean_sq = (cs2[hi] - cs2[lo]) / counts
    var = np.clip(mean_sq - means ** 2, 0, None)
    return var

# Downsample for pgfplots (every 5th sample -> ~1000 points)
step = 1
indices = np.arange(0, amp_masked.shape[0], step)
amp_downsampled = amp_masked[indices]

# Generate output file 1: Raw 52 subcarriers
raw_data = []
for i in range(52):
    for idx, sample_idx in enumerate(indices):
        raw_data.append({
            'n': sample_idx,
            'subcarrier': i,
            'amplitude': amp_downsampled[idx, i]
        })

raw_df = pd.DataFrame(raw_data)

os.makedirs(OUT_DIR, exist_ok=True)
raw_path = os.path.join(OUT_DIR, 'raw_52_subcarriers.csv')
raw_df.to_csv(raw_path, index=False, float_format='%.4f')
print(f"Wrote {len(raw_df)} rows to {raw_path}")

# Also copy to paper directory
paper_raw_path = os.path.join(PAPER_DIR, 'raw_52_subcarriers.csv')
raw_df.to_csv(paper_raw_path, index=False, float_format='%.4f')
print(f"Also copied raw data to {paper_raw_path}")

# Compute rolling variances for all 52 subcarriers with 3 window sizes
var_windows = [20, 200, 2000]
rv_data = []

for w in var_windows:
    rv = rolling_variance_2d(amp_masked, w)
    rv_downsampled = rv[indices]
    
    for i in range(52):
        for idx, sample_idx in enumerate(indices):
            rv_data.append({
                'n': sample_idx,
                'subcarrier': i,
                'window': w,
                'variance': rv_downsampled[idx, i]
            })

rv_df = pd.DataFrame(rv_data)

# Generate output file 2: 52x3 rolling variances
rv_path = os.path.join(OUT_DIR, 'rolling_variances_52x3.csv')
rv_df.to_csv(rv_path, index=False, float_format='%.4f')
print(f"Wrote {len(rv_df)} rows to {rv_path}")

# Also copy to paper directory
paper_rv_path = os.path.join(PAPER_DIR, 'rolling_variances_52x3.csv')
rv_df.to_csv(paper_rv_path, index=False, float_format='%.4f')
print(f"Also copied rolling variance data to {paper_rv_path}")

# Print statistics for each window size
for w in var_windows:
    rv = rolling_variance_2d(amp_masked, w)
    print(f"Window {w}: variance range [{rv.min():.4f}, {rv.max():.4f}], mean {rv.mean():.4f}")

print(f"\nGenerated files:")
print(f"  1. Raw 52 subcarriers: {raw_path}")
print(f"  2. Rolling variances (52x3): {rv_path}")
