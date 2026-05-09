#!/usr/bin/env python3
"""
CSI Studio — Unified real-time CSI collection, visualization, and recording.

Combines the functionality of collect_csi_std.py (standardized collection with
resampling, guard removal, metadata) and viewshape.py (real-time visualization)
into a single, polished tool.

Features:
  - Real-time CSI visualization (heatmap, subcarrier traces, mean+-std)
  - Auto-save every segment (default 10s) as standardized .npz + .meta.json
  - User-settable activity labels via toolbar text entry
  - Dynamic 2D PCA scatter updated every 1s, rolling 10-segment window
  - Variance tab with multiple rolling-window lengths (2s, 10s, 30s)
  - Day / Night theme toggle
  - Smooth trailing-interpolation animation for fluid 30 fps rendering
  - Publication-quality PDF export per panel

Usage:
    python csi_studio.py --rx-port COM5
    python csi_studio.py --rx-port /dev/ttyACM1 --tx-port /dev/ttyACM0 --label walk
    python csi_studio.py --rx-port COM5 --label sit --segment 10 --sr 150 --out-dir ./data
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

# Ensure local utils.py is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_this_dir, os.path.join(_this_dir, '..', 'train')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    print("ERROR: pyserial is required. Install with: pip install pyserial",
          file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox
    import matplotlib.patheffects as pe
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib",
          file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from utils import CSI_SUBCARRIER_MASK

# =====================================================================
# Constants
# =====================================================================
HEADER = ("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,"
          "channel,local_timestamp,sig_len,rx_state,len,first_word,data")

NUM_SUBCARRIERS = 52
FFT_SIZE = 64
DEFAULT_SR = 150
DEFAULT_SEGMENT_SEC = 10
HEATMAP_COLS = 80
TRACE_BINS = 120
STFT_NFFT = 64
STFT_HOP = 16
STFT_FREQ_BINS = STFT_NFFT // 2
PCA_ROLLING_SEGMENTS = 10
PCA_UPDATE_INTERVAL = 1.0
SLIDING_WINDOW_SEC = 10.0
BASELINE_WINDOW_SEC = 60.0
SMOOTH_ALPHA = 0.35

AMP_MIN, AMP_MAX = 0, 50
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK = 9
FONT_LEGEND = 9
FONT_STATS = 10
FONT_BTN = 8
FONT_SUPTITLE = 15

LABEL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<']
LABEL_CMAP = plt.cm.Set1

# =====================================================================
# Themes
# =====================================================================
THEMES = {
    'night': {
        'BG': '#0d1117', 'SURFACE': '#161b22', 'BORDER': '#30363d',
        'TEXT_PRIMARY': '#e6edf3', 'TEXT_SECONDARY': '#8b949e',
        'ACCENT': '#58a6ff', 'ACCENT_WARM': '#f0883e',
        'SUCCESS': '#3fb950', 'DANGER': '#f85149',
        'CTRL_BG': '#1a2332', 'CTRL_HOVER': '#2a4060',
        'TAB_ACTIVE': '#263d52', 'GRID': '#455a64',
        'CMAP_HEAT': 'inferno', 'CMAP_VAR': 'magma',
    },
    'day': {
        'BG': '#ffffff', 'SURFACE': '#f6f8fa', 'BORDER': '#d0d7de',
        'TEXT_PRIMARY': '#1f2328', 'TEXT_SECONDARY': '#656d76',
        'ACCENT': '#0969da', 'ACCENT_WARM': '#bf5700',
        'SUCCESS': '#1a7f37', 'DANGER': '#cf222e',
        'CTRL_BG': '#eaeef2', 'CTRL_HOVER': '#d0d7de',
        'TAB_ACTIVE': '#ddf4ff', 'GRID': '#d0d7de',
        'CMAP_HEAT': 'YlOrRd', 'CMAP_VAR': 'YlGnBu',
    },
}

T = dict(THEMES['night'])


# =====================================================================
# Raw CSI Buffer (thread-safe accumulator)
# =====================================================================
class RawCSIBuffer:
    """Thread-safe ring buffer that accumulates parsed CSI packets."""

    def __init__(self, maxlen=15000):
        self.lock = threading.Lock()
        self._timestamps = deque(maxlen=maxlen)
        self._rssi = deque(maxlen=maxlen)
        self._real = deque(maxlen=maxlen)
        self._imag = deque(maxlen=maxlen)
        self._amps = deque(maxlen=maxlen)
        self._wall_times = deque(maxlen=maxlen)
        self._count = 0
        self._errors = []

    @property
    def count(self):
        with self.lock:
            return self._count

    @property
    def errors(self):
        with self.lock:
            return list(self._errors)

    def add_line(self, text):
        """Parse one CSI_DATA line and append to buffers."""
        try:
            parts = text.split(",", 14)
            if len(parts) < 15:
                return
            rssi = float(parts[3])
            ts = int(parts[9])
            csi_str = parts[14].strip().strip('"').strip("'").strip('[]')
            csi_row = [int(x) for x in csi_str.split(",") if x.strip()]
            if len(csi_row) != 128:
                with self.lock:
                    self._errors.append(f"expected 128 I/Q, got {len(csi_row)}")
                return
            imag = csi_row[0::2]
            real = csi_row[1::2]
            amps_64 = [math.sqrt(r * r + i * i) for r, i in zip(real, imag)]
            amps_52 = [amps_64[k] for k in range(64) if CSI_SUBCARRIER_MASK[k]]
            wall_t = time.time()
            with self.lock:
                self._timestamps.append(ts)
                self._rssi.append(rssi)
                self._real.append(real)
                self._imag.append(imag)
                self._amps.append(amps_52)
                self._wall_times.append(wall_t)
                self._count += 1
        except Exception as e:
            with self.lock:
                self._errors.append(str(e))

    def snapshot_amps(self, max_age_sec=None):
        """Return (amps_array[N,52], wall_times[N]) for visualization."""
        with self.lock:
            if not self._amps:
                return np.empty((0, NUM_SUBCARRIERS)), np.array([])
            amps = np.array(list(self._amps), dtype=np.float64)
            wt = np.array(list(self._wall_times), dtype=np.float64)
        if max_age_sec is not None and len(wt) > 0:
            cutoff = wt[-1] - max_age_sec
            mask = wt >= cutoff
            amps = amps[mask]
            wt = wt[mask]
        return amps, wt

    def drain_segment(self, start_wall, end_wall):
        """Extract raw I/Q data for a time segment (non-destructive)."""
        with self.lock:
            wt = np.array(list(self._wall_times), dtype=np.float64)
            if len(wt) == 0:
                return None
            mask = (wt >= start_wall) & (wt < end_wall)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                return None
            real_list = [self._real[i] for i in idxs]
            imag_list = [self._imag[i] for i in idxs]
            rssi_list = [self._rssi[i] for i in idxs]
            ts_list = [self._timestamps[i] for i in idxs]
        return {
            'real': np.array(real_list, dtype=np.float64),
            'imag': np.array(imag_list, dtype=np.float64),
            'rssi': np.array(rssi_list, dtype=np.float64),
            'ts_us': np.array(ts_list, dtype=np.int64),
            'wall_times': wt[mask],
        }


# =====================================================================
# Standardization pipeline (resample + guard removal)
# =====================================================================
def resample_and_filter(real, imag, rssi, ts_us, guaranteed_sr,
                       wall_times=None):
    """Resample to uniform rate, remove guard subcarriers.

    Uses wall_times (time.time() floats) for resampling if provided,
    falling back to ESP32 ts_us.  Wall times are preferred because
    ESP32 local_timestamp can wrap or have discontinuities.
    """
    n_orig = len(ts_us)
    if n_orig < 2:
        return None

    if wall_times is not None and len(wall_times) == n_orig:
        ts_sec = wall_times.astype(np.float64)
        ts_sec = ts_sec - ts_sec[0]
    else:
        ts_sec = ts_us.astype(np.float64) / 1_000_000
    start, end = ts_sec[0], ts_sec[-1]
    duration = end - start
    if duration < 0.1:
        return None

    actual_sr = n_orig / duration
    max_n_out = int(guaranteed_sr * duration * 2)
    n_out_raw = int(np.ceil(duration * guaranteed_sr))
    if n_out_raw > max_n_out or n_out_raw > 1_000_000:
        return None
    n_out = n_out_raw
    if n_out < 2:
        return None

    target_t = start + np.arange(n_out) / guaranteed_sr
    dt = 1.0 / guaranteed_sr

    bin_edges = target_t - dt / 2
    bin_idx = np.searchsorted(bin_edges, ts_sec, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_out - 1)
    samples_per_bin = np.bincount(bin_idx, minlength=n_out).astype(np.int64)

    mag_raw = np.sqrt(real ** 2 + imag ** 2)
    phase_raw = np.arctan2(imag, real)

    channels = {'mag': mag_raw, 'phase': phase_raw}
    resampled = {}
    for k, src in channels.items():
        acc = np.zeros((n_out, 64), dtype=np.float64)
        np.add.at(acc, bin_idx, src)
        resampled[k] = acc

    rssi_acc = np.zeros(n_out, dtype=np.float64)
    np.add.at(rssi_acc, bin_idx, rssi)

    populated = samples_per_bin > 0
    for k in channels:
        resampled[k][populated] /= samples_per_bin[populated, None]
    rssi_acc[populated] /= samples_per_bin[populated]

    empty_mask = samples_per_bin == 0
    n_empty = int(empty_mask.sum())
    if n_empty > 0:
        valid_idx = np.where(populated)[0]
        empty_idx = np.where(empty_mask)[0]
        if len(valid_idx) >= 2:
            for k in channels:
                for sc in range(64):
                    resampled[k][empty_idx, sc] = np.interp(
                        target_t[empty_idx], target_t[valid_idx],
                        resampled[k][valid_idx, sc])
            rssi_acc[empty_idx] = np.interp(
                target_t[empty_idx], target_t[valid_idx], rssi_acc[valid_idx])

    mag_out = resampled['mag'][:, CSI_SUBCARRIER_MASK]
    phase_out = resampled['phase'][:, CSI_SUBCARRIER_MASK]
    resampled_ts = (target_t * 1_000_000).astype(np.int64)

    meta = {
        'original_samples': n_orig,
        'resampled_samples': n_out,
        'actual_sr': round(actual_sr, 2),
        'guaranteed_sr': guaranteed_sr,
        'sr_ratio': round(actual_sr / guaranteed_sr, 4),
        'duration_sec': round(duration, 3),
        'empty_bins': n_empty,
        'empty_bins_pct': round(100 * n_empty / n_out, 2) if n_out > 0 else 0,
        'subcarriers': int(CSI_SUBCARRIER_MASK.sum()),
    }

    return {
        'mag': mag_out,
        'phase': phase_out,
        'rssi': rssi_acc,
        'timestamp': resampled_ts,
        'meta': meta,
    }


# =====================================================================
# Segment saver (runs in its own thread)
# =====================================================================
class SegmentSaver:
    """Periodically saves standardized CSI segments with metadata."""

    def __init__(self, buf, out_dir, segment_sec, guaranteed_sr, get_label_fn):
        self.buf = buf
        self.out_dir = out_dir
        self.segment_sec = segment_sec
        self.sr = guaranteed_sr
        self.get_label = get_label_fn
        self.saved_files = []
        self.lock = threading.Lock()
        self._file_idx = 0
        self._running = False
        self.enabled = False
        self._segment_wall_start = None
        self._session_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        os.makedirs(out_dir, exist_ok=True)

    @property
    def file_count(self):
        with self.lock:
            return len(self.saved_files)

    @property
    def last_saved(self):
        with self.lock:
            return self.saved_files[-1] if self.saved_files else None

    def start(self, stop_event):
        self._running = True
        self._segment_wall_start = time.time()
        self._thread = threading.Thread(
            target=self._loop, args=(stop_event,), daemon=True)
        self._thread.start()

    def _loop(self, stop_event):
        while not stop_event.is_set():
            elapsed = time.time() - self._segment_wall_start
            remaining = self.segment_sec - elapsed
            if remaining > 0:
                stop_event.wait(timeout=min(remaining, 0.5))
                continue
            seg_end = time.time()
            seg_start = self._segment_wall_start
            self._segment_wall_start = seg_end
            if self.enabled:
                self._save_segment(seg_start, seg_end)
        if self._segment_wall_start is not None and self.enabled:
            self._save_segment(self._segment_wall_start, time.time())

    def _save_segment(self, seg_start, seg_end):
        raw = self.buf.drain_segment(seg_start, seg_end)
        if raw is None or len(raw['ts_us']) < 10:
            return

        result = resample_and_filter(
            raw['real'], raw['imag'], raw['rssi'], raw['ts_us'], self.sr,
            wall_times=raw['wall_times'])
        if result is None:
            return

        self._file_idx += 1
        label = self.get_label()
        wall_iso = datetime.fromtimestamp(
            seg_start, tz=timezone.utc).isoformat()

        base = f"{label}_{self._session_id}_{self._file_idx:04d}"
        npz_path = os.path.join(self.out_dir, base + ".npz")
        meta_path = os.path.join(self.out_dir, base + ".meta.json")

        np.savez_compressed(npz_path,
                            mag=result['mag'],
                            phase=result['phase'],
                            rssi=result['rssi'],
                            timestamp=result['timestamp'])

        meta = result['meta'].copy()
        meta['label'] = label
        meta['session_id'] = self._session_id
        meta['file_index'] = self._file_idx
        meta['segment_duration_sec'] = self.segment_sec
        meta['wall_start_time'] = wall_iso
        meta['wall_end_time'] = datetime.fromtimestamp(
            seg_end, tz=timezone.utc).isoformat()
        meta['output_shape'] = list(result['mag'].shape)
        meta['format'] = 'npz'
        meta['version'] = '2.0'

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        with self.lock:
            self.saved_files.append({
                'npz': npz_path, 'meta': meta_path,
                'label': label, 'shape': list(result['mag'].shape),
            })

        print(f"  \u2713 Saved segment {self._file_idx}: {base}.npz  "
              f"shape={result['mag'].shape}  label={label}  "
              f"sr_ratio={meta['sr_ratio']}")


# =====================================================================
# PCA tracker -- rolling window, updated every 1s from live buffer
# =====================================================================
class PCATracker:
    """Live 2D PCA from rolling 1-second snapshots of the CSI buffer.

    Keeps last max_snapshots feature vectors (default 100 = 10 seg x 10/seg).
    Each snapshot is a 1-second mean+std feature (104-dim).
    """

    def __init__(self, max_snapshots=PCA_ROLLING_SEGMENTS * 10):
        self._snapshots = deque(maxlen=max_snapshots)
        self._label_set = {}
        self.lock = threading.Lock()
        self._last_update = 0.0
        self._smooth_coords = None

    def maybe_update(self, buf, label, segment_sec):
        """Called from render loop. Adds a 1-second snapshot if due."""
        now = time.time()
        if now - self._last_update < PCA_UPDATE_INTERVAL:
            return
        self._last_update = now

        amps, wt = buf.snapshot_amps(max_age_sec=1.5)
        if len(amps) < 10:
            return
        cutoff = wt[-1] - 1.0
        chunk = amps[wt >= cutoff]
        if chunk.shape[0] < 5:
            return
        feat = np.concatenate([chunk.mean(axis=0), chunk.std(axis=0)])
        with self.lock:
            self._snapshots.append((label, feat))
            if label not in self._label_set:
                self._label_set[label] = len(self._label_set)

    def add_segment(self, label, mag_array):
        """Accept saved-segment data for PCA history."""
        if mag_array.shape[0] < 2:
            return
        feat = np.concatenate([mag_array.mean(axis=0), mag_array.std(axis=0)])
        with self.lock:
            self._snapshots.append((label, feat))
            if label not in self._label_set:
                self._label_set[label] = len(self._label_set)

    def get_embedding(self):
        """Return (coords[N,2], labels[N], label_set, ages[N]) or None."""
        if not HAS_SKLEARN:
            return None
        with self.lock:
            n = len(self._snapshots)
            if n < 3:
                return None
            labels = [s[0] for s in self._snapshots]
            feats = np.array([s[1] for s in self._snapshots])
            label_set = dict(self._label_set)
        try:
            pca = SkPCA(n_components=2)
            coords = pca.fit_transform(feats)
            ages = np.linspace(1.0, 0.0, n)
            if (self._smooth_coords is not None and
                    self._smooth_coords.shape == coords.shape):
                coords = (SMOOTH_ALPHA * coords +
                          (1 - SMOOTH_ALPHA) * self._smooth_coords)
            self._smooth_coords = coords.copy()
            return coords, labels, label_set, ages
        except Exception:
            return None


# =====================================================================
# Model Manager — dataset scan, RF training, live inference
# =====================================================================
class ModelManager:
    """Scan saved .npz files, train Random Forest, run live inference."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.lock = threading.Lock()
        # Dataset info: {label: [filepath, ...]}
        self.dataset = {}
        self.excluded_labels = set()
        # Model state
        self.model = None
        self.label_encoder = None
        self.classes = []
        self.deployed = False
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.n_train = 0
        self.n_test = 0
        # Last prediction
        self.last_probs = None
        self.last_pred = None
        self._last_segment_count = 0

    def scan(self):
        """Scan data_dir for .npz files, group by label from .meta.json."""
        dataset = {}
        if not os.path.isdir(self.data_dir):
            with self.lock:
                self.dataset = dataset
            return dataset
        for fname in os.listdir(self.data_dir):
            if not fname.endswith('.npz'):
                continue
            meta_path = os.path.join(
                self.data_dir, fname.replace('.npz', '.meta.json'))
            npz_path = os.path.join(self.data_dir, fname)
            label = 'unknown'
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    label = meta.get('label', 'unknown')
                except Exception:
                    pass
            else:
                # Infer label from filename: label_session_idx.npz
                parts = fname.rsplit('_', 2)
                if len(parts) >= 3:
                    label = parts[0]
            dataset.setdefault(label, []).append(npz_path)
        with self.lock:
            self.dataset = dataset
        return dataset

    def _extract_features(self, mag):
        """Extract feature vector from mag array (N, 52)."""
        if mag.shape[0] < 2:
            return None
        return np.concatenate([
            mag.mean(axis=0),        # 52
            mag.std(axis=0),         # 52
            np.percentile(mag, 25, axis=0),  # 52
            np.percentile(mag, 75, axis=0),  # 52
        ])  # 208-dim

    def get_included_labels(self):
        with self.lock:
            return [l for l in self.dataset if l not in self.excluded_labels]

    def train(self, n_estimators=100, max_depth=None, test_split=0.2):
        """Train RF on included labels. Returns (success, message)."""
        if not HAS_SKLEARN:
            return False, 'scikit-learn not installed'

        with self.lock:
            included = {l: files for l, files in self.dataset.items()
                        if l not in self.excluded_labels}

        if len(included) < 2:
            return False, f'Need >=2 labels, got {len(included)}'

        X, y = [], []
        for label, files in included.items():
            for fpath in files:
                try:
                    data = np.load(fpath)
                    feat = self._extract_features(data['mag'])
                    if feat is not None:
                        X.append(feat)
                        y.append(label)
                except Exception:
                    continue

        if len(X) < 10:
            return False, f'Too few valid samples ({len(X)})'

        X = np.array(X)
        y = np.array(y)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        if test_split > 0 and len(X) > 5:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, test_size=test_split, stratify=y_enc,
                random_state=42)
        else:
            X_tr, y_tr = X, y_enc
            X_te, y_te = X, y_enc

        md = max_depth if max_depth and max_depth > 0 else None
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=md,
            random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)

        train_acc = float(rf.score(X_tr, y_tr))
        test_acc = float(rf.score(X_te, y_te))

        with self.lock:
            self.model = rf
            self.label_encoder = le
            self.classes = list(le.classes_)
            self.train_accuracy = train_acc
            self.test_accuracy = test_acc
            self.n_train = len(X_tr)
            self.n_test = len(X_te)
            self.deployed = False
            self.last_probs = None
            self.last_pred = None

        msg = (f'Trained RF: {n_estimators} trees, '
               f'train={train_acc:.1%} ({len(X_tr)}), '
               f'test={test_acc:.1%} ({len(X_te)})')
        print(f'  [model] {msg}')
        return True, msg

    def deploy(self):
        with self.lock:
            if self.model is None:
                return False
            self.deployed = True
            self._last_segment_count = 0
        return True

    def undeploy(self):
        with self.lock:
            self.deployed = False
            self.last_probs = None
            self.last_pred = None

    def predict_segment(self, mag):
        """Predict probabilities for a single segment's mag array."""
        with self.lock:
            if not self.deployed or self.model is None:
                return None, None
            model = self.model
            le = self.label_encoder
        feat = self._extract_features(mag)
        if feat is None:
            return None, None
        probs = model.predict_proba(feat.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]
        with self.lock:
            self.last_probs = probs
            self.last_pred = pred_label
        return pred_label, probs

    def predict_from_buffer(self, buf, saver):
        """Predict from the latest 1-second buffer snapshot (live)."""
        with self.lock:
            if not self.deployed or self.model is None:
                return
        amps, wt = buf.snapshot_amps(max_age_sec=1.5)
        if len(amps) < 10:
            return
        cutoff = wt[-1] - 1.0
        chunk = amps[wt >= cutoff]
        if chunk.shape[0] < 5:
            return
        self.predict_segment(chunk)


# =====================================================================
# Serial reader thread
# =====================================================================
stop_event = threading.Event()


def serial_reader(rx_port, baud, buf, stop_evt):
    """Read CSI lines from serial into buffer until stopped."""
    print(f"  [serial] Opening {rx_port} at {baud} baud...")
    line_count = 0
    csi_count = 0
    non_csi_samples = []
    try:
        with serial.Serial(rx_port, baudrate=baud, timeout=0.05) as ser:
            print(f"  [serial] Port opened successfully")
            _ = ser.read(ser.in_waiting or 1)
            line_count = 0
            csi_count = 0
            last_print = time.time()
            non_csi_samples = []
            while not stop_evt.is_set():
                line = ser.readline()
                if not line:
                    # Periodic heartbeat to show reader is alive
                    if time.time() - last_print > 5.0:
                        print(f"  [serial] Waiting for data... lines={line_count}, CSI={csi_count}, buf.count={buf.count}")
                        last_print = time.time()
                    continue
                line_count += 1
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if text.startswith("CSI_DATA,"):
                    buf.add_line(text)
                    csi_count += 1
                    if csi_count % 100 == 0:
                        print(f"  [serial] Received {csi_count} CSI packets")
                        last_print = time.time()
                else:
                    # Collect first few non-CSI lines for debugging
                    if len(non_csi_samples) < 5 and text:
                        non_csi_samples.append(text[:80])  # Truncate long lines
    except serial.SerialException as e:
        print(f"[error] Serial: {e}", file=sys.stderr)
    except OSError as e:
        print(f"[error] OS: {e}", file=sys.stderr)

    # Print sample of non-CSI lines if we received data but no CSI
    if line_count > 0 and csi_count == 0 and non_csi_samples:
        print(f"  [serial] WARNING: Received {line_count} lines but no CSI_DATA packets")
        print(f"  [serial] Sample lines received:")
        for i, sample in enumerate(non_csi_samples):
            print(f"    [{i}] {sample}")

    print(f"  [serial] Reader stopped. Total lines: {line_count}, CSI packets: {csi_count}")


def reset_board(port, baud, label):
    """Toggle DTR/RTS to reset ESP32."""
    try:
        with serial.Serial(port, baudrate=baud) as s:
            s.dtr = False; s.rts = True
            time.sleep(0.05)
            s.dtr = True; s.rts = False
            time.sleep(0.05)
        print(f"  [reset] {label} at {port}")
    except Exception as e:
        print(f"  [warn] reset {label}: {e}", file=sys.stderr)


# =====================================================================
# Styling helpers
# =====================================================================
def _style_ax(ax):
    """Apply theme styling to an axis."""
    ax.set_facecolor(T['SURFACE'])
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(T['BORDER'])
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(direction='out', length=4, width=0.8,
                   colors=T['TEXT_SECONDARY'], labelsize=FONT_TICK)


def _set_axes_visible(axes_list, visible):
    """Show/hide axes, repositioning off-screen when hidden."""
    from matplotlib.transforms import Bbox
    _saved = getattr(_set_axes_visible, '_cache', {})
    _set_axes_visible._cache = _saved
    for ax in axes_list:
        ax.set_visible(visible)
        ax_id = id(ax)
        if visible:
            if ax_id in _saved:
                ax.set_position(_saved.pop(ax_id))
        else:
            if ax_id not in _saved:
                _saved[ax_id] = ax.get_position()
            ax.set_position(Bbox([[9, 9], [9.01, 9.01]]))


def _export_single_ax_pdf(fig, source_ax, title):
    """Export a single axes to a standalone publication-ready PDF."""
    counter = getattr(_export_single_ax_pdf, '_n', 0) + 1
    _export_single_ax_pdf._n = counter
    import tkinter.simpledialog as sd
    try:
        root = fig.canvas.get_tk_widget().winfo_toplevel()
    except Exception:
        root = None
    default = f'csi_{title.lower().replace(" ", "_")}_{counter:03d}'
    name = sd.askstring('Export PDF', 'Filename (without .pdf):',
                        initialvalue=default, parent=root)
    if not name:
        _export_single_ax_pdf._n = counter - 1
        return
    name = name.strip()
    if not name.endswith('.pdf'):
        name += '.pdf'
    pub_fig, pub_ax = plt.subplots(figsize=(7, 4))
    pub_fig.patch.set_facecolor('white')
    for line in source_ax.get_lines():
        xd, yd = line.get_data()
        pub_ax.plot(xd, yd, color=line.get_color(),
                    linewidth=max(line.get_linewidth(), 1.0),
                    alpha=line.get_alpha() or 1.0,
                    label=(line.get_label()
                           if not line.get_label().startswith('_') else None),
                    linestyle=line.get_linestyle())
    for im_artist in source_ax.get_images():
        data = im_artist.get_array()
        extent = im_artist.get_extent()
        pub_ax.imshow(data, aspect='auto', origin='lower',
                      cmap=im_artist.get_cmap(), extent=extent,
                      interpolation='bilinear',
                      vmin=im_artist.get_clim()[0],
                      vmax=im_artist.get_clim()[1])
    for coll in source_ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) > 0:
            colors = coll.get_facecolors()
            pub_ax.scatter(offsets[:, 0], offsets[:, 1],
                           s=coll.get_sizes(), c=colors,
                           edgecolors='none', alpha=0.8)
    pub_ax.set_xlim(source_ax.get_xlim())
    pub_ax.set_ylim(source_ax.get_ylim())
    pub_ax.set_xlabel(source_ax.get_xlabel(), fontsize=12, fontfamily='serif')
    pub_ax.set_ylabel(source_ax.get_ylabel(), fontsize=12, fontfamily='serif')
    pub_ax.set_title(title, fontsize=14, fontweight='semibold',
                     fontfamily='serif', pad=10)
    pub_ax.tick_params(labelsize=10, direction='out')
    for sp in ['top', 'right']:
        pub_ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        pub_ax.spines[sp].set_linewidth(0.8)
        pub_ax.spines[sp].set_color('#333333')
    if pub_ax.get_legend_handles_labels()[1]:
        pub_ax.legend(fontsize=10, framealpha=0.8, edgecolor='#cccccc')
    pub_ax.grid(True, alpha=0.15, linewidth=0.5, color='#888888')
    pub_fig.tight_layout()
    pub_fig.savefig(name, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(pub_fig)
    print(f"  [pdf] Exported: {name}")


def _make_pdf_btn(fig, ax_source, title, x, y, w, h):
    """Create a small PDF export button."""
    ax_btn = fig.add_axes([x, y, w, h])
    btn = Button(ax_btn, 'PDF', color=T['CTRL_BG'],
                 hovercolor=T['CTRL_HOVER'])
    btn.label.set_color(T['SUCCESS'])
    btn.label.set_fontsize(FONT_BTN)
    btn.on_clicked(lambda event: _export_single_ax_pdf(fig, ax_source, title))
    return ax_btn, btn


def _make_expand_btn(fig, graph_id, ax_target, other_axes, x, y, w, h, state):
    """Create a small expand/collapse button."""
    ax_btn = fig.add_axes([x, y, w, h])
    btn = Button(ax_btn, '⛶', color=T['CTRL_BG'],
                 hovercolor=T['CTRL_HOVER'])
    btn.label.set_color(T['TEXT_PRIMARY'])
    btn.label.set_fontsize(FONT_BTN + 2)
    
    # Store original position
    original_pos = ax_target.get_position()
    
    # Store original positions in state if not already stored
    if '_original_positions' not in state:
        state['_original_positions'] = {}
    state['_original_positions'][graph_id] = original_pos
    
    # Store button reference in state
    if '_expand_buttons' not in state:
        state['_expand_buttons'] = {}
    state['_expand_buttons'][graph_id] = btn
    
    # Store axes references for each graph
    if '_graph_axes' not in state:
        state['_graph_axes'] = {}
    state['_graph_axes'][graph_id] = {
        'target': ax_target,
        'others': other_axes
    }
    
    def toggle_expand(event):
        current = state['expanded_graph']
        L, R, Top, Bot = 0.055, 0.91, 0.92, 0.10
        
        if current == graph_id:
            # Collapse current
            state['expanded_graph'] = None
            btn.label.set_text('⛶')
            # Restore all axes visibility and positions
            for ax in other_axes:
                ax.set_visible(True)
            ax_target.set_visible(True)
            ax_target.set_position(state['_original_positions'][graph_id])
        else:
            # Collapse previously expanded graph if any
            if current is not None and current in state['_expand_buttons']:
                prev_btn = state['_expand_buttons'][current]
                prev_btn.label.set_text('⛶')
                prev_axes = state['_graph_axes'][current]
                for ax in prev_axes['others']:
                    ax.set_visible(True)
                prev_axes['target'].set_visible(True)
                prev_axes['target'].set_position(state['_original_positions'][current])
            
            # Expand this graph
            state['expanded_graph'] = graph_id
            btn.label.set_text('⛷')
            # Hide other axes
            for ax in other_axes:
                ax.set_visible(False)
            ax_target.set_visible(True)
            # Expand to full tab area
            ax_target.set_position([L, Bot, R - L, Top - Bot])
    
    btn.on_clicked(toggle_expand)
    return ax_btn, btn


def _compute_binned(data_win, ts_win, win_sec, nbins):
    """Shared binning helper."""
    bin_edges = np.linspace(0, win_sec, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    bin_idx = np.clip(np.digitize(ts_win, bin_edges) - 1, 0, nbins - 1)
    counts = np.bincount(bin_idx, minlength=nbins).astype(np.float64)
    pop = counts > 0
    return bin_centers, bin_idx, counts, pop


def _bin_average_1d(values, bin_idx, counts, pop, nbins):
    """Bin-average a 1D array, forward-fill empties."""
    acc = np.zeros(nbins, dtype=np.float64)
    np.add.at(acc, bin_idx, values)
    acc[pop] /= counts[pop]
    for b in np.where(~pop)[0]:
        if b > 0:
            acc[b] = acc[b - 1]
    return acc


def _bin_variance_2d(data_win, bin_idx, counts, pop, nbins, ncols):
    """Per-subcarrier bin variance."""
    s1 = np.zeros((nbins, ncols), dtype=np.float64)
    s2 = np.zeros((nbins, ncols), dtype=np.float64)
    np.add.at(s1, bin_idx, data_win[:, :ncols])
    np.add.at(s2, bin_idx, data_win[:, :ncols] ** 2)
    pop_v = counts > 1
    m = np.zeros_like(s1)
    m[pop_v] = s1[pop_v] / counts[pop_v, None]
    v = np.zeros_like(s1)
    v[pop_v] = s2[pop_v] / counts[pop_v, None] - m[pop_v] ** 2
    return np.clip(v, 0, None)


# =====================================================================
# Main visualization builder
# =====================================================================
def build_ui(initial_label, segment_sec, saver, pca_tracker, model_mgr=None):
    """Build the full matplotlib UI. Returns (fig, updater_fn, get_label_fn)."""

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(19, 10.5))
    fig.patch.set_facecolor(T['BG'])

    title_text = fig.text(
        0.025, 0.965, 'CSI Studio',
        fontsize=FONT_SUPTITLE + 2, fontweight='bold', color=T['ACCENT'],
        fontfamily='sans-serif',
        path_effects=[pe.withStroke(linewidth=2, foreground=T['BG'])])
    subtitle_text = fig.text(
        0.115, 0.968, 'Real-Time Collection & Analysis',
        fontsize=9, color=T['TEXT_SECONDARY'], fontfamily='sans-serif')

    state = {
        'active_tab': 0,
        'win_sec': SLIDING_WINDOW_SEC,
        'label': initial_label,
        'last_seq': 0,
        'fps_times': deque(maxlen=30),
        'theme': 'night',
        'saving': False,
        'expanded_graph': None,
        '_prev_stft': None,
        '_prev_mean': None,
        '_prev_std': None,
        '_prev_var': {},
        '_baseline': None,
        '_baseline_samples': 0,
    }

    sc_colors = plt.cm.turbo(np.linspace(0.05, 0.95, NUM_SUBCARRIERS))

    L, R, Top, Bot = 0.055, 0.91, 0.92, 0.10
    _pdf_x = R + 0.012
    _pdf_w = 0.033
    _pdf_h = 0.020

    # -- Status bar --
    ax_status = fig.add_axes([L, 0.045, R - L, 0.028])
    ax_status.axis('off')
    ax_status.set_facecolor(T['BG'])
    status_text = ax_status.text(
        0.5, 0.5, '', transform=ax_status.transAxes,
        ha='center', va='center', fontsize=FONT_STATS,
        color=T['TEXT_SECONDARY'], family='monospace')

    # -- Segment progress bar --
    ax_seg_bar = fig.add_axes([L, 0.035, R - L, 0.008])
    ax_seg_bar.set_xlim(0, 1); ax_seg_bar.set_ylim(0, 1)
    ax_seg_bar.axis('off'); ax_seg_bar.set_facecolor(T['BG'])
    seg_bar_bg = ax_seg_bar.barh(0.5, 1.0, height=0.8,
                                  color=T['BORDER'], left=0)[0]
    seg_bar_fg = ax_seg_bar.barh(0.5, 0.0, height=0.8,
                                  color=T['SUCCESS'], left=0)[0]

    # ================================================================
    # TAB 0: Signals (heatmap + traces + mean+-std)
    # ================================================================
    gs0 = fig.add_gridspec(3, 2, height_ratios=[3.0, 3.0, 2.5],
                           width_ratios=[1, 0.016],
                           left=L, right=R, top=Top, bottom=Bot,
                           hspace=0.45, wspace=0.03)
    ax_heat = fig.add_subplot(gs0[0, 0])
    ax_cbar = fig.add_subplot(gs0[0, 1])
    ax_lines = fig.add_subplot(gs0[1, 0])
    ax_mean = fig.add_subplot(gs0[2, 0])

    blank_stft = np.zeros((STFT_FREQ_BINS, HEATMAP_COLS))
    im = ax_heat.imshow(blank_stft, aspect='auto', origin='lower',
                        cmap=T['CMAP_HEAT'],
                        extent=[0, state['win_sec'], 0, STFT_FREQ_BINS],
                        interpolation='bilinear', vmin=0, vmax=AMP_MAX)
    ax_heat.set_ylabel('Frequency Bin', fontsize=FONT_LABEL,
                       color=T['TEXT_SECONDARY'])
    ax_heat.set_title('STFT Spectrogram (mean across subcarriers)',
                      fontsize=FONT_TITLE,
                      color=T['TEXT_PRIMARY'], fontweight='semibold', pad=6)
    _style_ax(ax_heat)
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label('dB', fontsize=FONT_LABEL - 2, color=T['TEXT_SECONDARY'])
    cbar.ax.tick_params(colors=T['TEXT_SECONDARY'], labelsize=FONT_TICK - 2)

    sc_line_indices = list(range(NUM_SUBCARRIERS))
    sc_line_artists = []
    for i in sc_line_indices:
        ln, = ax_lines.plot([], [], color=sc_colors[i], linewidth=0.4,
                            alpha=0.55)
        sc_line_artists.append(ln)
    ax_lines.set_ylabel('Amplitude', fontsize=FONT_LABEL,
                        color=T['TEXT_SECONDARY'])
    ax_lines.set_ylim(AMP_MIN, AMP_MAX)
    ax_lines.set_xlim(0, state['win_sec'])
    ax_lines.set_title('Subcarrier Traces', fontsize=FONT_TITLE,
                       color=T['TEXT_PRIMARY'], fontweight='semibold', pad=6)
    ax_lines.grid(True, alpha=0.08, color=T['GRID'], linewidth=0.4)
    _style_ax(ax_lines)

    mean_line, = ax_mean.plot([], [], color=T['ACCENT'], linewidth=1.8,
                              label='Mean')
    std_hi, = ax_mean.plot([], [], color=T['ACCENT'], linewidth=0.5,
                           alpha=0.25)
    std_lo, = ax_mean.plot([], [], color=T['ACCENT'], linewidth=0.5,
                           alpha=0.25)
    ax_mean.set_ylabel('Mean Amp', fontsize=FONT_LABEL,
                       color=T['TEXT_SECONDARY'])
    ax_mean.set_xlabel('Time (s)', fontsize=FONT_LABEL,
                       color=T['TEXT_SECONDARY'])
    ax_mean.set_xlim(0, state['win_sec'])
    ax_mean.set_ylim(AMP_MIN, AMP_MAX)
    ax_mean.grid(True, alpha=0.08, color=T['GRID'], linewidth=0.4)
    ax_mean.legend(loc='upper right', fontsize=FONT_LEGEND, framealpha=0.3)
    ax_mean.set_title('Mean +- Std', fontsize=FONT_TITLE,
                      color=T['TEXT_PRIMARY'], fontweight='semibold', pad=6)
    _style_ax(ax_mean)

    pb0_1, b0_1 = _make_pdf_btn(fig, ax_heat, 'Amplitude Heatmap',
                                 _pdf_x, 0.80, _pdf_w, _pdf_h)
    pb0_2, b0_2 = _make_pdf_btn(fig, ax_lines, 'Subcarrier Traces',
                                 _pdf_x, 0.52, _pdf_w, _pdf_h)
    pb0_3, b0_3 = _make_pdf_btn(fig, ax_mean, 'Mean Std Amplitude',
                                 _pdf_x, 0.24, _pdf_w, _pdf_h)
    
    # Expand buttons
    _expand_x = _pdf_x + _pdf_w + 0.005
    eb0_1, eb0_1_btn = _make_expand_btn(fig, 'heat', ax_heat, 
                                         [ax_cbar, ax_lines, ax_mean],
                                         _expand_x, 0.80, _pdf_w, _pdf_h, state)
    eb0_2, eb0_2_btn = _make_expand_btn(fig, 'lines', ax_lines,
                                         [ax_heat, ax_cbar, ax_mean],
                                         _expand_x, 0.52, _pdf_w, _pdf_h, state)
    eb0_3, eb0_3_btn = _make_expand_btn(fig, 'mean', ax_mean,
                                         [ax_heat, ax_cbar, ax_lines],
                                         _expand_x, 0.24, _pdf_w, _pdf_h, state)

    tab0_axes = [ax_heat, ax_cbar, ax_lines, ax_mean, pb0_1, pb0_2, pb0_3,
                 eb0_1, eb0_2, eb0_3]

    # ================================================================
    # TAB 1: Variance (single full-height heatmap)
    # ================================================================
    ax_var = fig.add_axes([L, Bot, R - L, Top - Bot])
    vblank = np.zeros((NUM_SUBCARRIERS, HEATMAP_COLS))
    var_im = ax_var.imshow(vblank, aspect='auto', origin='lower',
                           cmap=T['CMAP_VAR'], interpolation='bilinear',
                           vmin=0, vmax=AMP_MAX,
                           extent=[0, state['win_sec'], 0, NUM_SUBCARRIERS])
    ax_var.set_ylabel('Subcarrier', fontsize=FONT_LABEL,
                      color=T['TEXT_SECONDARY'])
    ax_var.set_xlabel('Time (s)', fontsize=FONT_LABEL,
                      color=T['TEXT_SECONDARY'])
    ax_var.set_title('Rolling Variance', fontsize=FONT_TITLE,
                     color=T['TEXT_PRIMARY'], fontweight='semibold', pad=6)
    _style_ax(ax_var)
    var_stat_text = ax_var.text(0.98, 0.96, '', transform=ax_var.transAxes,
                                fontsize=FONT_STATS, color=T['ACCENT_WARM'],
                                ha='right', va='top', family='monospace',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor=T['BG'], alpha=0.7))

    pb1_1, b1_1 = _make_pdf_btn(fig, ax_var, 'Rolling Variance',
                                 _pdf_x, 0.50, _pdf_w, _pdf_h)

    tab1_axes = [ax_var, pb1_1]

    # ================================================================
    # TAB 2: PCA (live 2D scatter, updated every 1s)
    # ================================================================
    ax_pca = fig.add_axes([L, Bot, R - L, Top - Bot])
    ax_pca.set_title('Live PCA Embedding (1s snapshots, rolling)',
                     fontsize=FONT_TITLE, color=T['TEXT_PRIMARY'],
                     fontweight='semibold', pad=8)
    ax_pca.set_xlabel('PC 1', fontsize=FONT_LABEL,
                      color=T['TEXT_SECONDARY'])
    ax_pca.set_ylabel('PC 2', fontsize=FONT_LABEL,
                      color=T['TEXT_SECONDARY'])
    ax_pca.grid(True, alpha=0.08, color=T['GRID'], linewidth=0.4)
    _style_ax(ax_pca)
    pca_status_text = ax_pca.text(
        0.5, 0.5, 'Collecting snapshots...',
        transform=ax_pca.transAxes, ha='center', va='center',
        fontsize=14, color=T['TEXT_SECONDARY'], style='italic')

    pb2_1, b2_1 = _make_pdf_btn(fig, ax_pca, 'PCA Embedding',
                                 _pdf_x, 0.50, _pdf_w, _pdf_h)

    tab2_axes = [ax_pca, pb2_1]

    # ================================================================
    # TAB 3: Model (dataset overview, train RF, deploy, live inference)
    # ================================================================
    _m_left = L
    _m_left_w = (R - L) * 0.38          # left column width
    _m_mid = _m_left + _m_left_w + 0.03 # right column start
    _m_right = R
    _ctrl_h = 0.12                       # control strip height

    # -- Left upper: Dataset overview --
    _ds_bot = Bot + _ctrl_h + 0.015
    ax_dataset = fig.add_axes([_m_left, _ds_bot,
                               _m_left_w, Top - _ds_bot])
    ax_dataset.set_xlim(0, 1); ax_dataset.set_ylim(0, 1)
    ax_dataset.axis('off')
    ax_dataset.set_facecolor(T['SURFACE'])
    for sp in ax_dataset.spines.values():
        sp.set_color(T['BORDER']); sp.set_linewidth(0.8)
        sp.set_visible(True)
    ax_dataset.set_title('Dataset', fontsize=FONT_TITLE,
                         color=T['TEXT_PRIMARY'], fontweight='bold',
                         pad=10, loc='left')
    dataset_text = ax_dataset.text(
        0.04, 0.93, 'Scanning...', transform=ax_dataset.transAxes,
        fontsize=FONT_STATS + 1, color=T['TEXT_SECONDARY'], va='top',
        family='monospace', linespacing=1.8)

    # -- Left lower: Control strip --
    ax_params = fig.add_axes([_m_left, Bot,
                              _m_left_w, _ctrl_h])
    ax_params.set_xlim(0, 1); ax_params.set_ylim(0, 1)
    ax_params.axis('off')
    ax_params.set_facecolor(T['SURFACE'])
    for sp in ax_params.spines.values():
        sp.set_color(T['BORDER']); sp.set_linewidth(0.8)
        sp.set_visible(True)
    model_status_text = ax_params.text(
        0.04, 0.15, 'No model trained', transform=ax_params.transAxes,
        fontsize=FONT_STATS, color=T['TEXT_SECONDARY'], va='center',
        family='monospace')

    # -- Right: Live prediction (full height, clear of left controls) --
    ax_probs = fig.add_axes([_m_mid, Bot,
                             _m_right - _m_mid, Top - Bot])
    ax_probs.set_facecolor(T['SURFACE'])
    ax_probs.set_title('Live Prediction', fontsize=FONT_TITLE,
                       color=T['TEXT_PRIMARY'], fontweight='bold',
                       pad=10, loc='left')
    ax_probs.set_xlim(0, 1.05)
    ax_probs.set_xlabel('Probability', fontsize=FONT_LABEL,
                        color=T['TEXT_SECONDARY'])
    _style_ax(ax_probs)
    prob_status_text = ax_probs.text(
        0.5, 0.5, 'No model deployed', transform=ax_probs.transAxes,
        ha='center', va='center', fontsize=16,
        color=T['TEXT_SECONDARY'], style='italic')

    # Model tab state
    _model_state = {
        'label_btns': {},
        'n_trees_val': 100,
        'max_depth_val': 0,
        'test_split_val': 0.2,
    }

    # Controls positioned inside the control strip using figure coords
    _bh = 0.024
    _box_w = 0.055
    _btn_w_ctrl = 0.065
    # Upper row (param boxes): placed at top of control strip
    _by_upper = Bot + _ctrl_h - _bh - 0.012
    # Lower row (action buttons): placed below param boxes
    _by_lower = Bot + 0.030

    # Parameter labels + text boxes (upper row of control strip)
    _param_items = [
        ('Trees', 0, '100', 'n_trees_val', lambda t: max(1, int(t))),
        ('Depth', 1, '0',   'max_depth_val', lambda t: max(0, int(t))),
        ('Test%', 2, '20',  'test_split_val',
         lambda t: max(0, min(90, int(t))) / 100.0),
    ]
    _param_boxes = []
    _param_box_axes = []
    _param_spacing = (_m_left_w - 0.01) / 3.0
    for plabel, pidx, pinit, pkey, pfn in _param_items:
        px = _m_left + 0.005 + pidx * _param_spacing
        fig.text(px, _by_upper + _bh + 0.004, plabel, fontsize=7,
                 color=T['TEXT_SECONDARY'], va='bottom')
        ax_p = fig.add_axes([px, _by_upper, _box_w, _bh])
        box = TextBox(ax_p, '', initial=pinit,
                      color=T['CTRL_BG'], hovercolor=T['CTRL_HOVER'])
        box.text_disp.set_color(T['ACCENT_WARM'])
        box.text_disp.set_fontsize(FONT_BTN + 1)

        def _make_param_cb(key, fn):
            def _cb(text):
                try:
                    _model_state[key] = fn(text.strip())
                except (ValueError, TypeError):
                    pass
            return _cb

        box.on_submit(_make_param_cb(pkey, pfn))
        _param_boxes.append(box)
        _param_box_axes.append(ax_p)

    # Action buttons: Train, Deploy, Rescan (lower row of control strip)
    _btn_specs = [
        ('Train', 0, T['SUCCESS'], 'bold'),
        ('Deploy', 1, T['ACCENT'], 'bold'),
        ('Rescan', 2, T['TEXT_SECONDARY'], 'normal'),
    ]
    _action_btns = []
    _action_btn_axes = []
    _btn_spacing = (_m_left_w - 0.01) / 3.0
    for blabel, bidx, bcolor, bweight in _btn_specs:
        bx = _m_left + 0.005 + bidx * _btn_spacing
        ax_b = fig.add_axes([bx, _by_lower, _btn_w_ctrl, _bh])
        btn = Button(ax_b, blabel,
                     color=T['CTRL_BG'], hovercolor=T['CTRL_HOVER'])
        btn.label.set_fontsize(FONT_BTN + 1)
        btn.label.set_color(bcolor)
        btn.label.set_fontweight(bweight)
        _action_btns.append(btn)
        _action_btn_axes.append(ax_b)

    train_btn, deploy_btn, rescan_btn = _action_btns

    def _on_train(event):
        if model_mgr is None:
            return
        model_mgr.scan()
        ms = _model_state
        ok, msg = model_mgr.train(
            n_estimators=ms['n_trees_val'],
            max_depth=ms['max_depth_val'] or None,
            test_split=ms['test_split_val'])
        model_status_text.set_text(msg)
        if ok:
            model_status_text.set_color(T['SUCCESS'])
            deploy_btn.label.set_text('Deploy')
            deploy_btn.label.set_color(T['ACCENT'])
        else:
            model_status_text.set_color(T['DANGER'])
        fig.canvas.draw_idle()

    train_btn.on_clicked(_on_train)

    def _on_deploy(event):
        if model_mgr is None:
            return
        if model_mgr.deployed:
            model_mgr.undeploy()
            deploy_btn.label.set_text('Deploy')
            deploy_btn.label.set_color(T['ACCENT'])
            prob_status_text.set_text('Model removed')
            prob_status_text.set_visible(True)
            model_status_text.set_text(
                f'RF ready  test={model_mgr.test_accuracy:.1%}')
        else:
            if model_mgr.deploy():
                deploy_btn.label.set_text('Undeploy')
                deploy_btn.label.set_color(T['DANGER'])
                prob_status_text.set_text('Deployed — waiting...')
                model_status_text.set_text(
                    f'DEPLOYED  test={model_mgr.test_accuracy:.1%}')
                model_status_text.set_color(T['SUCCESS'])
            else:
                model_status_text.set_text('Train a model first')
                model_status_text.set_color(T['DANGER'])
        fig.canvas.draw_idle()

    deploy_btn.on_clicked(_on_deploy)

    def _on_rescan(event):
        if model_mgr is not None:
            _refresh_dataset_display()
        fig.canvas.draw_idle()

    rescan_btn.on_clicked(_on_rescan)

    # Label toggle buttons (dynamically created per scan)
    _lbl_toggle_axes = []
    _lbl_toggle_btns = []

    def _refresh_dataset_display():
        if model_mgr is None:
            return
        model_mgr.scan()
        ds = model_mgr.dataset

        # Remove old toggle buttons
        for ax_old in _lbl_toggle_axes:
            ax_old.remove()
        _lbl_toggle_axes.clear()
        _lbl_toggle_btns.clear()
        _model_state['label_btns'].clear()

        sorted_labels = sorted(ds.keys(), key=lambda l: -len(ds[l]))
        total = sum(len(v) for v in ds.values())
        incl_labels = model_mgr.get_included_labels()
        incl_n = sum(len(ds.get(l, [])) for l in incl_labels)

        # Build clean text
        lines = []
        for lbl in sorted_labels:
            n = len(ds[lbl])
            excl = lbl in model_mgr.excluded_labels
            icon = '\u2717' if excl else '\u2713'
            tag = '  EXCL' if excl else ''
            lines.append(f' {icon}  {lbl:<14s} {n:>5d} files{tag}')
        lines.append(f' {"":─<36s}')
        lines.append(f'    {"Total":<14s} {total:>5d} files')
        lines.append(f'    {"Selected":<14s} {incl_n:>5d} files  '
                     f'({len(incl_labels)} labels)')
        dataset_text.set_text('\n'.join(lines))

        # Create toggle buttons aligned with dataset text rows
        _ds = ax_dataset.get_position()
        _btn_w = 0.048
        _btn_h = 0.022
        _btn_x = _ds.x0 + _ds.width - _btn_w - 0.008
        _row_h = 0.030
        _first_y = _ds.y0 + _ds.height - 0.048

        for i, lbl in enumerate(sorted_labels):
            y = _first_y - i * _row_h
            if y < _ds.y0 + 0.01:
                break
            excl = lbl in model_mgr.excluded_labels
            ax_t = fig.add_axes([_btn_x, y, _btn_w, _btn_h])
            label_txt = 'incl' if excl else 'excl'
            clr = T['SUCCESS'] if excl else T['DANGER']
            btn_t = Button(ax_t, label_txt,
                           color=T['CTRL_BG'], hovercolor=T['CTRL_HOVER'])
            btn_t.label.set_fontsize(FONT_BTN)
            btn_t.label.set_color(clr)
            _lbl_toggle_axes.append(ax_t)
            _lbl_toggle_btns.append(btn_t)

            def _make_toggle(lname, bt):
                def _cb(event):
                    if lname in model_mgr.excluded_labels:
                        model_mgr.excluded_labels.discard(lname)
                    else:
                        model_mgr.excluded_labels.add(lname)
                    _refresh_dataset_display()
                    fig.canvas.draw_idle()
                return _cb

            btn_t.on_clicked(_make_toggle(lbl, btn_t))

    # Initial scan
    if model_mgr is not None:
        _refresh_dataset_display()

    tab3_axes = [ax_dataset, ax_params, ax_probs] + \
                _param_box_axes + _action_btn_axes

    _set_axes_visible(tab1_axes, False)
    _set_axes_visible(tab2_axes, False)
    _set_axes_visible(tab3_axes, False)
    for ax_t in _lbl_toggle_axes:
        ax_t.set_visible(False)

    # ================================================================
    # Toolbar
    # ================================================================
    ax_toolbar_bg = fig.add_axes([0.0, 0.0, 1.0, 0.035])
    ax_toolbar_bg.set_facecolor(T['BG'])
    ax_toolbar_bg.set_xticks([]); ax_toolbar_bg.set_yticks([])
    for sp in ax_toolbar_bg.spines.values():
        sp.set_visible(False)

    _tab_w, _tab_h, _tab_y = 0.065, 0.022, 0.006
    _tab_gap = 0.003
    tab_names = ['Signals', 'Variance', 'PCA', 'Model']
    tab_colors_list = [T['ACCENT_WARM'], '#b388ff', T['SUCCESS'], T['ACCENT']]
    tab_groups = [tab0_axes, tab1_axes, tab2_axes, tab3_axes]
    tab_btn_list = []
    tab_ax_list = []

    for ti, tname in enumerate(tab_names):
        ax_tb = fig.add_axes([0.012 + ti * (_tab_w + _tab_gap), _tab_y,
                              _tab_w, _tab_h])
        is_active = (ti == 0)
        btn = Button(ax_tb, tname,
                     color=T['TAB_ACTIVE'] if is_active else T['CTRL_BG'],
                     hovercolor=T['CTRL_HOVER'])
        btn.label.set_fontsize(FONT_BTN)
        btn.label.set_color(
            tab_colors_list[ti] if is_active else T['TEXT_SECONDARY'])
        if is_active:
            btn.label.set_fontweight('bold')
        tab_btn_list.append(btn)
        tab_ax_list.append(ax_tb)

    def _switch_tab(idx):
        def _cb(event):
            state['active_tab'] = idx
            for i, grp in enumerate(tab_groups):
                _set_axes_visible(grp, i == idx)
            # Show/hide dynamic label toggle buttons for Model tab
            for ax_t in _lbl_toggle_axes:
                ax_t.set_visible(idx == 3)
            # Reset expanded graph when leaving tab 0
            if idx != 0 and state['expanded_graph'] is not None:
                # Collapse the expanded graph
                graph_id = state['expanded_graph']
                if graph_id in state['_expand_buttons']:
                    btn = state['_expand_buttons'][graph_id]
                    btn.label.set_text('⛶')
                if graph_id in state['_graph_axes']:
                    axes_info = state['_graph_axes'][graph_id]
                    for ax in axes_info['others']:
                        ax.set_visible(True)
                    axes_info['target'].set_visible(True)
                    axes_info['target'].set_position(state['_original_positions'][graph_id])
                state['expanded_graph'] = None
            for i, (b, bax) in enumerate(zip(tab_btn_list, tab_ax_list)):
                if i == idx:
                    bax.set_facecolor(T['TAB_ACTIVE'])
                    b.label.set_fontweight('bold')
                    b.label.set_color(tab_colors_list[i])
                else:
                    bax.set_facecolor(T['CTRL_BG'])
                    b.label.set_fontweight('normal')
                    b.label.set_color(T['TEXT_SECONDARY'])
            # Rescan dataset when switching to Model tab
            if idx == 3 and model_mgr is not None:
                _refresh_dataset_display()
            ax_status.set_visible(True)
            ax_seg_bar.set_visible(True)
            ax_toolbar_bg.set_visible(True)
            fig.canvas.draw_idle()
        return _cb

    for ti in range(len(tab_names)):
        tab_btn_list[ti].on_clicked(_switch_tab(ti))

    _sep1_x = 0.012 + len(tab_names) * (_tab_w + _tab_gap) + 0.004
    fig.text(_sep1_x, 0.017, '|', fontsize=10, color=T['BORDER'],
             va='center')

    # Window buttons
    _win_x0 = _sep1_x + 0.012
    fig.text(_win_x0 - 0.002, 0.017, 'Window:', fontsize=7,
             color=T['TEXT_SECONDARY'], va='center')
    _win_options = [5.0, 10.0, 20.0, 30.0, 60.0]
    _win_btns = []
    for wi, wval in enumerate(_win_options):
        ax_w = fig.add_axes([_win_x0 + 0.038 + wi * 0.036, _tab_y,
                             0.033, _tab_h])
        b = Button(ax_w, f'{int(wval)}s', color=T['CTRL_BG'],
                   hovercolor=T['CTRL_HOVER'])
        b.label.set_fontsize(FONT_BTN - 1)
        b.label.set_color(T['ACCENT_WARM'] if wval == state['win_sec']
                          else T['TEXT_SECONDARY'])
        _win_btns.append(b)

    def _make_win_cb(wval, wi):
        def _cb(event):
            state['win_sec'] = wval
            for j, b in enumerate(_win_btns):
                b.label.set_color(T['ACCENT_WARM'] if j == wi
                                  else T['TEXT_SECONDARY'])
            for ax in [ax_heat, ax_lines, ax_mean, ax_var]:
                ax.set_xlim(0, wval)
            fig.canvas.draw_idle()
        return _cb

    for wi, wval in enumerate(_win_options):
        _win_btns[wi].on_clicked(_make_win_cb(wval, wi))

    _sep2_x = _win_x0 + 0.038 + len(_win_options) * 0.036 + 0.006
    fig.text(_sep2_x, 0.017, '|', fontsize=10, color=T['BORDER'],
             va='center')

    # Label quick-select buttons + custom entry
    _lbl_x0 = _sep2_x + 0.010
    fig.text(_lbl_x0 - 0.002, 0.017, 'Label:', fontsize=7,
             color=T['TEXT_SECONDARY'], va='center')
    _LABEL_PRESETS = ['walk', 'sit', 'stand', 'empty']
    _lbl_btns = []
    _lbl_btn_axes = []
    _lbl_bw = 0.038

    def _set_label(new_label):
        state['label'] = new_label
        # Highlight active label button
        for j, lb in enumerate(_lbl_btns):
            if _LABEL_PRESETS[j] == new_label:
                lb.label.set_color(T['ACCENT_WARM'])
                lb.label.set_fontweight('bold')
            else:
                lb.label.set_color(T['TEXT_SECONDARY'])
                lb.label.set_fontweight('normal')
        label_box.set_val(new_label)

    for li, lname in enumerate(_LABEL_PRESETS):
        ax_lb = fig.add_axes([_lbl_x0 + 0.030 + li * (_lbl_bw + 0.002),
                              _tab_y, _lbl_bw, _tab_h])
        lb = Button(ax_lb, lname, color=T['CTRL_BG'],
                    hovercolor=T['CTRL_HOVER'])
        lb.label.set_fontsize(FONT_BTN)
        if lname == initial_label:
            lb.label.set_color(T['ACCENT_WARM'])
            lb.label.set_fontweight('bold')
        else:
            lb.label.set_color(T['TEXT_SECONDARY'])
        _lbl_btns.append(lb)
        _lbl_btn_axes.append(ax_lb)

    def _make_lbl_cb(lname):
        def _cb(event):
            _set_label(lname)
        return _cb

    for li, lname in enumerate(_LABEL_PRESETS):
        _lbl_btns[li].on_clicked(_make_lbl_cb(lname))

    # Custom label text box (smaller, after presets)
    _custom_x = _lbl_x0 + 0.030 + len(_LABEL_PRESETS) * (_lbl_bw + 0.002) + 0.004
    ax_label_box = fig.add_axes([_custom_x, _tab_y, 0.065, _tab_h])
    label_box = TextBox(ax_label_box, '', initial=initial_label,
                        color=T['CTRL_BG'], hovercolor=T['CTRL_HOVER'])
    label_box.text_disp.set_color(T['ACCENT_WARM'])
    label_box.text_disp.set_fontsize(FONT_BTN + 1)
    label_box.text_disp.set_fontweight('bold')

    def _on_label_change(text):
        new = text.strip() or 'unlabeled'
        state['label'] = new
        for j, lb in enumerate(_lbl_btns):
            if _LABEL_PRESETS[j] == new:
                lb.label.set_color(T['ACCENT_WARM'])
                lb.label.set_fontweight('bold')
            else:
                lb.label.set_color(T['TEXT_SECONDARY'])
                lb.label.set_fontweight('normal')
    label_box.on_submit(_on_label_change)

    # Day/Night toggle
    _sep3_x = _custom_x + 0.075
    fig.text(_sep3_x, 0.017, '|', fontsize=10, color=T['BORDER'],
             va='center')
    ax_theme_btn = fig.add_axes([_sep3_x + 0.008, _tab_y, 0.05, _tab_h])
    theme_btn = Button(ax_theme_btn, 'Day',
                       color=T['CTRL_BG'], hovercolor=T['CTRL_HOVER'])
    theme_btn.label.set_fontsize(FONT_BTN)
    theme_btn.label.set_color(T['ACCENT'])

    def _toggle_theme(event):
        new = 'day' if state['theme'] == 'night' else 'night'
        state['theme'] = new
        T.update(THEMES[new])
        theme_btn.label.set_text('Night' if new == 'day' else 'Day')
        fig.patch.set_facecolor(T['BG'])
        ax_toolbar_bg.set_facecolor(T['BG'])
        ax_status.set_facecolor(T['BG'])
        ax_seg_bar.set_facecolor(T['BG'])
        status_text.set_color(T['TEXT_SECONDARY'])
        title_text.set_color(T['ACCENT'])
        subtitle_text.set_color(T['TEXT_SECONDARY'])
        for ax in [ax_heat, ax_lines, ax_mean, ax_pca, ax_var, ax_probs]:
            _style_ax(ax)
        for ax in [ax_dataset, ax_params]:
            ax.set_facecolor(T['SURFACE'])
        im.set_cmap(T['CMAP_HEAT'])
        var_im.set_cmap(T['CMAP_VAR'])
        mean_line.set_color(T['ACCENT'])
        std_hi.set_color(T['ACCENT'])
        std_lo.set_color(T['ACCENT'])
        # Reset smooth caches so new theme applies cleanly
        state['_prev_stft'] = None
        state['_prev_var'] = {}
        state['_baseline'] = None
        state['_baseline_samples'] = 0
        fig.canvas.draw_idle()

    theme_btn.on_clicked(_toggle_theme)

    # Save / View-only toggle
    _sep4_x = _sep3_x + 0.066
    fig.text(_sep4_x, 0.017, '|', fontsize=10, color=T['BORDER'],
             va='center')
    ax_save_btn = fig.add_axes([_sep4_x + 0.008, _tab_y, 0.055, _tab_h])
    save_btn = Button(ax_save_btn, 'View Only',
                      color=T['CTRL_BG'], hovercolor=T['CTRL_HOVER'])
    save_btn.label.set_fontsize(FONT_BTN)
    save_btn.label.set_color(T['TEXT_SECONDARY'])
    save_btn.label.set_fontweight('normal')

    def _toggle_saving(event):
        state['saving'] = not state['saving']
        if state['saving']:
            save_btn.label.set_text('Save: ON')
            save_btn.label.set_color(T['SUCCESS'])
            if saver is not None:
                saver.enabled = True
        else:
            save_btn.label.set_text('View Only')
            save_btn.label.set_color(T['TEXT_SECONDARY'])
            if saver is not None:
                saver.enabled = False
        fig.canvas.draw_idle()

    save_btn.on_clicked(_toggle_saving)

    rec_dot = fig.text(0.955, 0.017, '\u25cf  REC', fontsize=8,
                       color=T['DANGER'], fontweight='bold', va='center',
                       fontfamily='monospace')

    fig._studio_widgets = (tab_btn_list, _win_btns, b0_1, b0_2, b0_3,
                           b1_1, b2_1, label_box, theme_btn, save_btn,
                           _lbl_btns, ax_toolbar_bg, rec_dot,
                           train_btn, deploy_btn, rescan_btn,
                           _param_boxes, _lbl_toggle_btns)

    # ================================================================
    # Update function (called every frame ~20 fps)
    # ================================================================
    _alpha = SMOOTH_ALPHA
    _min_pkt_delta = 3  # skip frame if fewer than 3 new packets

    def update(buf):
        seq = buf.count
        if seq < 2:
            return
        delta = seq - state['last_seq']
        if delta < _min_pkt_delta:
            return
        state['last_seq'] = seq
        state['fps_times'].append(time.time())

        win_sec = state['win_sec']

        # Update baseline (long-term running average over BASELINE_WINDOW_SEC)
        baseline_win = BASELINE_WINDOW_SEC
        amps_bl, wt_bl = buf.snapshot_amps(max_age_sec=baseline_win + 1.0)
        if len(amps_bl) >= 10:
            ncols_bl = min(NUM_SUBCARRIERS, amps_bl.shape[1])
            baseline_mean = amps_bl[:, :ncols_bl].mean(axis=0)
            if state['_baseline'] is None:
                state['_baseline'] = baseline_mean.copy()
                state['_baseline_samples'] = len(amps_bl)
            else:
                # Exponential moving average for smooth baseline updates
                alpha_bl = 0.02
                state['_baseline'] = (alpha_bl * baseline_mean +
                                     (1 - alpha_bl) * state['_baseline'])
                state['_baseline_samples'] += len(amps_bl)

        # -- Tab 0: Signals --
        if state['active_tab'] == 0:
            amps, wt = buf.snapshot_amps(max_age_sec=win_sec + 1.0)
            if len(amps) < 2:
                return
            ncols = min(NUM_SUBCARRIERS, amps.shape[1])
            t_rel = wt - wt[0]
            t_now = t_rel[-1]
            t_start = max(0.0, t_now - win_sec)
            mask = t_rel >= t_start
            data_win = amps[mask]
            ts_win = t_rel[mask] - t_start
            if len(data_win) < 2:
                return

            # Subtract baseline if available
            if state['_baseline'] is not None and len(state['_baseline']) >= ncols:
                data_win = data_win - state['_baseline'][:ncols]
                # Take absolute value to measure deviation from baseline
                data_win = np.abs(data_win)

            # STFT spectrogram (mean amplitude across subcarriers)
            mean_sig = data_win[:, :ncols].mean(axis=1)
            n_samples = len(mean_sig)
            nfft = STFT_NFFT
            hop = STFT_HOP
            n_frames = max(1, (n_samples - nfft) // hop + 1)
            if n_frames >= 2 and n_samples >= nfft:
                window = np.hanning(nfft)
                spec = np.zeros((STFT_FREQ_BINS, n_frames))
                for fi in range(n_frames):
                    seg = mean_sig[fi * hop: fi * hop + nfft]
                    if len(seg) < nfft:
                        break
                    ft = np.abs(np.fft.rfft(seg * window))[:STFT_FREQ_BINS]
                    spec[:, fi] = ft
                spec_db = 20.0 * np.log10(spec + 1e-10)
                spec_db = np.clip(spec_db, 0, AMP_MAX)
                if n_frames != HEATMAP_COLS:
                    x_old = np.linspace(0, 1, n_frames)
                    x_new = np.linspace(0, 1, HEATMAP_COLS)
                    disp = np.zeros((STFT_FREQ_BINS, HEATMAP_COLS))
                    for row in range(STFT_FREQ_BINS):
                        disp[row] = np.interp(x_new, x_old, spec_db[row])
                else:
                    disp = spec_db
                prev = state['_prev_stft']
                if prev is not None and prev.shape == disp.shape:
                    disp = _alpha * disp + (1 - _alpha) * prev
                state['_prev_stft'] = disp.copy()
                im.set_data(disp)
                im.set_extent([0, win_sec, 0, STFT_FREQ_BINS])

            # Subcarrier traces (higher-res bins for smooth curves)
            tbc, tbi, tct, tpo = _compute_binned(
                data_win, ts_win, win_sec, TRACE_BINS)
            for li, sci in enumerate(sc_line_indices):
                if sci < ncols:
                    sc_bin = _bin_average_1d(
                        data_win[:, sci], tbi, tct, tpo, TRACE_BINS)
                    sc_line_artists[li].set_data(
                        tbc, np.clip(sc_bin, AMP_MIN, AMP_MAX))

            # Mean+-std with smoothing (same high-res bins)
            row_means = data_win[:, :ncols].mean(axis=1)
            bm = _bin_average_1d(row_means, tbi, tct, tpo, TRACE_BINS)
            bsq = _bin_average_1d(
                row_means ** 2, tbi, tct, tpo, TRACE_BINS)
            bstd = np.sqrt(np.clip(bsq - bm ** 2, 0, None))
            if (state['_prev_mean'] is not None and
                    len(state['_prev_mean']) == TRACE_BINS):
                bm = _alpha * bm + (1 - _alpha) * state['_prev_mean']
                bstd = _alpha * bstd + (1 - _alpha) * state['_prev_std']
            state['_prev_mean'] = bm.copy()
            state['_prev_std'] = bstd.copy()
            mean_line.set_data(tbc, bm)
            std_hi.set_data(tbc,
                            np.clip(bm + bstd, AMP_MIN, AMP_MAX))
            std_lo.set_data(tbc,
                            np.clip(bm - bstd, AMP_MIN, AMP_MAX))

        # -- Tab 1: Variance (single window) --
        elif state['active_tab'] == 1:
            vw = win_sec
            amps_v, wt_v = buf.snapshot_amps(max_age_sec=vw + 1.0)
            if len(amps_v) >= 5:
                ncols = min(NUM_SUBCARRIERS, amps_v.shape[1])
                tr = wt_v - wt_v[0]
                tn = tr[-1]
                ts_ = max(0.0, tn - vw)
                dw = amps_v[tr >= ts_]
                tw = tr[tr >= ts_] - ts_
                if len(dw) >= 3:
                    # Subtract baseline if available
                    if state['_baseline'] is not None and len(state['_baseline']) >= ncols:
                        dw = dw - state['_baseline'][:ncols]
                        # Take absolute value to measure deviation from baseline
                        dw = np.abs(dw)
                    bc, bi, ct, po = _compute_binned(
                        dw, tw, vw, HEATMAP_COLS)
                    vdata = _bin_variance_2d(
                        dw, bi, ct, po, HEATMAP_COLS, ncols)
                    vt = vdata.T
                    prev_v = state['_prev_var'].get(0)
                    if prev_v is not None and prev_v.shape == vt.shape:
                        vt = _alpha * vt + (1 - _alpha) * prev_v
                    state['_prev_var'][0] = vt.copy()
                    var_im.set_data(vt)
                    var_im.set_extent([0, vw, 0, NUM_SUBCARRIERS])
                    ax_var.set_xlim(0, vw)
                    var_stat_text.set_text(
                        f'avg={vt.mean():.1f}  max={vt.max():.1f}')

        # -- Tab 2: PCA --
        elif state['active_tab'] == 2:
            pca_tracker.maybe_update(buf, state['label'], segment_sec)
            emb = pca_tracker.get_embedding()
            if emb is not None:
                coords, labels, label_set, ages = emb
                ax_pca.cla()
                ax_pca.set_title(
                    'Live PCA Embedding',
                    fontsize=FONT_TITLE, color=T['TEXT_PRIMARY'],
                    fontweight='semibold', pad=8)
                ax_pca.set_xlabel('PC 1', fontsize=FONT_LABEL,
                                  color=T['TEXT_SECONDARY'])
                ax_pca.set_ylabel('PC 2', fontsize=FONT_LABEL,
                                  color=T['TEXT_SECONDARY'])
                ax_pca.grid(True, alpha=0.08, color=T['GRID'],
                            linewidth=0.4)
                _style_ax(ax_pca)

                for lbl, lidx in label_set.items():
                    idxs = [i for i, l in enumerate(labels) if l == lbl]
                    if not idxs:
                        continue
                    c = LABEL_CMAP(lidx % 9)
                    m = LABEL_MARKERS[lidx % len(LABEL_MARKERS)]
                    pts = coords[idxs]
                    pt_ages = ages[idxs]
                    # Tail line connecting sequential points (fading)
                    if len(pts) > 1:
                        for k in range(len(pts) - 1):
                            a = 0.08 + 0.35 * (1.0 - pt_ages[k])
                            ax_pca.plot(
                                [pts[k, 0], pts[k + 1, 0]],
                                [pts[k, 1], pts[k + 1, 1]],
                                color=c, linewidth=0.8, alpha=a)
                    # Scatter dots with age-based size and alpha
                    alphas = 0.15 + 0.75 * (1.0 - pt_ages)
                    sizes = 15 + 55 * (1.0 - pt_ages)
                    ax_pca.scatter(pts[:, 0], pts[:, 1],
                                   c=[c] * len(idxs), marker=m,
                                   s=sizes, alpha=float(alphas.mean()),
                                   edgecolors='white', linewidth=0.3,
                                   label=f'{lbl} ({len(idxs)})',
                                   zorder=3)
                ax_pca.legend(loc='upper right', fontsize=FONT_LEGEND,
                              framealpha=0.5, edgecolor=T['BORDER'])
                pca_status_text.set_visible(False)
            else:
                n_snap = len(pca_tracker._snapshots)
                pca_status_text.set_text(
                    f'Collecting snapshots... ({n_snap}, need >=3)')
                pca_status_text.set_visible(True)

        # -- Tab 3: Model (live inference) --
        elif state['active_tab'] == 3:
            if model_mgr is not None and model_mgr.deployed:
                model_mgr.predict_from_buffer(buf, saver)
                probs = model_mgr.last_probs
                pred = model_mgr.last_pred
                if probs is not None and len(model_mgr.classes) > 0:
                    ax_probs.cla()
                    ax_probs.set_title('Live Prediction',
                                       fontsize=FONT_TITLE,
                                       color=T['TEXT_PRIMARY'],
                                       fontweight='semibold', pad=6)
                    _style_ax(ax_probs)
                    classes = model_mgr.classes
                    y_pos = np.arange(len(classes))
                    colors = []
                    for ci, cls in enumerate(classes):
                        if cls == pred:
                            colors.append(T['SUCCESS'])
                        else:
                            colors.append(T['ACCENT'])
                    bars = ax_probs.barh(y_pos, probs, height=0.6,
                                         color=colors, edgecolor='none')
                    ax_probs.set_yticks(y_pos)
                    ax_probs.set_yticklabels(classes, fontsize=FONT_LABEL,
                                              color=T['TEXT_PRIMARY'])
                    ax_probs.set_xlim(0, 1.05)
                    ax_probs.set_xlabel('Probability', fontsize=FONT_LABEL,
                                        color=T['TEXT_SECONDARY'])
                    # Annotate bars with percentage
                    for bi, (bar, p) in enumerate(zip(bars, probs)):
                        ax_probs.text(
                            bar.get_width() + 0.02, bi,
                            f'{p:.1%}', va='center',
                            fontsize=FONT_STATS, fontweight='bold',
                            color=T['TEXT_PRIMARY'])
                    # Show predicted label prominently
                    ax_probs.text(
                        0.98, 0.02,
                        f'Prediction: {pred}',
                        transform=ax_probs.transAxes,
                        ha='right', va='bottom',
                        fontsize=FONT_TITLE + 2, fontweight='bold',
                        color=T['SUCCESS'],
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor=T['BG'], alpha=0.8))
                    prob_status_text.set_visible(False)
                else:
                    prob_status_text.set_text(
                        'Deployed — waiting for data...')
                    prob_status_text.set_visible(True)

        # -- Segment progress bar --
        if saver is not None and saver._segment_wall_start is not None:
            if state['saving']:
                seg_elapsed = time.time() - saver._segment_wall_start
                seg_frac = min(seg_elapsed / segment_sec, 1.0)
                seg_bar_fg.set_width(seg_frac)
                seg_bar_fg.set_color(T['SUCCESS'])
                if int(time.time() * 2) % 2 == 0:
                    rec_dot.set_color(T['DANGER'])
                else:
                    rec_dot.set_color('#661a17' if state['theme'] == 'night'
                                      else '#ffcccc')
                rec_dot.set_text('\u25cf  REC')
            else:
                seg_bar_fg.set_width(0)
                rec_dot.set_color(T['TEXT_SECONDARY'])
                rec_dot.set_text('VIEW')

        # -- Status text --
        fps = 0.0
        ft = state['fps_times']
        if len(ft) >= 2:
            dt_fps = ft[-1] - ft[0]
            if dt_fps > 0:
                fps = (len(ft) - 1) / dt_fps
        n_saved = saver.file_count if saver is not None else 0
        tab_tag = ['SIG', 'VAR', 'PCA', 'MDL'][state['active_tab']]
        baseline_status = ('BL:ON' if state['_baseline'] is not None
                           else f'BL:CALC({state["_baseline_samples"]/150:.0f}s)')
        status_text.set_text(
            f'[{tab_tag}]  Pkts: {seq:,}  |  '
            f'FPS: {fps:.0f}  |  '
            f'Label: {state["label"]}  |  '
            f'{baseline_status}  |  '
            f'Saved: {n_saved}  |  Win: {win_sec:.0f}s')

        fig.canvas.draw_idle()

    def get_label():
        return state['label']

    return fig, update, get_label


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="CSI Studio - Real-time collection, visualization, "
                    "and recording.")
    ap.add_argument("--rx-port", required=True,
                    help="Serial port of receiver ESP32")
    ap.add_argument("--tx-port", default=None,
                    help="(Optional) sender ESP32 port (reset only)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", default="unlabeled",
                    help="Initial activity label (changeable in UI)")
    ap.add_argument("--segment", type=float, default=DEFAULT_SEGMENT_SEC,
                    help=f"Segment duration in seconds "
                         f"(default: {DEFAULT_SEGMENT_SEC})")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR,
                    help=f"Guaranteed SR for saved files "
                         f"(default: {DEFAULT_SR})")
    ap.add_argument("--out-dir", default="./csi_studio_data",
                    help="Output directory for saved segments")
    ap.add_argument("--no-reset", action="store_true",
                    help="Skip board reset pulses")
    args = ap.parse_args()

    w = 62
    print(f"\u2554{'=' * w}\u2557")
    print(f"\u2551  CSI Studio - Real-Time Collection & Analysis"
          f"{' ' * (w - 48)}\u2551")
    print(f"\u2560{'=' * w}\u2563")
    print(f"\u2551  Receiver : {args.rx_port:<{w - 14}}\u2551")
    print(f"\u2551  Baud     : {args.baud:<{w - 14}}\u2551")
    print(f"\u2551  Label    : {args.label:<{w - 14}}\u2551")
    seg_str = f"{args.segment}s"
    print(f"\u2551  Segment  : {seg_str:<{w - 14}}\u2551")
    sr_str = f"{args.sr} Hz"
    print(f"\u2551  SR       : {sr_str:<{w - 14}}\u2551")
    print(f"\u2551  Out Dir  : {args.out_dir:<{w - 14}}\u2551")
    if not HAS_SKLEARN:
        print(f"\u2551  PCA      : DISABLED (pip install scikit-learn)"
              f"{' ' * (w - 49)}\u2551")
    else:
        print(f"\u2551  PCA      : Enabled (1s live, rolling {PCA_ROLLING_SEGMENTS} seg)"
              f"{' ' * max(0, w - 46 - len(str(PCA_ROLLING_SEGMENTS)))}\u2551")
    print(f"\u255a{'=' * w}\u255d")

    ports = [p.device for p in list_ports.comports()]
    print(f"  Available ports: {', '.join(ports) or 'none'}")

    if not args.no_reset:
        reset_board(args.rx_port, args.baud, "receiver")
        if args.tx_port:
            reset_board(args.tx_port, args.baud, "sender")

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    buf = RawCSIBuffer(maxlen=15000)
    pca_tracker = PCATracker()
    model_mgr = ModelManager(args.out_dir)

    # Mutable label ref so saver always gets the current UI label
    _label_ref = [args.label]

    saver = SegmentSaver(buf, args.out_dir, args.segment, args.sr,
                         lambda: _label_ref[0])

    fig, update_fn, get_label_fn = build_ui(
        args.label, args.segment, saver, pca_tracker, model_mgr)

    # Now wire the UI's live label back into the saver's ref
    _label_ref[0] = get_label_fn()
    saver.get_label = get_label_fn

    # Wire PCA: saved segments also feed PCA
    _orig_save = saver._save_segment

    def _save_with_pca(seg_start, seg_end):
        _orig_save(seg_start, seg_end)
        last = saver.last_saved
        if last is not None:
            try:
                data = np.load(last['npz'])
                pca_tracker.add_segment(last['label'], data['mag'])
            except Exception:
                pass

    saver._save_segment = _save_with_pca

    t_serial = threading.Thread(
        target=serial_reader,
        args=(args.rx_port, args.baud, buf, stop_event),
        daemon=True)
    t_serial.start()
    saver.start(stop_event)

    print("  [info] Serial reader started")
    print("  [info] Segment saver started")
    print("  [info] Visualization starting...")

    plt.ion()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    target_dt = 0.050  # ~20 fps (lighter on CPU)

    try:
        tk_widget = fig.canvas.get_tk_widget()
    except Exception:
        tk_widget = None

    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()
            update_fn(buf)
            elapsed = time.time() - t0
            sleep_t = max(0.002, target_dt - elapsed)
            if tk_widget is not None:
                end_t = time.time() + sleep_t
                while time.time() < end_t:
                    tk_widget.update()
                    time.sleep(0.001)
            else:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("\n  [info] Interrupted")
    finally:
        stop_event.set()
        time.sleep(0.3)

    summary = {
        'session_id': saver._session_id,
        'label_initial': args.label,
        'segment_sec': args.segment,
        'guaranteed_sr': args.sr,
        'subcarriers': NUM_SUBCARRIERS,
        'total_packets': buf.count,
        'total_segments_saved': saver.file_count,
        'files': saver.saved_files,
        'errors': len(buf.errors),
        'completed_at': datetime.now(timezone.utc).isoformat(),
    }
    summary_path = os.path.join(
        args.out_dir, f"session_{saver._session_id}.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  [done] Session summary: {summary_path}")
    except Exception:
        pass

    print(f"  [done] {saver.file_count} segments saved to {args.out_dir}")
    print(f"  [done] {buf.count:,} total packets, "
          f"{len(buf.errors)} errors")
    sys.exit(0)


if __name__ == "__main__":
    main()
