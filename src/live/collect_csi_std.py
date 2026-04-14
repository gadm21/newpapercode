#!/usr/bin/env python3
"""
Standardized CSI data collector for ESP32-C6 WiFi sensing.

Produces publication-quality CSI recordings with:
  - Unified file duration (~1 min each, configurable)
  - Real-clock timestamps (ISO 8601 in filename + metadata)
  - Uniform resampling to a guaranteed sampling rate (default 150 Hz)
  - Guard subcarrier removal (64 -> 52 valid LLTF subcarriers)
  - Per-file JSON metadata sidecar (sr, start_time, end_time, sr_ratio, etc.)
  - Output shape: (guaranteed_sr * duration, 52) per file

Each output file is a NumPy .npz archive containing:
  - 'mag':       float64 (N, 52)  — amplitude of 52 valid subcarriers
  - 'phase':     float64 (N, 52)  — phase angle (radians)
  - 'rssi':      float64 (N,)     — RSSI per sample
  - 'timestamp': int64   (N,)     — uniform timestamps (microseconds)

Plus a companion .meta.json file with collection metadata.

Usage:
    python collect_csi_std.py --rx-port COM5 --label work --duration 60 --repeats 5
    python collect_csi_std.py --rx-port /dev/ttyACM1 --tx-port /dev/ttyACM0 \\
        --label empty --duration 60 --repeats 3 --out-dir ./data/office_loc

Future extensions (see roadmap at bottom):
  - PyPI library (wifisense) for collection, visualization, training, FL
  - Camera mode with face detection flag
  - Foundational model cross-task pretraining
"""

import argparse
import json
import os
import sys
import time
import threading
import signal
from datetime import datetime, timezone

import numpy as np

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    print("ERROR: pyserial is required. Install with: pip install pyserial",
          file=sys.stderr)
    sys.exit(1)

# =========================================================================
# Constants
# =========================================================================
HEADER = ("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,"
          "channel,local_timestamp,sig_len,rx_state,len,first_word,data")

CSI_SUBCARRIER_MASK = np.array([
    False, False, False, False, False, False,           # 0-5:  lower guard
    True,  True,  True,  True,  True,  True,            # 6-11
    True,  True,  True,  True,  True,  True,            # 12-17
    True,  True,  True,  True,  True,  True,            # 18-23
    True,  True,  True,  True,  True,  True,            # 24-29
    True,  True,                                         # 30-31
    False,                                               # 32: DC
    True,  True,  True,  True,  True,  True,            # 33-38
    True,  True,  True,  True,  True,  True,            # 39-44
    True,  True,  True,  True,  True,  True,            # 45-50
    True,  True,  True,  True,  True,  True,            # 51-56
    True,  True,                                         # 57-58
    False, False, False, False, False,                   # 59-63: upper guard
], dtype=bool)  # 52 True values

DEFAULT_GUARANTEED_SR = 150
DEFAULT_FILE_DURATION = 60  # seconds


# =========================================================================
# Raw serial reader (collects into memory buffer)
# =========================================================================
class RawCSIBuffer:
    """Thread-safe buffer that accumulates raw CSI packets from serial."""

    def __init__(self):
        self.timestamps = []   # ESP32 local_timestamp (microseconds)
        self.rssi_vals = []
        self.real_rows = []
        self.imag_rows = []
        self.errors = []
        self.lock = threading.Lock()
        self._count = 0

    @property
    def count(self):
        with self.lock:
            return self._count

    def add_line(self, text):
        """Parse one CSI_DATA line and append to buffers."""
        try:
            parts = text.split(",", 14)  # 15 fields total
            if len(parts) < 15:
                return
            rssi = float(parts[3])
            ts = int(parts[9])
            csi_str = parts[14].strip()
            csi_row = [int(x) for x in csi_str[1:-1].split(",")]
            if len(csi_row) != 128:
                with self.lock:
                    self.errors.append(f"expected 128, got {len(csi_row)}")
                return
            imag = csi_row[0::2]
            real = csi_row[1::2]
            with self.lock:
                self.timestamps.append(ts)
                self.rssi_vals.append(rssi)
                self.real_rows.append(real)
                self.imag_rows.append(imag)
                self._count += 1
        except Exception as e:
            with self.lock:
                self.errors.append(str(e))

    def to_arrays(self):
        """Convert buffers to numpy arrays. Returns (real, imag, rssi, ts)."""
        with self.lock:
            if not self.real_rows:
                empty = np.empty((0, 64), dtype=np.float64)
                return empty, empty, np.array([]), np.array([], dtype=np.int64)
            real = np.array(self.real_rows, dtype=np.float64)
            imag = np.array(self.imag_rows, dtype=np.float64)
            rssi = np.array(self.rssi_vals, dtype=np.float64)
            ts = np.array(self.timestamps, dtype=np.int64)
            return real, imag, rssi, ts


# =========================================================================
# Resampling + guard removal (standardization)
# =========================================================================
def resample_and_filter(real, imag, rssi, ts_us, guaranteed_sr):
    """Resample to uniform rate, remove guard subcarriers.

    Parameters
    ----------
    real, imag : (N, 64) float64
    rssi : (N,) float64
    ts_us : (N,) int64   — ESP32 microsecond timestamps
    guaranteed_sr : int

    Returns
    -------
    dict with keys: mag, phase, rssi, timestamp, metadata
    """
    n_orig = len(ts_us)
    if n_orig == 0:
        return None

    ts_sec = ts_us.astype(np.float64) / 1_000_000
    start, end = ts_sec[0], ts_sec[-1]
    duration = end - start
    if duration < 0.1:
        return None

    actual_sr = n_orig / duration
    n_out = int(np.ceil(duration * guaranteed_sr))
    if n_out < 2:
        return None

    target_t = start + np.arange(n_out) / guaranteed_sr
    dt = 1.0 / guaranteed_sr

    # Vectorized bin assignment
    bin_edges = target_t - dt / 2
    bin_idx = np.searchsorted(bin_edges, ts_sec, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_out - 1)
    samples_per_bin = np.bincount(bin_idx, minlength=n_out).astype(np.int64)

    # Compute mag and phase from raw I/Q
    mag_raw = np.sqrt(real ** 2 + imag ** 2)
    phase_raw = np.arctan2(imag, real)

    # Bin-average all channels
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

    # Interpolate empty bins
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

    # Apply subcarrier mask (64 -> 52)
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
        'mag': mag_out,        # (N, 52)
        'phase': phase_out,    # (N, 52)
        'rssi': rssi_acc,      # (N,)
        'timestamp': resampled_ts,  # (N,)
        'meta': meta,
    }


# =========================================================================
# Serial reader thread
# =========================================================================
stop_event = threading.Event()


def serial_reader(rx_port, baud, buf, duration):
    """Read CSI lines from serial into buffer for `duration` seconds."""
    end_time = time.time() + duration
    try:
        with serial.Serial(rx_port, baudrate=baud, timeout=0.05) as ser:
            _ = ser.read(ser.in_waiting or 1)  # drain boot garbage
            while not stop_event.is_set() and time.time() < end_time:
                line = ser.readline()
                if not line:
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if text.startswith("CSI_DATA,"):
                    buf.add_line(text)
    except serial.SerialException as e:
        print(f"[error] Serial: {e}", file=sys.stderr)


def rate_monitor(buf):
    """Print live CSI rate every second."""
    last = 0
    t0 = time.time()
    while not stop_event.is_set():
        time.sleep(1.0)
        cur = buf.count
        elapsed = time.time() - t0
        rate = cur - last
        print(f"  [rate] {rate:5d} CSI/s  total={cur:7d}  elapsed={elapsed:5.1f}s")
        last = cur


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


# =========================================================================
# Save standardized output
# =========================================================================
def save_standardized(result, out_dir, label, file_idx, wall_start_iso):
    """Save .npz data + .meta.json sidecar."""
    os.makedirs(out_dir, exist_ok=True)

    # Filename: label_YYYYMMDD_HHMMSS_NNN.npz
    ts_str = wall_start_iso.replace(":", "").replace("-", "")[:15]
    base = f"{label}_{ts_str}_{file_idx:03d}"
    npz_path = os.path.join(out_dir, base + ".npz")
    meta_path = os.path.join(out_dir, base + ".meta.json")

    np.savez_compressed(npz_path,
                        mag=result['mag'],
                        phase=result['phase'],
                        rssi=result['rssi'],
                        timestamp=result['timestamp'])

    meta = result['meta'].copy()
    meta['label'] = label
    meta['file_index'] = file_idx
    meta['wall_start_time'] = wall_start_iso
    meta['wall_end_time'] = datetime.now(timezone.utc).isoformat()
    meta['output_shape'] = list(result['mag'].shape)
    meta['format'] = 'npz'
    meta['version'] = '1.0'

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  [saved] {npz_path}  shape={result['mag'].shape}  "
          f"sr_ratio={meta['sr_ratio']}")
    return npz_path, meta_path


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Standardized CSI collector: fixed-duration files with "
                    "uniform sampling, guard removal, and metadata.")
    ap.add_argument("--rx-port", required=True,
                    help="Serial port of receiver ESP32")
    ap.add_argument("--tx-port", default=None,
                    help="(Optional) sender ESP32 port (reset only)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True,
                    help="Activity/location label for this recording")
    ap.add_argument("--duration", type=float, default=DEFAULT_FILE_DURATION,
                    help=f"Duration per file in seconds (default: {DEFAULT_FILE_DURATION})")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Number of files to collect")
    ap.add_argument("--sr", type=int, default=DEFAULT_GUARANTEED_SR,
                    help=f"Guaranteed sampling rate (default: {DEFAULT_GUARANTEED_SR})")
    ap.add_argument("--out-dir", default="./csi_standardized",
                    help="Output directory")
    ap.add_argument("--no-reset", action="store_true",
                    help="Skip board reset")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="Pause between repeats (seconds)")
    args = ap.parse_args()

    print(f"[config] rx={args.rx_port}  baud={args.baud}  label={args.label}")
    print(f"[config] duration={args.duration}s  repeats={args.repeats}  sr={args.sr} Hz")
    print(f"[config] out_dir={args.out_dir}")

    ports = [p.device for p in list_ports.comports()]
    print(f"[info] Available ports: {', '.join(ports) or 'none'}")

    if not args.no_reset:
        reset_board(args.rx_port, args.baud, "receiver")
        if args.tx_port:
            reset_board(args.tx_port, args.baud, "sender")

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    all_meta = []

    for i in range(args.repeats):
        stop_event.clear()
        buf = RawCSIBuffer()
        wall_start = datetime.now(timezone.utc).isoformat()

        print(f"\n{'='*60}")
        print(f"  Recording {i+1}/{args.repeats}  label={args.label}  "
              f"start={wall_start}")
        print(f"{'='*60}")

        t_reader = threading.Thread(
            target=serial_reader,
            args=(args.rx_port, args.baud, buf, args.duration),
            daemon=True)
        t_rate = threading.Thread(target=rate_monitor, args=(buf,), daemon=True)

        t_reader.start()
        t_rate.start()

        deadline = time.time() + args.duration
        try:
            while t_reader.is_alive():
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(0.2, max(0.0, remaining)))
        finally:
            stop_event.set()
            t_reader.join()
            time.sleep(0.1)

        # Convert raw buffer to arrays
        real, imag, rssi, ts = buf.to_arrays()
        print(f"  [raw] {len(ts)} packets collected, {len(buf.errors)} errors")

        if len(ts) == 0:
            print("  [warn] No data collected, skipping file")
            continue

        # Standardize: resample + filter
        result = resample_and_filter(real, imag, rssi, ts, args.sr)
        if result is None:
            print("  [warn] Resampling failed (too few samples), skipping")
            continue

        # Save
        npz, meta_path = save_standardized(
            result, args.out_dir, args.label, i + 1, wall_start)
        all_meta.append(meta_path)

        # Pause between repeats
        if i < args.repeats - 1 and args.pause > 0:
            print(f"  [pause] {args.pause}s ...")
            time.sleep(args.pause)

    # Save session summary
    summary_path = os.path.join(args.out_dir, "session_summary.json")
    summary = {
        'label': args.label,
        'repeats': args.repeats,
        'duration_per_file': args.duration,
        'guaranteed_sr': args.sr,
        'subcarriers': int(CSI_SUBCARRIER_MASK.sum()),
        'output_shape_per_file': f"({int(args.duration * args.sr)}, 52)",
        'files': all_meta,
        'collected_at': datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] {len(all_meta)} files saved to {args.out_dir}")
    print(f"[done] Session summary: {summary_path}")


if __name__ == "__main__":
    main()


# =========================================================================
# ROADMAP — Future extensions
# =========================================================================
#
# 2. PyPI library (wifisense):
#    - Package structure: wifisense/collect, wifisense/viz, wifisense/train,
#      wifisense/fl (Flower-based federated learning)
#    - CLI: wifisense collect --port COM5 --label walk --duration 60
#           wifisense train --data ./data --model conv1d
#           wifisense viz --file recording.npz
#           wifisense fl --server --data ./data --rounds 10
#    - pip install wifisense
#
# 3. Foundational model (cross-task):
#    - Self-supervised pretraining on unlabeled CSI (e.g., masked autoencoder
#      on rolling-variance windows across all environments)
#    - Fine-tune on downstream tasks (HAR, occupancy, localization)
#    - Transfer across environments with few-shot adaptation
#
# 4. Camera mode with face detection:
#    - Use OpenCV + Haar/DNN face detector
#    - Run in a separate thread during collection
#    - Add 'face_present' flag (0/1) column to each CSI sample
#    - Enables automatic ground-truth labeling for occupancy detection
#    - Implementation sketch:
#        import cv2
#        cap = cv2.VideoCapture(0)
#        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
#            'haarcascade_frontalface_default.xml')
#        while collecting:
#            ret, frame = cap.read()
#            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#            face_flag = 1 if len(faces) > 0 else 0
#            buffer.set_face_flag(face_flag)
