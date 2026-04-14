---
language:
- en
license: cc-by-4.0
task_categories:
- time-series-classification
tags:
- wifi-sensing
- csi
- indoor-localization
- occupancy-detection
- esp32
- smart-home
- indoor-sensing
pretty_name: "Office Localization — WiFi CSI Indoor Localization (Office)"
size_categories:
- 1M<n<10M
---

# Office Localization — WiFi CSI Indoor Localization (Office Environment)

## Dataset Description

**Office Localization** is a WiFi Channel State Information (CSI) dataset for zone-level indoor localization and occupancy region detection, collected in an office environment using two ESP32-C6 microcontrollers operating as commodity 802.11n access points. It contains **4 region/occupancy classes** recorded across **2 temporal sessions per class**, totaling approximately **1.6 million CSI packets** and **~124 minutes** of continuous recording.

This dataset is part of the research paper:

> **WiFi Sensing-Based Human Activity Recognition For Smart Home Applications Using Commodity Access Points**
> Gad Gad, Iqra Batool, Mostafa M. Fouda, Shikhar Verma, Zubair Md Fadlullah
> IEEE, 2026

📄 [Paper](https://gadm21.github.io/WifiSensingESP32HAR/IEEE_2026__wifi_sensing_.pdf) · ⚡ [GitHub](https://github.com/gadm21/WifiSensingESP32HAR) · 🌐 [Project Page](https://gadm21.github.io/WifiSensingESP32HAR/)

## Region / Occupancy Classes

| Label | Description |
|-------|-------------|
| `empty` | No human present in the sensing area |
| `one` | Person present in Zone 1 of the office |
| `two` | Person present in Zone 2 of the office |
| `five` | Person present in Zone 5 of the office |

The zone labels correspond to distinct spatial regions within the office. The task is to determine **where** a person is located (or if the room is empty) based solely on how their body perturbs the WiFi channel between the transmitter and receiver.

## Collection Setup

| Parameter | Value |
|-----------|-------|
| **Hardware** | 2 × ESP32-C6 (TX: AP mode, RX: STA mode) |
| **WiFi Standard** | 802.11n, 20 MHz bandwidth, HT-LTF |
| **Subcarriers** | 64 total (52 LLTF data subcarriers extracted) |
| **Packet Rate** | ~200 packets/sec (irregular, resampled to 150 Hz) |
| **Transport** | UART serial @ 115200 baud |
| **Environment** | Office room with desks, chairs, and typical office furniture |
| **TX–RX Distance** | ~3 meters, line-of-sight |
| **Recorded** | October 2025 |

## Data Organization

| File | Label | Split | Approx. Packets |
|------|-------|-------|-----------------|
| `empty_1.csv` | empty | Train | ~210K |
| `empty_2.csv` | empty | Test | ~210K |
| `five_1.csv` | five | Train | ~150K |
| `five_2.csv` | five | Test | ~150K |
| `one_1.csv` | one | Train | ~150K |
| `one_2.csv` | one | Test | ~150K |
| `two_1.csv` | two | Train | ~150K |
| `two_2.csv` | two | Test | ~150K |

**Split strategy**: File-based temporal holdout. The first recording session per label is used for training and the second for testing. This ensures the model generalizes to temporally distinct data collected at a different time.

## CSV Format

Each CSV file contains one row per received CSI packet with the following columns:

| Column | Description |
|--------|-------------|
| `type` | Packet type (always `CSI_DATA`) |
| `seq` | Sequence number / local timestamp |
| `mac` | Transmitter MAC address |
| `rssi` | Received Signal Strength Indicator (dBm) |
| `rate` | PHY rate index |
| `noise_floor` | Noise floor estimate (dBm) |
| `fft_gain` | FFT gain applied by hardware |
| `agc_gain` | Automatic Gain Control value |
| `channel` | WiFi channel number |
| `local_timestamp` | ESP32 local timestamp (µs) |
| `sig_len` | Signal length |
| `rx_state` | Receiver state |
| `len` | CSI data length (128 = 64 subcarriers × 2 components) |
| `first_word` | Header word |
| `data` | Raw CSI data as `[I₀, Q₀, I₁, Q₁, ..., I₆₃, Q₆₃]` — 128 signed integers representing in-phase and quadrature components for 64 subcarriers |

### Extracting CSI Amplitude

```python
import numpy as np

# Parse the 'data' column (strip brackets, split by comma)
raw = "[0,0,-5,12,-6,14,...]"  # example
iq = np.array([int(x) for x in raw.strip('"[]').split(',')])
iq = iq.reshape(64, 2)  # 64 subcarriers × (I, Q)
amplitude = np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)  # |H(f)|

# Select 52 LLTF data subcarriers (indices 6–31 and 33–58)
lltf_mask = list(range(6, 32)) + list(range(33, 59))
csi_amplitude = amplitude[lltf_mask]  # shape: (52,)
```

## Recommended Preprocessing Pipeline

1. **Load** CSV and parse the `data` column into complex I/Q arrays
2. **Select** 52 LLTF subcarriers (discard guard/null subcarriers)
3. **Resample** to a uniform 150 Hz sample rate (original rate is irregular ~100–200 Hz)
4. **Feature extraction**: Rolling variance with window W ∈ {20, 200, 2000} (recommended: W=200)
5. **Windowing**: Segment into fixed-length windows (e.g., 100 samples = 0.67s at 150 Hz)

## Benchmark Results

Best results from the paper using rolling-variance features (W=200):

| Classifier | Accuracy |
|-----------|----------|
| Random Forest | 89.1% |
| XGBoost | 88.6% |
| Conv1D | 95.7% |
| CNN-LSTM | 96.7% |
| PCA + KNN | 84.1% |

Office Localization achieves excellent results with deep learning models, demonstrating that commodity WiFi CSI can perform zone-level indoor localization without any dedicated infrastructure — just two off-the-shelf ESP32-C6 boards.

## Use Cases

- **Smart building management**: Automatically determine which zones are occupied
- **Energy optimization**: Zone-aware HVAC and lighting control
- **Elderly care**: Non-intrusive monitoring of movement between rooms/zones
- **Security**: Detect unauthorized presence in restricted zones

## Citation

```bibtex
@article{gad2026wifisensing,
  title={WiFi Sensing-Based Human Activity Recognition For Smart Home Applications Using Commodity Access Points},
  author={Gad, Gad and Batool, Iqra and Fouda, Mostafa M. and Verma, Shikhar and Fadlullah, Zubair Md},
  journal={IEEE},
  year={2026}
}
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
