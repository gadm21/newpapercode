---
language:
- en
license: cc-by-4.0
task_categories:
- time-series-classification
tags:
- wifi-sensing
- csi
- human-activity-recognition
- esp32
- smart-home
- indoor-sensing
pretty_name: "Office HAR — WiFi CSI Human Activity Recognition (Office)"
size_categories:
- 100K<n<1M
---

# Office HAR — WiFi CSI Human Activity Recognition (Office Environment)

## Dataset Description

**Office HAR** is a WiFi Channel State Information (CSI) dataset for human activity recognition collected in an office environment using two ESP32-C6 microcontrollers operating as commodity 802.11n access points. It contains **4 activity classes** with approximately **0.8 million CSI packets** and **~66 minutes** of continuous recording.

This dataset is part of the research paper:

> **WiFi Sensing-Based Human Activity Recognition For Smart Home Applications Using Commodity Access Points**
> Gad Gad, Iqra Batool, Mostafa M. Fouda, Shikhar Verma, Zubair Md Fadlullah
> IEEE, 2026

📄 [Paper](https://gadm21.github.io/WifiSensingESP32HAR/IEEE_2026__wifi_sensing_.pdf) · ⚡ [GitHub](https://github.com/gadm21/WifiSensingESP32HAR) · 🌐 [Project Page](https://gadm21.github.io/WifiSensingESP32HAR/)

## Activity Classes

| Label | Description |
|-------|-------------|
| `eat` | Eating a meal at a desk |
| `empty` | No human present in the sensing area |
| `watch` | Watching a screen (seated, minimal motion) |
| `work` | Working at a desk (typing, mouse use) |

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

| File | Label | Approx. Packets |
|------|-------|-----------------|
| `eat_1.csv` | eat | ~192K |
| `empty_1.csv` | empty | ~192K |
| `watch_1.csv` | watch | ~192K |
| `work.csv` | work | ~192K |

**Split strategy**: Percentage-based temporal split. The first 80% of each recording is used for training and the remaining 20% for testing. This preserves temporal ordering — the model never sees future data during training.

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
| Random Forest | 93.3% |
| XGBoost | 91.4% |
| Conv1D | 94.2% |
| CNN-LSTM | 93.3% |
| PCA + KNN | 85.6% |

Office HAR demonstrates strong performance across all classifiers. The 4-class problem in an office setting is well-suited for practical deployment in workplace occupancy analytics and smart building management.

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
