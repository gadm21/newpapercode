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
pretty_name: "Home HAR — WiFi CSI Human Activity Recognition (Home)"
size_categories:
- 1M<n<10M
---

# Home HAR — WiFi CSI Human Activity Recognition (Home Environment)

## Dataset Description

**Home HAR** is a WiFi Channel State Information (CSI) dataset for human activity recognition collected in a residential (home) environment using two ESP32-C6 microcontrollers operating as commodity 802.11n access points. It contains **7 activity classes** recorded across **3 temporally separated data-collection sessions**, totaling approximately **2.7 million CSI packets** and **~465 minutes** of continuous recording.

This dataset is part of the research paper:

> **WiFi Sensing-Based Human Activity Recognition For Smart Home Applications Using Commodity Access Points**
> Gad Gad, Iqra Batool, Mostafa M. Fouda, Shikhar Verma, Zubair Md Fadlullah
> IEEE, 2026

📄 [Paper](https://gadm21.github.io/WifiSensingESP32HAR/IEEE_2026__wifi_sensing_.pdf) · ⚡ [GitHub](https://github.com/gadm21/WifiSensingESP32HAR) · 🌐 [Project Page](https://gadm21.github.io/WifiSensingESP32HAR/)

## Activity Classes

| Label | Description |
|-------|-------------|
| `drink` | Drinking from a cup/glass |
| `eat` | Eating a meal at a table |
| `empty` | No human present in the sensing area |
| `sleep` | Sleeping / lying still in bed |
| `smoke` | Smoking (arm/hand motions) |
| `watch` | Watching TV (seated, minimal motion) |
| `work` | Working at a desk (typing, mouse use) |

## Collection Setup

| Parameter | Value |
|-----------|-------|
| **Hardware** | 2 × ESP32-C6 (TX: AP mode, RX: STA mode) |
| **WiFi Standard** | 802.11n, 20 MHz bandwidth, HT-LTF |
| **Subcarriers** | 64 total (52 LLTF data subcarriers extracted) |
| **Packet Rate** | ~200 packets/sec (irregular, resampled to 150 Hz) |
| **Transport** | UART serial @ 115200 baud |
| **Environment** | Residential home, single room |
| **TX–RX Distance** | ~3 meters, line-of-sight |

## Data Sessions

| Session | Period | Role | Files |
|---------|--------|------|-------|
| **Session 1** (`data1/`) | October 2025 | Train | 7 files (1 per activity) |
| **Session 2** (`data2/`) | October 2025 | Train | 7 files (1 per activity) |
| **Session 3** (`data3/`) | February 2026 | Test | 14 files (2 per activity) |

The **~3.5-month temporal gap** between training (Sessions 1–2) and test (Session 3) evaluates model robustness to environmental drift — changes in furniture placement, ambient RF interference, seasonal temperature variations, etc.

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
| Random Forest | 46.3% |
| XGBoost | 47.8% |
| Conv1D | 52.9% |

*Note: Home HAR is the most challenging dataset due to 7 fine-grained classes and the large temporal gap between train and test sessions.*

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
