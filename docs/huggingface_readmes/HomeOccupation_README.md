---
language:
- en
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- wifi-sensing
- csi
- occupancy-detection
- esp32
- smart-home
- indoor-sensing
pretty_name: "Home Occupation — WiFi CSI Occupancy Detection (Home)"
size_categories:
- 1M<n<10M
---

# Home Occupation — WiFi CSI Occupancy Detection (Home Environment)

## Dataset Description

**Home Occupation** is a WiFi Channel State Information (CSI) dataset for room-level occupancy detection collected in a residential (home) environment using two ESP32-C6 microcontrollers operating as commodity 802.11n access points. It contains **3 occupancy classes** recorded across **3 temporal sessions per class**, totaling approximately **1.7 million CSI packets** and **~150 minutes** of continuous recording.

This dataset is part of the research paper:

> **WiFi Sensing-Based Human Activity Recognition For Smart Home Applications Using Commodity Access Points**
> Gad Gad, Iqra Batool, Mostafa M. Fouda, Shikhar Verma, Zubair Md Fadlullah
> IEEE, 2026

📄 [Paper](https://gadm21.github.io/WifiSensingESP32HAR/IEEE_2026__wifi_sensing_.pdf) · ⚡ [GitHub](https://github.com/gadm21/WifiSensingESP32HAR) · 🌐 [Project Page](https://gadm21.github.io/WifiSensingESP32HAR/)

## Occupancy Classes

| Label | Description |
|-------|-------------|
| `empty` | No human present in the sensing area |
| `sleep` | Person sleeping / lying still in bed (minimal motion) |
| `work` | Person working at a desk (typing, mouse use, subtle motion) |

This is a **coarse-grained occupancy task** — the goal is to detect whether the room is empty, or if someone is present and performing a low-motion or moderate-motion activity. This is directly relevant to smart-home energy management, HVAC automation, and security systems.

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

## Data Organization

| File | Label | Split | Approx. Packets |
|------|-------|-------|-----------------|
| `empty_1.csv` | empty | Train | ~193K |
| `empty_2.csv` | empty | Train | ~193K |
| `empty_3.csv` | empty | Test | ~193K |
| `sleep_1.csv` | sleep | Train | ~193K |
| `sleep_2.csv` | sleep | Train | ~193K |
| `sleep_3.csv` | sleep | Test | ~193K |
| `work_1.csv` | work | Train | ~193K |
| `work_2.csv` | work | Train | ~193K |
| `work_3.csv` | work | Test | ~193K |

**Split strategy**: File-based temporal holdout. The last recording session per label is used as the test set, ensuring the model is evaluated on temporally distinct data.

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
| Random Forest | 98.7% |
| Conv1D | 100.0% |
| PCA + KNN | 69.9% |

Home Occupation achieves near-perfect classification, demonstrating that WiFi CSI can reliably distinguish between empty rooms, sleeping, and active desk work — even with commodity hardware.

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
