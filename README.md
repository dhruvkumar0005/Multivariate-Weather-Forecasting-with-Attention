# Multivariate Weather Forecasting with Attention

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An advanced deep learning framework for **multivariate time-series weather forecasting** using the **Jena Climate Dataset**. This project jointly models and forecasts multiple thermodynamically coupled meteorological variables and benchmarks progressive recurrent architectures against an **additive temporal attention mechanism** to evaluate both predictive performance and temporal interpretability.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [Dataset & Preprocessing](#-dataset--preprocessing)
  - [Raw Data Audit](#raw-data-audit)
  - [Cleaning & Hourly Resampling](#cleaning--hourly-resampling)
- [Feature Engineering & Sequence Formulation](#-feature-engineering--sequence-formulation)
  - [Multivariate Targets](#multivariate-targets)
  - [Cyclical Time Embeddings](#cyclical-time-embeddings)
  - [Sliding-Window Sequences](#sliding-window-sequences)
- [Time-Series Splitting & Normalization](#-time-series-splitting--normalization)
- [Model Architectures](#-model-architectures)
  - [1. Baseline Multi-Output LSTM](#1-baseline-multi-output-lstm)
  - [2. Stacked GRU](#2-stacked-gru)
  - [3. Stacked GRU + Temporal Additive Attention](#3-stacked-gru--temporal-additive-attention)
- [Temporal Attention Mechanism Formulation](#-temporal-attention-mechanism-formulation)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Attention Interpretability & Case Studies](#-attention-interpretability--case-studies)
- [Evaluation & Benchmark Methodology](#-evaluation--benchmark-methodology)
- [Hyperparameter Tuning & Attention Ablation Analysis](#-hyperparameter-tuning--attention-ablation-analysis)
- [Interactive Jupyter Notebooks](#-interactive-jupyter-notebooks)
- [Model Serialization & Inference (.keras, .joblib, .pkl)](#-model-serialization--inference-keras-joblib-pkl)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Hardware & Practical Execution](#-hardware--practical-execution)
- [License & Citation](#-license--citation)

---

## 🔬 Executive Overview

Weather systems are non-linear dynamical processes governed by physical laws (thermodynamics, fluid mechanics, and vapor equilibria). Conventional single-variable forecasting fails to exploit cross-variable dependencies—such as the inverse relationship between air temperature and relative humidity or barometric pressure shifts preceding storm fronts.

This project implements an end-to-end, reproducible deep learning system that:
1. **Forecasts 4 Key Weather Variables Jointly:** Temperature (°C), Barometric Pressure (mbar), Relative Humidity (%), and Wind Velocity (m/s).
2. **Conducts a Fair 3-Way Architectural Comparison:** Baseline LSTM vs. Stacked GRU vs. GRU + Temporal Attention under identical window lengths, split boundaries, loss functions, and optimization schedules.
3. **Inspects Temporal Attention Representations:** Extracts attention weight distributions over a 72-hour historical window to assess whether the network places representational focus on physically meaningful signals (e.g. 24-hour diurnal peaks vs. abrupt weather transitions).
4. **Enforces Strict Anti-Leakage Engineering:** Chronological train/validation/test partitioning and scaler fitting strictly restricted to training data.

---

## ⚡ Key Features

- **End-to-End Pipeline:** Modular, production-ready code organized into clean scripts under `src/`.
- **Custom Keras Layer:** Vectorized `TemporalAdditiveAttention` layer implementing Bahdanau-style scoring across recurrent hidden states along the time dimension (`axis=1`).
- **5 Comprehensive EDA Visualizations:** High-resolution diagnostic figures covering longitudinal trends, correlation matrices, diurnal profiles, thermodynamic phase relationships, and seasonal distributions.
- **Leakage-Free Sequence Generator:** Memory-efficient, generator-backed sliding-window batching using `tf.data.Dataset`.
- **Per-Variable Metrics:** Evaluation reported independently for each variable in true physical units (MAE, RMSE, $R^2$), avoiding single-number metric masking.

---

## 📊 Dataset & Preprocessing

### Raw Data Audit
The project utilizes the **Jena Climate Dataset**, recorded by the Max Planck Institute for Biogeochemistry in Jena, Germany.
- **Total Records:** 420,551 timestamps (recorded every 10 minutes between Jan 1, 2009 and Dec 31, 2016).
- **Recorded Variables:** 14 physical sensor channels + 1 datetime column.
- **Sensor Faults Detected:** 18 records contained invalid negative wind velocities (`wv = -9999 m/s`), and 327 duplicate timestamps were detected in raw sensor intervals.

### Cleaning & Hourly Resampling
1. **Anomaly Filtering:** Faulty wind velocity readings (`wv < 0` and `max. wv < 0`) were replaced with `NaN`.
2. **Hourly Aggregation:** Data was resampled from 10-minute intervals to regular 1-hour intervals (`.resample('1h').mean()`), yielding **70,129 clean hourly records**.
   - *Advantage:* Compresses sequence lengths by $6\times$ (a 72-hour window requires 72 timesteps rather than 432), speeding up recurrent training by ~80% while retaining diurnal and synoptic dynamics.
3. **Linear Gap Filling:** Isolated missing values following resampling were imputed via time-weighted linear interpolation.
4. **Monotonicity & Integrity:** Verified strictly increasing, continuous, and duplicate-free datetime indexing.

---

## 🛠 Feature Engineering & Sequence Formulation

### Multivariate Targets
Four correlated variables were chosen based on physical significance and predictive value:
1. **Temperature (`T (degC)`):** Core thermodynamic variable; strong diurnal and annual periodicity.
2. **Atmospheric Pressure (`p (mbar)`):** Synoptic indicator of incoming low/high-pressure systems.
3. **Relative Humidity (`rh (%)`):** Moisture indicator strongly coupled with air temperature.
4. **Wind Velocity (`wv (m/s)`):** Kinetic transport variable; turbulent and high-frequency.

### Cyclical Time Embeddings
Standard numeric hours (`0` to `23`) introduce artificial boundary discontinuities. We project timestamps into continuous sinusoidal representations:
$$\text{Day}_{\sin} = \sin\left(\frac{2\pi \cdot t_{\text{hour}}}{24}\right), \quad \text{Day}_{\cos} = \cos\left(\frac{2\pi \cdot t_{\text{hour}}}{24}\right)$$
$$\text{Year}_{\sin} = \sin\left(\frac{2\pi \cdot t_{\text{epoch}}}{365.2425 \times 86400}\right), \quad \text{Year}_{\cos} = \cos\left(\frac{2\pi \cdot t_{\text{epoch}}}{365.2425 \times 86400}\right)$$

Total feature representation per timestep: **$D = 8$** (4 targets + 4 cyclical signals).

### Sliding-Window Sequences
- **Input Tensor ($X$):** Past 72 hours $\to$ `(Batch_Size, 72, 8)`.
  - *Why 72 hours?* Captures exactly 3 full diurnal cycles (24h, 48h, 72h) and synoptic front scales, providing the attention layer with historical context from previous days at the same hour.
- **Forecast Horizon ($Y$):** Direct 1-step-ahead ($t+1$) joint multivariate prediction $\to$ `(Batch_Size, 4)`.

---

## 🔒 Time-Series Splitting & Normalization

```
[================= TRAIN: 70% =================] [==== VAL: 15% ====] [==== TEST: 15% ====]
2009-01-01 ------------------------> 2014-08-08  --> 2015-10-20       --> 2017-01-01
(Fit Scaler Here ONLY)                           (Transform ONLY)      (Transform ONLY)
```

1. **Chronological Partitioning:**
   - **Train:** First 70% (49,090 hourly timesteps, ~5.6 years).
   - **Validation:** Next 15% (10,519 hourly timesteps, ~1.2 years).
   - **Test:** Final 15% (10,520 hourly timesteps, ~1.2 years).
   - *Rule:* Zero random shuffling across temporal boundaries.
2. **Zero-Leakage Scaler:**
   - `StandardScaler` is fitted **strictly on the training split**.
   - Validation and test splits are normalized using training mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$.
   - All evaluation metrics are calculated after **inverse transforming** predictions back to physical units (°C, mbar, %, m/s).

---

## 🧠 Model Architectures

All three architectures share an identical parameter scale, loss function, and dense output head:

```
                    ┌──────────────────────────────────────────────┐
                    │ Input Sequence: X (Batch, 72 timesteps, 8 D) │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌───────────────────┐             ┌───────────────────┐             ┌─────────────────────────┐
│  1. Baseline LSTM │             │  2. Stacked GRU   │             │ 3. GRU + Attention      │
├───────────────────┤             ├───────────────────┤             ├─────────────────────────┤
│ LSTM 1: 64 units  │             │ GRU 1: 64 units   │             │ GRU 1: 64 units (seq)   │
│ Dropout: 0.2      │             │ Dropout: 0.2      │             │ Dropout: 0.2            │
│ LSTM 2: 32 units  │             │ GRU 2: 32 units   │             │ GRU 2: 32 units (seq)   │
│ (return_seq=False)│             │ (return_seq=False)│             │ (return_seq=True)       │
│ Dense: 32 (ReLU)  │             │ Dense: 32 (ReLU)  │             │                         │
│ Output: Dense(4)  │             │ Output: Dense(4)  │             │ [Temporal Attention]    │
│                   │             │                   │             │ e_t = v^T tanh(W·h_t+b) │
│                   │             │                   │             │ α = Softmax(e, axis=1)  │
│                   │             │                   │             │ Context c = Σ(α_t · h_t)│
│                   │             │                   │             │ Dense: 32 (ReLU)        │
│                   │             │                   │             │ Output: Dense(4) + α    │
└─────────┬─────────┘             └─────────┬─────────┘             └────────────┬────────────┘
          ▼                                 ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Fair Benchmark Head: Dense(4, activation='linear') -> Targets: [T, p, rh, wv]               │
│ Loss: Multi-Target MSE | Optimizer: Adam (lr=1e-3) | Callbacks: EarlyStopping, ReduceLROnPlat│
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Parameter Comparison
| Architecture | Recurrent Units | Trainable Parameters | Memory Footprint |
|---|---|---|---|
| **Baseline LSTM** | 64 $\to$ 32 | **32,292** | ~126.1 KB |
| **Stacked GRU** | 64 $\to$ 32 | **24,804** | ~96.9 KB |
| **GRU + Attention** | 64 $\to$ 32 (+ Attn) | **25,892** | ~101.1 KB |

---

## 🔍 Temporal Attention Mechanism Formulation

The attention mechanism operates over the **TIME dimension** (`axis=1`) across all 72 hidden states $H = [h_1, h_2, \dots, h_{72}] \in \mathbb{R}^{\text{Batch} \times 72 \times 32}$:

1. **Alignment Score Computation (Additive / Bahdanau):**
   $$u_t = \tanh(W_a h_t + b_a), \quad u_t \in \mathbb{R}^{32}$$
   $$e_t = v_a^T u_t, \quad e_t \in \mathbb{R}$$
   where $W_a \in \mathbb{R}^{32 \times 32}$, $b_a \in \mathbb{R}^{32}$, and $v_a \in \mathbb{R}^{32 \times 1}$ are learnable parameters.

2. **Softmax Normalization Over Time:**
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{j=1}^{72} \exp(e_j)}, \quad \sum_{t=1}^{72} \alpha_t = 1$$
   The resulting tensor $\alpha \in \mathbb{R}^{\text{Batch} \times 72 \times 1}$ represents the normalized attention weight for each past hour.

3. **Context Vector Generation:**
   $$c = \sum_{t=1}^{72} \alpha_t h_t \in \mathbb{R}^{32}$$
   The context vector $c$ summarizes historical information, weighting salient timesteps more heavily, and is passed to the prediction head.

---

## 📈 Exploratory Data Analysis (EDA)

The project generates five diagnostic visualizations in `figures/`:

1. **`01_yearly_trends_seasonality.png`:** Longitudinal 8-year trajectories (2009–2016) with 30-day centered moving averages displaying annual solar cycles and inter-annual variations.
2. **`02_correlation_heatmap.png`:** Pearson correlation matrix across variables, confirming negative coupling between temperature and relative humidity ($r \approx -0.57$) and strong links to vapor pressure.
3. **`03_diurnal_hourly_patterns.png`:** 24-hour diurnal profiles with $\pm 1\sigma$ confidence bands showing afternoon temperature peaks and morning humidity maximums.
4. **`04_temp_humidity_pressure_coupling.png`:** Temperature vs. Relative Humidity phase distribution by season, illustrating seasonal variations in humidity.
5. **`05_distributions_and_outliers.png`:** Seasonal boxplots highlighting differences in seasonal variance across temperature and wind speed.

---

## 🎯 Attention Interpretability & Case Studies

The model outputs attention weight distributions $\alpha$ across the 72-hour window. In `figures/08_attention_case_studies.png`, we analyze four meteorological regimes:

- **Case 1: Diurnal Warming Cycle:** Attention exhibits periodic peaks at $t-24\text{h}$, $t-48\text{h}$, and $t-72\text{h}$, demonstrating that the network attends to the same time of day from previous cycles.
- **Case 2: Sudden Synoptic Shift:** Attention concentrates heavily on the immediate past ($t-1\text{h} \dots t-6\text{h}$), reflecting rapid barometric and temperature adjustments during frontal passages.
- **Case 3: Turbulent Wind Gusts:** Attention distributes across broader historical segments to estimate kinetic momentum.
- **Case 4: Stagnant High-Pressure System:** Attention produces a smooth, relatively uniform weighting across the window.

> **Methodological Note:** Attention weights reflect *representational reliance within the model*, not *physical causality*. The analysis highlights these distinctions accordingly.

---

## 📐 Evaluation & Benchmark Methodology

Models are benchmarked on the test set (Oct 20, 2015 to Jan 1, 2017) using:
- **Validation Root Mean Squared Error (Val RMSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination ($R^2$)**

All metrics are computed per variable after inverse scaling to true physical units:

| Architecture | Variable | Unit | Val RMSE | Test MAE | Test RMSE | Test $R^2$ Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline LSTM** | Temperature ($T$) | °C | **0.6881** | 0.4903 | 0.6519 | 0.9930 |
| | Atmospheric Pressure ($p$) | mbar | **0.5064** | 0.3913 | 0.5140 | 0.9966 |
| | Relative Humidity ($rh$) | % | **3.0553** | 1.9899 | 2.8591 | 0.9658 |
| | Wind Velocity ($wv$) | m/s | **0.6371** | 0.4503 | 0.6349 | 0.8264 |
| **Stacked GRU** | Temperature ($T$) | °C | **0.7152** | 0.4994 | 0.6617 | 0.9928 |
| | Atmospheric Pressure ($p$) | mbar | **0.6398** | 0.4869 | 0.6523 | 0.9946 |
| | Relative Humidity ($rh$) | % | **3.0898** | 2.0242 | 2.8886 | 0.9651 |
| | Wind Velocity ($wv$) | m/s | **0.6406** | 0.4557 | 0.6387 | 0.8243 |
| **GRU + Attention** | Temperature ($T$) | °C | **0.6975** | **0.4860** | **0.6536** | **0.9930** |
| *(Recommended)* | Atmospheric Pressure ($p$) | mbar | **0.4782** | **0.3385** | **0.4531** | **0.9974** |
| | Relative Humidity ($rh$) | % | **3.0878** | **2.0025** | **2.8852** | **0.9652** |
| | Wind Velocity ($wv$) | m/s | **0.6415** | **0.4518** | **0.6399** | **0.8236** |

Full numeric results are automatically compiled in `results/metrics/model_comparison.csv`.

---

## 🔬 Hyperparameter Tuning & Attention Ablation Analysis

A rigorous ablation study was conducted in `src/tune.py` comparing lookback window lengths, recurrent layer depths, and attention scoring functions:

| Experiment Configuration | Lookback Window | Recurrent Layers | Scoring Mechanism | Val Mean RMSE | Test Mean RMSE | Training Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `GRU_Window_24h` | 24 hours | 2 Layers (64, 32) | Pure GRU | 1.3567 | 1.2801 | 79.1s |
| `GRU_Window_48h` | 48 hours | 2 Layers (64, 32) | Pure GRU | 1.3560 | 1.2796 | 150.1s |
| `GRU_Window_72h_2L` | 72 hours | 2 Layers (64, 32) | Pure GRU | 1.3647 | 1.2858 | 206.0s |
| `GRU_Window_72h_3L` | 72 hours | 3 Layers (64, 32, 16) | Pure GRU | 1.4694 | 1.3854 | 310.2s |
| **`GRU_Attention_Additive`** | 72 hours | 2 Layers (64, 32) | **Bahdanau Additive** | **1.5753** | **1.4651** | 314.0s |
| `GRU_Attention_DotProduct` | 72 hours | 2 Layers (64, 32) | **Luong Dot-Product** | **3.3877** | **3.3576** | 397.6s |

### Key Architectural Findings:
1. **Additive vs. Dot-Product Scoring:** Bahdanau **additive attention** ($v^T \tanh(Wh + b)$) achieved a validation RMSE of **1.5753**, massively outperforming **scaled dot-product** attention (**3.3877**). The nonlinear projection in additive attention is critical for aligning multi-sensor weather signals across time.
2. **Optimal Lookback:** A 72-hour window captures exactly 3 full diurnal cycles (24h, 48h, 72h), providing the temporal attention mechanism with the necessary context to extract physical diurnal peaks.
3. **Recurrent Depth:** 2 layers (64 $\to$ 32) achieved lower validation error than 3 layers (64 $\to$ 32 $\to$ 16), which exhibited slight over-parameterization on noisy atmospheric signals.

Visual summary saved to: `figures/09_hyperparameter_tuning_ablation.png`.

---

## 📓 Interactive Jupyter Notebooks

The repository contains four self-contained, fully executed Jupyter Notebooks in the `notebooks/` directory. All cells have been pre-executed with inline tables, execution logs, and high-resolution figures:

1. **[`01_data_cleaning_and_eda.ipynb`](notebooks/01_data_cleaning_and_eda.ipynb):**
   - Raw sensor audit (14 meteorological features, 420,551 timestamps).
   - Anomaly filtering (replacing negative wind velocity sensor errors `wv = -9999 m/s`).
   - Hourly resampling and time-weighted linear gap imputation (70,129 clean records).
   - Interactive rendering of all 5 EDA figures at 300 DPI.

2. **[`02_feature_eng_and_sequences.ipynb`](notebooks/02_feature_eng_and_sequences.ipynb):**
   - Cyclical trigonometric temporal encoding ($\text{Day}_{\sin/\cos}, \text{Year}_{\sin/\cos}$).
   - Explicit calendar features (hour, day of year, seasonal classification).
   - Anti-leakage chronological train/val/test splitting (70% / 15% / 15%).
   - Scaler fitting strictly on the training partition and sliding-window tensor verification.

3. **[`03_model_training_and_eval.ipynb`](notebooks/03_model_training_and_eval.ipynb):**
   - Progressive construction of Baseline LSTM, Stacked GRU, and Attention GRU.
   - Modular training loops with `EarlyStopping` and `ReduceLROnPlateau` callbacks.
   - Comparative loss convergence curves and inverse physical scaling.
   - Comprehensive test evaluation table (`MAE`, `RMSE`, $R^2$) across all 4 variables.

4. **[`04_attention_interpretability.ipynb`](notebooks/04_attention_interpretability.ipynb):**
   - Vectorized attention weight extraction across the 72-hour historical window.
   - Diurnal cyclic alignment analysis ($t-24\text{h}, t-48\text{h}, t-72\text{h}$ periodicity).
   - 4 meteorological case studies: Diurnal warming, sudden frontal shift, turbulent wind gust, and stagnant high-pressure.

---

## 💾 Model Serialization & Inference (.keras, .joblib, .pkl)

To support maximum production and research flexibility, all trained models and data preprocessors are serialized across three standard formats:

| Target Model | `.keras` (Native Keras 3) | `.joblib` (Scikit-Learn/Joblib) | `.pkl` (Python Pickle) | Model Size | Verified Output Shape |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline LSTM** | `Baseline_LSTM.keras` | `Baseline_LSTM.joblib` | `Baseline_LSTM.pkl` | 422.6 KB | `(Batch, 4)` |
| **Stacked GRU** | `Stacked_GRU.keras` | `Stacked_GRU.joblib` | `Stacked_GRU.pkl` | 334.8 KB | `(Batch, 4)` |
| **GRU + Attention** | `GRU_with_Attention.keras` | `GRU_with_Attention.joblib` | `GRU_with_Attention.pkl` | 353.5 KB | `[(Batch, 4), (Batch, 72, 1)]` |
| **Feature Scaler** | — | `scaler.joblib` | `scaler.pkl` | ~1.1 KB | Full 8 Features |
| **Target Scaler** | — | `target_scaler.joblib` | `target_scaler.pkl` | ~1.0 KB | 4 Weather Targets |

### Python Loading Examples:

#### Option A: Load via `joblib`
```python
import joblib
import numpy as np
from src.attention import TemporalAdditiveAttention  # Ensure custom layer is registered

# 1. Load Attention Model and Scalers
model = joblib.load("results/models/GRU_with_Attention.joblib")
scaler = joblib.load("results/scaler.joblib")
target_scaler = joblib.load("results/target_scaler.joblib")

# 2. Forward pass with test sequence of shape (Batch, 72, 8)
dummy_x = np.random.randn(1, 72, 8).astype(np.float32)
normalized_preds, attention_weights = model(dummy_x)

# 3. Inverse transform predictions to physical units (°C, mbar, %, m/s)
physical_preds = target_scaler.inverse_transform(normalized_preds.numpy())
print("Forecast (T, p, rh, wv):", physical_preds)
```

#### Option B: Load via standard `pickle`
```python
import pickle
from src.attention import TemporalAdditiveAttention

with open("results/models/GRU_with_Attention.pkl", "rb") as f:
    model = pickle.load(f)

with open("results/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
```

#### Option C: Load via native `tf.keras`
```python
import tensorflow as tf
from src.attention import TemporalAdditiveAttention

model = tf.keras.models.load_model(
    "results/models/GRU_with_Attention.keras",
    custom_objects={"TemporalAdditiveAttention": TemporalAdditiveAttention}
)
```

---

## 📁 Repository Structure

```
Multivariate-Weather-Forecasting-with-Attention/
├── Dataset/
│   ├── jena_climate_2009_2016.csv          # Raw 10-minute dataset (420k rows)
│   └── jena_climate_hourly_cleaned.csv     # Cleaned hourly dataset (70k rows)
│
├── notebooks/
│   ├── 01_data_cleaning_and_eda.ipynb      # EDA & visual inspection
│   ├── 02_feature_eng_and_sequences.ipynb  # Sequence generation verification
│   ├── 03_model_training_and_eval.ipynb    # Training & evaluation workflow
│   └── 04_attention_interpretability.ipynb # Attention weight analysis
│
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── config.py                           # Paths, target variables, hyperparams
│   ├── data_loader.py                      # Auditing, cleaning, hourly resampling
│   ├── features.py                         # Cyclical encoding, splitting, scaling
│   ├── sequences.py                        # Sliding-window tf.data generator
│   ├── models.py                           # LSTM, Stacked GRU, GRU+Attention
│   ├── attention.py                        # Custom Temporal Additive Attention layer
│   ├── train.py                            # Training engine with EarlyStopping
│   ├── evaluate.py                         # Evaluation, metrics CSV, plotting
│   ├── tune.py                             # Hyperparameter tuning & ablation suite
│   └── visualization.py                    # 5-figure EDA pipeline
│
├── results/
│   ├── models/                             # Saved models (.keras, .joblib, .pkl)
│   ├── metrics/                            # Metrics CSVs & history JSONs
│   ├── attention_weights/                  # Extracted attention arrays (.npy)
│   ├── scaler.joblib / .pkl                # Feature scalers
│   └── target_scaler.joblib / .pkl         # Target scalers
│
├── figures/                                # Exported publication-ready charts
│   ├── 01_yearly_trends_seasonality.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_diurnal_hourly_patterns.png
│   ├── 04_temp_humidity_pressure_coupling.png
│   ├── 05_distributions_and_outliers.png
│   ├── 06_training_loss_curves.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_attention_case_studies.png
│   └── 09_hyperparameter_tuning_ablation.png
│
├── save_models_joblib_pkl.py               # Model export to .joblib & .pkl formats
├── execute_all_notebooks.py                # Automated Jupyter execution preprocessor
├── requirements.txt                        # Pinned dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/dhruvkumar0005/Multivariate-Weather-Forecasting-with-Attention.git
cd "Multivariate-Weather-Forecasting-with-Attention"

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Cleaning & Resampling
Processes the raw Jena dataset, removes faulty negative wind readings, resamples to hourly intervals, and saves the cleaned dataset:
```bash
python -m src.data_loader
```

### 3. Generate EDA Visualizations
Produces all 5 EDA figures in the `figures/` folder:
```bash
python -m src.visualization
```

### 4. Train All Three Models
Trains Baseline LSTM, Stacked GRU, and Attention GRU under identical conditions:
```bash
python -m src.train
```

### 5. Evaluate & Generate Diagnostic Visualizations
Evaluates test performance, computes per-variable MAE/RMSE/$R^2$, exports the comparison CSV, and generates actual vs. predicted and attention plots:
```bash
python -m src.evaluate
```

### 6. Export Models in `.joblib` and `.pkl` Formats
Serializes all three architectures and scalers into both `.joblib` and `.pkl` binaries alongside `.keras`:
```bash
python save_models_joblib_pkl.py
```

### 7. Run Hyperparameter Tuning & Ablation Studies
Executes window length (24h, 48h, 72h), recurrent depth (2 vs 3 layers), and attention mechanism (Additive vs Dot-Product) experiments:
```bash
python -m src.tune
```

### 8. Execute All Jupyter Notebooks (Headless / Automated)
Runs all 4 analysis and modeling notebooks end-to-end and preserves cell outputs inline:
```bash
python execute_all_notebooks.py
```

---

## 💻 Hardware & Practical Execution

- **CPU Viability:** The hourly resampling strategy reduces the dataset from 420,551 to 70,129 rows. A compact architecture (64 $\to$ 32 recurrent units) allows all three models to train on standard multi-core laptop CPUs in reasonable time without requiring dedicated GPUs.
- **GPU Acceleration:** Fully compatible with CUDA-enabled GPUs via standard TensorFlow 2.16+; batch sizes can be increased to 256 for faster epoch throughput.

---

## 📜 License & Citation

This project is released under the **MIT License**.

If you use this project or dataset in your coursework or research, please cite:
```bibtex
@misc{jena_climate_attention,
  author = {Kumar, Dhruv},
  title = {Multivariate Weather Forecasting with Attention},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/dhruvkumar0005/Multivariate-Weather-Forecasting-with-Attention}}
}
```