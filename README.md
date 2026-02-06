<p align="center">
  <h1 align="center">Kloiya River — Flow Rate Prediction with LSTM</h1>
  <p align="center">
    <em>Deep learning approach for hydrological time-series forecasting on 58 years of river flow data</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  </p>
</p>

---

## Overview

This project builds an **LSTM (Long Short-Term Memory)** neural network to predict daily river flow rates for the **Kloiya River**, using historical hydrological data spanning from **April 1964 to July 2022** (~20,000+ daily observations). It also includes a statistical analysis suite for flow duration curves, distribution analysis, and hydropower optimization scenarios.

> **Context:** This project was developed as part of a research effort exploring the viability of machine learning in hydrological forecasting and its applications to hydropower efficiency optimization.

---

## Features

### LSTM Flow Rate Model
- **Multi-layer LSTM** architecture with 3 stacked layers (64 hidden units)
- **28-day sliding window** sequences for temporal pattern learning
- **Min-Max normalization** applied independently to train/validation/test splits
- **Training pipeline** with:
  - Adam optimizer with learning rate scheduling (`ReduceLROnPlateau`)
  - Early stopping with patience of 10 epochs
  - Best model checkpointing
- **Evaluation metrics:** MSE, RMSE, MAE, MAPE, and R² Score
- **GPU-accelerated** training with automatic CUDA detection

### Statistical Analysis
- Computes **mean, median, standard deviation, variance, coefficient of variation**, and flow range for any date interval
- Generates **Flow Duration Curves (FDCs)** across multiple historical periods (1982–2022, 1992–2022, 2002–2022, etc.)
- Produces **flow rate distribution histograms** for custom date ranges
- Full time-series visualization from 1964–2022

### Hydropower Optimization
- Models **turbine efficiency** as a function of flow rate
- Compares three optimization scenarios:
  - Baseline (no optimization)
  - Q80+ optimization (high-flow regime)
  - Q60+ extended optimization
- Calculates **power output gains** and efficiency improvements in kW

---

## Model Architecture

```
Input (28-day sequence)
        │
        ▼
┌──────────────────┐
│   LSTM Layer 1   │  (input_size=1, hidden_size=64)
├──────────────────┤
│   LSTM Layer 2   │  (hidden_size=64)
├──────────────────┤
│   LSTM Layer 3   │  (hidden_size=64)
└──────────────────┘
        │
        ▼  (last timestep output)
┌──────────────────┐
│  Fully Connected │  (64 → 1)
└──────────────────┘
        │
        ▼
   Predicted Flow Rate
```

---

## Hyperparameters

| Parameter | Value |
|:---|:---|
| Sequence Length | 28 days |
| Hidden Size | 64 |
| LSTM Layers | 3 |
| Batch Size | 128 |
| Learning Rate | 0.0003 |
| Max Epochs | 80 |
| LR Scheduler | ReduceLROnPlateau (patience=10, factor=0.3) |
| Early Stopping | Patience of 10 epochs |

---

## Data

The dataset contains **20,153 daily flow rate observations** (m³/s) for the Kloiya River:

| Field | Description |
|:---|:---|
| `Date` | Observation date (`YYYY/MM/DD`) |
| `Value` | River flow rate in m³/s |

- **Date Range:** April 16, 1964 — July 22, 2022
- **Preprocessing:** Linear interpolation is applied to fill missing dates, creating a continuous daily time series
- **Split:** 70% train · 15% validation · 15% test (sequential, no shuffling)

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

### Training the Model

Open `fr-lstm-model.ipynb` in Jupyter or VS Code and run all cells sequentially. The notebook will:

1. Load and preprocess the Kloiya River dataset
2. Create 28-day sliding window sequences
3. Split into train/validation/test sets (70/15/15)
4. Train the LSTM model with early stopping
5. Save the best model to `best_model.pth`
6. Evaluate and visualize predictions against actual test data

### Running Statistical Analysis

```bash
python std_analysis/analysis.py
```

Generates flow duration curves, distribution histograms, and computes descriptive statistics for configurable date ranges.

### Running Hydropower Optimization

```bash
python std_analysis/prediction.py
```

Outputs a comparison table of power output under different optimization scenarios.

---

## Reports

| Document | Link |
|:---|:---|
| Full Group Report | [View on Google Drive](https://drive.google.com/file/d/16eneVEujviDUxeveyxRqlOzv8Uhy56Jz/view?usp=sharing) |
| Solo Work Report | [View on Google Drive](https://drive.google.com/file/d/1SdAOeUYottZNUEnVOe5oAUZP-mff_Ict/view?usp=sharing) |
| Architecture Diagram | [Architecture.pdf](https://github.com/user-attachments/files/25060205/Architecture.pdf) |

---

## License

This project is for academic and research purposes.
