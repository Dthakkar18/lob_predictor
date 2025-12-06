# LOB Micro-Price Movement Predictor

Goal: Predict short-term mid-price movements (up / down / flat) from limit order book (LOB) snapshots using deep learning.

## Structure

- `data/raw/`        – raw limit order book data dumps
- `data/processed/`  – cleaned & prepared sequences (numpy / torch files)
- `notebooks/`       – exploratory analysis and result visualization
- `src/data/`        – loading, preprocessing, sequence creation
- `src/models/`      – baseline and deep learning models
- `src/training/`    – training scripts
- `src/eval/`        – evaluation & backtesting scripts
- `configs/`         – YAML configs for experiments
- `scripts/`         – helper shell scripts to run pipelines
