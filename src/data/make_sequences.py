from pathlib import Path

import numpy as np
import pandas as pd


# Hyperparameters for sequence generation
T_HISTORY = 50      # number of past steps (T)
H_HORIZON = 5       # number of steps into the future (H)
EPSILON = 0.001     # threshold for up/down in terms of % move (0.1%)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_clean_lob(symbol: str = "BTC", timeframe: str = "1min") -> pd.DataFrame:
    """
    Load the cleaned LOB features parquet file created in Phase 1.
    """
    filename = f"clean_{symbol}_{timeframe}.parquet"
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at {path}")

    df = pd.read_parquet(path)

    # Ensure sorted by time, just in case
    if "system_time" in df.columns:
        df = df.sort_values("system_time").reset_index(drop=True)

    return df


def build_sequences_and_labels(df: pd.DataFrame):
    """
    Turn a cleaned LOB dataframe into:
      - X: [num_samples, T_HISTORY, num_features]
      - y: [num_samples] with labels in {-1, 0, +1}
    """

    # 1) Choose which columns are features.
    # For now all numeric columns except the time column.
    time_col = "system_time"
    feature_cols = [c for c in df.columns if c != time_col]

    # Convert to numpy for speed
    features = df[feature_cols].to_numpy(dtype=np.float32)

    # 2) Get mid price series.
    # If there is a 'midpoint' column, use it. Otherwise, you would compute from bids/asks.
    if "midpoint" not in df.columns:
        raise KeyError("Expected 'midpoint' column in dataframe.")
    mid = df["midpoint"].to_numpy(dtype=np.float32)

    n = len(df)

    # 3) Compute future mid price shifted by H steps
    # future_mid[i] = mid[i + H_HORIZON]
    future_mid = np.roll(mid, -H_HORIZON)

    # For the last H_HORIZON points, future_mid is invalid (wrapped), so we mark them as NaN
    future_mid[-H_HORIZON:] = np.nan

    # 4) Compute percentage return over horizon H:
    # r_t = (mid_{t+H} - mid_t) / mid_t
    returns = (future_mid - mid) / (mid + 1e-9)

    # 5) Create labels: -1 (down), 0 (flat), +1 (up)
    labels = np.zeros_like(returns, dtype=np.int8)
    labels[returns > EPSILON] = 1
    labels[returns < -EPSILON] = -1

    # We can't use samples where future_mid is NaN (incomplete horizon)
    valid_mask = ~np.isnan(future_mid)

    # 6) Build sliding windows of length T_HISTORY
    X_list = []
    y_list = []

    # We also need full history, so index must be >= T_HISTORY-1
    start_index = T_HISTORY - 1
    end_index = n - H_HORIZON  # last index that still has a valid label

    for idx in range(start_index, end_index):
        if not valid_mask[idx]:
            continue

        # window covers [idx - T_HISTORY + 1, ..., idx]
        start = idx - T_HISTORY + 1
        end = idx + 1  # python slicing is exclusive at end

        seq = features[start:end, :]         # shape [T_HISTORY, num_features]
        label = labels[idx]                  # scalar in {-1, 0, 1}

        X_list.append(seq)
        y_list.append(label)

    X = np.stack(X_list, axis=0)  # [num_samples, T_HISTORY, num_features]
    y = np.array(y_list, dtype=np.int8)

    return X, y, feature_cols


def time_based_split(X, y, train_frac=0.7, val_frac=0.15):
    """
    Split sequences into train/val/test in time order (no shuffling).
    """
    n = X.shape[0]

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main(symbol="BTC", timeframe="1min"):
    df = load_clean_lob(symbol, timeframe)
    X, y, feature_cols = build_sequences_and_labels(df)

    print(f"Built sequences for {symbol} {timeframe}:")
    print(f"  X shape: {X.shape}  (num_samples, T={T_HISTORY}, num_features={X.shape[-1]})")
    print(f"  y shape: {y.shape}")
    print(f"  Label distribution: {np.bincount(y + 1)}  # index 0=label -1, 1=0, 2=+1")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_based_split(X, y)

    out_name = f"sequences_{symbol}_{timeframe}_T{T_HISTORY}_H{H_HORIZON}.npz"
    out_path = PROCESSED_DIR / out_name

    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_cols=np.array(feature_cols),
        T_HISTORY=T_HISTORY,
        H_HORIZON=H_HORIZON,
        EPSILON=EPSILON,
    )

    print(f"Saved sequences to {out_path}")


if __name__ == "__main__":
    main("BTC", "1min")
