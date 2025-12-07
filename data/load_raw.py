import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "crypto_lob_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_raw_lob(symbol: str = "BTC", timeframe: str = "1min") -> pd.DataFrame:
    """
    Load raw LOB features for a given symbol and timeframe.
    Expects files like BTC_1min.csv, ETH_5min.csv, etc.
    """
    filename = f"{symbol}_{timeframe}.csv"
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Raw file not found at {path}")

    # Reading in the csv file
    df = pd.read_csv(path)
    return df


def clean_lob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the Kaggle crypto LOB features:
    - Parse system_time as datetime
    - Sort by time
    - Make numeric columns
    - Drop obvious junk columns like 'Unnamed: 0' if present
    """
    # Drop index like column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Expect a 'system_time' column (at least this is what the crypto kaggle dataset uses)
    time_col = "system_time"
    if time_col not in df.columns:
        raise KeyError(f"Expected a '{time_col}' column in the data")

    # Parse timestamp into a datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)

    # Make all the non time columns to numeric (they should already be but still)
    non_numeric = [time_col]
    numeric_cols = []
    for col in df.columns:
        if col not in non_numeric:
            numeric_cols.append(col)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing midpoint/spread (aka critical columns) (these are also just for this crypto dataset)
    required_cols = ["midpoint", "spread"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected required column '{col}' in data")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df


def main(symbol: str = "BTC", timeframe: str = "1min"):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_lob(symbol=symbol, timeframe=timeframe)
    df_clean = clean_lob(df_raw)

    out_name = f"clean_{symbol}_{timeframe}.parquet"
    out_path = PROCESSED_DIR / out_name

    # Make df into the binary Parquet format
    # This kinda format is a "columnar storage format" which is best for efficient data storage and retrieval
    df_clean.to_parquet(out_path, index=False)

    print(f"Saved cleaned LOB data to {out_path}")


if __name__ == "__main__":
    # Right now the functions are defaulting BTC 1min
    # You can put the other coins/timeframes if you want
    main("BTC", "1min")
    
