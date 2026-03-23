from pathlib import Path
import pandas as pd

def load_split(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    return pd.read_csv(file_path, sep=";", names=["text", "emotion"])

def load_datasets(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_df = load_split(data_dir / "train.txt")
    val_df = load_split(data_dir / "val.txt")
    test_df = load_split(data_dir / "test.txt")
    return train_df, val_df, test_df