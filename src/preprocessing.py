import re
import pandas as pd

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def add_clean_text_column(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df[text_column].apply(clean_text)
    return df