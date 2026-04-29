import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def split_features_labels(df: pd.DataFrame, text_column: str = "clean_text", label_column: str = "emotion"):
    X = df[text_column]
    y = df[label_column]
    return X, y

def vectorize_datasets(X_train, X_val, X_test, **vectorizer_kwargs):
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_val_vec, X_test_vec, vectorizer