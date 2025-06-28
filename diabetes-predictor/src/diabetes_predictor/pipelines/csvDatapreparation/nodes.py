import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def _drop_unused_column(df: pd.DataFrame, col: str = "HbA1c_level") -> pd.DataFrame:
    """Usuń zbędne dane."""
    return df.drop(columns=[col], errors="ignore")

def _normalize_smoking(df: pd.DataFrame, col: str = "smoking_history") -> pd.DataFrame:
    """palenie na dwie kategorie true/false"""
    df[col] = df[col].apply(
        lambda x: "smoker" if x in ("former", "current") else "non-smoker"
    )
    return df

def _build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer do skali i one-hot encodingu."""
    num_cols = ["age", "blood_glucose_level", "bmi"]
    cat_cols = ["gender", "smoking_history"]
    return ColumnTransformer(
        transformers=[
            ("scale", MinMaxScaler(), num_cols),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough"
    )

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:

    # Kopiuje dane, usuwa HbA1c_level
    # Normalizuje historię palenia
    df = data.copy()
    df = _drop_unused_column(df)
    df = _normalize_smoking(df)

    #  Rozdziela X i y
    X = df[["age", "blood_glucose_level", "bmi", "gender", "smoking_history", "hypertension", "heart_disease"]]
    y = df["diabetes"].copy()

    # Tworzy i dopasowuje ColumnTransformer
    preprocessor = _build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Transformuje X, odtwarza nazwy kolumn i dokleja kolumnę 'diabetes'
    num_cols = ["age", "blood_glucose_level", "bmi"]
    cat_cols = ["gender", "smoking_history"]
    ohe_names = list(preprocessor.named_transformers_["ohe"].get_feature_names_out(cat_cols))
    passthrough = ["hypertension", "heart_disease"]
    all_cols = num_cols + ohe_names + passthrough

    df_out = pd.DataFrame(X_transformed, columns=all_cols, index=df.index)
    df_out["diabetes"] = y.values

    # Zapisuje przetwornik na dysk
    joblib.dump(preprocessor, "data/06_models/preprocessor.pkl")

    return df_out
