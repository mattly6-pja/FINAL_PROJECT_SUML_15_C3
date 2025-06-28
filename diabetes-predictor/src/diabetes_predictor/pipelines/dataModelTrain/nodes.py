import os

import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, predict_model, pull, save_model
import joblib

def split_data(data: pd.DataFrame, test_size: float = 0.2):
    """
    Dzieli ramkę danych na zbiór tren. i testowy.
    Zwraca słownik z X_train, X_test, y_train, y_test.
    """
    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def train_model(split_output: dict, top_n: int = 15):
    os.makedirs("data/08_reporting", exist_ok=True)
    """
    Trenuje modele za pomocą PyCaret, wybiera top_n, ewaluje je na zbiorze testowym,
    zapisuje metryki i wybiera najlepszy model według Recall.
    """
    X_train = split_output["X_train"]
    X_test  = split_output["X_test"]
    y_train = split_output["y_train"]
    y_test  = split_output["y_test"]

    # Łączymy X_train i y_train w jedną ramkę
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.columns = list(X_train.columns) + ["diabetes"]

    # Inicjalizacja eksperymentu PyCaret
    setup(
        data=train_df,
        target="diabetes",
        session_id=123,
        fix_imbalance=True,
        verbose=False
    )

    # PORÓWNUJEMY i wybieramy TOP_N modeli
    top_models = compare_models(n_select=top_n)

    metrics_list = []
    model_objs   = []

    for idx, m in enumerate(top_models):
        model_name = type(m).__name__
        # zapis każdego z topowych modeli
        save_model(m, f"data/06_models/{model_name}_{idx}")

        # ewaluacja na zbiorze testowym
        test_df = X_test.copy()
        test_df["diabetes"] = y_test.values
        predict_model(m, data=test_df)
        m_metrics = pull()
        m_metrics["model_name"] = model_name

        metrics_list.append(m_metrics)
        model_objs.append((model_name, m))

    # Agregacja metryk
    metrics_df = pd.concat(metrics_list, ignore_index=True)
    # Porządkujemy kolumny F1 przed Recall
    metrics_df["F1"] = pd.to_numeric(metrics_df["F1"], errors="coerce")
    metrics_df["Recall"] = pd.to_numeric(metrics_df["Recall"], errors="coerce")
    f1 = metrics_df.pop("F1")
    metrics_df.insert(1, "F1", f1)
    metrics_df = metrics_df.sort_values(by="F1", ascending=False)
    # Zapis metryk
    metrics_df.to_csv("data/08_reporting/best_models_metrics.csv", index=False)

    # Wybór najlepszego modelu na podstawie Recall spośród top3
    top3 = metrics_df.head(3)
    best_name = top3.sort_values(by="Recall", ascending=False).iloc[0]["model_name"]
    best_model = dict(model_objs)[best_name]

    # Zapis finalnego, najlepszego modelu
    joblib.dump(best_model, "data/06_models/best_model.pkl")

    return best_model
