import numpy as np
import shap
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Wczytanie modelu i preprocesora ---
model = joblib.load("diabetes-predictor/data/06_models/best_model.pkl")
preprocessor = joblib.load("diabetes-predictor/data/06_models/preprocessor.pkl")

st.title("Sprawdź jakie jest Twoje ryzyko zachorowania na cukrzycę!")

# --- Słowniki mapujące wejścia ---
gender_dict = {"Kobieta": "Female", "Mężczyzna": "Male"}
smoking_dict = {
    "Nigdy nie paliłam/em": "non-smoker",
    "Paliłam/em w przeszłości, ale obecnie nie palę": "smoker",
    "Palę obecnie": "smoker",
}
binary_dict = {"Nie": 0, "Tak": 1}

# --- Formularz użytkownika ---
selected_gender = st.selectbox("Płeć", list(gender_dict.keys()))
gender = gender_dict[selected_gender]
age = st.slider("Wiek (lata)", 0, 100, 30)
blood_glucose_level = st.number_input(
    "Średni poziom glukozy (mg/dL)", 50, 300, 80
)
height = st.number_input("Wzrost (cm)", 30.0, 250.0, 170.0, 0.5)
weight = st.number_input("Waga (kg)", 2.5, 300.0, 60.0, 0.5)
selected_smoking = st.selectbox(
    "Historia palenia", list(smoking_dict.keys())
)
smoking_history = smoking_dict[selected_smoking]
hypertension = binary_dict[
    st.selectbox("Nadciśnienie tętnicze?", list(binary_dict.keys()))
]
heart_disease = binary_dict[
    st.selectbox("Choroba serca?", list(binary_dict.keys()))
]

# --- Obliczenie BMI ---
bmi = weight / ((height / 100) ** 2)

# --- Przygotowanie DataFrame dla modelu ---
input_data = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "smoking_history": smoking_history,
    "bmi": bmi,
    "blood_glucose_level": blood_glucose_level,
}])

transformed = pd.DataFrame(
    preprocessor.transform(input_data),
        columns = [
            *preprocessor.named_transformers_["scale"].get_feature_names_out(
                ["age", "blood_glucose_level", "bmi"]
        ),
            *preprocessor.named_transformers_["ohe"].get_feature_names_out(
                ["gender", "smoking_history"]
        ),
        "hypertension",
        "heart_disease",
        ],
        index=input_data.index,
)

# --- Mapa przyjaznych nazw cech ---
feature_labels = {
    "age": "Wiek",
    "blood_glucose_level": "Średni poziom glukozy",
    "bmi": "BMI",
    "gender_Female": "Płeć",
    "gender_Male":   "Płeć",
    "smoking_history_non-smoker": "Historia palenia: Nigdy nie paliłam/em",
    "smoking_history_smoker":     "Historia palenia: Palę/paliłem",
    "hypertension": "Nadciśnienie",
    "heart_disease": "Choroba serca",
}

if st.button("Przewiduj ryzyko cukrzycy"):
    # --- 1) Predykcja i wykres kołowy ---
    proba = model.predict_proba(transformed)[0][1]
    percent = int(round(proba * 100))

    fig, ax = plt.subplots()
    size = 0.3
    ax.pie(
        [percent, 100 - percent],
        labels=[f"Ryzyko\n{percent}%", f"Brak ryzyka\n{100 - percent}%"],
        colors=["red", "lightgray"],
        radius=1,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=size, edgecolor="white"),
    )
    st.pyplot(fig)

    # --- 2) SHAP: obliczenie raw log-odds ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    shap_contrib = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    shap_df = pd.DataFrame({
        "feature": transformed.columns,
        "shap": shap_contrib
    })

    # --- 3) Sumy SHAP dla procentowych udziałów ---
    risk_sum = shap_df[shap_df.shap > 0].shap.sum()
    protect_sum = -shap_df[shap_df.shap < 0].shap.sum()

    # --- 4) Przygotowanie rozkładu procentowego ---
    df_break = shap_df.copy()
    # Pokazujemy tylko wybraną płeć
    df_break = df_break[~df_break.feature.str.startswith("gender_") | (df_break.feature == f"gender_{gender}")]
    risk_df_break = df_break[df_break.shap > 0].copy()
    protect_df_break = df_break[df_break.shap < 0].copy()

    # Filtrowanie zerowych udziałów (po zaokrągleniu)
    risk_df_break["contrib_%"] = risk_df_break.shap / risk_sum * percent
    protect_df_break["contrib_%"] = -protect_df_break.shap / protect_sum * (100 - percent)
    risk_df_break = risk_df_break[ risk_df_break["contrib_%"].round(1) > 0 ]
    protect_df_break = protect_df_break[ protect_df_break["contrib_%"].round(1) > 0 ]

    # --- 5) Wizualne wyświetlenie w dwóch kolorowych panelach ---
    col1, col2 = st.columns(2)

    # Czynniki ochronne
    protect_lines = []
    for _, row in protect_df_break.sort_values("contrib_%", ascending=False).iterrows():
        feat = row["feature"]
        pct = row["contrib_%"]
        label = feature_labels.get(feat, feat)
        # wartość i opis
        if feat == "age":
            val = f"{age} lat"
            desc = "Dla osób w Twoim wieku ryzyko cukrzycy jest niższe."
        elif feat == "blood_glucose_level":
            val = f"{blood_glucose_level} mg/dL"
            desc = "Prawidłowy poziom glukozy pomaga utrzymać zdrowie."
        elif feat == "bmi":
            val = f"{bmi:.1f}"
            desc = "Twoje BMI jest w normie i obniża ryzyko cukrzycy."
        elif feat.startswith("gender_"):
            val = selected_gender
            desc = "Twoja płeć jest czynnikiem prognostycznym."
        elif feat.startswith("smoking_history_"):
            val = selected_smoking
            desc = "Brak historii palenia to wsparcie dla zdrowia."
        elif feat == "hypertension":
            val = "Tak" if hypertension == 1 else "Nie"
            desc = "Brak nadciśnienia zmniejsza ryzyko cukrzycy."
        elif feat == "heart_disease":
            val = "Tak" if heart_disease == 1 else "Nie"
            desc = "Brak chorób serca obniża ryzyko cukrzycy."
        protect_lines.append(f"✅ {label}: {val} ({pct:.1f}%) - {desc}")

    # Czynniki ryzyka
    risk_lines = []
    for _, row in risk_df_break.sort_values("contrib_%", ascending=False).iterrows():
        feat = row["feature"]
        pct = row["contrib_%"]
        label = feature_labels.get(feat, feat)
        if feat == "age":
            val = f"{age} lat"
            desc = "Starszy wiek podwyższa ryzyko cukrzycy."
        elif feat == "blood_glucose_level":
            val = f"{blood_glucose_level} mg/dL"
            desc = "Podwyższony poziom glukozy to poważny czynnik ryzyka cukrzycy – kontroluj go regularnie."
        elif feat == "bmi":
            val = f"{bmi:.1f}"
            desc = "Wyższe BMI zwiększa ryzyko cukrzycy."
        elif feat.startswith("gender_"):
            val = selected_gender
            desc = "Niektóre różnice płciowe wpływają na ryzyko."
        elif feat.startswith("smoking_history_"):
            val = selected_smoking
            desc = "Palenie zwiększa ryzyko cukrzycy."
        elif feat == "hypertension":
            val = "Tak" if hypertension == 1 else "Nie"
            desc = "Nadciśnienie podwyższa ryzyko cukrzycy."
        elif feat == "heart_disease":
            val = "Tak" if heart_disease == 1 else "Nie"
            desc = "Obecność chorób serca może zwiększać ryzyko cukrzycy."
        risk_lines.append(f"⚠️ {label}: {val} ({pct:.1f}%) - {desc}")

    # Wyświetlenie paneli
    if protect_lines:
        with col1:
            st.info("### Czynniki ochronne\n\n" + "\n\n".join(protect_lines))
    else:
        with col1:
            st.info("### Czynniki ochronne\nBrak czynników ochronnych.")

    if risk_lines:
        with col2:
            st.error("### Czynniki ryzyka\n\n" + "\n\n".join(risk_lines))
    else:
        with col2:
            st.error("### Czynniki ryzyka\nBrak czynników ryzyka.")

    # --- 6) Disclaimer ---
    st.markdown("##### *to nie jest porada medyczna")
