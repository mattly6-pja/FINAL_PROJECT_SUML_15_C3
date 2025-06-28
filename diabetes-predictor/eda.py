import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("data/01_raw/diabetes-dataset.csv")

profile = ProfileReport(df, title="Eksploracyjna analiza danych", explorative=True)

profile.to_file("docs/eda.html")