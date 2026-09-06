"""
Convierte el archivo original de UCI (processed.cleveland.data, sin encabezados)
en un CSV con nombres de columna legibles. NO se imputan valores faltantes ni se
transforma el target aqui a proposito: esa limpieza se realiza explicitamente
dentro del notebook como parte del proyecto integrador.
"""
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parent / "raw" / "processed.cleveland.data"
OUT_PATH = Path(__file__).parent / "heart_disease.csv"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

df = pd.read_csv(RAW_PATH, header=None, names=COLUMN_NAMES, na_values="?")
assert df.shape == (303, 14), f"Forma inesperada: {df.shape}"

df.to_csv(OUT_PATH, index=False)
print(f"Guardado: {OUT_PATH} -> shape={df.shape}")
print(df.isna().sum()[df.isna().sum() > 0])
