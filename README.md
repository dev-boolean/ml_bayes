# Clasificación de Enfermedades Cardíacas con Modelos Bayesianos

**Proyecto integrador — curso de Aprendizaje Automático (Capítulo 4: Clasificador Bayesiano)**
**Autores:** Andrés Parejo, Santiago Hurtado, Juan Marín

📖 **Sitio publicado (GitHub Pages):** https://dev-boolean.github.io/ml_bayes/

## Qué hace este proyecto

Aplica un clasificador bayesiano (`GaussianNB`) dentro de un `Pipeline` de scikit-learn para
predecir la presencia o ausencia de enfermedad cardíaca en pacientes a partir de indicadores
clínicos (edad, presión arterial, colesterol, tipo de dolor torácico, etc.), usando el *Heart
Disease Dataset* de la UCI (303 pacientes, Cleveland Clinic Foundation).

El notebook cubre el flujo completo de un proyecto de clasificación supervisada:

- **EDA:** distribuciones numéricas y categóricas por diagnóstico, correlaciones.
- **Preprocesamiento:** imputación de valores faltantes, codificación one-hot de variables
  categóricas y escalado, todo dentro de un único `ColumnTransformer` + `Pipeline` (sin fuga de
  datos entre entrenamiento y prueba).
- **Modelado:** `GaussianNB` como modelo principal, con validación cruzada estratificada.
- **Evaluación:** matriz de confusión, accuracy/precision/recall/F1, curva ROC y AUC,
  distribución de probabilidades predichas y las densidades gaussianas condicionales que el
  propio modelo aprende por variable (la parte más "bayesiana" del análisis).
- **Interpretabilidad:** importancia de variables por permutación (AUC), ya que `GaussianNB` no
  expone coeficientes ni `feature_importances_`.
- **Comparación:** `BernoulliNB` y `LogisticRegression` como referencia, para contrastar el costo
  de las asunciones ingenuas de Naive Bayes.
- **Discusión:** fortalezas/limitaciones del modelo bayesiano e interpretabilidad aplicada al
  contexto médico.

**Restricciones didácticas respetadas:** uso obligatorio de `Pipeline`; sin `GridSearchCV`
(`GaussianNB` no tiene hiperparámetros de complejidad que ajustar); trabajo documentado y
reproducible (semilla fija `RANDOM_STATE = 42`).

Enunciado del proyecto: [lihkir.github.io/MachineLearning — 4.4 Proyecto Integrador](https://lihkir.github.io/MachineLearning/bayes_model.html#proyecto-integrador-de-aprendizaje-automatico)

## Estructura del repositorio

```
ml_bayes/
├── docs/                   # Sitio publicado (Jupyter Book compilado) — servido por GitHub Pages
├── notebooks/
│   └── proyecto_bayes_enfermedad_cardiaca.ipynb   # Notebook fuente del proyecto
├── data/
│   ├── raw/                                # Datos originales de UCI, sin procesar
│   ├── prepare_raw_csv.py                  # Script que arma heart_disease.csv con encabezados
│   └── heart_disease.csv                   # Dataset listo para el notebook
├── _config.yml             # Configuración del Jupyter Book (título, autores, tema)
├── _toc.yml                # Tabla de contenidos del libro (una sola página)
└── requirements.txt
```

## Cómo reproducir o recompilar el sitio

```bash
pip install -r requirements.txt

# Editar/ejecutar el notebook directamente
jupyter lab notebooks/proyecto_bayes_enfermedad_cardiaca.ipynb

# Recompilar el sitio (genera _build/html/)
jupyter-book build .

# Publicar los cambios en docs/ (lo que GitHub Pages sirve)
rm -rf docs/*
cp -r _build/html/. docs/
```

`_config.yml` tiene `execute_notebooks: 'off'`: el build reutiliza los resultados ya guardados en
el notebook en vez de re-ejecutarlo. GitHub Pages está configurado para servir desde la carpeta
`docs/` de la rama `main` (Settings → Pages → Branch: `main` / `docs`).

## Dataset

**Heart Disease UCI (subconjunto Cleveland)** — 303 pacientes, 13 variables clínicas + diagnóstico.
Fuente: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
