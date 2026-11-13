# Predicción de la Pobreza Estructural en Argentina mediante Machine Learning (EPH 2016–2026)

**Autor:** Marcelo Gabriel Ulrich (Licenciatura en Ciencia de Datos — EEyN, UNSAM)  
**Directores:** César Germán Santamaría | Agustín Moro

Pipeline reproducible de ingeniería de datos, modelado predictivo, recalibración bayesiana e interpretabilidad SHAP para estimar la condición de pobreza monetaria de los hogares argentinos utilizando **exclusivamente indicadores estructurales y no monetarios** de la Encuesta Permanente de Hogares (EPH — INDEC) entre 2016 y 2026.

---

## 1. Descripción Metodológica

El proyecto aborda dos problemas complementarios a partir de microdatos trimestrales de la EPH:

1. **Focalización microeconómica a nivel hogar (Régimen A):** Clasificación binaria de hogares bajo la línea de pobreza priorizando sensibilidad (`F2-Score` y `PR-AUC`) mediante optimización de umbral $\tau$.
2. **Medición macroeconómica agregada (Régimen B/C):** Estimación de la tasa semestral de pobreza en hogares ponderada por `PONDIH` mediante recalibración bayesiana de probabilidades y ensambles en espacio *logit*, contrastada frente a las publicaciones oficiales del INDEC.

### Decisiones clave de diseño
* **Unidad de análisis y etiqueta oficial:** Consolidación a nivel **hogar** (`codusu` + `nro_hogar`). La línea de pobreza se calcula de manera individualizada según la composición etaria y por sexo de cada hogar (`linea_pobreza = CBT_trimestral × adequi_hogar`), descartando la aproximación fija de "Hogar Tipo 2" (`3.09` adultos equivalentes).
* **Prevención estricta de *Data Leakage*:** Se eliminan 137 variables crudas vinculadas a ingresos directos (`ITF`, `P21`, `P47T`, `IPCF`), deciles, transferencias y subsidios (Módulo V completo), privación alimentaria coyuntural (Módulo VII) y ponderadores muestrales (`PONDERA`, `PONDII`, `PONDIIO`), conservando únicamente `PONDIH` como metadato de expansión para evaluación macro.
* **Cobertura temporal y regional:** Se excluyen el **1T 2016** (emergencia estadística INDEC) y el **2T 2020** (quiebre de relevamiento por ASPO). Se filtran hogares respondentes con `ITF > 0` y se particiona el Gran Buenos Aires en `CABA` y `Conurbano` (7 regiones estadísticas).
* **Esquema de validación sin fuga por rotación 2-2-2:** Partición temporal con bloque de desarrollo (`2016–2021`) mediante *Purged Walk-Forward Validation* (ventanas móviles de 8 trimestres purgando hogares reincidentes del panel rotativo) y bloque de evaluación *Out-of-Sample* (`2022–2026`).

---

## 2. Estructura del Repositorio

```text
ProyectoCIAI/
├── data/
│   ├── eph raw/                                  # [Manual] Microdatos crudos de EPH (.xls / .xlsx)
│   ├── serie_cbt.csv                             # [Manual / Versionado] Serie mensual CBT (INDEC)
│   ├── eph_hogares_consolidada_2016_2026.parquet # [Versionado] Corpus limpio (60 cols, 0 NaNs)
│   ├── serie_pobreza_hogares.csv                 # Auditoría semestral Target vs. INDEC
│   └── resumen_metricas_pobreza.csv              # Resumen estadístico de desviaciones del Target
├── notebooks/
│   ├── limpieza.py                               # Consolidación EPH + CBT y feature engineering
│   ├── analisis_nans.py                          # Depuración residual de registros incompletos
│   ├── analisis_target.py                        # Validación del target reconstruido vs. INDEC
│   ├── prueba_solapamiento.py                    # Auditoría de solapamiento y purga del panel 2-2-2
│   ├── eda.py                                    # Análisis exploratorio univariado y dimensional
│   ├── entrenamiento.py                          # Tuning bayesiano (W&B) con Purged Walk-Forward CV
│   ├── reentrenamiento_oos.py                    # Reentrenamiento Top-3 (2016-2021) e inferencia OOS
│   ├── evaluacion_oos_indec.py                   # Evaluación micro/macro, recalibración y gráficos
│   ├── shap_analysis.py                          # Explicabilidad SHAP recalibrada y ponderada
│   ├── modelos/                                  # [Generado localmente] Checkpoints .joblib / .pt
│   ├── predicciones/                             # Predicciones OOS por familia (.parquet)
│   └── resultados/                               # Métricas CSV, matrices de confusión y carpeta shap/
├── reports/
│   ├── figures/                                  # Figuras de EDA en alta resolución (300 DPI)
│   └── tables/                                   # Tablas resumen de Information Value (IV) y regiones
├── .gitignore
└── requirements.txt
```

> **Nota sobre archivos pesados:** El directorio `notebooks/modelos/` (checkpoints `.joblib` y `.pt`) y las bases crudas en `data/eph raw/` no se incluyen en la versión liviana del repositorio, pero se regeneran íntegramente siguiendo los pasos detallados a continuación.

---

## 3. Requisitos e Instalación

* **Python:** 3.10 o superior (desarrollado y probado en Python 3.12).
* **Hardware recomendado:** CPU multinúcleo para Random Forest / XGBoost y GPU NVIDIA con soporte CUDA para el entrenamiento de la red neuronal tabular (`TabularMLP` en PyTorch) y cálculo de SHAP. El código detecta automáticamente `cuda` o `cpu`.

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd ProyectoCIAI

# 2. Crear y activar entorno virtual
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

*(Nota: para el cálculo de interpretabilidad en `shap_analysis.py`, asegurarse de tener instalado `shap`: `pip install shap` en caso de no figurar en el entorno).*

---

## 4. Obtención y Preparación Manual de Datos Crudos

Dado que las librerías automatizadas de consulta (`pyeph`, documentada en `notebooks/descarga.ipynb`) presentaron inconsistencias en ondas históricas recientes y cambios de esquema entre años, **la descarga de las fuentes primarias se realizó de forma manual** desde los servidores oficiales del INDEC.

Existen dos caminos para reproducir el proyecto:
* **Vía rápida (desde el dataset consolidado):** Utilizar directamente `data/eph_hogares_consolidada_2016_2026.parquet` y `data/serie_cbt.csv` ya incluidos en el repositorio, saltando directamente al **Paso 3** de ejecución.
* **Vía completa (reconstrucción desde cero):** Seguir las instrucciones de descarga manual detalladas debajo.

### 4.1. Microdatos de la EPH (`data/eph raw/`)
1. Ingresar al portal oficial de [Bases de Datos de la EPH — INDEC](https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos).
2. Descargar los archivos comprimidos (`.zip`) de microdatos trimestrales correspondientes a **Hogares** e **Individuos** desde el **2° Trimestre de 2016** hasta el **2° Trimestre de 2026** (omitiendo 1T 2016 y 2T 2020).
3. Descomprimir todos los archivos y ubicar las planillas Excel (`.xls` o `.xlsx`) **directamente dentro de la carpeta `data/eph raw/`** (sin subcarpetas por año), respetando la nomenclatura estándar de INDEC esperada por `limpieza.py`:
   * Bases de hogar: `usu_hogar_T<trimestre><año_2_digitos>.xls` o `.xlsx` (ej. `usu_hogar_T216.xlsx`, `usu_hogar_T423.xlsx`).
   * Bases individuales: `usu_individual_T<trimestre><año_2_digitos>.xls` o `.xlsx` (ej. `usu_individual_T216.xlsx`, `usu_individual_T423.xlsx`).

### 4.2. Serie Histórica de Canasta Básica Total (`data/serie_cbt.csv`)
1. Descargar la planilla oficial de valorización mensual de canastas del INDEC:  
   `https://www.indec.gob.ar/ftp/cuadros/sociedad/serie_cba_cbt.xls`
2. Abrir el archivo y seleccionar la hoja **`CBA-CBT`** (Canasta Básica Alimentaria y Canasta Básica Total para el Adulto Equivalente, GBA).
3. Conservar únicamente la **Columna A (Mes/Período)** y la **Columna D (CBT — Canasta Básica Total en pesos por adulto equivalente)**.
4. Eliminar filas de título, notas al pie o celdas vacías, renombrar los encabezados exactamente como `Mes,CBT` y formatear las fechas como `MM/DD/YY` (primer día de cada mes):
   ```csv
   Mes,CBT
   04/01/16,3663.66
   05/01/16,3830.77
   ...
   07/01/26,506380.55
   ```
5. Exportar el archivo en formato CSV separado por comas (UTF-8) en `data/serie_cbt.csv`.

---

## 5. Orden de Ejecución del Pipeline

### Paso 1: Consolidación, Ingeniería de Features y Etiquetado (Opcional si se usa el `.parquet` versionado)
Ejecutar desde la **raíz del repositorio** para procesar las planillas de `data/eph raw/`, cruzar con `data/serie_cbt.csv`, construir las 53 variables predictoras estructurales y auditar el target:

```bash
# 1. Consolidación incremental a Parquet (excluye leakage y etiqueta pobreza oficial)
python notebooks/limpieza.py

# 2. Limpieza de columnas espurias residuales y control estricto de 0 NaNs
python notebooks/analisis_nans.py

# 3. Auditoría de la tasa de pobreza reconstruida frente a las series oficiales de INDEC
python notebooks/analisis_target.py
```
**Salidas:** `data/eph_hogares_consolidada_2016_2026.parquet`, `data/serie_pobreza_hogares.csv` y `data/resumen_metricas_pobreza.csv`.

---

### Paso 2: Análisis Exploratorio de Datos (EDA) y Auditoría de Panel
Ejecutar desde la **raíz del repositorio**:

```bash
# Verifica porcentajes de solapamiento de hogares en ventanas móviles de 8 trimestres
python notebooks/prueba_solapamiento.py

# Genera figuras de composición dimensional, Information Value (IV) y correlaciones
python notebooks/eda.py
```
**Salidas:** `reports/figures/eda_*.png` y `reports/tables/eda_*.csv`.

---

### Paso 3: Búsqueda Bayesiana de Hiperparámetros (Opcional)
Los mejores hiperparámetros y umbrales $\tau$ obtenidos durante la sintonización ya se encuentran fijados en `notebooks/reentrenamiento_oos.py` (`CONFIGS_TOP3`). Si se desea repetir la búsqueda bayesiana mediante *Purged Walk-Forward Validation* (2016–2021), se requiere una cuenta activa en Weights & Biases (`wandb login`):

```bash
python notebooks/entrenamiento.py --model logistic --project eph-pobreza-ml --count 30
python notebooks/entrenamiento.py --model random_forest --project eph-pobreza-ml --count 30
python notebooks/entrenamiento.py --model xgboost --project eph-pobreza-ml --count 30
python notebooks/entrenamiento.py --model mlp --project eph-pobreza-ml --count 30
```

---

### Paso 4: Reentrenamiento Final (2016–2021 Purgado) e Inferencia OOS (2022–2026)
Ejecutar desde el directorio `notebooks/` para que los artefactos se guarden en `notebooks/modelos/` y `notebooks/predicciones/`:

```bash
cd notebooks

# Entrena el Top-3 de las 4 familias (o especificar: logistic, random_forest, xgboost, mlp)
python reentrenamiento_oos.py --model all
```
**Salidas:**
* `notebooks/modelos/preprocessor_fitted.joblib`, `feature_names_raw.joblib` y checkpoints `<familia>_cfg{1,2,3}.{joblib|pt}`.
* `notebooks/predicciones/predicciones_oos_<familia>.parquet`.

---

### Paso 5: Evaluación Micro/Macro, Recalibración y Comparación con INDEC
Desde `notebooks/`:

```bash
python evaluacion_oos_indec.py
```
**Salidas en `notebooks/resultados/`:**
* `metricas_micro_hogar_oos.csv` (Accuracy, Precision, Recall, Especificidad, F1, F2, PR-AUC, ROC-AUC, Brier).
* `metricas_macro_vs_indec.csv` (MAE, RMSE, Sesgo MBE y Precisión Direccional frente al INDEC oficial).
* `tasas_semestrales_pobreza_hogares.csv` y gráficos comparativos (`evolucion_pobreza_*.png`, `matrices_confusion_pre_calibracion.png`).

---

### Paso 6: Interpretabilidad con SHAP Recalibrado y Ponderado
Desde `notebooks/` (requiere haber ejecutado el Paso 4 para contar con los modelos serializados):

```bash
# Calcula valores SHAP en probabilidad recalibrada ponderada por PONDIH y consolida comparativas
python shap_analysis.py --model all --consolidar
```
*(Opcional: agregar `--all-configs` para promediar las 3 configuraciones por familia en lugar de usar solo `cfg1`, o `--sample-per-sem 0` para evaluar el corpus OOS completo).*

**Salidas en `notebooks/resultados/shap/`:** Gráficos *beeswarm*, evolución longitudinal por dimensión socioeconómica, perfiles por estratos de riesgo y tablas CSV comparativas entre familias.

---

## 6. Resumen de Variables del Corpus Consolidado

El dataset `data/eph_hogares_consolidada_2016_2026.parquet` contiene **60 columnas** sin valores faltantes:
* **Metadatos y Target (7):** `codusu`, `nro_hogar`, `ano4`, `trimestre`, `pondih` (factor de expansión), `linea_pobreza` y `pobre` (target binario).
* **Predictores Estructurales (53):**
  * *Hábitat y Calidad Habitacional (30):* `iv1`–`iv12_3`, `ii1`–`ii9`, `personas_por_cuarto`, `hacinamiento_critico`.
  * *Inserción Laboral (10):* `cant_ocupados`, `cant_desocupados`, `cant_activos`, `cant_asalariados`, `cant_informales`, `tasa_informalidad_hogar`, `tasa_desocupacion_hogar`, `jefe_ocupado`, `jefe_desocupado`, `jefe_informal`.
  * *Estructura Demográfica (9):* `cant_miembros`, `cant_menores_14`, `cant_adultos_mayores`, `cant_edad_activa`, `adequi_hogar`, `ratio_dependencia_demog`, `ratio_dependencia_econom`, `jefe_mujer`, `jefe_edad`.
  * *Capital Humano y Salud (3):* `max_educ_hogar`, `jefe_nivel_ed`, `prop_salud_publica`.
  * *Contexto Regional (1):* `region_estudio` (`CABA`, `Conurbano`, `NOA`, `NEA`, `Cuyo`, `Pampeana`, `Patagonia`).
