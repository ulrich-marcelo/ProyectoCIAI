"""
===============================================================================
PROYECTO: Predicción de la Pobreza mediante Machine Learning (EPH 2016-2026)
AUTOR: Marcelo Gabriel Ulrich | Licenciatura en Ciencia de Datos - UNSAM
DIRECTORES: César Germán Santamaría | Agustín Moro
===============================================================================
Script de consolidación, ingeniería de características estructurales y 
construcción del corpus unificado en formato Apache Parquet.
"""

import os
import re
import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xlrd

# Silenciar advertencias de formato de archivos heredados de INDEC
warnings.filterwarnings("ignore")
devnull = open(os.devnull, 'w')
new_defaults = list(xlrd.open_workbook.__defaults__)
new_defaults[1] = devnull
xlrd.open_workbook.__defaults__ = tuple(new_defaults)


# =============================================================================
# 1. DEFINICIÓN DE RUTAS Y PARÁMETROS
# =============================================================================
# Rutas dinámicas compatibles con ejecución en Jupyter o script .py
BASE_DIR = Path.cwd()
DATA_DIR =  '../data/eph raw/'
CBT_FILE = '../data/serie_cbt.csv'
OUTPUT_PARQUET = '../data/eph_consolidada_2016_2026.parquet'

AÑOS = range(2016, 2027)
TRIMESTRES = range(1, 5)


# =============================================================================
# 2. DEFINICIÓN DE VARIABLES A EXCLUIR (MONETARIAS Y NO COMPATIBLES)
# =============================================================================
# Se eliminan montos en pesos, deciles de ingreso, variables administrativas,
# cat_inac (suprimida en 2026) y nuevas variables incorporadas desde 2023/2024.
# IMPORTANTE: Se conservan 'codusu', 'nro_hogar', 'ano4', 'trimestre', 'pondera',
# 'pondih', 'itf' para procesamiento, split temporal y evaluación ponderada.
COLS_TO_DROP_INICIAL = {
    # Ingresos individuales y deciles (Target Leakage)
    "p21", "tot_p12", "p47t", "ipcf", "t_vi", "decocur", "rdecocur", "adecocur",
    "gdecocur", "pdecocur", "idecocur", "decinndr", "rdecindr", "gdecindr",
    "pdecindr", "idecindr", "adecindr", "decifr", "idecifr", "rdecifr", "gdecifr",
    "pdecifr", "adecifr", "deccfr", "ideccfr", "rdeccfr", "gdeccfr", "pdeccfr",
    "adeccfr", "p_deccf", "p_rdeccf", "p_gdeccf", "p_pdeccf", "p_ideccf", "p_adeccf",
    "pondiio", "pondii",

    # Montos de ingresos no laborales del hogar y personas
    "v2_m", "v3_m", "v4_m", "v5_m", "v8_m", "v9_m", "v10_m", "v11_m", "v12_m",
    "v18_m", "v19_am", "v21_m", "v2_01_m", "v2_02_m", "v2_03_m", "v5_01_m",
    "v5_02_m", "v5_03_m", "v11_01_m", "v11_02_m", "v21_01_m", "v21_02_m",
    "v21_03_m", "v22_01_m", "v22_02_m", "v22_03_m",

    # Variables operativas, fechas y clasificaciones muy granulares
    "realizada", "h15", "imputa", "pp04b_cod", "pp04d_cod", "pp11b_cod", "pp11d_cod",
    "pp04b3_mes", "pp04b3_ano", "pp04b3_dia", "pp05b2_mes", "pp05b2_ano", "pp05b2_dia",
    "pp11b2_mes", "pp11b2_ano", "pp11b2_dia", "pp11g_ano", "pp11g_mes", "pp11g_dia",
    "ch05", "ch15_cod", "ch16_cod", "pp08d1", "pp08d4", "pp08f1", "pp08f2",
    "pp08j1", "pp08j2", "pp08j3", "pp09a_esp", "pp09c_esp",

    # Discontinuada en 2026
    "cat_inac",

    # Variables nuevas post-2023 (evitan matriz con 100% de nulos entre 2016 y 2023)
    "empleo", "sector", "pp02a", "pp02b", "pp02d", "pp02f", "pp02g", "pp03k",
    "pp04a1", "pp05b3", "pp05i", "pp05j", "pp05k", "pp06e1", "pp06k", "pp06k_sem",
    "pp06k_mes", "pp06l", "pp07f1_1", "pp07f1_2", "pp07f1_3", "pp07i2", "pp07i3",
    "pp07i4", "pp07l", "pp07m", "pp08g", "pp08g_dsem", "pp08g_dmes", "pp08h",
    "pp10b1", "pp10b2", "pp10b3", "pp10b4", "pp10b5", "pp10b6", "pp10b7",
    "pp10b8", "pp10b9", "pp10b10", "pp11l2"
}


# =============================================================================
# 3. FUNCIONES DE INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================

def calcular_adulto_equivalente(df_i: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el coeficiente calórico oficial de Adulto Equivalente (INDEC)
    según Sexo (CH04: 1=Varón, 2=Mujer) y Edad (CH06).
    """
    cond_v = df_i['ch04'] == 1
    conds_v = [
        (df_i['ch06'] < 1), (df_i['ch06'] == 1), (df_i['ch06'] == 2),
        (df_i['ch06'] == 3), (df_i['ch06'] == 4), (df_i['ch06'] == 5),
        (df_i['ch06'] == 6), (df_i['ch06'] == 7), (df_i['ch06'] == 8),
        (df_i['ch06'] == 9), (df_i['ch06'] >= 10) & (df_i['ch06'] <= 12),
        (df_i['ch06'] >= 13) & (df_i['ch06'] <= 14), (df_i['ch06'] >= 15) & (df_i['ch06'] <= 17),
        (df_i['ch06'] >= 18) & (df_i['ch06'] <= 29), (df_i['ch06'] >= 30) & (df_i['ch06'] <= 60),
        (df_i['ch06'] >= 61) & (df_i['ch06'] <= 75), (df_i['ch06'] > 75)
    ]
    vals_v = [0.35, 0.37, 0.46, 0.51, 0.55, 0.60, 0.64, 0.66, 0.68, 0.69, 
              0.79, 0.98, 1.03, 1.02, 1.00, 0.83, 0.74]
    
    cond_m = df_i['ch04'] == 2
    conds_m = [
        (df_i['ch06'] < 1), (df_i['ch06'] == 1), (df_i['ch06'] == 2),
        (df_i['ch06'] == 3), (df_i['ch06'] == 4), (df_i['ch06'] == 5),
        (df_i['ch06'] == 6), (df_i['ch06'] == 7), (df_i['ch06'] == 8),
        (df_i['ch06'] == 9), (df_i['ch06'] >= 10) & (df_i['ch06'] <= 12),
        (df_i['ch06'] >= 13) & (df_i['ch06'] <= 14), (df_i['ch06'] >= 15) & (df_i['ch06'] <= 17),
        (df_i['ch06'] >= 18) & (df_i['ch06'] <= 29), (df_i['ch06'] >= 30) & (df_i['ch06'] <= 60),
        (df_i['ch06'] >= 61) & (df_i['ch06'] <= 75), (df_i['ch06'] > 75)
    ]
    vals_m = [0.35, 0.37, 0.46, 0.51, 0.55, 0.60, 0.64, 0.66, 0.68, 0.69, 
              0.70, 0.73, 0.73, 0.76, 0.77, 0.67, 0.63]

    df_i['adequi'] = np.where(cond_v, np.select(conds_v, vals_v, default=1.00),
                              np.select(conds_m, vals_m, default=0.77))
    return df_i


def reconstruir_variables_vivienda(df_h: pd.DataFrame) -> pd.DataFrame:
    """
    Armoniza subvariables abiertas post-2020 (V2, V5, V11, V21, V22).
    Si var_madre no existe o está vacía, se consolida:
    1 = Percibe algún concepto; 2 = No percibe ninguno.
    """
    mapeo = {
        'v2':  ['v2_01', 'v2_02', 'v2_03'],
        'v5':  ['v5_01', 'v5_02', 'v5_03'],
        'v11': ['v11_01', 'v11_02'],
        'v21': ['v21_01', 'v21_02', 'v21_03'],
        'v22': ['v22_01', 'v22_02', 'v22_03']
    }
    
    for var_madre, subvars in mapeo.items():
        presentes = [s for s in subvars if s in df_h.columns]
        if presentes:
            es_si = (df_h[presentes] == 1).any(axis=1)
            es_no = (df_h[presentes].isin([0, 2])).all(axis=1)
            
            if var_madre not in df_h.columns:
                df_h[var_madre] = np.nan
                
            df_h.loc[df_h[var_madre].isna() & es_si, var_madre] = 1
            df_h.loc[df_h[var_madre].isna() & es_no, var_madre] = 2
            
    return df_h


def extraer_agregados_hogar(df_i: pd.DataFrame) -> pd.DataFrame:
    """
    Computa métricas estructurales agregadas del hogar y atributos
    del Jefe/a de Hogar de forma 100% vectorizada.
    """
    # Llaves únicas presentes en personas
    hh_keys = df_i[['codusu', 'nro_hogar']].drop_duplicates().set_index(['codusu', 'nro_hogar'])
    
    # 1. Agregados generales
    adequi_h = df_i.groupby(['codusu', 'nro_hogar'])['adequi'].sum().rename('adequi_hogar')
    miembros = df_i.groupby(['codusu', 'nro_hogar'])['componente'].count().rename('cant_miembros')
    ocupados = df_i[df_i['estado'] == 1].groupby(['codusu', 'nro_hogar'])['componente'].count().rename('cant_ocupados')
    desocupados = df_i[df_i['estado'] == 2].groupby(['codusu', 'nro_hogar'])['componente'].count().rename('cant_desocupados')
    
    # Segmentación etaria
    menores_14 = df_i[df_i['ch06'] < 14].groupby(['codusu', 'nro_hogar'])['componente'].count().rename('menores_14')
    mayores_65 = df_i[df_i['ch06'] >= 65].groupby(['codusu', 'nro_hogar'])['componente'].count().rename('mayores_65')
    activos_15_64 = df_i[(df_i['ch06'] >= 15) & (df_i['ch06'] <= 64)].groupby(['codusu', 'nro_hogar'])['componente'].count().rename('activos_15_64')
    
    # Clima educativo: Máximo nivel educativo de adultos (>= 18 años)
    max_educ = df_i[df_i['ch06'] >= 18].groupby(['codusu', 'nro_hogar'])['nivel_ed'].max().rename('max_educ_hogar')
    
    # Informalidad laboral en el hogar (asalariados sin descuento jubilatorio: cat_ocup == 3 y pp07h == 2)
    asalariados = df_i[(df_i['estado'] == 1) & (df_i['cat_ocup'] == 3)].groupby(['codusu', 'nro_hogar'])['componente'].count().rename('cant_asalariados')
    informales = df_i[(df_i['estado'] == 1) & (df_i['cat_ocup'] == 3) & (df_i.get('pp07h', 0) == 2)].groupby(['codusu', 'nro_hogar'])['componente'].count().rename('cant_informales')
    
    # Consolidación vectorizada
    agg = pd.concat([
        hh_keys, adequi_h, miembros, ocupados, desocupados,
        menores_14, mayores_65, activos_15_64, max_educ, asalariados, informales
    ], axis=1).reset_index()
    
    cols_zero = ['cant_ocupados', 'cant_desocupados', 'menores_14', 'mayores_65', 
                 'activos_15_64', 'cant_asalariados', 'cant_informales']
    agg[cols_zero] = agg[cols_zero].fillna(0)
    agg['max_educ_hogar'] = agg['max_educ_hogar'].fillna(1)
    
    # Ratios socioeconómicos derivados
    agg['ratio_dependencia_demog'] = (agg['menores_14'] + agg['mayores_65']) / np.maximum(agg['activos_15_64'], 1)
    agg['ratio_dependencia_econom'] = agg['cant_miembros'] / np.maximum(agg['cant_ocupados'], 1)
    agg['sin_ocupados'] = (agg['cant_ocupados'] == 0).astype(float)
    agg['tasa_informalidad_hogar'] = np.where(
        agg['cant_asalariados'] > 0,
        agg['cant_informales'] / agg['cant_asalariados'],
        0.0
    )
    
    # 2. Atributos estructurales del Jefe/a de Hogar (ch03 == 1)
    jefes = df_i[df_i['ch03'] == 1].drop_duplicates(subset=['codusu', 'nro_hogar'])
    jefes_features = pd.DataFrame({
        'codusu': jefes['codusu'],
        'nro_hogar': jefes['nro_hogar'],
        'jefe_sexo': jefes['ch04'].astype(float),
        'jefe_edad': jefes['ch06'].astype(float),
        'jefe_nivel_ed': jefes['nivel_ed'].astype(float),
        'jefe_estado': jefes['estado'].astype(float),
        'jefe_es_informal': ((jefes['estado'] == 1) & (jefes['cat_ocup'] == 3) & (jefes.get('pp07h', 0) == 2)).astype(float),
        'jefe_es_cuentapropia': ((jefes['estado'] == 1) & (jefes['cat_ocup'] == 2)).astype(float)
    })
    
    return agg.merge(jefes_features, on=['codusu', 'nro_hogar'], how='left')


def procesar_trimestre(df_i: pd.DataFrame, df_h: pd.DataFrame, cbt_valor: float) -> pd.DataFrame:
    """
    Ejecuta el pipeline de transformación, etiquetado y merge para un trimestre.
    """
    # 1. Adulto equivalente
    df_i = calcular_adulto_equivalente(df_i)
    
    # 2. Reconstrucción de variables de vivienda
    df_h = reconstruir_variables_vivienda(df_h)
    
    # 3. Agregados a nivel hogar
    hogar_agg = extraer_agregados_hogar(df_i)
    
    # 4. Enriquecimiento del hogar y cómputo de la línea de pobreza
    h_completo = df_h.merge(hogar_agg, on=['codusu', 'nro_hogar'], how='left')
    
    # Hacinamiento (personas por habitación de uso exclusivo / dormitorio)
    habitaciones = pd.to_numeric(h_completo.get('ii1', np.nan), errors='coerce').fillna(1)
    h_completo['hacinamiento'] = h_completo['cant_miembros'] / np.maximum(habitaciones, 1)
    
    # Línea de Pobreza Total del Hogar
    h_completo['linea_pobreza_hogar'] = cbt_valor * h_completo['adequi_hogar']
    
    # Etiquetado objetivo: 'esPobre'
    # Respondentes completos (ITF > 0): 1 = Pobre, 0 = No Pobre
    # No respuesta de ingresos (ITF == 0 o NaN): NaN (subconjunto de imputación)
    h_completo['esPobre'] = np.nan
    itf_num = pd.to_numeric(h_completo['itf'], errors='coerce').fillna(0)
    respondentes = itf_num > 0
    h_completo.loc[respondentes, 'esPobre'] = (
        itf_num[respondentes] < h_completo.loc[respondentes, 'linea_pobreza_hogar']
    ).astype(float)
    
    # 5. Evitar colisiones de columnas antes del merge final
    cols_duplicadas = set(df_i.columns).intersection(set(h_completo.columns)) - {'codusu', 'nro_hogar'}
    h_para_unir = h_completo.drop(columns=list(cols_duplicadas))
    
    # 6. Merge final: Individuos left join Hogar Enriquecido
    df_final_trim = df_i.merge(h_para_unir, on=['codusu', 'nro_hogar'], how='left')
    
    return df_final_trim


# =============================================================================
# 4. EJECUCIÓN PRINCIPAL CON ESCRITURA INCREMENTAL EN DISCO
# =============================================================================

def ejecutar_procesamiento():
    print("--- INICIANDO PIPELINE DE LIMPIEZA Y CONSOLIDACIÓN EPH (2016-2026) ---")
    
    # Cargar y promediar la Canasta Básica Total (CBT) trimestral
    if not CBT_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo CBT en: {CBT_FILE}")
        
    cbt = pd.read_csv(CBT_FILE)
    cbt['Mes_dt'] = pd.to_datetime(cbt['Mes'], format='%m/%d/%y')
    cbt['ano4'] = cbt['Mes_dt'].dt.year
    cbt['trimestre'] = cbt['Mes_dt'].dt.quarter
    cbt_trim = cbt.groupby(['ano4', 'trimestre'])['CBT'].mean().reset_index()
    print("Serie CBT trimestral cargada exitosamente.")
    
    # Archivos disponibles
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"El directorio de microdatos no existe: {DATA_DIR}")
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.xls', '.xlsx'))]
    print(f"Archivos encontrados en carpeta cruda: {len(all_files)}")
    
    writer = None
    columnas_esquema_fijo = None
    total_filas = 0
    
    for año in AÑOS:
        año_corto = str(año)[-2:]
        for trim in TRIMESTRES:
            # En 2016, INDEC no publicó el T1
            if año == 2016 and trim == 1:
                continue
                
            re_hogar = re.compile(rf"(?i)usu_hogar_t{trim}{año_corto}\.xlsx?")
            re_indiv = re.compile(rf"(?i)usu_individual_t{trim}{año_corto}\.xlsx?")
            
            file_h = next((f for f in all_files if re_hogar.search(f)), None)
            file_i = next((f for f in all_files if re_indiv.search(f)), None)
            
            if not file_h or not file_i:
                continue
                
            # Obtener valor de la CBT correspondiente
            cbt_row = cbt_trim[(cbt_trim['ano4'] == año) & (cbt_trim['trimestre'] == trim)]
            if cbt_row.empty:
                print(f"[ALERTA] Sin dato CBT para T{trim} {año}. Se omite.")
                continue
            cbt_valor = cbt_row['CBT'].values[0]
            
            # Carga optimizada descartando variables excluidas
            df_h = pd.read_excel(
                DATA_DIR / file_h,
                usecols=lambda c: c.lower() not in COLS_TO_DROP_INICIAL
            )
            df_i = pd.read_excel(
                DATA_DIR / file_i,
                usecols=lambda c: c.lower() not in COLS_TO_DROP_INICIAL
            )
            
            df_h.columns = df_h.columns.str.lower()
            df_i.columns = df_i.columns.str.lower()
            
            # Limpieza básica de strings en identificadores
            df_h['codusu'] = df_h['codusu'].astype(str).str.strip()
            df_i['codusu'] = df_i['codusu'].astype(str).str.strip()
            
            # Procesar el trimestre
            merged_trim = procesar_trimestre(df_i, df_h, cbt_valor)
            
            # --- GARANTIZAR HOMOGENEIDAD Y TIPO DE DATOS PARA PARQUET ---
            if columnas_esquema_fijo is None:
                # Fijamos el orden y conjunto de columnas con el primer trimestre válido
                columnas_esquema_fijo = list(merged_trim.columns)
            else:
                # Armonizamos columnas (NaN si falta en algún trimestre; descarta columnas raras)
                merged_trim = merged_trim.reindex(columns=columnas_esquema_fijo)
                
            # Casteo estricto de tipos para evitar choques en el esquema de PyArrow
            for col in columnas_esquema_fijo:
                if col in ['codusu']:
                    merged_trim[col] = merged_trim[col].astype(str)
                elif col in ['ano4', 'trimestre', 'nro_hogar', 'componente']:
                    merged_trim[col] = pd.to_numeric(merged_trim[col], errors='coerce').fillna(0).astype('int32')
                else:
                    # Todas las variables estructurales, ratios y esPobre en float32
                    merged_trim[col] = pd.to_numeric(merged_trim[col], errors='coerce').astype('float32')
            
            # Escritura en bloques (streaming) al archivo Parquet
            table = pa.Table.from_pandas(merged_trim, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema, compression='snappy')
                
            writer.write_table(table)
            total_filas += len(merged_trim)
            print(f"✓ Procesado y anexado con éxito: T{trim} {año} ({len(merged_trim):,} personas)")
            
            # Liberar memoria
            del df_h, df_i, merged_trim, table
            gc.collect()
            
    if writer:
        writer.close()
        print(f"\n=================================================================")
        print(f"¡CONSOLIDACIÓN FINALIZADA!")
        print(f"Archivo generado: {OUTPUT_PARQUET.resolve()}")
        print(f"Total de registros consolidados: {total_filas:,}")
        print(f"=================================================================\n")
    else:
        print("No se encontraron trimestres válidos para procesar.")


# =============================================================================
# 5. VERIFICACIÓN Y AUDITORÍA DEL DATASET GENERADO
# =============================================================================

def auditar_dataset():
    if not OUTPUT_PARQUET.exists():
        print("El archivo parquet consolidado aún no ha sido generado.")
        return
        
    df = pd.read_parquet(OUTPUT_PARQUET)
    
    print("--- 1. INFORMACIÓN GENERAL ---")
    print(f"Filas totales (individuos): {df.shape[0]:,}")
    print(f"Columnas totales: {df.shape[1]}")
    print(f"Uso de memoria: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n")
    
    print("--- 2. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (esPobre) ---")
    # esPobre: 1.0 (Pobre), 0.0 (No Pobre), NaN (No Respuesta ITF == 0)
    conteo = df['esPobre'].value_counts(dropna=False)
    pct = df['esPobre'].value_counts(dropna=False, normalize=True) * 100
    resumen_target = pd.DataFrame({'Casos': conteo, 'Porcentaje (%)': pct.round(2)})
    resumen_target.index = ['No Pobre (0.0)' if i == 0.0 else 'Pobre (1.0)' if i == 1.0 else 'No Respuesta (NaN)' for i in resumen_target.index]
    print(resumen_target)
    print()
    
    print("--- 3. VERIFICACIÓN DE VARIABLES ESTRUCTURALES CLAVE ---")
    cols_check = [c for c in ['cant_miembros', 'hacinamiento', 'ratio_dependencia_demog', 
                             'max_educ_hogar', 'tasa_informalidad_hogar'] if c in df.columns]
    print(df[cols_check].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].round(2))


if __name__ == "__main__":
    ejecutar_procesamiento()
    auditar_dataset()