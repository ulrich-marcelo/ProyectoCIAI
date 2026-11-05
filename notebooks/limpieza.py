"""
===============================================================================
PROYECTO: Predicción de la Pobreza mediante Machine Learning (EPH 2016-2026)
AUTOR: Marcelo Gabriel Ulrich | Licenciatura en Ciencia de Datos - UNSAM
DIRECTORES: César Germán Santamaría | Agustín Moro
===============================================================================
Script de consolidación a nivel HOGAR con escritura incremental en Parquet.
- Exclusión metodológica del 2T 2020 (ASPO).
- Filtrado estricto de respondentes (ITF > 0).
- Separación regional en 7 zonas (CABA, Conurbano y 5 regiones INDEC).
- Supresión total de ponderadores, Módulo V (sustento/transferencias) y Módulo VII.
- Depuración estricta: archivo resultante 100% libre de NaNs.
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

warnings.filterwarnings("ignore")

import xlrd

xlrd_log = open(os.devnull, 'w')
if hasattr(xlrd, 'open_workbook'):
    defaults = list(xlrd.open_workbook.__defaults__)
    defaults[1] = xlrd_log
    xlrd.open_workbook.__defaults__ = tuple(defaults)

# =============================================================================
# 1. PARÁMETROS Y LISTA DE EXCLUSIÓN TOTAL
# =============================================================================
BASE_DIR = Path.cwd()
DATA_DIR = Path("data/eph raw/")
CBT_FILE = Path("data/serie_cbt.csv")
OUTPUT_PARQUET = Path("data/eph_hogares_consolidada_2016_2026.parquet")

AÑOS = range(2016, 2027)
TRIMESTRES = range(1, 5)

COLS_TO_DROP = {
    # 1. Ponderadores muestrales (Target Leakage crítico)
    "pondera", "pondih", "pondiio", "pondii",
    
    # 2. Ingresos individuales directos y deciles (Target Leakage)
    "p21", "tot_p12", "p47t", "ipcf", "t_vi", "decocur", "rdecocur", "adecocur",
    "gdecocur", "pdecocur", "idecocur", "decinndr", "rdecindr", "gdecindr",
    "pdecindr", "idecindr", "adecindr", "decifr", "idecifr", "rdecifr", "gdecifr",
    "pdecifr", "adecifr", "deccfr", "ideccfr", "rdeccfr", "gdeccfr", "pdeccfr",
    "adeccfr", "p_deccf", "p_rdeccf", "p_gdeccf", "p_pdeccf", "p_ideccf", "p_adeccf",
    
    # 3. Módulo V completo: Banderas de sustento, transferencias y montos
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", 
    "v13", "v14", "v15", "v16", "v17", "v18", "v19_a", "v19_b", "v19_am", "v21", "v22",
    "v2_01", "v2_02", "v2_03", "v5_01", "v5_02", "v5_03", "v11_01", "v11_02", 
    "v21_01", "v21_02", "v21_03", "v22_01", "v22_02", "v22_03",
    "v1_m", "v2_m", "v3_m", "v4_m", "v5_m", "v8_m", "v9_m", "v10_m", "v11_m", 
    "v12_m", "v18_m", "v21_m", "v2_01_m", "v2_02_m", "v2_03_m", "v5_01_m", 
    "v5_02_m", "v5_03_m", "v11_01_m", "v11_02_m", "v21_01_m", "v21_02_m", 
    "v21_03_m", "v22_01_m", "v22_02_m", "v22_03_m",

    # 4. Módulo VII: Privación alimentaria y estrategias de consumo coyuntural
    "vii1_1", "vii1_2", "vii2_1", "vii2_2", "vii2_3", "vii2_4",

    # 5. Módulo IX redundante
    "ix_tot", "ix_men10", "ix_mayeq10",

    # 6. Variables de texto abierto y operativas
    "mas_500", "iv1_esp", "iv3_esp", "iv7_esp", "ii7_esp", "ii8_esp", 
    "pp09a_esp", "pp09c_esp", "realizada", "h15", "imputa", "cba", "indigencia",
    "pp04b_cod", "pp04d_cod", "pp11b_cod", "pp11d_cod",
    "pp04b3_mes", "pp04b3_ano", "pp04b3_dia", "pp05b2_mes", "pp05b2_ano", "pp05b2_dia",
    "pp06c", "pp06d", "pp11b2_mes", "pp11b2_ano", "pp11b2_dia", "pp11g_ano", 
    "pp11g_mes", "pp11g_dia", "ch05", "ch15_cod", "ch16_cod",
    "pp08d1", "pp08d4", "pp08f1", "pp08f2", "pp08j1", "pp08j2", "pp08j3",
    "pp09a", "pp09b", "pp09c", "pp02c1", "pp02c2", "pp02c3", "pp02c4", 
    "pp02c5", "pp02c6", "pp02c7", "pp02c8",
    "cat_inac", "empleo", "sector", "pp02a", "pp02b", "pp02d", "pp02f", "pp02g", 
    "pp03k", "pp04a1", "pp05b3", "pp05i", "pp05j", "pp05k", "pp06e1", "pp06k", 
    "pp06k_sem", "pp06k_mes", "pp06l", "pp07f1_1", "pp07f1_2", "pp07f1_3", 
    "pp07i2", "pp07i3", "pp07i4", "pp07l", "pp07m", "pp08g", "pp08g_dsem", 
    "pp08g_dmes", "pp08h", "pp10b1", "pp10b2", "pp10b3", "pp10b4", "pp10b5", 
    "pp10b6", "pp10b7", "pp10b8", "pp10b9", "pp10b10", "pp11l2"
}

# =============================================================================
# 2. ADULTO EQUIVALENTE OFICIAL (INDEC)
# =============================================================================
def calcular_adulto_equivalente(df_i: pd.DataFrame) -> pd.DataFrame:
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

    adequi = np.where(cond_v, np.select(conds_v, vals_v, default=1.00),
                      np.select(conds_m, vals_m, default=0.77))
    
    return df_i.assign(adequi=adequi)

# =============================================================================
# 3. AGREGACIÓN VECTORIZADA (INDIVIDUO -> HOGAR)
# =============================================================================
def procesar_agregados_hogar(df_i: pd.DataFrame) -> pd.DataFrame:
    df_i = calcular_adulto_equivalente(df_i)
    
    df_i = df_i.assign(
        es_menor_14=(df_i['ch06'] < 14).astype(int),
        es_adulto_mayor=(df_i['ch06'] >= 65).astype(int),
        es_edad_activa=((df_i['ch06'] >= 15) & (df_i['ch06'] <= 64)).astype(int),
        es_ocupado=(df_i['estado'] == 1).astype(int),
        es_desocupado=(df_i['estado'] == 2).astype(int),
        es_activo=(df_i['estado'].isin([1, 2])).astype(int),
        es_asalariado=((df_i['estado'] == 1) & (df_i.get('cat_ocup', 0) == 3)).astype(int),
        es_informal=((df_i['estado'] == 1) & (df_i.get('cat_ocup', 0) == 3) & (df_i.get('pp07h', 0) == 2)).astype(int),
        solo_salud_publica=(df_i.get('ch08', 0) == 4).astype(int)
    )
    
    hogar_agg = df_i.groupby(['codusu', 'nro_hogar']).agg(
        cant_miembros=('ch06', 'count'),
        cant_menores_14=('es_menor_14', 'sum'),
        cant_adultos_mayores=('es_adulto_mayor', 'sum'),
        cant_edad_activa=('es_edad_activa', 'sum'),
        cant_ocupados=('es_ocupado', 'sum'),
        cant_desocupados=('es_desocupado', 'sum'),
        cant_activos=('es_activo', 'sum'),
        cant_asalariados=('es_asalariado', 'sum'),
        cant_informales=('es_informal', 'sum'),
        adequi_hogar=('adequi', 'sum'),
        prop_salud_publica=('solo_salud_publica', 'mean'),
        max_educ_hogar=('nivel_ed', lambda s: s[df_i.loc[s.index, 'ch06'] >= 18].max() if (df_i.loc[s.index, 'ch06'] >= 18).any() else 1)
    )
    
    hogar_agg = hogar_agg.assign(
        max_educ_hogar=hogar_agg['max_educ_hogar'].fillna(1),
        ratio_dependencia_demog=lambda df: (df['cant_menores_14'] + df['cant_adultos_mayores']) / np.maximum(df['cant_edad_activa'], 1),
        ratio_dependencia_econom=lambda df: df['cant_miembros'] / np.maximum(df['cant_ocupados'], 1),
        tasa_informalidad_hogar=lambda df: np.where(df['cant_asalariados'] > 0, df['cant_informales'] / df['cant_asalariados'], 0.0),
        tasa_desocupacion_hogar=lambda df: np.where(df['cant_activos'] > 0, df['cant_desocupados'] / df['cant_activos'], 0.0)
    ).reset_index()

    jefes = df_i[df_i['ch03'] == 1].drop_duplicates(subset=['codusu', 'nro_hogar'])
    jefes_features = pd.DataFrame({
        'codusu': jefes['codusu'],
        'nro_hogar': jefes['nro_hogar'],
        'jefe_mujer': (jefes['ch04'] == 2).astype(float),
        'jefe_edad': jefes['ch06'].astype(float),
        'jefe_nivel_ed': jefes['nivel_ed'].astype(float),
        'jefe_ocupado': (jefes['estado'] == 1).astype(float),
        'jefe_desocupado': (jefes['estado'] == 2).astype(float),
        'jefe_informal': ((jefes['estado'] == 1) & (jefes.get('cat_ocup', 0) == 3) & (jefes.get('pp07h', 0) == 2)).astype(float)
    })
    
    return hogar_agg.merge(jefes_features, on=['codusu', 'nro_hogar'], how='left')

# =============================================================================
# 4. PIPELINE PRINCIPAL: STREAMING INCREMENTAL A PARQUET
# =============================================================================
def ejecutar_pipeline():
    print("--- INICIANDO PIPELINE DE CONSOLIDACIÓN (ESTRUCTURAL PURO, CERO NaNs) ---")
    
    if not CBT_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {CBT_FILE}")
        
    cbt = pd.read_csv(CBT_FILE)
    cbt['Mes_dt'] = pd.to_datetime(cbt['Mes'], format='%m/%d/%y')
    cbt['ano4'] = cbt['Mes_dt'].dt.year
    cbt['trimestre'] = cbt['Mes_dt'].dt.quarter
    cbt_trim = cbt.groupby(['ano4', 'trimestre'])['CBT'].mean().reset_index()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No existe el directorio de datos crudos: {DATA_DIR}")
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.xls', '.xlsx'))]
    
    writer = None
    columnas_fijas = None
    total_hogares = 0
    
    for año in AÑOS:
        año_corto = str(año)[-2:]
        for trim in TRIMESTRES:
            # Exclusión de 1T 2016 (emergencia estadística) y 2T 2020 (quiebre ASPO)
            if (año == 2016 and trim == 1) or (año == 2020 and trim == 2):
                continue
                
            re_hogar = re.compile(rf"(?i)usu_hogar_t{trim}{año_corto}\.xlsx?")
            re_indiv = re.compile(rf"(?i)usu_individual_t{trim}{año_corto}\.xlsx?")
            
            file_h = next((f for f in all_files if re_hogar.search(f)), None)
            file_i = next((f for f in all_files if re_indiv.search(f)), None)
            
            if not file_h or not file_i:
                continue
                
            cbt_row = cbt_trim[(cbt_trim['ano4'] == año) & (cbt_trim['trimestre'] == trim)]
            if cbt_row.empty:
                print(f"[ALERTA] Sin CBT para T{trim} {año}. Se omite.")
                continue
            cbt_valor = float(cbt_row['CBT'].values[0])
            
            df_h = pd.read_excel(DATA_DIR / file_h, usecols=lambda c: c.lower() not in COLS_TO_DROP)
            df_i = pd.read_excel(DATA_DIR / file_i, usecols=lambda c: c.lower() not in COLS_TO_DROP)
            
            df_h.columns = df_h.columns.str.lower().str.strip()
            df_i.columns = df_i.columns.str.lower().str.strip()
            
            df_h['codusu'] = df_h['codusu'].astype(str).str.strip()
            df_i['codusu'] = df_i['codusu'].astype(str).str.strip()
            df_h['nro_hogar'] = pd.to_numeric(df_h['nro_hogar'], errors='coerce').fillna(1).astype(int)
            df_i['nro_hogar'] = pd.to_numeric(df_i['nro_hogar'], errors='coerce').fillna(1).astype(int)
            
            # FILTRO ESTRICTO: Solo respondentes con ITF > 0
            if 'itf' not in df_h.columns:
                print(f"[ALERTA] Columna 'itf' ausente en T{trim} {año}. Se omite.")
                continue
                
            df_h['itf'] = pd.to_numeric(df_h['itf'], errors='coerce').fillna(0.0)
            df_h = df_h[df_h['itf'] > 0].copy()
            if df_h.empty:
                continue
            
            # PARTICIÓN REGIONAL EN 7 ZONAS
            df_h['aglomerado'] = pd.to_numeric(df_h.get('aglomerado', np.nan), errors='coerce')
            df_h['region'] = pd.to_numeric(df_h.get('region', np.nan), errors='coerce')

            conds_region = [
                df_h['aglomerado'] == 32,
                df_h['aglomerado'] == 33,
                df_h['region'] == 40,
                df_h['region'] == 41,
                df_h['region'] == 42,
                df_h['region'] == 43,
                df_h['region'] == 44
            ]
            choices_region = [
                'CABA', 'Conurbano', 'NOA', 'NEA', 'Cuyo', 'Pampeana', 'Patagonia'
            ]
            
            df_h['region_estudio'] = np.select(conds_region, choices_region, default='Otro')
            df_h = df_h.drop(columns=[col for col in ['aglomerado', 'region'] if col in df_h.columns])
            
            df_i_agg = procesar_agregados_hogar(df_i)
            hogar_consolidado = df_h.merge(df_i_agg, on=['codusu', 'nro_hogar'], how='inner')
            if hogar_consolidado.empty:
                continue
            
            # Ratios habitacionales
            habitaciones = pd.to_numeric(hogar_consolidado.get('ii1', np.nan), errors='coerce').fillna(1)
            hogar_consolidado['personas_por_cuarto'] = hogar_consolidado['cant_miembros'] / np.maximum(habitaciones, 1)
            hogar_consolidado['hacinamiento_critico'] = (hogar_consolidado['personas_por_cuarto'] > 3).astype(float)
            
            # TARGET OFICIAL INDEC
            hogar_consolidado['linea_pobreza'] = (cbt_valor * hogar_consolidado['adequi_hogar']).astype('float32')
            hogar_consolidado['pobre'] = (hogar_consolidado['itf'] < hogar_consolidado['linea_pobreza']).astype('int32')
            
            # Eliminación de ITF post-etiquetado
            hogar_consolidado = hogar_consolidado.drop(columns=['itf'])

            if 'ano4' not in hogar_consolidado.columns:
                hogar_consolidado['ano4'] = año
            if 'trimestre' not in hogar_consolidado.columns:
                hogar_consolidado['trimestre'] = trim
                
            # Homogeneización de tipos de datos
            if columnas_fijas is None:
                columnas_fijas = list(hogar_consolidado.columns)
            else:
                hogar_consolidado = hogar_consolidado.reindex(columns=columnas_fijas)
                
            for col in columnas_fijas:
                if col in ['codusu', 'region_estudio']:
                    hogar_consolidado[col] = hogar_consolidado[col].astype(str)
                elif col in ['ano4', 'trimestre', 'nro_hogar', 'pobre']:
                    hogar_consolidado[col] = pd.to_numeric(hogar_consolidado[col], errors='coerce').fillna(0).astype('int32')
                else:
                    hogar_consolidado[col] = pd.to_numeric(hogar_consolidado[col], errors='coerce').astype('float32')
            
            # DEPURACIÓN ESTRICTA: Eliminación de filas con datos incompletos
            hogar_consolidado = hogar_consolidado.dropna().reset_index(drop=True)
            if hogar_consolidado.empty:
                continue

            # Control de calidad previo al guardado
            assert hogar_consolidado.isna().sum().sum() == 0, f"Error: Se detectaron NaNs en T{trim} {año}"

            table = pa.Table.from_pandas(hogar_consolidado, preserve_index=False)
            if writer is None:
                OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema, compression='snappy')
            
            writer.write_table(table)
            total_hogares += len(hogar_consolidado)
            print(f"✓ Guardado: T{trim} {año} ({len(hogar_consolidado):,} hogares respondentes sin NaNs)")
            
            del df_h, df_i, df_i_agg, hogar_consolidado, table
            gc.collect()
            
    if writer:
        writer.close()
        print(f"\n¡Proceso finalizado! Archivo guardado: {OUTPUT_PARQUET.resolve()}")
        print(f"Total de observaciones consolidadas: {total_hogares:,}\n")

if __name__ == "__main__":
    ejecutar_pipeline()