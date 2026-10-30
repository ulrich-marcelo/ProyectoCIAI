"""
===============================================================================
PROYECTO: Predicción de Pobreza mediante Machine Learning (EPH 2016-2026)
AUTOR: Marcelo Gabriel Ulrich | EEyN - UNSAM
DIRECTORES: César Germán Santamaría | Agustín Moro
===============================================================================
Módulo de Análisis Exploratorio de Datos (EDA):
1. Dinámica temporal y no respuesta a ingresos.
2. Demografía, dependencia y hacinamiento.
3. Clima educativo e informalidad laboral.
4. Matriz de correlaciones estructurales.
===============================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración estética general para gráficos académicos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

PARQUET_PATH = Path("data/eph_hogares_consolidada_2016_2026.parquet")
OUTPUT_FIGS = Path("reports/figures/")
OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE SUBMUESTRAS
# =============================================================================
print("Cargando dataset...")
df = pd.read_parquet(PARQUET_PATH)

# Identificador temporal cronológico
df['semestre'] = np.where(df['trimestre'].isin([1, 2]), 1, 2)
df['periodo_sem'] = df['ano4'].astype(str) + "-S" + df['semestre'].astype(str)
df['periodo_trim'] = df['ano4'].astype(str) + "-T" + df['trimestre'].astype(str)

# Subconjuntos: Total de relevamiento vs. Respondentes válidos
df_respondentes = df.dropna(subset=['pobre_oficial']).copy()
df_respondentes['pobre_oficial'] = df_respondentes['pobre_oficial'].astype(int)

# =============================================================================
# FIGURA 1: EVOLUCIÓN TEMPORAL DE LA POBREZA Y TASA DE NO RESPUESTA
# =============================================================================
def plot_dinamica_temporal_y_no_respuesta(df_all):
    resumen_periodo = []
    
    for p, subdf in df_all.groupby('periodo_trim'):
        total_h = len(subdf)
        no_resp_h = (subdf['itf'] == 0).sum() if 'itf' in subdf.columns else subdf['pobre_oficial'].isna().sum()
        pct_no_resp = (no_resp_h / total_h) * 100
        
        # Pobreza ponderada por PONDIH en respondentes
        resp = subdf.dropna(subset=['pobre_oficial'])
        if len(resp) > 0 and resp['pondih'].sum() > 0:
            tasa_pob = np.average(resp['pobre_oficial'], weights=resp['pondih']) * 100
        else:
            tasa_pob = np.nan
            
        resumen_periodo.append({
            'periodo': p,
            'pct_no_respuesta': pct_no_resp,
            'tasa_pobreza_hogares': tasa_pob
        })
        
    df_temp = pd.DataFrame(resumen_periodo).sort_values('periodo')
    
    fig, ax1 = plt.subplots(figsize=(14, 5))
    
    color_pob = '#c0392b'
    ax1.set_xlabel('Trimestre de Relevamiento')
    ax1.set_ylabel('Tasa de Pobreza en Hogares (%) [PONDIH]', color=color_pob, fontweight='bold')
    line1 = ax1.plot(df_temp['periodo'], df_temp['tasa_pobreza_hogares'], color=color_pob, marker='o', lw=2, label='Pobreza Hogares')
    ax1.tick_params(axis='y', labelcolor=color_pob)
    ax1.set_xticks(range(0, len(df_temp), 2))
    ax1.set_xticklabels(df_temp['periodo'].iloc[::2], rotation=45, ha='right')
    
    # Eje secundario para la no respuesta
    ax2 = ax1.twinx()
    color_nr = '#2980b9'
    ax2.set_ylabel('Tasa de No Respuesta a Ingresos (%)', color=color_nr, fontweight='bold')
    line2 = ax2.plot(df_temp['periodo'], df_temp['pct_no_respuesta'], color=color_nr, linestyle='--', marker='s', lw=1.8, label='No Respuesta ITF')
    ax2.tick_params(axis='y', labelcolor=color_nr)
    ax2.grid(False)
    
    plt.title('Evolución de la Incidencia de Pobreza y Tasa de No Respuesta (EPH 2016-2026)', pad=15, fontweight='bold')
    fig.tight_layout()
    plt.savefig(OUTPUT_FIGS / "eda_01_dinamica_temporal.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# FIGURA 2: DEMOGRAFÍA, HACINAMIENTO Y CARGA FAMILIAR
# =============================================================================
def plot_demografia_y_hacinamiento(df_resp, incluir_media=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    
    labels = {0: 'No Pobre', 1: 'Pobre'}
    df_plot = df_resp.copy()
    df_plot['Condición'] = df_plot['pobre_oficial'].map(labels)
    paleta = {'No Pobre': '#27ae60', 'Pobre': '#e74c3c'}
    
    mean_props = dict(marker='D', markeredgecolor='black', markerfacecolor='gold', markersize=6)
    
    # 1. Tamaño del hogar
    sns.boxplot(
        data=df_plot, x='Condición', y='cant_miembros',
        hue='Condición', palette=paleta, legend=False,
        ax=axes[0], showmeans=incluir_media, meanprops=mean_props if incluir_media else None
    )
    axes[0].set_title('Cantidad de Miembros en el Hogar')
    axes[0].set_ylabel('Miembros')
    axes[0].set_ylim(0, 14)
    
    # 2. Ratio de dependencia demográfica
    sns.boxplot(
        data=df_plot, x='Condición', y='ratio_dependencia_demog',
        hue='Condición', palette=paleta, legend=False,
        ax=axes[1], showmeans=incluir_media, meanprops=mean_props if incluir_media else None
    )
    axes[1].set_title('Ratio de Dependencia Demográfica\n(Menores + Mayores) / Activos')
    axes[1].set_ylabel('Ratio')
    axes[1].set_ylim(0, 4)
    
    # 3. Personas por cuarto (Hacinamiento)
    sns.boxplot(
        data=df_plot, x='Condición', y='personas_por_cuarto',
        hue='Condición', palette=paleta, legend=False,
        ax=axes[2], showmeans=incluir_media, meanprops=mean_props if incluir_media else None
    )
    axes[2].axhline(3.0, color='black', linestyle=':', label='Umbral Crítico (>3)')
    axes[2].set_title('Personas por Habitación de Dormir')
    axes[2].set_ylabel('Ratio de Hacinamiento')
    axes[2].set_ylim(0, 6)
    axes[2].legend(loc='upper right')
    
    fig.suptitle('Contraste Demográfico y de Hacinamiento según Condición de Pobreza', fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_FIGS / "eda_02_demografia_hacinamiento.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# FIGURA 3: CAPITAL HUMANO, MERCADO LABORAL Y SALUD
# =============================================================================
def plot_educacion_informalidad_salud(df_resp):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    
    df_local = df_resp.copy()
    
    # Mapeo descriptivo del nivel educativo
    educ_map = {
        1: 'Primaria Inc.', 2: 'Primaria Comp.', 3: 'Secundaria Inc.',
        4: 'Secundaria Comp.', 5: 'Superior Inc.', 6: 'Superior Comp.', 7: 'Sin Inst.'
    }
    
    # 1. Gradiente Educativo (Máximo nivel de adultos en el hogar)
    df_local['educ_desc'] = df_local['max_educ_hogar'].map(educ_map)
    orden_educ = ['Sin Inst.', 'Primaria Inc.', 'Primaria Comp.', 'Secundaria Inc.', 'Secundaria Comp.', 'Superior Inc.', 'Superior Comp.']
    
    pob_por_educ = df_local.groupby('educ_desc', observed=False).apply(
        lambda s: np.average(s['pobre_oficial'], weights=s['pondih']) * 100
    ).reindex(orden_educ)
    
    pob_por_educ.plot(kind='bar', color='#34495e', ax=axes[0])
    axes[0].set_title('Tasa de Pobreza por Máx. Nivel\nEducativo de Adultos en el Hogar')
    axes[0].set_ylabel('Pobreza (%)')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=35)
    plt.setp(axes[0].get_xticklabels(), ha='right')
    
    # 2. Tasa de Informalidad en el Hogar (por tramos)
    df_local['tramo_informalidad'] = pd.cut(
        df_local['tasa_informalidad_hogar'],
        bins=[-0.01, 0.0, 0.33, 0.66, 1.0],
        labels=['Sin Informales (0%)', 'Baja (1-33%)', 'Media (34-66%)', 'Alta (67-100%)']
    )
    pob_por_inf = df_local.groupby('tramo_informalidad', observed=False).apply(
        lambda s: np.average(s['pobre_oficial'], weights=s['pondih']) * 100
    )
    pob_por_inf.plot(kind='bar', color='#d35400', ax=axes[1])
    axes[1].set_title('Tasa de Pobreza según Nivel de\nInformalidad Laboral del Hogar')
    axes[1].set_ylabel('Pobreza (%)')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=30)
    plt.setp(axes[1].get_xticklabels(), ha='right')
    
    # 3. Dependencia de Salud Pública Exclusiva
    df_local['tramo_salud'] = pd.cut(
        df_local['prop_salud_publica'],
        bins=[-0.01, 0.0, 0.5, 0.99, 1.0],
        labels=['Cobertura Privada', 'Mayoría Obra Social', 'Mayoría Pública', '100% Salud Pública']
    )
    pob_por_salud = df_local.groupby('tramo_salud', observed=False).apply(
        lambda s: np.average(s['pobre_oficial'], weights=s['pondih']) * 100
    )
    pob_por_salud.plot(kind='bar', color='#16a085', ax=axes[2])
    axes[2].set_title('Tasa de Pobreza según Proporción con\nCobertura de Salud Pública Exclusiva')
    axes[2].set_ylabel('Pobreza (%)')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=30)
    plt.setp(axes[2].get_xticklabels(), ha='right')
    
    fig.suptitle('Gradientes de Vulnerabilidad Estructural: Capital Humano, Empleo y Salud', fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_FIGS / "eda_03_gradientes_vulnerabilidad.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# FIGURA 4: MATRIZ DE CORRELACIÓN ENTRE VARIABLES CONTINUAS CLAVE
# =============================================================================
def plot_matriz_correlacion(df_resp):
    mapa_nombres = {
        'pobre_oficial': 'Pobre',
        'cant_miembros': 'Miembros',
        'adequi_hogar': 'Adequi',
        'cant_menores_14': 'Menores 14',
        'ratio_dependencia_demog': 'Dep. Demog.',
        'ratio_dependencia_econom': 'Dep. Econ.',
        'personas_por_cuarto': 'Pers./Cuarto',
        'tasa_informalidad_hogar': 'Informalidad',
        'max_educ_hogar': 'Máx. Educ.',
        'prop_salud_publica': 'Salud Pública'
    }
    
    cols_existentes = [c for c in mapa_nombres.keys() if c in df_resp.columns]
    nombres_legibles = [mapa_nombres[c] for c in cols_existentes]
    
    corr_matrix = df_resp[cols_existentes].corr(method='spearman')
    
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
        vmin=-0.6, vmax=0.8, cbar_kws={'shrink': 0.8},
        xticklabels=nombres_legibles, yticklabels=nombres_legibles
    )
    plt.title('Matriz de Correlación de Spearman entre Predictores Estructurales y Pobreza', pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "eda_04_matriz_correlacion.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    plot_dinamica_temporal_y_no_respuesta(df)
    # Si deseas volver a mostrar las medias en los boxplots, cambia a: incluir_media=True
    plot_demografia_y_hacinamiento(df_respondentes, incluir_media=False)
    plot_educacion_informalidad_salud(df_respondentes)
    plot_matriz_correlacion(df_respondentes)
    print("✓ Gráficos del EDA generados y guardados con éxito en:", OUTPUT_FIGS.resolve())