"""
===============================================================================
AUDITORÍA DE SOLAPAMIENTO: PORCENTAJES DE FUGA Y PURGA (ESQUEMA 8 TRIMESTRES)
===============================================================================
"""
from pathlib import Path
import pandas as pd

PARQUET_FILE = Path("../data/eph_hogares_consolidada_2016_2026.parquet")
if not PARQUET_FILE.exists():
    PARQUET_FILE = Path("data/eph_hogares_consolidada_2016_2026.parquet")


def auditar_porcentajes_solapamiento(
    parquet_path: Path = PARQUET_FILE,
    max_ano4: int = 2021,
    train_quarters: int = 8,
) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {parquet_path.resolve()}")

    df = pd.read_parquet(parquet_path, columns=["ano4", "trimestre", "codusu"])
    df_dev = df[df["ano4"] <= max_ano4].reset_index(drop=True) if max_ano4 else df.reset_index(drop=True)

    periodos = (
        df_dev[["ano4", "trimestre"]]
        .drop_duplicates()
        .sort_values(by=["ano4", "trimestre"])
        .values.tolist()
    )

    reporte = []

    for i in range(train_quarters, len(periodos)):
        val_ano, val_trim = periodos[i]
        train_window = periodos[i - train_quarters : i]

        # Máscaras temporales
        train_mask = pd.Series(False, index=df_dev.index)
        for a, t in train_window:
            train_mask |= (df_dev["ano4"] == a) & (df_dev["trimestre"] == t)
        val_mask = (df_dev["ano4"] == val_ano) & (df_dev["trimestre"] == val_trim)

        # Hogares únicos (codusu) en cada partición
        hogares_val = set(df_dev.loc[val_mask, "codusu"].unique())
        hogares_train = set(df_dev.loc[train_mask, "codusu"].unique())

        # 1. % de hogares de validación presentes en train (potencial leakage sin purga)
        solapados = hogares_val.intersection(hogares_train)
        pct_val_en_train = (len(solapados) / len(hogares_val) * 100) if hogares_val else 0.0

        # 2. % de filas de train descartadas al aplicar la purga
        n_train_pre = train_mask.sum()
        train_mask_purged = train_mask & (~df_dev["codusu"].isin(hogares_val))
        n_train_post = train_mask_purged.sum()
        pct_train_descartado = ((n_train_pre - n_train_post) / n_train_pre * 100) if n_train_pre > 0 else 0.0

        reporte.append({
            "fold": len(reporte) + 1,
            "train_ventana": f"{train_window[0][0]}T{train_window[0][1]} a {train_window[-1][0]}T{train_window[-1][1]}",
            "val_trimestre": f"{val_ano}T{val_trim}",
            "%_val_en_train": round(pct_val_en_train, 2),
            "%_train_descartado": round(pct_train_descartado, 2),
        })

    return pd.DataFrame(reporte)


if __name__ == "__main__":
    df_pct = auditar_porcentajes_solapamiento()
    print("\n" + df_pct.to_string(index=False))