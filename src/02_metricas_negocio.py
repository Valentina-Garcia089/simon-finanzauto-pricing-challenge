from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_INPUT = BASE_DIR / "data" / "processed" / "inventario_enriquecido.csv"
RAW_FALLBACK = BASE_DIR / "data" / "raw" / "inventario_subastas_.csv"
METRICS_OUTPUT = BASE_DIR / "data" / "processed" / "inventario_con_metricas.csv"


def cargar_dataset_base() -> pd.DataFrame:
    #Dataset enriquecido del Paso 2
    ruta = PROCESSED_INPUT if PROCESSED_INPUT.exists() else RAW_FALLBACK
    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]
    df["es_vendido"] = df["Estado_Subasta"] == "Vendido"
    return df


def calcular_modulo_a_subastas(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("MÓDULO A: ANALÍTICA DE SUBASTAS & RECUPERACIÓN DE CAPITAL")
    print("=" * 75)

    win_rate_global = df["es_vendido"].mean() * 100
    print(
        f"Win Rate Global: {win_rate_global:.2f}% "
        f"({df['es_vendido'].sum()} vendidos de {len(df)} publicados)"
    )

    #Win Rate por segmentos clave
    for col in ["Marca", "Linea", "Canal_Asignado", "Estado_Vehiculo"]:
        resumen_wr = (
            df.groupby(col, as_index=False)
            .agg(
                total_publicados=("ID_Vehiculo", "count"),
                vehiculos_vendidos=("es_vendido", "sum"),
            )
        )
        resumen_wr["win_rate_pct"] = (
            resumen_wr["vehiculos_vendidos"] / resumen_wr["total_publicados"]
        ) * 100
        resumen_wr = resumen_wr.sort_values("win_rate_pct", ascending=False)
        print(f"\n--- Win Rate por {col} ---")
        print(resumen_wr.to_string(index=False))

    #% Recuperación de Activo y Margen Bruto
    df["pct_recuperacion_activo"] = np.where(
        df["es_vendido"] & (df["Costo_Adquisicion_COP"] > 0),
        df["Precio_Final_Venta_COP"] / df["Costo_Adquisicion_COP"],
        np.nan,
    )
    df["margen_bruto_cop"] = np.where(
        df["es_vendido"],
        df["Precio_Final_Venta_COP"] - df["Costo_Adquisicion_COP"],
        np.nan,
    )

    #Spread de Puja Precio_Reserva_COP > 0
    df["spread_puja"] = np.where(
        df["es_vendido"] & (df["Precio_Reserva_COP"] > 0),
        (df["Precio_Final_Venta_COP"] - df["Precio_Reserva_COP"]) / df["Precio_Reserva_COP"],
        np.nan,
    )

    #Ratio Precio de Reserva vs Valor Fasecolda para todos los vehiculos
    df["ratio_reserva_fasecolda"] = np.where(
        df["Valor_Fasecolda_COP"] > 0,
        df["Precio_Reserva_COP"] / df["Valor_Fasecolda_COP"],
        np.nan,
    )

    vendidos = df[df["es_vendido"]].copy()
    print("\n.......Resumen en vehículos VENDIDOS......")
    print(
        vendidos[["pct_recuperacion_activo", "spread_puja", "margen_bruto_cop"]]
        .describe()
        .to_string()
    )

    print("\n.......Ratio Precio_Reserva / Valor_Fasecolda por Estado_Subasta......")
    print(
        df.groupby("Estado_Subasta", as_index=False)["ratio_reserva_fasecolda"]
        .agg(conteo="count", promedio="mean", mediana="median")
        .to_string(index=False)
    )

    #Correlaciones y regresión lineal simple para explicar que impacta el margen bruto en Vendidos
    vars_corr = [
        "Kilometraje",
        "Dias_en_Inventario",
        "Costo_Adquisicion_COP",
        "Valor_Fasecolda_COP",
        "Precio_Reserva_COP",
        "Precio_Final_Venta_COP",
        "pct_recuperacion_activo",
        "margen_bruto_cop",
    ]
    corr_margen = vendidos[vars_corr].corr()["margen_bruto_cop"].sort_values()
    print("\n.......Correlaciones con margen_bruto_cop.........")
    print(corr_margen.to_string())

    #Regresión lineal simple: variables operativas
    X_reg = pd.DataFrame(index=vendidos.index)
    X_reg["Kilometraje"] = vendidos["Kilometraje"]
    X_reg["Dias_en_Inventario"] = vendidos["Dias_en_Inventario"]
    X_reg["es_siniestro_parcial"] = (
        vendidos["Estado_Vehiculo"].astype(str).str.strip() == "Siniestro Parcial"
    ).astype(int)

    modelo_reg = LinearRegression()
    modelo_reg.fit(X_reg, vendidos["margen_bruto_cop"])
    r2_score = modelo_reg.score(X_reg, vendidos["margen_bruto_cop"])

    print(
        f"\n........Regresión Lineal Simple sobre margen_bruto_cop (R² = {r2_score:.4f})......."
    )
    print(f"Intercepto: {modelo_reg.intercept_:,.2f} COP")
    for nombre_var, coef in zip(X_reg.columns, modelo_reg.coef_):
        print(f"  Efecto de {nombre_var}: {coef:,.2f} COP por unidad")

    return df



def clasificar_valuacion_ratio(ratio: float) -> str:
    if pd.isna(ratio):
        return "Sin Dato"
    if ratio > 1.05:
        return "Overpriced"
    if ratio >= 0.90:
        return "Fair Value"
    return "Underpriced"


def calcular_modulo_b_benchmarking_finanzas(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("MÓDULO B: BENCHMARKING VS. MERCADO & COSTO FINANCIERO")
    print("=" * 75)

    #Clasificación de Valuación (Reserva vs Fasecolda y Costo vs Fasecolda
    df["clasificacion_valuacion"] = df["ratio_reserva_fasecolda"].apply(
        clasificar_valuacion_ratio
    )
    df["ratio_costo_fasecolda"] = np.where(
        df["Valor_Fasecolda_COP"] > 0,
        df["Costo_Adquisicion_COP"] / df["Valor_Fasecolda_COP"],
        np.nan,
    )

    wr_valuacion = (
        df.groupby("clasificacion_valuacion", as_index=False)
        .agg(
            total_vehiculos=("ID_Vehiculo", "count"),
            vendidos=("es_vendido", "sum"),
            ratio_reserva_fasecolda_prom=("ratio_reserva_fasecolda", "mean"),
        )
    )
    wr_valuacion["win_rate_pct"] = (
        wr_valuacion["vendidos"] / wr_valuacion["total_vehiculos"]
    ) * 100
    print("\n--- Win Rate por Clasificación de Valuación (Reserva vs. Fasecolda) ---")
    print(wr_valuacion.to_string(index=False))

    #Buckets de Antigüedad
    bins_aging = [-1, 30, 60, 90, 10**9]
    labels_aging = ["<= 30 días", "31-60 días", "61-90 días", "> 90 días"]
    df["bucket_aging"] = pd.cut(
        df["Dias_en_Inventario"], bins=bins_aging, labels=labels_aging
    )

    #Holding Cost
    tasa_mensual = 0.015
    df["holding_cost_cop"] = (
        df["Costo_Adquisicion_COP"] * tasa_mensual * (df["Dias_en_Inventario"] / 30.0)
    )

    # Margen Neto Ajustado
    if "margen_bruto_cop" not in df.columns:
        df["margen_bruto_cop"] = np.where(
            df["es_vendido"],
            df["Precio_Final_Venta_COP"] - df["Costo_Adquisicion_COP"],
            np.nan,
        )

    df["margen_neto_ajustado_cop"] = np.where(
        df["es_vendido"],
        df["margen_bruto_cop"] - df["holding_cost_cop"],
        np.nan,
    )

    print("\n......Resumen de Aging y Holding Cost en todo el inventario con 350 vehículos.......")
    resumen_aging = (
        df.groupby("bucket_aging", observed=False, as_index=False)
        .agg(
            vehiculos=("ID_Vehiculo", "count"),
            vendidos=("es_vendido", "sum"),
            costo_adquisicion_total=("Costo_Adquisicion_COP", "sum"),
            holding_cost_total=("holding_cost_cop", "sum"),
            margen_bruto_vendidos=("margen_bruto_cop", "sum"),
            margen_neto_vendidos=("margen_neto_ajustado_cop", "sum"),
        )
    )
    resumen_aging["win_rate_pct"] = (
        resumen_aging["vendidos"] / resumen_aging["vehiculos"]
    ) * 100
    print(resumen_aging.to_string(index=False))
    print(
        f"\nHolding Cost acumulado total del inventario: {df['holding_cost_cop'].sum():,.0f} COP"
    )

    return df



def calcular_modulo_c_compradores(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("MÓDULO C: PERFILACIÓN & COMPORTAMIENTO DE COMPRADORES")
    print("=" * 75)

    vendidos_comp = df[df["es_vendido"] & df["ID_Comprador"].notna()].copy()

    por_comprador = (
        vendidos_comp.groupby(["ID_Comprador", "Tipo_Comprador"], as_index=False)
        .agg(
            vehiculos_comprados=("ID_Vehiculo", "count"),
            compra_total_cop=("Precio_Final_Venta_COP", "sum"),
            ticket_promedio_cop=("Precio_Final_Venta_COP", "mean"),
            margen_bruto_cop=("margen_bruto_cop", "sum"),
            margen_neto_ajustado_cop=("margen_neto_ajustado_cop", "sum"),
        )
        .sort_values("margen_bruto_cop", ascending=False)
        .reset_index(drop=True)
    )

    total_compradores = len(por_comprador)
    top_n = max(1, int(np.ceil(0.20 * total_compradores)))
    por_comprador["es_top_20_pct"] = por_comprador.index < top_n

    por_comprador["pct_acumulado_margen"] = (
        por_comprador["margen_bruto_cop"].cumsum()
        / por_comprador["margen_bruto_cop"].sum()
        * 100
    )
    por_comprador["pct_acumulado_volumen"] = (
        por_comprador["vehiculos_comprados"].cumsum()
        / por_comprador["vehiculos_comprados"].sum()
        * 100
    )

    top_20_df = por_comprador[por_comprador["es_top_20_pct"]]
    pct_vol_top20 = (
        top_20_df["vehiculos_comprados"].sum()
        / por_comprador["vehiculos_comprados"].sum()
        * 100
    )
    pct_margen_top20 = (
        top_20_df["margen_bruto_cop"].sum()
        / por_comprador["margen_bruto_cop"].sum()
        * 100
    )

    tasa_recompra = (por_comprador["vehiculos_comprados"] >= 2).mean() * 100

    print(f"Total de compradores únicos (en ventas): {total_compradores}")
    print(f"Top 20% de compradores: {top_n} compradores")
    print(f"  - % de volumen de unidades generado por Top 20%: {pct_vol_top20:.2f}%")
    print(f"  - % de margen bruto generado por Top 20%: {pct_margen_top20:.2f}%")
    print(
        f"Tasa de recompra (>= 2 vehículos adquiridos): {tasa_recompra:.2f}% "
        f"({(por_comprador['vehiculos_comprados'] >= 2).sum()} de {total_compradores} compradores)"
    )

    print("\n.......Métricas por Arquetipo / Tipo_Comprador.......")
    por_tipo = (
        vendidos_comp.groupby("Tipo_Comprador", as_index=False)
        .agg(
            transacciones=("ID_Vehiculo", "count"),
            compradores_unicos=("ID_Comprador", "nunique"),
            ticket_promedio_cop=("Precio_Final_Venta_COP", "mean"),
            margen_bruto_total_cop=("margen_bruto_cop", "sum"),
            margen_bruto_prom_cop=("margen_bruto_cop", "mean"),
            margen_neto_prom_cop=("margen_neto_ajustado_cop", "mean"),
        )
        .sort_values("margen_bruto_total_cop", ascending=False)
    )
    print(por_tipo.to_string(index=False))

    return por_comprador


def main() -> None:
    df = cargar_dataset_base()
    df = calcular_modulo_a_subastas(df)
    df = calcular_modulo_b_benchmarking_finanzas(df)
    _ = calcular_modulo_c_compradores(df)

    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_OUTPUT, index=False)


if __name__ == "__main__":
    main()