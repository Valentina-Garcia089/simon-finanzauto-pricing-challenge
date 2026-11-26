from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_INPUT = BASE_DIR / "data" / "processed" / "inventario_con_metricas.csv"
ENRICHED_INPUT = BASE_DIR / "data" / "processed" / "inventario_enriquecido.csv"
RAW_INPUT = BASE_DIR / "data" / "raw" / "inventario_subastas_.csv"
SIMULATION_OUTPUT = BASE_DIR / "data" / "processed" / "simulacion_pricing_y_canales.csv"

TASA_HOLDING_MENSUAL = 0.015


def cargar_datos_para_simulacion() -> pd.DataFrame:
    for ruta in [METRICS_INPUT, ENRICHED_INPUT, RAW_INPUT]:
        if ruta.exists():
            df = pd.read_csv(ruta)
            df.columns = [c.strip() for c in df.columns]
            break
    else:
        raise FileNotFoundError("No se encontró ningún archivo base de inventario")

    df["es_vendido"] = df["Estado_Subasta"] == "Vendido"
    if "holding_cost_cop" not in df.columns:
        df["holding_cost_cop"] = (
            df["Costo_Adquisicion_COP"] * TASA_HOLDING_MENSUAL * (df["Dias_en_Inventario"] / 30.0)
        )
    if "participacion_mercado_runt_pct" not in df.columns:
        conteo_linea = df.groupby(["Marca", "Linea"])["ID_Vehiculo"].transform("count")
        df["participacion_mercado_runt_pct"] = (conteo_linea / len(df)) * 100.0

    return df


def calcular_precio_reserva_dinamico(row: pd.Series) -> dict:
    dias = float(row["Dias_en_Inventario"])
    estado = str(row["Estado_Vehiculo"]).strip()
    fasecolda = float(row["Valor_Fasecolda_COP"])
    costo = float(row["Costo_Adquisicion_COP"])
    reserva_actual = float(row["Precio_Reserva_COP"])
    holding_actual = float(row["holding_cost_cop"])
    liq_pct = float(row.get("participacion_mercado_runt_pct", 1.0))

    #Factor base según estado físico del vehículo
    factor_estado = {
        "Excelente": 1.00,
        "Bueno": 0.96,
        "Regular": 0.90,
        "Siniestro Parcial": 0.75,
    }.get(estado, 0.95)

    #Ajuste fino por demanda/liquidez en mercado
    ajuste_demanda = 0.015 if liq_pct >= 1.5 else (-0.015 if liq_pct < 0.4 else 0.0)

    if dias <= 30:
        tramo = "0-30 días (Alineación Mercado)"
        reserva_base = fasecolda * np.clip(factor_estado + ajuste_demanda, 0.70, 1.02)
        descuento_pct = 0.0
    elif dias <= 45:
        tramo = "31-45 días (Ajuste Leve -4%)"
        descuento_pct = 0.04
        ancla = reserva_actual if reserva_actual > 0 else fasecolda * factor_estado
        reserva_base = ancla * (1.0 - descuento_pct)
    elif dias <= 60:
        tramo = "46-60 días (Escalonado -8.5%)"
        descuento_pct = 0.085
        ancla = reserva_actual if reserva_actual > 0 else fasecolda * factor_estado
        reserva_base = ancla * (1.0 - descuento_pct)
    else:
        tramo = "> 60 días (Liquidación Acelerada)"
        #Reducción progresiva adicional por exceso de días sobre 60
        exceso_meses = max(0.0, (dias - 60.0) / 30.0)
        descuento_pct = min(0.25, 0.12 + 0.04 * exceso_meses)
        ancla = reserva_actual if reserva_actual > 0 else fasecolda * factor_estado
        reserva_base = ancla * (1.0 - descuento_pct)

    #Piso financiero: protege costo de adquisición + holding cost acumulado
    piso_financiero = costo + holding_actual
    techo_mercado = fasecolda * 0.98 if dias > 45 else fasecolda * 1.03

    if estado == "Siniestro Parcial" or dias > 60:
        precio_sugerido = np.clip(reserva_base, min(piso_financiero, costo * 0.92), techo_mercado)
    else:
        precio_sugerido = np.clip(reserva_base, min(piso_financiero, techo_mercado), techo_mercado)

    return {
        "tramo_pricing_dinamico": tramo,
        "descuento_aplicado_pct": round(descuento_pct * 100, 2),
        "piso_financiero_cop": round(float(piso_financiero), 2),
        "precio_reserva_sugerido_cop": round(float(precio_sugerido), 2),
    }


def asignar_canal_optimo(row: pd.Series) -> str:
    dias = float(row["Dias_en_Inventario"])
    estado = str(row["Estado_Vehiculo"]).strip()
    linea = str(row["Linea"]).strip()
    canal_actual = str(row["Canal_Asignado"]).strip()

    #Siniestro parcial, canal especializado de reventa / dealers con descuento
    if estado == "Siniestro Parcial":
        return "Subasta Virtual (Nicho Dealers / Descuento)"

    #Alta antiguedad (>45 días) en líneas de vocación flotillera, Venta Directa Flotas
    lineas_flotilleras = {"Hilux", "Duster", "Tracker", "Versa"}
    if dias > 45 and linea in lineas_flotilleras:
        return "Venta Directa Flotas"

    #Antiguedad > 45 días en gama media/compacta en buen estado, Concesionario Aliado
    lineas_retail = {"Onix", "Picanto", "Kwid", "Mazda 3", "CX-5", "Tucson", "Sportage", "Corolla"}
    if dias > 45 and estado in {"Excelente", "Bueno"} and linea in lineas_retail:
        return "Concesionario Aliado"

    #Antiguedad > 60 días en estado Regular, liquidación acelerada por canal aliado/directo
    if dias > 60 and estado == "Regular":
        return "Concesionario Aliado"

    return canal_actual


def simular_impacto_financiero(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 75)
    print("MÓDULO D: PRICING DINÁMICO, REASIGNACIÓN DE CANALES Y BACKTESTING")
    print("=" * 75)

    no_vendidos = df[~df["es_vendido"]].copy()

    #Aplicar función de Pricing Dinámico
    pricing_df = no_vendidos.apply(calcular_precio_reserva_dinamico, axis=1).apply(pd.Series)
    no_vendidos = pd.concat([no_vendidos, pricing_df], axis=1)

    #Aplicar árbol de decisión de reasignación de canal
    no_vendidos["canal_sugerido"] = no_vendidos.apply(asignar_canal_optimo, axis=1)
    no_vendidos["reasignacion_canal"] = (
        no_vendidos["Canal_Asignado"] != no_vendidos["canal_sugerido"]
    )

    #Estimación conservadora de aceleración de salida:
    #Si > 60 días y con ajuste/reasignación: se estima salida en 15 días adicionales en vez de seguir acumulando meses
    #Si 31-60 días: ahorro estimado de ~5 días de inmovilización
    # -Si <= 30 días: ahorro estimado de 7 días por mejor alineación de reserva desde el inicio
    def estimar_dias_ahorrados(r: pd.Series) -> float:
        dias = float(r["Dias_en_Inventario"])
        if dias > 60:
            return max(15.0, (dias - 60.0) * 0.65 + 15.0)
        if dias > 30:
            return 15.0
        return 7.0

    no_vendidos["dias_holding_ahorrados"] = no_vendidos.apply(estimar_dias_ahorrados, axis=1)
    no_vendidos["ahorro_holding_cost_cop"] = (
        no_vendidos["Costo_Adquisicion_COP"]
        * TASA_HOLDING_MENSUAL
        * (no_vendidos["dias_holding_ahorrados"] / 30.0)
    )

    #Probabilidad esperada de colocación tras ajuste de precio y canal
    prob_colocacion = np.where(
        no_vendidos["reasignacion_canal"],
        0.75,
        np.where(no_vendidos["descuento_aplicado_pct"] > 0, 0.65, 0.55),
    )
    no_vendidos["prob_colocacion_estimada"] = prob_colocacion
    no_vendidos["capital_liberado_esperado_cop"] = (
        no_vendidos["precio_reserva_sugerido_cop"] * no_vendidos["prob_colocacion_estimada"]
    )

    # Totales de impacto
    total_no_vendidos = len(no_vendidos)
    reasignados = int(no_vendidos["reasignacion_canal"].sum())
    holding_actual_total = no_vendidos["holding_cost_cop"].sum()
    ahorro_holding_total = no_vendidos["ahorro_holding_cost_cop"].sum()
    capital_liberado_total = no_vendidos["capital_liberado_esperado_cop"].sum()

    print(f"Vehículos no vendidos analizados: {total_no_vendidos}")
    print(f"Vehículos con reasignación de canal sugerida: {reasignados}")
    print(
        f"Holding Cost actual acumulado en no vendidos: "
        f"${holding_actual_total:,.0f} COP (${holding_actual_total / 1e6:,.2f} millones)"
    )
    print(
        f"Ahorro estimado en Holding Cost: "
        f"${ahorro_holding_total:,.0f} COP (${ahorro_holding_total / 1e6:,.2f} millones)"
    )
    print(
        f"Capital esperado liberado por colocación estimada: "
        f"${capital_liberado_total:,.0f} COP (${capital_liberado_total / 1e6:,.2f} millones)"
    )

    print("\n......Resumen por Tramo de Pricing Dinámico.......")
    resumen_tramos = (
        no_vendidos.groupby("tramo_pricing_dinamico", as_index=False)
        .agg(
            vehiculos=("ID_Vehiculo", "count"),
            descuento_prom_pct=("descuento_aplicado_pct", "mean"),
            reserva_actual_prom=("Precio_Reserva_COP", "mean"),
            reserva_sugerida_prom=("precio_reserva_sugerido_cop", "mean"),
            ahorro_holding_millones=("ahorro_holding_cost_cop", lambda s: s.sum() / 1e6),
            capital_liberado_millones=(
                "capital_liberado_esperado_cop",
                lambda s: s.sum() / 1e6,
            ),
        )
    )
    print(resumen_tramos.to_string(index=False))

    print("\n.......Reasignación de Canales.......")
    matriz_canales = (
        no_vendidos.groupby(["Canal_Asignado", "canal_sugerido"], as_index=False)
        .agg(
            vehiculos=("ID_Vehiculo", "count"),
            capital_liberado_millones=(
                "capital_liberado_esperado_cop",
                lambda s: s.sum() / 1e6,
            ),
        )
    )
    print(matriz_canales.to_string(index=False))

    return no_vendidos


def main() -> None:
    df = cargar_datos_para_simulacion()
    sim_df = simular_impacto_financiero(df)
    SIMULATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(SIMULATION_OUTPUT, index=False)


if __name__ == "__main__":
    main()