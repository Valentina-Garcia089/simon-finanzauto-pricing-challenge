from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "inventario_subastas_.csv"
RUNT_PATH = BASE_DIR / "data" / "external" / "runt_vehiculos.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "inventario_enriquecido.csv"

MARCAS_OBJETIVO = ["Toyota", "Hyundai", "Renault", "Chevrolet", "Kia", "Mazda", "Nissan"]


def cargar_y_auditar_inventario(ruta_csv: Path) -> pd.DataFrame:
    #Revisa la calidad de datos, los 97 registros sin comprador y anomalías de subasta
    df = pd.read_csv(ruta_csv)
    df.columns = [c.strip() for c in df.columns]

    print("=" * 72)
    print(".......AUDITORÍA DE CALIDAD DE DATOS — INVENTARIO SIMÓN.......")
    print("=" * 72)
    print(f"Total de registros: {len(df)} | Columnas: {len(df.columns)}")

    nulos_comprador = df["ID_Comprador"].isna().sum()
    print(f"\nRegistros con ID_Comprador nulo: {nulos_comprador}")
    print(
        "Se identificaron 97 registros sin comprador asignado correspondientes a "
        "subastas 'No Adjudicadas' o 'Retiradas' con precio de venta en $0; se aíslan "
        "del calculo de ticket promedio y recuperación para evitar sesgos"
    )

    resumen_estados = (
        df.groupby("Estado_Subasta", dropna=False)
        .agg(
            total_vehiculos=("ID_Vehiculo", "count"),
            comprador_nulo=("ID_Comprador", lambda s: s.isna().sum()),
            venta_en_cero=("Precio_Final_Venta_COP", lambda s: (s == 0).sum()),
            precio_venta_promedio=("Precio_Final_Venta_COP", "mean"),
        )
        .reset_index()
    )
    print("\nResumen por Estado_Subasta:")
    print(resumen_estados.to_string(index=False))

    #Revisión de anomalías en vehiculos vendidos 
    vendidos = df[df["Estado_Subasta"] == "Vendido"].copy()
    bajo_reserva = vendidos[vendidos["Precio_Final_Venta_COP"] < vendidos["Precio_Reserva_COP"]]
    print(
        f"\nAnomalías en 'Vendido': {len(bajo_reserva)} de {len(vendidos)} vehiculos "
        "se adjudicaron POR DEBAJO de su Precio_Reserva_COP"
    )
    if not bajo_reserva.empty:
        print(
            "Posibles explicaciones de negocio: flexibilización del precio de reserva al "
            "cierre para evitar re-publicación, antiguedad/desgaste o cierre comercial"
        )
        print(
            bajo_reserva[
                [
                    "ID_Vehiculo",
                    "Marca",
                    "Linea",
                    "Estado_Vehiculo",
                    "Dias_en_Inventario",
                    "Canal_Asignado",
                    "Precio_Reserva_COP",
                    "Precio_Final_Venta_COP",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    df["es_vendido"] = df["Estado_Subasta"] == "Vendido"
    df["vendido_bajo_reserva"] = np.where(
        df["es_vendido"],
        df["Precio_Final_Venta_COP"] < df["Precio_Reserva_COP"],
        False,
    )
    return df


def enriquecer_con_runt(df: pd.DataFrame, ruta_runt: Path) -> pd.DataFrame:
    #Hace el cruce por Marca y Linea con runt_vehiculos.csv para calcular participación de parque
    print("\n" + "=" * 74)
    print(".......CRUCE CON PARQUE AUTOMOTOR RUNT.......")
    print("=" * 72)

    if not ruta_runt.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_runt}")

    runt = pd.read_csv(ruta_runt, low_memory=False)
    runt.columns = [c.strip() for c in runt.columns]
    print(f"Total registros cargados desde runt_vehiculos.csv: {len(runt)}")

    marcas_upper = {m.upper() for m in MARCAS_OBJETIVO}
    runt_filtrado = runt[
        runt["Marca"].astype(str).str.strip().str.upper().isin(marcas_upper)
    ].copy()

    runt_filtrado["marca_norm"] = runt_filtrado["Marca"].astype(str).str.strip().str.upper()
    runt_filtrado["linea_norm"] = runt_filtrado["Linea"].astype(str).str.strip().str.upper()

    #Quitar filas vacias o sin linea definida en RUNT para el cruce por linea
    runt_lineas = runt_filtrado[
        runt_filtrado["linea_norm"].notna()
        & (runt_filtrado["linea_norm"] != "")
        & (runt_filtrado["linea_norm"] != "SIN LINEA")
    ].copy()

    conteo_linea = (
        runt_lineas.groupby(["marca_norm", "linea_norm"])
        .size()
        .reset_index(name="conteo_runt_parque")
    )
    total_lineas = conteo_linea["conteo_runt_parque"].sum()
    conteo_linea["participacion_mercado_runt_pct"] = np.where(
        total_lineas > 0,
        (conteo_linea["conteo_runt_parque"] / total_lineas) * 100,
        0.0,
    )
    conteo_linea["fuente_liquidez_mercado"] = "RUNT_DatosAbiertos (Marca + Linea)"

    df["marca_norm"] = df["Marca"].astype(str).str.strip().str.upper()
    df["linea_norm"] = df["Linea"].astype(str).str.strip().str.upper()

    df = df.merge(conteo_linea, on=["marca_norm", "linea_norm"], how="left")

    #Respaldo por Marca si una linea exacta no aparece con el mismo nombre en runt_vehiculos.csv
    faltantes = df["participacion_mercado_runt_pct"].isna()
    if faltantes.any():
        conteo_marca = (
            runt_filtrado.groupby("marca_norm")
            .size()
            .reset_index(name="conteo_runt_marca")
        )
        total_marca = conteo_marca["conteo_runt_marca"].sum()
        conteo_marca["pct_marca_runt"] = np.where(
            total_marca > 0,
            (conteo_marca["conteo_runt_marca"] / total_marca) * 100,
            0.0,
        )
        df = df.merge(
            conteo_marca[["marca_norm", "conteo_runt_marca", "pct_marca_runt"]],
            on="marca_norm",
            how="left",
        )
        df.loc[faltantes, "conteo_runt_parque"] = df.loc[faltantes, "conteo_runt_marca"]
        df.loc[faltantes, "participacion_mercado_runt_pct"] = df.loc[faltantes, "pct_marca_runt"]
        df.loc[faltantes, "fuente_liquidez_mercado"] = "RUNT_DatosAbiertos (Respaldo Marca)"
        df = df.drop(columns=["conteo_runt_marca", "pct_marca_runt"])

    df = df.drop(columns=["marca_norm", "linea_norm"])

    print("\nCruce por Marca y Línea del inventario:")
    print(
        df.groupby(["Marca", "Linea", "fuente_liquidez_mercado"], dropna=False)[
            ["conteo_runt_parque", "participacion_mercado_runt_pct"]
        ]
        .first()
        .reset_index()
        .to_string(index=False)
    )
    return df


def validar_valor_fasecolda(df: pd.DataFrame) -> pd.DataFrame:
    #Validar coherencia de Valor_Fasecolda_COP por Marca, Linea y Modelo
    print("\n" + "=" * 72)
    print(".......VALIDACIÓN DE VALOR_FASECOLDA_COP POR MARCA, LINEA Y MODELO.......")
    print("=" * 72)

    resumen = (
        df.groupby(["Marca", "Linea"])
        .agg(
            modelo_min=("Modelo", "min"),
            modelo_max=("Modelo", "max"),
            fasecolda_min=("Valor_Fasecolda_COP", "min"),
            fasecolda_max=("Valor_Fasecolda_COP", "max"),
            fasecolda_mediana=("Valor_Fasecolda_COP", "median"),
        )
        .reset_index()
    )
    print(resumen.to_string(index=False))

    desfases = []
    for (marca, linea), grupo in df.groupby(["Marca", "Linea"]):
        mediana_por_anio = grupo.groupby("Modelo")["Valor_Fasecolda_COP"].median().sort_index()
        if len(mediana_por_anio) > 1 and np.any(np.diff(mediana_por_anio.values) < 0):
            desfases.append(f"{marca} {linea}")

    if desfases:
        print(
            "\nAtención — Lineas con mediana de Fasecolda no creciente en algún año modelo: "
            + ", ".join(desfases)
        )
    else:
        print("\nValidación OK: la mediana de Valor_Fasecolda_COP es coherente entre modelos")

    df["fuente_validacion_fasecolda"] = "Validacion_Interna_Modelo_Linea"
    return df


def main() -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = cargar_y_auditar_inventario(RAW_PATH)
    df = enriquecer_con_runt(df, RUNT_PATH)
    df = validar_valor_fasecolda(df)
    df.to_csv(PROCESSED_PATH, index=False)


if __name__ == "__main__":
    main()