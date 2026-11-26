from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Simón Movilidad & Finanzauto — Analítica y Pricing Dinámico",
    page_icon="🚗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "data" / "processed" / "inventario_con_metricas.csv"
ENRICHED_PATH = BASE_DIR / "data" / "processed" / "inventario_enriquecido.csv"
RAW_PATH = BASE_DIR / "data" / "raw" / "inventario_subastas_.csv"

TASA_HOLDING_MENSUAL = 0.015


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    for ruta in [METRICS_PATH, ENRICHED_PATH, RAW_PATH]:
        if ruta.exists():
            df = pd.read_csv(ruta)
            df.columns = [c.strip() for c in df.columns]
            break
    else:
        st.error("No se encontró el dataset de inventario.")
        st.stop()

    df["es_vendido"] = df["Estado_Subasta"] == "Vendido"
    df["holding_cost_cop"] = (
        df["Costo_Adquisicion_COP"] * TASA_HOLDING_MENSUAL * (df["Dias_en_Inventario"] / 30.0)
    )
    df["pct_recuperacion"] = np.where(
        df["es_vendido"] & (df["Costo_Adquisicion_COP"] > 0),
        (df["Precio_Final_Venta_COP"] / df["Costo_Adquisicion_COP"]) * 100.0,
        np.nan,
    )
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
    df["ratio_reserva_fasecolda"] = np.where(
        df["Valor_Fasecolda_COP"] > 0,
        df["Precio_Reserva_COP"] / df["Valor_Fasecolda_COP"],
        np.nan,
    )
    if "clasificacion_valuacion" not in df.columns:
        df["clasificacion_valuacion"] = np.where(
            df["ratio_reserva_fasecolda"] > 1.05,
            "Overpriced",
            np.where(df["ratio_reserva_fasecolda"] >= 0.90, "Fair Value", "Underpriced"),
        )
    if "bucket_aging" not in df.columns:
        df["bucket_aging"] = pd.cut(
            df["Dias_en_Inventario"],
            bins=[-1, 30, 60, 90, 10**9],
            labels=["<= 30 días", "31-60 días", "61-90 días", "> 90 días"],
        )
    return df


df_all = cargar_datos()

# Barra lateral con filtros globales
st.sidebar.title("Filtros de Inventario")
marcas_sel = st.sidebar.multiselect(
    "Marca",
    options=sorted(df_all["Marca"].dropna().unique()),
    default=sorted(df_all["Marca"].dropna().unique()),
)
canales_sel = st.sidebar.multiselect(
    "Canal Asignado",
    options=sorted(df_all["Canal_Asignado"].dropna().unique()),
    default=sorted(df_all["Canal_Asignado"].dropna().unique()),
)
estados_sel = st.sidebar.multiselect(
    "Estado del Vehículo",
    options=sorted(df_all["Estado_Vehiculo"].dropna().unique()),
    default=sorted(df_all["Estado_Vehiculo"].dropna().unique()),
)

df = df_all[
    df_all["Marca"].isin(marcas_sel)
    & df_all["Canal_Asignado"].isin(canales_sel)
    & df_all["Estado_Vehiculo"].isin(estados_sel)
].copy()

if df.empty:
    st.warning("No hay registros con los filtros seleccionados. Ajusta los filtros en la barra lateral")
    st.stop()

st.title("Simón Movilidad & Finanzauto — Tablero de Subastas, Pricing y Rentabilidad")
st.caption(
    "Diagnóstico financiero, benchmarking Fasecolda/RUNT, concentración de compradores "
    "y simulador de pricing dinámico sobre el inventario de subastas"
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Vista CFO / Resumen Ejecutivo",
        "2. Benchmarking Fasecolda & Subastas",
        "3. Compradores Pareto",
        "4. Simulador de Pricing Dinámico",
    ]
)


with tab1:
    win_rate = df["es_vendido"].mean() * 100.0
    rec_prom = df.loc[df["es_vendido"], "pct_recuperacion"].mean()
    cap_inmov_gt60_m = df.loc[df["Dias_en_Inventario"] > 60, "Costo_Adquisicion_COP"].sum() / 1e6
    holding_total_m = df["holding_cost_cop"].sum() / 1e6

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate", f"{win_rate:.1f}%")
    c2.metric("% Recuperación (Vendidos)", f"{rec_prom:.1f}%" if pd.notna(rec_prom) else "N/A")
    c3.metric("Capital Inmovilizado > 60 días", f"${cap_inmov_gt60_m:,.1f}M")
    c4.metric("Holding Cost Acumulado (1.5%/mes)", f"${holding_total_m:,.1f}M")

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        aging_df = (
            df.groupby("bucket_aging", observed=False, as_index=False)
            .agg(
                vehiculos=("ID_Vehiculo", "count"),
                holding_cost_m=("holding_cost_cop", lambda s: s.sum() / 1e6),
                margen_neto_m=("margen_neto_ajustado_cop", lambda s: s.sum() / 1e6),
            )
        )
        fig_aging = go.Figure()
        fig_aging.add_bar(
            x=aging_df["bucket_aging"].astype(str),
            y=aging_df["holding_cost_m"],
            name="Holding Cost (M COP)",
        )
        fig_aging.add_bar(
            x=aging_df["bucket_aging"].astype(str),
            y=aging_df["margen_neto_m"],
            name="Margen Neto Ajustado Vendidos (M COP)",
        )
        fig_aging.update_layout(
            title="Aging vs. Holding Cost y Margen Neto Ajustado (Millones COP)",
            barmode="group",
            xaxis_title="Tramo de Antigüedad (Días)",
            yaxis_title="Millones COP",
        )
        st.plotly_chart(fig_aging, use_container_width=True)

    with col_g2:
        vend_aging = (
            df[df["es_vendido"]]
            .groupby("bucket_aging", observed=False, as_index=False)["margen_neto_ajustado_cop"]
            .mean()
        )
        vend_aging["margen_neto_prom_m"] = vend_aging["margen_neto_ajustado_cop"] / 1e6
        fig_mn = px.bar(
            vend_aging,
            x="bucket_aging",
            y="margen_neto_prom_m",
            title="Margen Neto Ajustado Promedio por Vehículo Vendido según Aging",
            labels={"bucket_aging": "Tramo de Antigüedad", "margen_neto_prom_m": "Promedio (Millones COP)"},
        )
        st.plotly_chart(fig_mn, use_container_width=True)








with tab2:
    st.subheader("Dispersión Precio de Reserva vs. Valor Fasecolda")
    color_by = st.radio(
        "Colorear puntos por:",
        options=["Estado_Subasta", "Canal_Asignado"],
        horizontal=True,
    )
    fig_scatter = px.scatter(
        df,
        x="Valor_Fasecolda_COP",
        y="Precio_Reserva_COP",
        color=color_by,
        hover_data=["ID_Vehiculo", "Marca", "Linea", "Modelo", "Dias_en_Inventario", "Estado_Vehiculo"],
        title=f"Precio de Reserva vs. Valor Fasecolda (Coloreado por {color_by})",
        labels={
            "Valor_Fasecolda_COP": "Valor Fasecolda (COP)",
            "Precio_Reserva_COP": "Precio de Reserva (COP)",
        },
    )
    max_val = max(df["Valor_Fasecolda_COP"].max(), df["Precio_Reserva_COP"].max())
    min_val = min(df["Valor_Fasecolda_COP"].min(), df["Precio_Reserva_COP"].min())
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Referencia 1:1 (Reserva = Fasecolda)",
            line=dict(dash="dash", color="gray"),
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Win Rate por Clasificación de Valuación frente a Fasecolda")
    wr_val = (
        df.groupby("clasificacion_valuacion", as_index=False)
        .agg(vehiculos=("ID_Vehiculo", "count"), vendidos=("es_vendido", "sum"))
    )
    wr_val["win_rate_pct"] = (wr_val["vendidos"] / wr_val["vehiculos"]) * 100.0
    fig_wr_val = px.bar(
        wr_val,
        x="clasificacion_valuacion",
        y="win_rate_pct",
        text=wr_val["win_rate_pct"].map(lambda x: f"{x:.1f}%"),
        title="Win Rate (%) según Clasificación de Valuación",
        labels={"clasificacion_valuacion": "Clasificación", "win_rate_pct": "Win Rate (%)"},
    )
    st.plotly_chart(fig_wr_val, use_container_width=True)








with tab3:
    vend_comp = df[df["es_vendido"] & df["ID_Comprador"].notna()].copy()
    if vend_comp.empty:
        st.info("No hay compras registradas con los filtros actuales.")
    else:
        por_comp = (
            vend_comp.groupby(["ID_Comprador", "Tipo_Comprador"], as_index=False)
            .agg(
                vehiculos=("ID_Vehiculo", "count"),
                margen_bruto_cop=("margen_bruto_cop", "sum"),
                ticket_promedio_cop=("Precio_Final_Venta_COP", "mean"),
            )
            .sort_values("margen_bruto_cop", ascending=False)
            .reset_index(drop=True)
        )
        n_comp = len(por_comp)
        top_n = max(1, int(np.ceil(0.20 * n_comp)))
        pct_vol_top20 = (por_comp.iloc[:top_n]["vehiculos"].sum() / por_comp["vehiculos"].sum()) * 100.0
        pct_mar_top20 = (
            por_comp.iloc[:top_n]["margen_bruto_cop"].sum() / por_comp["margen_bruto_cop"].sum()
        ) * 100.0

        counts_comp = por_comp.groupby("ID_Comprador")["vehiculos"].sum()
        tasa_recompra = ((counts_comp >= 2).mean()) * 100.0

        k1, k2, k3 = st.columns(3)
        k1.metric("Volumen del Top 20% Compradores", f"{pct_vol_top20:.1f}%")
        k2.metric("Margen Bruto del Top 20% Compradores", f"{pct_mar_top20:.1f}%")
        k3.metric("Tasa de Recompra (>= 2 vehículos)", f"{tasa_recompra:.1f}%")

        por_comp["rank_pct"] = ((por_comp.index + 1) / n_comp) * 100.0
        por_comp["pct_acum_margen"] = (
            por_comp["margen_bruto_cop"].cumsum() / por_comp["margen_bruto_cop"].sum()
        ) * 100.0

        fig_pareto = px.line(
            por_comp,
            x="rank_pct",
            y="pct_acum_margen",
            title="Curva de Pareto Acumulada de Margen Bruto por Comprador",
            labels={"rank_pct": "% Acumulado de Compradores", "pct_acum_margen": "% Acumulado de Margen Bruto"},
        )
        fig_pareto.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="Top 20%")
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.subheader("Rentabilidad y Ticket Promedio por Tipo de Comprador")
        arquetipos = (
            vend_comp.groupby("Tipo_Comprador", as_index=False)
            .agg(
                vehiculos_comprados=("ID_Vehiculo", "count"),
                ticket_promedio_m=("Precio_Final_Venta_COP", lambda s: s.mean() / 1e6),
                margen_bruto_total_m=("margen_bruto_cop", lambda s: s.sum() / 1e6),
                margen_neto_prom_m=("margen_neto_ajustado_cop", lambda s: s.mean() / 1e6),
            )
            .sort_values("margen_bruto_total_m", ascending=False)
        )
        st.dataframe(arquetipos, use_container_width=True)









with tab4:
    st.subheader("Simulador Interactivo de Descuentos por Antigüedad (Días en Inventario)")
    st.write(
        "Ajusta los descuentos escalonados por tramo de días para observar en vivo el impacto "
        "en la reserva sugerida, ahorro en Holding Cost (1.5%/mes) y capital esperado liberado sobre vehículos no vendidos."
    )

    s1, s2, s3 = st.columns(3)
    desc_31_45 = s1.slider("Descuento 31–45 días (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)
    desc_46_60 = s2.slider("Descuento 46–60 días (%)", min_value=0.0, max_value=30.0, value=8.5, step=0.5)
    desc_gt60 = s3.slider("Descuento > 60 días (%)", min_value=0.0, max_value=40.0, value=14.0, step=0.5)

    no_vend = df[~df["es_vendido"]].copy()
    if no_vend.empty:
        st.info("No hay vehículos no vendidos bajo los filtros actuales.")
    else:
        def aplicar_slider(row: pd.Series) -> pd.Series:
            dias = float(row["Dias_en_Inventario"])
            reserva = float(row["Precio_Reserva_COP"])
            if dias > 60:
                d = desc_gt60 / 100.0
                dias_ahorrados = 20.0
            elif dias > 45:
                d = desc_46_60 / 100.0
                dias_ahorrados = 15.0
            elif dias > 30:
                d = desc_31_45 / 100.0
                dias_ahorrados = 10.0
            else:
                d = 0.0
                dias_ahorrados = 5.0

            nueva_reserva = reserva * (1.0 - d)
            ahorro_hc = row["Costo_Adquisicion_COP"] * TASA_HOLDING_MENSUAL * (dias_ahorrados / 30.0)
            cap_lib = nueva_reserva * 0.65
            return pd.Series([nueva_reserva, ahorro_hc, cap_lib], index=["nueva_reserva", "ahorro_hc", "cap_lib"])

        sim_res = no_vend.apply(aplicar_slider, axis=1)
        no_vend = pd.concat([no_vend, sim_res], axis=1)

        m1, m2, m3 = st.columns(3)
        m1.metric("Vehículos No Vendidos Simulados", f"{len(no_vend)}")
        m2.metric("Ahorro Estimado Holding Cost", f"${no_vend['ahorro_hc'].sum() / 1e6:,.2f}M")
        m3.metric("Capital Esperado Liberado", f"${no_vend['cap_lib'].sum() / 1e6:,.2f}M")

        fig_sim = go.Figure()
        fig_sim.add_bar(
            x=no_vend["ID_Vehiculo"].head(25),
            y=no_vend["Precio_Reserva_COP"].head(25) / 1e6,
            name="Reserva Actual (M COP)",
        )
        fig_sim.add_bar(
            x=no_vend["ID_Vehiculo"].head(25),
            y=no_vend["nueva_reserva"].head(25) / 1e6,
            name="Reserva Simulada (M COP)",
        )
        fig_sim.update_layout(
            title="Comparación de Reserva Actual vs. Simulada",
            barmode="group",
            xaxis_title="ID Vehículo",
            yaxis_title="Millones COP",
        )
        st.plotly_chart(fig_sim, use_container_width=True)