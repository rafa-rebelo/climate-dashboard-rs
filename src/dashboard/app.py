"""
Agente 4 — Especialista em Visualização
Dashboard principal do Monitor Hidrometeorológico RS.
Multi-página: Visão Geral → Rios → Chuva → Previsões → Alertas → Sistema.
Auto-refresh 10 min, tema dark, mobile-friendly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — permite importar database.*
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC  = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página (deve ser o primeiro comando Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Monitor Hidrometeorológico RS",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Monitor Hidrometeorológico RS — Plataforma operacional de análise climática.",
    },
)

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# folium + streamlit-folium + plugins
try:
    import folium
    from folium.plugins import HeatMap as _FoliumHeatMap
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

# data_loader — GitHub Raw com fallback local
try:
    from dashboard.utils.data_loader import (
        DataSource, fetch_accumulated_rain, fetch_river_status,
        fetch_nwp_forecasts, fetch_progress,
        fetch_gpm_precip, fetch_inmet_historico, fetch_inmet_stations,
    )
    _LOADER_OK = True
except ImportError:
    _LOADER_OK = False

# streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_OK = True
except ImportError:
    _AUTOREFRESH_OK = False

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_DB_PATH = os.getenv("DB_PATH", str(_ROOT / "data" / "monitor_rs.duckdb"))
_REFRESH_MS = 10 * 60 * 1000  # 10 minutos

RIOS_COTAS: dict[str, dict[str, float]] = {
    "Sinos":   {"atencao": 4.5, "alerta": 5.5, "emergencia": 6.5, "lat": -29.76, "lon": -51.15},
    "Taquari": {"atencao": 5.0, "alerta": 6.5, "emergencia": 8.5, "lat": -29.67, "lon": -51.87},
    "Jacuí":   {"atencao": 7.0, "alerta": 8.5, "emergencia": 10.0,"lat": -29.78, "lon": -51.73},
    "Guaíba":  {"atencao": 3.0, "alerta": 3.6, "emergencia": 4.5, "lat": -30.03, "lon": -51.18},
}

# 10 pontos NWP do config.yaml — grade de previsão para o RS
NWP_POINTS: list[dict] = [
    {"nome": "Porto Alegre",  "lat": -30.03, "lon": -51.23},
    {"nome": "Caxias do Sul", "lat": -29.16, "lon": -51.18},
    {"nome": "Pelotas",       "lat": -31.77, "lon": -52.34},
    {"nome": "Santa Maria",   "lat": -29.68, "lon": -53.80},
    {"nome": "Passo Fundo",   "lat": -28.26, "lon": -52.41},
    {"nome": "Lajeado",       "lat": -29.47, "lon": -51.96},
    {"nome": "São Leopoldo",  "lat": -29.76, "lon": -51.15},
    {"nome": "Novo Hamburgo", "lat": -29.68, "lon": -51.13},
    {"nome": "Ijuí",          "lat": -28.39, "lon": -53.91},
    {"nome": "Bagé",          "lat": -31.33, "lon": -54.11},
]
# lookup rápido nome → coordenadas
_NWP_COORDS: dict[str, dict] = {p["nome"]: p for p in NWP_POINTS}

STATUS_COLORS = {
    "NORMAL":     "#22c55e",
    "ATENCAO":    "#f59e0b",
    "ALERTA":     "#ef4444",
    "EMERGENCIA": "#7c3aed",
}
STATUS_EMOJI = {
    "NORMAL":     "🟢",
    "ATENCAO":    "🟡",
    "ALERTA":     "🔴",
    "EMERGENCIA": "🟣",
}

# ---------------------------------------------------------------------------
# CSS dark theme
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
/* Fundo principal dark */
.stApp { background-color: #0f172a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background-color: #1e293b; }
.stMetric { background-color: #1e293b; border-radius: 8px; padding: 8px; }
div[data-testid="metric-container"] { background-color: #1e293b; border-radius: 8px; padding: 10px; }
.status-card { background-color: #1e293b; border-radius: 12px; padding: 16px; margin: 8px 0; }
.alert-badge { border-radius: 6px; padding: 2px 10px; font-weight: bold; font-size: 0.85em; }
/* Plotly backgrounds */
.js-plotly-plot .plotly .bg { fill: #0f172a !important; }
/* Mobile */
@media (max-width: 768px) {
    .block-container { padding: 0.5rem; }
    div[data-testid="metric-container"] { padding: 6px; }
}
/* Windy period selector — pills horizontais */
div[data-testid="stRadio"] div[role="radiogroup"] { gap: 6px; }
/* Windy legend / card lateral */
.windy-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 12px; margin: 6px 0; font-size: 0.83em; color: #cbd5e1;
}
</style>
"""

# ---------------------------------------------------------------------------
# Helpers de banco
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_conn() -> Optional[duckdb.DuckDBPyConnection]:
    """Retorna conexão DuckDB read-only ou None se banco não existe."""
    db = Path(_DB_PATH)
    if not db.exists():
        return None
    try:
        return duckdb.connect(str(db), read_only=True)
    except duckdb.IOException:
        # Banco aberto em outro processo — abre nova conexão
        return duckdb.connect(str(db))


def _query(sql: str, params: Optional[list] = None) -> pd.DataFrame:
    """Executa query SQL e retorna DataFrame. Retorna DataFrame vazio em caso de erro."""
    conn = _get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        if params:
            return conn.execute(sql, params).df()
        return conn.execute(sql).df()
    except duckdb.CatalogException:
        return pd.DataFrame()
    except Exception as exc:
        logger.warning(f"Query falhou: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Queries principais
# ---------------------------------------------------------------------------

def load_river_status() -> pd.DataFrame:
    """Carrega status atual de todos os rios com % da cota calculada dinamicamente."""
    df = _query("""
        SELECT river, segment, level_m, status, trend,
               municipalities_risk, updated_at AS ts
        FROM river_status
        ORDER BY river
    """)
    if df.empty:
        return df
    # Calcula pct_cota_atencao a partir das cotas fixas
    def _pct(row: pd.Series) -> float:
        cfg = RIOS_COTAS.get(row["river"])
        if cfg is None or not row["level_m"]:
            return 0.0
        return (row["level_m"] / cfg["atencao"]) * 100.0

    df["pct_cota_atencao"] = df.apply(_pct, axis=1)
    df["flow_cms"] = None  # coluna não existe na tabela, preenche como None
    return df


def load_river_levels(river: str, hours: int = 72) -> pd.DataFrame:
    """Carrega série temporal de níveis de um rio."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return _query("""
        SELECT rl.river, rl.station_id, rl.ts, rl.level_m, rl.flow_cms,
               s.lat, s.lon, s.name AS station_name
        FROM river_levels rl
        LEFT JOIN stations s ON s.station_id = rl.station_id
        WHERE rl.river = ?
          AND rl.ts >= ?
        ORDER BY rl.ts
    """, [river, cutoff])


def load_rain_accumulated() -> tuple[pd.DataFrame, "DataSource"]:
    """
    Carrega acumulados de chuva — GitHub Raw primeiro, DuckDB como fallback.

    Returns:
        Tupla (DataFrame, DataSource) com dados e metadados de origem.
    """
    if _LOADER_OK:
        df, src = fetch_accumulated_rain()
        if not df.empty:
            # Parquet do GitHub não inclui metadados de estação — enriquecer via DuckDB
            if "station_name" not in df.columns:
                stations = _query(
                    "SELECT station_id, lat, lon, name AS station_name,"
                    "       municipality, river FROM stations"
                )
                if not stations.empty:
                    df = df.merge(stations, on="station_id", how="left")
                else:
                    for _col in ("lat", "lon", "station_name", "municipality", "river"):
                        df[_col] = None
            return df, src

    # Fallback DuckDB
    df = _query("""
        SELECT ra.station_id, ra.date, ra.rain_1h, ra.rain_3h, ra.rain_6h,
               ra.rain_12h, ra.rain_24h, ra.rain_48h, ra.rain_72h,
               s.lat, s.lon, s.name AS station_name, s.municipality,
               s.river
        FROM rain_accumulated ra
        JOIN stations s ON s.station_id = ra.station_id
        WHERE (ra.station_id, ra.date) IN (
            SELECT station_id, MAX(date) FROM rain_accumulated GROUP BY station_id
        )
        ORDER BY ra.rain_24h DESC
    """)
    from datetime import datetime, timezone
    from dashboard.utils.data_loader import DataSource as _DS
    return df, _DS(source="local", url=_DB_PATH)


def load_forecasts(location: str = "Porto Alegre") -> pd.DataFrame:
    """
    Carrega previsões NWP para uma localidade — parquet GitHub → DuckDB.

    Args:
        location: Nome da localidade (ex: "Porto Alegre").

    Returns:
        DataFrame filtrado para a localidade, ordenado por valid_ts.
    """
    if _LOADER_OK:
        df_all, _ = fetch_nwp_forecasts()
        if not df_all.empty and "location_name" in df_all.columns:
            now = pd.Timestamp.now(tz="UTC")
            df = df_all[
                (df_all["location_name"] == location) &
                (df_all["valid_ts"] >= now)
            ].sort_values("valid_ts").head(168)
            if not df.empty:
                return df

    # Fallback DuckDB
    return _query("""
        SELECT location_name, valid_ts, rain_mm, temperature,
               wind_speed, cape_j_kg, lifted_index, k_index, model_source
        FROM forecasts
        WHERE location_name = ?
          AND valid_ts >= now()
        ORDER BY valid_ts
        LIMIT 168
    """, [location])


def load_all_forecasts_summary() -> pd.DataFrame:
    """
    Agrega previsões NWP para todos os 10 pontos da grade RS.

    Retorna um DataFrame com chuva acumulada 6h/24h/48h, temperatura média
    e CAPE máximo por localidade — usado para colorir o mapa de previsões.

    Returns:
        DataFrame com colunas: location_name, lat, lon,
        rain_6h, rain_24h, rain_48h, temp_mean, cape_max.
    """
    now_utc = datetime.now(timezone.utc)
    h6  = now_utc + timedelta(hours=6)
    h24 = now_utc + timedelta(hours=24)
    h48 = now_utc + timedelta(hours=48)

    # Tenta parquet do GitHub primeiro
    if _LOADER_OK:
        df_all, _ = fetch_nwp_forecasts()
        if not df_all.empty and "valid_ts" in df_all.columns:
            df_all["valid_ts"] = pd.to_datetime(df_all["valid_ts"], utc=True,
                                                 errors="coerce")
            fut = df_all[df_all["valid_ts"] >= pd.Timestamp(now_utc)]
            if not fut.empty:
                df = fut.groupby("location_name").apply(
                    lambda g: pd.Series({
                        "rain_6h":   g.loc[g["valid_ts"] <= pd.Timestamp(h6),  "rain_mm"].sum(),
                        "rain_24h":  g.loc[g["valid_ts"] <= pd.Timestamp(h24), "rain_mm"].sum(),
                        "rain_48h":  g.loc[g["valid_ts"] <= pd.Timestamp(h48), "rain_mm"].sum(),
                        "temp_mean": g.loc[g["valid_ts"] <= pd.Timestamp(h24), "temperature"].mean(),
                        "cape_max":  g.loc[g["valid_ts"] <= pd.Timestamp(h24), "cape_j_kg"].max()
                            if "cape_j_kg" in g.columns else None,
                    })
                ).reset_index()
                coords = pd.DataFrame(NWP_POINTS).rename(columns={"nome": "location_name"})
                return df.merge(coords, on="location_name", how="left")

    # Fallback DuckDB
    df = _query("""
        SELECT location_name,
               SUM(CASE WHEN valid_ts <= ? THEN COALESCE(rain_mm, 0) ELSE 0 END) AS rain_6h,
               SUM(CASE WHEN valid_ts <= ? THEN COALESCE(rain_mm, 0) ELSE 0 END) AS rain_24h,
               SUM(CASE WHEN valid_ts <= ? THEN COALESCE(rain_mm, 0) ELSE 0 END) AS rain_48h,
               AVG(CASE WHEN valid_ts <= ? THEN temperature END)                  AS temp_mean,
               MAX(CASE WHEN valid_ts <= ? THEN cape_j_kg END)                    AS cape_max
        FROM forecasts
        WHERE valid_ts >= ?
        GROUP BY location_name
    """, [h6, h24, h48, h24, h24, now_utc])

    if df.empty:
        return df

    coords = pd.DataFrame(NWP_POINTS).rename(columns={"nome": "location_name"})
    return df.merge(coords, on="location_name", how="left")


def load_alerts(limit: int = 50, active_only: bool = False) -> pd.DataFrame:
    """Carrega log de alertas."""
    where = "WHERE resolved = FALSE" if active_only else ""
    return _query(f"""
        SELECT id, ts, alert_type, severity, location, river,
               message, level_m, threshold_m, telegram_sent, resolved
        FROM alerts_log
        {where}
        ORDER BY ts DESC
        LIMIT {limit}
    """)


def load_stations() -> pd.DataFrame:
    """Carrega todas as estações ativas com coordenadas."""
    return _query("""
        SELECT station_id, name, source, lat, lon, river, municipality, active
        FROM stations
        WHERE active = TRUE AND lat IS NOT NULL AND lon IS NOT NULL
        ORDER BY source, river
    """)


def load_system_stats() -> dict:
    """Carrega métricas operacionais do sistema."""
    stats: dict[str, int] = {}
    for table in ["stations", "rain_readings", "river_levels", "forecasts",
                  "alerts_log", "rain_accumulated", "model_metrics"]:
        df = _query(f"SELECT COUNT(*) AS n FROM {table}")
        stats[table] = int(df["n"].iloc[0]) if not df.empty else 0
    return stats


@st.cache_data(ttl=1800)
def load_gpm_precip() -> pd.DataFrame:
    """
    Carrega grade de precipitação satélite GPM/CHIRPS (GitHub Raw → local).

    Returns:
        DataFrame com lat, lon, precip_mm, timestamp, source (17k+ pontos RS).
    """
    if not _LOADER_OK:
        return pd.DataFrame()
    df, src = fetch_gpm_precip()
    if not df.empty:
        logger.info(f"GPM precip: {len(df):,} pontos ← {src.label}")
    return df


@st.cache_data(ttl=600)
def load_inmet_latest() -> pd.DataFrame:
    """
    Carrega histórico INMET e retorna leitura mais recente por estação.

    Returns:
        DataFrame com station_id, lat, lon, precip_mm, temp_c, humidity_pct
        (uma linha por estação, ~98 linhas).
    """
    if not _LOADER_OK:
        return pd.DataFrame()
    df, _ = fetch_inmet_historico()
    if df.empty:
        return df
    # Obtém leitura mais recente por estação
    ts_col = next((c for c in ("ts", "timestamp", "date_utc") if c in df.columns), None)
    if ts_col:
        df = df.sort_values(ts_col).groupby("station_id").last().reset_index()
    return df


# ---------------------------------------------------------------------------
# Componentes de visualização
# ---------------------------------------------------------------------------

def _river_gauge(river: str, level_m: float, status: str,
                 trend: str, pct: float) -> go.Figure:
    """Gauge Plotly para nível do rio com limiar colorido."""
    cfg = RIOS_COTAS.get(river, {"atencao": 3.0, "alerta": 5.0, "emergencia": 7.0})
    max_val = cfg["emergencia"] * 1.2

    trend_arrow = {"SUBINDO": "▲ subindo", "DESCENDO": "▼ descendo",
                   "ESTAVEL": "→ estável"}.get(trend, "→")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level_m,
        title={"text": f"{river}<br><span style='font-size:0.75em'>{trend_arrow}</span>",
               "font": {"color": "#e2e8f0", "size": 16}},
        number={"suffix": " m", "font": {"color": "#e2e8f0", "size": 24}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#94a3b8",
                     "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": STATUS_COLORS.get(status, "#22c55e")},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, cfg["atencao"]], "color": "#14532d"},
                {"range": [cfg["atencao"], cfg["alerta"]], "color": "#713f12"},
                {"range": [cfg["alerta"], cfg["emergencia"]], "color": "#7f1d1d"},
                {"range": [cfg["emergencia"], max_val], "color": "#3b0764"},
            ],
            "threshold": {
                "line": {"color": "#f59e0b", "width": 3},
                "thickness": 0.75,
                "value": cfg["atencao"],
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=220,
        margin={"t": 60, "b": 10, "l": 20, "r": 20},
    )
    return fig


def _river_timeseries(df: pd.DataFrame, river: str) -> go.Figure:
    """Série temporal de nível + vazão do rio (eixos duplos)."""
    cfg = RIOS_COTAS.get(river, {"atencao": 3.0, "alerta": 5.0, "emergencia": 7.0})

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Nível
    for station_id, grp in df.groupby("station_id"):
        fig.add_trace(go.Scatter(
            x=grp["ts"], y=grp["level_m"],
            name=f"Nível {grp['station_name'].iloc[0] if 'station_name' in grp else station_id}",
            mode="lines",
            line={"width": 2},
        ), secondary_y=False)

    # Linhas de limiar
    for label, key, color in [
        ("Atenção", "atencao", "#f59e0b"),
        ("Alerta",  "alerta",  "#ef4444"),
        ("Emerg.",  "emergencia", "#7c3aed"),
    ]:
        fig.add_hline(y=cfg[key], line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font_color=color)

    # Vazão (eixo secundário)
    if "flow_cms" in df.columns and df["flow_cms"].notna().any():
        for station_id, grp in df.groupby("station_id"):
            g = grp.dropna(subset=["flow_cms"])
            if not g.empty:
                fig.add_trace(go.Scatter(
                    x=g["ts"], y=g["flow_cms"],
                    name=f"Vazão {station_id}",
                    mode="lines",
                    line={"dash": "dot", "width": 1.5, "color": "#38bdf8"},
                    opacity=0.7,
                ), secondary_y=True)

    fig.update_layout(
        title=f"Rio {river} — últimas horas",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0"},
        legend={"bgcolor": "#1e293b", "bordercolor": "#334155"},
        hovermode="x unified",
        height=350,
        margin={"t": 50, "b": 40, "l": 60, "r": 60},
    )
    fig.update_xaxes(gridcolor="#1e293b", linecolor="#334155")
    fig.update_yaxes(title_text="Nível (m)", gridcolor="#334155",
                     linecolor="#334155", secondary_y=False)
    fig.update_yaxes(title_text="Vazão (m³/s)", gridcolor="#334155",
                     linecolor="#334155", secondary_y=True)
    return fig


def _rain_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Gráfico de barras agrupado: chuva acumulada top-20 estações."""
    if df.empty:
        return go.Figure()
    top = df.nlargest(20, "rain_24h")
    fig = go.Figure()
    periodos = [("rain_1h", "1h"), ("rain_6h", "6h"),
                ("rain_24h", "24h"), ("rain_72h", "72h")]
    colors = ["#38bdf8", "#818cf8", "#f59e0b", "#ef4444"]
    for (col, label), color in zip(periodos, colors):
        if col in top.columns:
            x_labels = (top["station_name"].fillna(top["station_id"])
                        if "station_name" in top.columns else top["station_id"])
            fig.add_trace(go.Bar(
                name=label,
                x=x_labels,
                y=top[col],
                marker_color=color,
            ))
    fig.update_layout(
        barmode="group",
        title="Chuva acumulada — top 20 estações",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0"},
        legend={"bgcolor": "#1e293b"},
        height=380,
        margin={"t": 50, "b": 80, "l": 60, "r": 20},
        xaxis={"tickangle": -45, "gridcolor": "#334155"},
        yaxis={"title": "mm", "gridcolor": "#334155"},
    )
    return fig


def _forecast_chart(df: pd.DataFrame, location: str) -> go.Figure:
    """
    Meteograma multi-eixo estilo HCMR: chuva (barras) + temperatura + vento.

    Args:
        df: DataFrame NWP com colunas valid_ts, rain_mm, temperature, wind_speed,
            model_source.
        location: Nome da localidade para o título.

    Returns:
        Figura Plotly com até 3 eixos Y independentes.
    """
    if df.empty:
        return go.Figure()

    _COLORS = {"openmeteo": "#38bdf8", "noaa": "#f59e0b", "ecmwf": "#a78bfa"}

    fig = go.Figure()

    for model, grp in df.groupby("model_source"):
        color = _COLORS.get(model, "#94a3b8")
        grp = grp.sort_values("valid_ts")

        if "rain_mm" in grp.columns:
            fig.add_trace(go.Bar(
                x=grp["valid_ts"],
                y=grp["rain_mm"].clip(lower=0),
                name=f"Chuva {model}",
                marker_color=color,
                opacity=0.75,
                yaxis="y1",
            ))

        if "temperature" in grp.columns:
            fig.add_trace(go.Scatter(
                x=grp["valid_ts"],
                y=grp["temperature"],
                name=f"Temp {model}",
                mode="lines",
                line={"color": "#fb923c", "width": 2, "dash": "dot"},
                yaxis="y2",
            ))

        if "wind_speed" in grp.columns:
            fig.add_trace(go.Scatter(
                x=grp["valid_ts"],
                y=grp["wind_speed"],
                name=f"Vento {model}",
                mode="lines",
                line={"color": "#a3e635", "width": 1.5, "dash": "dash"},
                opacity=0.75,
                yaxis="y3",
            ))

    fig.update_layout(
        title=f"Meteograma — {location}",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0", "size": 12},
        legend={"bgcolor": "rgba(30,41,59,0.27)", "bordercolor": "#334155",
                "borderwidth": 1, "x": 1.18, "y": 1},
        hovermode="x unified",
        height=480,
        margin={"t": 60, "b": 50, "l": 60, "r": 150},
        xaxis={"gridcolor": "#334155", "linecolor": "#334155", "tickangle": -30},
        yaxis={
            "title": "Chuva (mm)",
            "titlefont": {"color": "#38bdf8"},
            "tickfont": {"color": "#38bdf8"},
            "gridcolor": "#334155",
        },
        yaxis2={
            "title": "Temp (°C)",
            "titlefont": {"color": "#fb923c"},
            "tickfont": {"color": "#fb923c"},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        yaxis3={
            "title": "Vento (km/h)",
            "titlefont": {"color": "#a3e635"},
            "tickfont": {"color": "#a3e635"},
            "overlaying": "y",
            "side": "right",
            "position": 0.90,
            "showgrid": False,
            "anchor": "free",
        },
    )
    return fig


def _wind_rose(df: pd.DataFrame, location: str) -> go.Figure:
    """
    Rosa dos ventos estilo Windy com plotly polar, colorida por velocidade.

    Args:
        df: DataFrame NWP com colunas wind_speed e wind_direction_deg.
        location: Nome da localidade para o título.

    Returns:
        Figura Plotly polar. Retorna figura vazia com aviso se dados insuficientes.
    """
    _DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

    fig_empty = go.Figure()
    fig_empty.update_layout(
        title=f"Rosa dos Ventos — {location}",
        paper_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        height=320,
        annotations=[{"text": "Sem dados de direção", "showarrow": False,
                      "x": 0.5, "y": 0.5, "font": {"color": "#94a3b8", "size": 14}}],
    )

    if df.empty or "wind_direction_deg" not in df.columns or "wind_speed" not in df.columns:
        return fig_empty

    df2 = df[["wind_speed", "wind_direction_deg"]].dropna().copy()
    if len(df2) < 3:
        return fig_empty

    df2["sector"] = ((df2["wind_direction_deg"] + 11.25) % 360 // 22.5).astype(int)
    df2["sector_name"] = df2["sector"].map(lambda i: _DIRS[i % 16])

    grp = df2.groupby("sector_name").agg(
        freq=("wind_speed", "count"),
        speed_mean=("wind_speed", "mean"),
    ).reset_index()
    grp["pct"] = grp["freq"] / grp["freq"].sum() * 100

    fig = go.Figure(go.Barpolar(
        r=grp["pct"],
        theta=grp["sector_name"],
        marker_color=grp["speed_mean"],
        marker_colorscale=[[0, "#1d4ed8"], [0.4, "#16a34a"],
                           [0.75, "#ca8a04"], [1, "#dc2626"]],
        marker_showscale=True,
        marker_colorbar={
            "title": "km/h", "thickness": 12,
            "tickfont": {"color": "#e2e8f0", "size": 10},
            "titlefont": {"color": "#e2e8f0"},
        },
        opacity=0.85,
        name="Frequência (%)",
    ))
    fig.update_layout(
        title=f"Rosa dos Ventos — {location}",
        paper_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        polar={
            "bgcolor": "#1e293b",
            "angularaxis": {"tickfont": {"size": 10, "color": "#94a3b8"},
                            "linecolor": "#334155"},
            "radialaxis": {"tickfont": {"size": 9, "color": "#94a3b8"},
                           "gridcolor": "#334155", "linecolor": "#334155",
                           "ticksuffix": "%"},
        },
        height=320,
        margin={"t": 50, "b": 20, "l": 20, "r": 60},
    )
    return fig


def _inmet_rain_color(mm: float) -> str:
    """
    Escala de cores Windy para chuva de estações pontuais.

    Args:
        mm: Milímetros acumulados no período selecionado.

    Returns:
        Cor hex para o marker.
    """
    if mm <= 0:    return "#555555"
    if mm < 5:     return "#4fc3f7"
    if mm < 20:    return "#1565c0"
    if mm < 40:    return "#2e7d32"
    if mm < 60:    return "#f9a825"
    return "#c62828"


def _build_map(stations_df: pd.DataFrame,
               rain_df: pd.DataFrame,
               river_status_df: pd.DataFrame,
               period: str = "rain_24h",
               gpm_df: Optional[pd.DataFrame] = None,
               inmet_df: Optional[pd.DataFrame] = None,
               show_layers: Optional[dict] = None) -> Optional["folium.Map"]:
    """
    Mapa Folium estilo Windy: HeatMap GPM + círculos INMET + marcadores de rios.

    Args:
        stations_df: DataFrame de estações (com lat/lon/source).
        rain_df: DataFrame de acumulados de chuva por estação (ANA + INMET).
        river_status_df: DataFrame de status atual dos rios.
        period: Coluna de acumulado a exibir (ex: "rain_24h").
        gpm_df: DataFrame GPM/CHIRPS com lat, lon, precip_mm (17k+ pontos RS).
        show_layers: Dict com booleanos para cada camada:
            {"heatmap": True, "inmet": True, "rios": True, "nwp": False}

    Returns:
        Objeto folium.Map ou None se folium não estiver disponível.
    """
    if not _FOLIUM_OK:
        return None

    period_label = period.replace("rain_", "")
    layers = show_layers or {"heatmap": True, "inmet": True, "rios": True, "nwp": False}

    m = folium.Map(
        location=[-29.5, -53.0],
        zoom_start=7,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # ── Legenda flutuante (canto inferior direito) ──────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;right:15px;z-index:9999;
                background:#1e293bdd;border:1px solid #475569;border-radius:8px;
                padding:10px 14px;font-size:11px;color:#e2e8f0;line-height:1.85;
                backdrop-filter:blur(4px)">
        <b style="color:#f8fafc;font-size:12px">Precipitação</b><br>
        <span style="color:#555555">⬤</span>&nbsp;Sem chuva<br>
        <span style="color:#4fc3f7">⬤</span>&nbsp;&lt; 5 mm<br>
        <span style="color:#1565c0">⬤</span>&nbsp;5 – 20 mm<br>
        <span style="color:#2e7d32">⬤</span>&nbsp;20 – 40 mm<br>
        <span style="color:#f9a825">⬤</span>&nbsp;40 – 60 mm<br>
        <span style="color:#c62828">⬤</span>&nbsp;&gt; 60 mm
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── HeatMap GPM/CHIRPS ────────────────────────────────────────────────
    if layers.get("heatmap", True) and gpm_df is not None and not gpm_df.empty:
        _gpm_valid = gpm_df.dropna(subset=["lat", "lon", "precip_mm"])
        if not _gpm_valid.empty:
            _max_p = _gpm_valid["precip_mm"].max()
            if _max_p > 0:
                heat_data = [
                    [float(r.lat), float(r.lon),
                     float(r.precip_mm) / _max_p]
                    for r in _gpm_valid.itertuples()
                ]
            else:
                heat_data = [[float(r.lat), float(r.lon), 0.0]
                             for r in _gpm_valid.itertuples()]
            _FoliumHeatMap(
                heat_data,
                gradient={
                    "0.0": "#313695", "0.2": "#4575b4", "0.4": "#74add1",
                    "0.6": "#abd9e9", "0.7": "#fee090", "0.8": "#f46d43",
                    "0.9": "#d73027", "1.0": "#a50026",
                },
                radius=15,
                blur=20,
                min_opacity=0.3,
                name="HeatMap GPM/CHIRPS",
            ).add_to(m)

    # ── Camada estações INMET (círculos coloridos por intensidade) ──────────
    rain_layer = folium.FeatureGroup(
        name=f"Estações chuva {period_label}", show=layers.get("inmet", True)
    )
    if not rain_df.empty and period in rain_df.columns:
        ts_col = next((c for c in ("max_ts", "ts", "updated_at") if c in rain_df.columns), None)
        for _, row in rain_df.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            val    = float(row.get(period) or 0)
            color  = _rain_color_windy(val) if val > 0 else "#64748b"
            radius = max(8, min(45, 8 + val * 0.55))
            name   = (row.get("station_name") or row.get("name")
                      or str(row.get("station_id", "—")))
            mun    = str(row.get("municipality") or "")
            rv     = str(row.get("river") or "")
            r1h    = float(row.get("rain_1h") or 0)
            r6h    = float(row.get("rain_6h") or 0)
            r24h   = float(row.get("rain_24h") or 0)
            r72h   = float(row.get("rain_72h") or 0)
            ts_str = ""
            if ts_col and not pd.isna(row.get(ts_col)):
                try:
                    ts_str = pd.to_datetime(row[ts_col]).strftime("%d/%m %H:%M")
                except Exception:
                    pass

            meta_line = ""
            if mun:
                meta_line = f"<span style='color:#94a3b8'>{mun}" + (f" · {rv}" if rv else "") + "</span><br>"

            popup_html = (
                f"<div style='font-family:sans-serif;background:#1e293b;color:#e2e8f0;"
                f"border-radius:8px;padding:12px;min-width:180px;font-size:13px'>"
                f"<b style='font-size:15px;color:#f8fafc'>{name}</b><br>"
                f"{meta_line}"
                f"<hr style='border-color:#334155;margin:6px 0'>"
                f"<table style='width:100%;border-collapse:collapse'>"
                f"<tr><td style='color:#94a3b8'>1h</td>"
                f"    <td style='text-align:right;color:#38bdf8'><b>{r1h:.1f} mm</b></td></tr>"
                f"<tr><td style='color:#94a3b8'>6h</td>"
                f"    <td style='text-align:right;color:#38bdf8'><b>{r6h:.1f} mm</b></td></tr>"
                f"<tr><td style='color:#94a3b8'>24h</td>"
                f"    <td style='text-align:right;color:#38bdf8'><b>{r24h:.1f} mm</b></td></tr>"
                f"<tr><td style='color:#94a3b8'>72h</td>"
                f"    <td style='text-align:right;color:#38bdf8'><b>{r72h:.1f} mm</b></td></tr>"
                f"</table>"
                + (f"<hr style='border-color:#334155;margin:6px 0'>"
                   f"<small style='color:#64748b'>{ts_str}</small>" if ts_str else "")
                + "</div>"
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color=color,
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.72,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{name}: {val:.1f} mm/{period_label}",
            ).add_to(rain_layer)
    rain_layer.add_to(m)

    # ── Camada de rios (círculos + labels coloridos por status) ────────────
    rivers_layer = folium.FeatureGroup(name="Status dos Rios", show=True)
    if not river_status_df.empty:
        for _, row in river_status_df.iterrows():
            river = row.get("river", "")
            cfg   = RIOS_COTAS.get(river)
            if cfg is None:
                continue
            status = row.get("status", "NORMAL")
            color  = STATUS_COLORS.get(status, "#22c55e")
            level  = float(row.get("level_m") or 0)
            pct    = float(row.get("pct_cota_atencao") or 0)
            popup_html = (
                f"<div style='font-family:sans-serif;background:#1e293b;color:#e2e8f0;"
                f"border-radius:8px;padding:12px;min-width:160px;font-size:13px'>"
                f"<b style='font-size:15px;color:#f8fafc'>Rio {river}</b>"
                f"<hr style='border-color:#334155;margin:6px 0'>"
                f"<span style='color:{color};font-weight:bold;font-size:14px'>{status}</span><br>"
                f"<span style='color:#94a3b8'>Nível:</span> <b>{level:.2f} m</b><br>"
                f"<span style='color:#94a3b8'>% cota atenção:</span> <b>{pct:.0f}%</b>"
                f"</div>"
            )
            folium.CircleMarker(
                location=[cfg["lat"], cfg["lon"]],
                radius=18,
                color=color,
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.30,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"Rio {river}: {level:.2f} m ({status})",
            ).add_to(rivers_layer)
            folium.Marker(
                location=[cfg["lat"], cfg["lon"]],
                icon=folium.DivIcon(
                    html=(
                        f"<div style='font-size:9px;font-weight:bold;color:{color};"
                        f"text-shadow:0 0 4px #000;white-space:nowrap;"
                        f"margin-top:20px;margin-left:-22px'>{river}</div>"
                    ),
                    icon_size=(80, 14),
                    icon_anchor=(0, 0),
                ),
            ).add_to(rivers_layer)
    rivers_layer.add_to(m)

    # ── Camada de estações INMET (círculos coloridos por chuva) ────────────
    inmet_layer = folium.FeatureGroup(
        name="Estações INMET (chuva recente)", show=layers.get("inmet_pts", False)
    )
    _inmet_src = inmet_df if (inmet_df is not None and not inmet_df.empty) else None
    if _inmet_src is not None:
        for row in _inmet_src.itertuples():
            if pd.isna(row.lat) or pd.isna(row.lon):
                continue
            mm    = float(getattr(row, "precip_mm", 0) or 0)
            color = _inmet_rain_color(mm)
            rad   = max(6, min(25, 6 + mm * 0.32))
            name  = str(getattr(row, "station_id", "?"))
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=rad,
                color=color,
                weight=0,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=f"INMET {name}: {mm:.1f} mm",
            ).add_to(inmet_layer)
    elif not stations_df.empty and "source" in stations_df.columns:
        for row in stations_df[stations_df["source"] == "INMET"].itertuples():
            if pd.isna(row.lat) or pd.isna(row.lon):
                continue
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=4,
                color="#94a3b8",
                weight=0,
                fill=True,
                fill_color="#94a3b8",
                fill_opacity=0.5,
                tooltip=f"INMET: {getattr(row, 'name', getattr(row, 'station_id', '?'))}",
            ).add_to(inmet_layer)
    inmet_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _rain_color_windy(mm: float) -> str:
    """
    Retorna cor hex estilo Windy baseada na intensidade de chuva acumulada.

    Args:
        mm: milímetros acumulados.

    Returns:
        Cor hex correspondente à faixa de intensidade.
    """
    if mm < 5:
        return "#93c5fd"   # azul claro  — traço/fraca
    if mm < 20:
        return "#1d4ed8"   # azul escuro — fraca/moderada
    if mm < 40:
        return "#16a34a"   # verde       — moderada
    if mm < 60:
        return "#ca8a04"   # amarelo     — forte
    return "#dc2626"       # vermelho    — muito forte / extrema


def _build_forecast_map(summary_df: pd.DataFrame,
                        selected_loc: str) -> Optional["folium.Map"]:
    """
    Cria mapa Folium estilo Windy com círculos coloridos pela chuva prevista 24h.

    Cada um dos 10 pontos NWP recebe um círculo proporcional ao acumulado,
    colorido pela escala Windy (azul claro → azul → verde → amarelo → vermelho).
    O ponto selecionado recebe marcador especial de destaque.

    Args:
        summary_df: DataFrame de load_all_forecasts_summary().
        selected_loc: Nome da localidade selecionada no dropdown.

    Returns:
        Objeto folium.Map ou None se folium não estiver disponível.
    """
    if not _FOLIUM_OK:
        return None

    m = folium.Map(
        location=[-29.8, -52.0],
        zoom_start=7,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # Camada de legenda estilo Windy — injetada como HTML
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:#1e293b;border:1px solid #475569;border-radius:8px;
                padding:10px 14px;font-size:12px;color:#e2e8f0;line-height:1.8">
        <b>Chuva prevista 24h</b><br>
        <span style="color:#93c5fd">●</span> 0–5 mm &nbsp;
        <span style="color:#1d4ed8">●</span> 5–20 mm<br>
        <span style="color:#16a34a">●</span> 20–40 mm &nbsp;
        <span style="color:#ca8a04">●</span> 40–60 mm<br>
        <span style="color:#dc2626">●</span> &gt;60 mm
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Pontos sem dados no banco — usa os NWP_POINTS como fallback
    existing = set(summary_df["location_name"].tolist()) if not summary_df.empty else set()
    all_points = {p["nome"]: p for p in NWP_POINTS}

    for nome, pt in all_points.items():
        # Tenta pegar dados do banco; se não tiver, exibe círculo cinza
        row = (summary_df[summary_df["location_name"] == nome].iloc[0]
               if nome in existing else None)

        rain_24h = float(row["rain_24h"]) if row is not None and row["rain_24h"] else 0.0
        rain_6h  = float(row["rain_6h"])  if row is not None and row["rain_6h"]  else 0.0
        rain_48h = float(row["rain_48h"]) if row is not None and row["rain_48h"] else 0.0
        temp     = float(row["temp_mean"]) if row is not None and row.get("temp_mean") else None
        cape     = float(row["cape_max"])  if row is not None and row.get("cape_max")  else None

        color    = _rain_color_windy(rain_24h) if row is not None else "#64748b"
        # Raio proporcional: mínimo 12px, máximo 55px
        radius   = max(12, min(55, 12 + rain_24h * 0.6))
        is_sel   = (nome == selected_loc)

        popup_html = (
            f"<div style='font-family:sans-serif;min-width:160px'>"
            f"<b style='font-size:14px'>{nome}</b><br>"
            f"<hr style='margin:4px 0'>"
            f"🌧 6h: <b>{rain_6h:.1f} mm</b><br>"
            f"🌧 24h: <b>{rain_24h:.1f} mm</b><br>"
            f"🌧 48h: <b>{rain_48h:.1f} mm</b><br>"
            + (f"🌡 Temp: <b>{temp:.1f} °C</b><br>" if temp is not None else "")
            + (f"⚡ CAPE: <b>{cape:.0f} J/kg</b>" if cape is not None else "")
            + "</div>"
        )

        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=radius,
            color="#ffffff" if is_sel else color,
            weight=3 if is_sel else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.75 if row is not None else 0.25,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{'★ ' if is_sel else ''}{nome}: {rain_24h:.1f} mm/24h",
        ).add_to(m)

        # Label com nome da cidade
        folium.Marker(
            location=[pt["lat"], pt["lon"]],
            icon=folium.DivIcon(
                html=(
                    f"<div style='font-size:10px;font-weight:bold;"
                    f"color:{'#fff' if is_sel else '#cbd5e1'};"
                    f"text-shadow:0 0 4px #000;white-space:nowrap;"
                    f"margin-top:{int(radius)+4}px;margin-left:-30px'>"
                    f"{nome}</div>"
                ),
                icon_size=(120, 20),
                icon_anchor=(0, 0),
            ),
        ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------

def _page_overview(river_status_df: pd.DataFrame,
                   rain_df: pd.DataFrame,
                   stations_df: pd.DataFrame) -> None:
    """Página principal: KPIs de rios + mapa estilo Windy + painel lateral com legenda."""
    st.markdown("## 🗺️ Visão Geral")

    # ── KPIs dos rios ──────────────────────────────────────────────────────
    if river_status_df.empty:
        st.info("Nenhum dado de rios disponível. Execute o coletor ANA primeiro.")
    else:
        cols = st.columns(len(river_status_df))
        for i, (_, row) in enumerate(river_status_df.iterrows()):
            status = row.get("status", "NORMAL")
            emoji  = STATUS_EMOJI.get(status, "⚪")
            level  = row.get("level_m") or 0.0
            pct    = row.get("pct_cota_atencao") or 0.0
            trend  = {"SUBINDO": "▲", "DESCENDO": "▼", "ESTAVEL": "→"}.get(
                row.get("trend", ""), "→")
            with cols[i]:
                st.metric(
                    label=f"{emoji} Rio {row['river']}",
                    value=f"{level:.2f} m",
                    delta=f"{pct:.0f}% cota {trend}",
                    delta_color="inverse" if status in ("NORMAL",) else "off",
                )

    st.markdown("---")

    # ── Seletor de período (pills horizontais) ─────────────────────────────
    _period_opts = {
        "1h": "rain_1h", "6h": "rain_6h", "24h": "rain_24h",
        "48h": "rain_48h", "72h": "rain_72h",
    }
    periodo = st.radio(
        "Acumulado exibido no mapa",
        list(_period_opts.keys()),
        index=2,
        horizontal=True,
    )
    period_col = _period_opts[periodo]

    # ── Dados satélite / INMET (carregados em background) ─────────────────
    gpm_df    = load_gpm_precip()
    inmet_df  = load_inmet_latest()

    # ── Toggles de camadas (sidebar) ──────────────────────────────────────
    with st.sidebar:
        st.markdown("#### 🗂️ Camadas do mapa")
        show_heatmap  = st.checkbox("HeatMap GPM/CHIRPS",       value=True)
        show_inmet_pt = st.checkbox("Pontos INMET (chuva)",     value=False)
        show_rios     = st.checkbox("Status dos Rios",          value=True)
        show_layers = {
            "heatmap":   show_heatmap,
            "inmet_pts": show_inmet_pt,
            "rios":      show_rios,
        }

    # ── Layout mapa (3/4) + painel lateral (1/4) ──────────────────────────
    col_map, col_right = st.columns([3, 1])

    with col_map:
        if _FOLIUM_OK:
            m = _build_map(
                stations_df, rain_df, river_status_df,
                period=period_col,
                gpm_df=gpm_df if not gpm_df.empty else None,
                inmet_df=inmet_df if not inmet_df.empty else None,
                show_layers=show_layers,
            )
            if m is not None:
                st_folium(m, use_container_width=True, height=560, returned_objects=[])
        else:
            st.warning("Instale `streamlit-folium`: `pip install streamlit-folium`")
        if not gpm_df.empty:
            src_label = "CHIRPS/GPM"
            st.caption(f"Precipitação satélite: {len(gpm_df):,} pontos RS ← {src_label}")

    with col_right:
        # Legenda de chuva
        st.markdown(
            "<div class='windy-card'><b style='color:#f8fafc'>Chuva acumulada</b></div>",
            unsafe_allow_html=True,
        )
        _scale = [
            ("#64748b", "Sem dados"),
            ("#93c5fd", "< 5 mm"),
            ("#1d4ed8", "5 – 20 mm"),
            ("#16a34a", "20 – 40 mm"),
            ("#ca8a04", "40 – 60 mm"),
            ("#dc2626", "> 60 mm"),
        ]
        for color, label in _scale:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
                f"<span style='width:13px;height:13px;border-radius:50%;"
                f"background:{color};flex-shrink:0;display:inline-block'></span>"
                f"<span style='font-size:0.82em;color:#cbd5e1'>{label}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Legenda de status dos rios
        st.markdown(
            "<div class='windy-card'><b style='color:#f8fafc'>Status dos Rios</b></div>",
            unsafe_allow_html=True,
        )
        for status, color in STATUS_COLORS.items():
            emoji = STATUS_EMOJI.get(status, "⚪")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
                f"<span style='width:13px;height:13px;border-radius:50%;"
                f"background:{color};flex-shrink:0;display:inline-block'></span>"
                f"<span style='font-size:0.82em;color:#cbd5e1'>{emoji} {status}</span></div>",
                unsafe_allow_html=True,
            )

        # Cards de rios monitorados
        if not river_status_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='windy-card'><b style='color:#f8fafc'>Monitoramento</b></div>",
                unsafe_allow_html=True,
            )
            for _, row in river_status_df.iterrows():
                status = row.get("status", "NORMAL")
                color  = STATUS_COLORS.get(status, "#22c55e")
                level  = row.get("level_m") or 0.0
                st.markdown(
                    f"<div style='background:#1e293b;border-left:3px solid {color};"
                    f"border-radius:6px;padding:6px 10px;margin:4px 0;font-size:0.82em'>"
                    f"<b style='color:#f8fafc'>{row['river']}</b><br>"
                    f"<span style='color:#94a3b8'>{level:.2f} m</span> &nbsp;"
                    f"<span style='color:{color}'>{status}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Tabela top-5 acumulados ────────────────────────────────────────────
    if not rain_df.empty:
        sort_col = period_col if period_col in rain_df.columns else "rain_24h"
        st.markdown(f"### 🌧️ Maiores acumulados {periodo}")
        _wanted = ["station_name", "municipality", "river",
                   "rain_1h", "rain_6h", "rain_24h", "rain_72h"]
        _avail  = [c for c in _wanted if c in rain_df.columns]
        top5 = rain_df.nlargest(5, sort_col)[_avail].rename(columns={
            "station_name": "Estação",
            "municipality": "Município",
            "river": "Rio",
            "rain_1h": "1h (mm)",
            "rain_6h": "6h (mm)",
            "rain_24h": "24h (mm)",
            "rain_72h": "72h (mm)",
        })
        st.dataframe(top5, use_container_width=True, hide_index=True)


def _page_rivers(river_status_df: pd.DataFrame) -> None:
    """Página de rios: gauges + séries temporais."""
    st.markdown("## 🌊 Monitoramento de Rios")

    col_gauges = st.columns(4)
    for i, (_, row) in enumerate(river_status_df.iterrows() if not river_status_df.empty
                                   else iter([])):
        with col_gauges[i % 4]:
            fig = _river_gauge(
                river=row.get("river", ""),
                level_m=row.get("level_m") or 0.0,
                status=row.get("status", "NORMAL"),
                trend=row.get("trend", "ESTAVEL"),
                pct=row.get("pct_cota_atencao") or 0.0,
            )
            st.plotly_chart(fig, use_container_width=True)

    if river_status_df.empty:
        st.info("Sem dados de rios.")
        return

    st.markdown("---")
    st.markdown("### 📈 Série histórica")

    selected_river = st.selectbox(
        "Selecione o rio",
        options=river_status_df["river"].tolist() if not river_status_df.empty
                else list(RIOS_COTAS.keys()),
    )
    hours_back = st.slider("Janela temporal (horas)", 12, 240, 72, step=12)

    df_levels = load_river_levels(selected_river, hours=hours_back)
    if df_levels.empty:
        st.info(f"Sem histórico de níveis para {selected_river}.")
    else:
        fig = _river_timeseries(df_levels, selected_river)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Dados brutos")
        disp = df_levels[["ts", "station_id", "station_name",
                           "level_m", "flow_cms"]].copy()
        disp["ts"] = disp["ts"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(disp, use_container_width=True, hide_index=True)


def _page_rain(rain_df: pd.DataFrame) -> None:
    """Página de chuva: acumulados e mapa de intensidade."""
    st.markdown("## 🌧️ Chuva Acumulada")

    if rain_df.empty:
        st.info("Sem dados de chuva acumulada. Execute o rain_accumulator.")
        return

    # Métricas resumo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estações com dados", len(rain_df))
    c2.metric("Maior 1h (mm)",  f"{rain_df['rain_1h'].max():.1f}")
    c3.metric("Maior 24h (mm)", f"{rain_df['rain_24h'].max():.1f}")
    c4.metric("Maior 72h (mm)", f"{rain_df['rain_72h'].max():.1f}")

    # Período selecionado
    periodo = st.radio(
        "Período",
        ["1h", "3h", "6h", "12h", "24h", "48h", "72h"],
        horizontal=True,
        index=4,
    )
    col_map = {
        "1h": "rain_1h", "3h": "rain_3h", "6h": "rain_6h",
        "12h": "rain_12h", "24h": "rain_24h",
        "48h": "rain_48h", "72h": "rain_72h",
    }
    col = col_map[periodo]

    if _FOLIUM_OK and "lat" in rain_df.columns:
        col_has = rain_df[col].notna() & (rain_df[col] > 0)
        df_plot = rain_df[col_has].copy()
        if not df_plot.empty:
            m2 = folium.Map(location=[-29.8, -51.5], zoom_start=7,
                            tiles="CartoDB dark_matter", prefer_canvas=True)
            for _, row in df_plot.iterrows():
                val = row[col]
                color = ("#38bdf8" if val < 10 else
                         "#f59e0b" if val < 30 else
                         "#ef4444" if val < 60 else "#7c3aed")
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=max(4, min(35, val / 3)),
                    color=color, fill=True, fill_color=color, fill_opacity=0.65,
                    tooltip=f"{row.get('station_name', row['station_id'])}: {val:.1f} mm/{periodo}",
                ).add_to(m2)
            st_folium(m2, use_container_width=True, height=480,
                      returned_objects=[])

    # Gráfico de barras
    fig = _rain_bar_chart(rain_df)
    st.plotly_chart(fig, use_container_width=True)

    # Tabela completa
    with st.expander("Tabela completa"):
        _meta   = ["station_name", "municipality", "river"]
        _rain_c = ["rain_1h", "rain_3h", "rain_6h", "rain_24h", "rain_48h", "rain_72h"]
        _cols   = [c for c in _meta + _rain_c if c in rain_df.columns]
        _rename = {
            "station_name": "Estação", "municipality": "Município", "river": "Rio",
            "rain_1h": "1h", "rain_3h": "3h", "rain_6h": "6h",
            "rain_24h": "24h", "rain_48h": "48h", "rain_72h": "72h",
        }
        disp = rain_df[_cols].rename(columns=_rename).copy()
        st.dataframe(disp, use_container_width=True, hide_index=True)


def _wind_forecast_chart(df: pd.DataFrame, location: str) -> go.Figure:
    """
    Gráfico de velocidade e direção do vento (Y1 velocidade, Y2 direção).

    Args:
        df: DataFrame NWP com colunas wind_speed e wind_direction_deg.
        location: Nome da localidade para o título.

    Returns:
        Figure Plotly com dois eixos Y.
    """
    fig = go.Figure()
    has_speed = "wind_speed" in df.columns
    has_dir   = "wind_direction_deg" in df.columns or "wind_direction" in df.columns
    dir_col   = "wind_direction_deg" if "wind_direction_deg" in df.columns else "wind_direction"

    if has_speed:
        fig.add_trace(go.Scatter(
            x=df["valid_ts"],
            y=df["wind_speed"],
            name="Velocidade (km/h)",
            line={"color": "#4ade80", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(74,222,128,0.15)",
            yaxis="y1",
        ))

    if has_dir and dir_col in df.columns:
        # Cor do ponto por quadrante
        dirs = df[dir_col].fillna(0)
        def _dir_color(d: float) -> str:
            d = d % 360
            if d < 90:   return "#38bdf8"   # N→L  azul
            if d < 180:  return "#fb923c"   # L→S  laranja
            if d < 270:  return "#f43f5e"   # S→O  vermelho
            return "#a78bfa"                # O→N  roxo
        pt_colors = [_dir_color(float(d)) for d in dirs]

        fig.add_trace(go.Scatter(
            x=df["valid_ts"],
            y=dirs,
            name="Direção (°)",
            mode="markers",
            marker={"color": pt_colors, "size": 7, "symbol": "arrow-up",
                    "angle": dirs.tolist()},
            yaxis="y2",
        ))

    _DARK = "#0f172a"
    fig.update_layout(
        paper_bgcolor=_DARK,
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0", "size": 12},
        height=280,
        margin={"t": 30, "b": 50, "l": 50, "r": 60},
        legend={"bgcolor": "rgba(30,41,59,0.27)"},
        title={"text": f"Vento — {location}", "font": {"size": 14, "color": "#f8fafc"}},
        xaxis={"gridcolor": "#334155"},
        yaxis={"title": "km/h", "gridcolor": "#334155", "titlefont": {"color": "#4ade80"},
               "tickfont": {"color": "#4ade80"}},
        yaxis2={"title": "Direção (°)", "overlaying": "y", "side": "right",
                "range": [0, 360], "showgrid": False,
                "titlefont": {"color": "#94a3b8"}, "tickfont": {"color": "#94a3b8"}},
    )
    if not has_speed and not has_dir:
        fig.add_annotation(text="Dados de vento indisponíveis",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"color": "#64748b", "size": 14})
    return fig


def _solar_chart(df: pd.DataFrame, location: str) -> go.Figure:
    """
    Gráfico de radiação solar prevista com bandas noturnas (20h–06h) hachuradas.

    Args:
        df: DataFrame NWP com colunas shortwave_radiation (W/m²) e valid_ts.
        location: Nome da localidade.

    Returns:
        Figure Plotly com área preenchida amarela e bandas noturnas.
    """
    fig = go.Figure()
    _DARK = "#0f172a"
    rad_col = next(
        (c for c in ("shortwave_radiation", "solar_radiation", "radiation_w_m2")
         if c in df.columns),
        None,
    )

    if rad_col:
        fig.add_trace(go.Scatter(
            x=df["valid_ts"],
            y=df[rad_col].clip(lower=0),
            name="Radiação (W/m²)",
            line={"color": "#fbbf24", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(251,191,36,0.20)",
        ))

        # Bandas noturnas (20h–06h UTC-3 = 23h–09h UTC)
        if "valid_ts" in df.columns and len(df) > 1:
            ts_vals = pd.to_datetime(df["valid_ts"])
            t_min, t_max = ts_vals.min(), ts_vals.max()
            import datetime as _dt
            cur = t_min.normalize()
            while cur <= t_max:
                night_start = cur + pd.Timedelta(hours=23)
                night_end   = cur + pd.Timedelta(hours=33)  # +1 day 09h
                if night_end > t_min and night_start < t_max:
                    fig.add_vrect(
                        x0=max(night_start, t_min),
                        x1=min(night_end, t_max),
                        fillcolor="rgba(15,23,42,0.55)",
                        layer="below",
                        line_width=0,
                    )
                cur += pd.Timedelta(days=1)

    fig.update_layout(
        paper_bgcolor=_DARK,
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0", "size": 12},
        height=240,
        margin={"t": 30, "b": 50, "l": 60, "r": 20},
        legend={"bgcolor": "rgba(30,41,59,0.27)"},
        title={"text": f"Radiação Solar — {location}", "font": {"size": 14, "color": "#f8fafc"}},
        xaxis={"gridcolor": "#334155"},
        yaxis={"title": "W/m²", "gridcolor": "#334155",
               "titlefont": {"color": "#fbbf24"}, "tickfont": {"color": "#fbbf24"}},
    )
    if not rad_col:
        fig.add_annotation(text="Dados de radiação solar indisponíveis",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"color": "#64748b", "size": 14})
    return fig


def _page_forecasts() -> None:
    """Página de previsões NWP: mapa Windy + meteograma multi-eixo + rosa dos ventos."""
    st.markdown("## 🔮 Previsões Meteorológicas")

    all_locations = [p["nome"] for p in NWP_POINTS]
    c_loc, c_per = st.columns([2, 3])
    with c_loc:
        loc = st.selectbox("Localidade", all_locations, index=0)
    with c_per:
        _period_fc_opts = {
            "6h": 6, "24h": 24, "36h": 36, "48h": 48, "72h": 72, "7d": 168,
        }
        periodo_fc = st.radio(
            "Horizonte",
            list(_period_fc_opts.keys()),
            index=1,
            horizontal=True,
        )
        _hours_fc = _period_fc_opts[periodo_fc]

    @st.cache_data(ttl=600)
    def _cached_summary() -> pd.DataFrame:
        return load_all_forecasts_summary()

    summary_df = _cached_summary()
    df_fc_all  = load_forecasts(loc)

    # Filtrar pelo horizonte selecionado
    if not df_fc_all.empty and "valid_ts" in df_fc_all.columns:
        t0_fc  = pd.to_datetime(df_fc_all["valid_ts"].min())
        df_fc  = df_fc_all[
            df_fc_all["valid_ts"] <= t0_fc + timedelta(hours=_hours_fc)
        ].copy()
    else:
        df_fc = df_fc_all.copy()

    # ── Métricas de instabilidade + acumulados ────────────────────────────
    if not df_fc.empty:
        latest   = df_fc.iloc[0]
        cape     = latest.get("cape_j_kg")
        li       = latest.get("lifted_index")
        ki       = latest.get("k_index")
        t0       = df_fc["valid_ts"].iloc[0]
        rain_6h  = df_fc[df_fc["valid_ts"] <= t0 + timedelta(hours=6)]["rain_mm"].sum()
        rain_24h = df_fc[df_fc["valid_ts"] <= t0 + timedelta(hours=24)]["rain_mm"].sum()
        wind_max = df_fc["wind_speed"].max() if "wind_speed" in df_fc.columns else None

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("CAPE (J/kg)",     f"{cape:.0f}"      if cape     is not None else "N/D",
                  help="< 1000 fraco | 1000-2500 moderado | > 2500 severo")
        c2.metric("Lifted Index",    f"{li:.1f}"         if li       is not None else "N/D",
                  help="> 0 estável | -2 a 0 leve | < -6 severo")
        c3.metric("K-Index",         f"{ki:.0f}"         if ki       is not None else "N/D",
                  help="< 20 baixo | 20-30 moderado | > 35 alto")
        c4.metric("Chuva 6h",        f"{rain_6h:.1f} mm" if rain_6h  else "N/D")
        c5.metric("Chuva 24h",       f"{rain_24h:.1f} mm"if rain_24h else "N/D")
        c6.metric("Vento máx.",      f"{wind_max:.0f} km/h" if wind_max is not None else "N/D")
    else:
        st.info(f"Sem previsões para {loc}. Execute o coletor Open-Meteo/NOAA.")

    st.markdown("---")

    # ── Mapa Windy ────────────────────────────────────────────────────────
    st.markdown("### 🗺️ Chuva prevista 24h — grade RS")

    if _FOLIUM_OK:
        fc_map = _build_forecast_map(summary_df, loc)
        if fc_map is not None:
            st_folium(fc_map, use_container_width=True, height=480, returned_objects=[])
        else:
            st.info("Mapa não disponível.")
    else:
        st.warning("Instale `streamlit-folium`: `pip install streamlit-folium`")
        if not summary_df.empty:
            _cols_disp = [c for c in ["location_name", "rain_6h", "rain_24h",
                                       "rain_48h", "temp_mean"] if c in summary_df.columns]
            disp_sum = summary_df[_cols_disp].copy()
            disp_sum.columns = ["Cidade", "6h (mm)", "24h (mm)", "48h (mm)", "Temp °C"][
                :len(_cols_disp)]
            st.dataframe(disp_sum, use_container_width=True, hide_index=True)

    st.markdown("---")

    if not df_fc.empty:
        # ── Meteograma multi-eixo + Rosa dos ventos ───────────────────────
        st.markdown(f"### 📈 Meteograma — {loc}")
        fig_fc = _forecast_chart(df_fc, loc)
        st.plotly_chart(fig_fc, use_container_width=True)

        col_rose, col_info = st.columns([1, 2])
        with col_rose:
            fig_rose = _wind_rose(df_fc, loc)
            st.plotly_chart(fig_rose, use_container_width=True)
        with col_info:
            st.markdown("#### Resumo por modelo")
            for model, grp in df_fc.groupby("model_source"):
                r_total = grp["rain_mm"].clip(lower=0).sum() if "rain_mm" in grp.columns else 0
                t_min   = grp["temperature"].min() if "temperature" in grp.columns else None
                t_max   = grp["temperature"].max() if "temperature" in grp.columns else None
                w_max   = grp["wind_speed"].max()  if "wind_speed"  in grp.columns else None
                temp_str = f"{t_min:.0f}–{t_max:.0f} °C" if t_min is not None else "N/D"
                wind_str = f"{w_max:.0f} km/h"            if w_max  is not None else "N/D"
                color = {"openmeteo": "#38bdf8", "noaa": "#f59e0b",
                         "ecmwf": "#a78bfa"}.get(model, "#94a3b8")
                st.markdown(
                    f"<div style='background:#1e293b;border-left:3px solid {color};"
                    f"border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.9em'>"
                    f"<b style='color:{color}'>{model.upper()}</b><br>"
                    f"<span style='color:#94a3b8'>Chuva total:</span> <b>{r_total:.1f} mm</b> &nbsp;"
                    f"<span style='color:#94a3b8'>Temp:</span> <b>{temp_str}</b> &nbsp;"
                    f"<span style='color:#94a3b8'>Vento max:</span> <b>{wind_str}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with st.expander("Ver dados tabulares"):
            _fc_cols = [c for c in ["valid_ts", "rain_mm", "temperature",
                                     "wind_speed", "cape_j_kg", "model_source"]
                        if c in df_fc.columns]
            disp = df_fc[_fc_cols].copy()
            if "valid_ts" in disp.columns:
                disp["valid_ts"] = disp["valid_ts"].dt.strftime("%d/%m %H:%M")
            _rename_fc = {"valid_ts": "Data/Hora", "rain_mm": "Chuva (mm)",
                          "temperature": "Temp (°C)", "wind_speed": "Vento (km/h)",
                          "cape_j_kg": "CAPE (J/kg)", "model_source": "Modelo"}
            disp.columns = [_rename_fc.get(c, c) for c in disp.columns]
            st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(f"### 💨 Vento — {loc}")
        st.plotly_chart(_wind_forecast_chart(df_fc, loc), use_container_width=True)

        st.markdown(f"### ☀️ Radiação Solar — {loc}")
        st.plotly_chart(_solar_chart(df_fc, loc), use_container_width=True)

    # ── Comparativo dos 10 pontos NWP ────────────────────────────────────
    if not summary_df.empty:
        st.markdown("---")
        st.markdown("### 📊 Comparativo — 10 pontos NWP")
        fig_cmp = go.Figure()
        periodos = [("rain_6h", "6h"), ("rain_24h", "24h"), ("rain_48h", "48h")]
        colors   = ["#38bdf8", "#818cf8", "#f59e0b"]
        for (col, label), color in zip(periodos, colors):
            if col in summary_df.columns:
                fig_cmp.add_trace(go.Bar(
                    name=label,
                    x=summary_df["location_name"],
                    y=summary_df[col].fillna(0).clip(lower=0),
                    marker_color=color,
                ))
        fig_cmp.update_layout(
            barmode="group",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font={"color": "#e2e8f0"},
            legend={"bgcolor": "#1e293b"},
            height=320,
            margin={"t": 20, "b": 60, "l": 50, "r": 20},
            xaxis={"tickangle": -30, "gridcolor": "#334155"},
            yaxis={"title": "mm", "gridcolor": "#334155"},
        )
        st.plotly_chart(fig_cmp, use_container_width=True)


def _page_alerts() -> None:
    """Página de alertas: histórico e ativos."""
    st.markdown("## 🚨 Alertas")

    tab1, tab2 = st.tabs(["Alertas Ativos", "Histórico"])

    with tab1:
        df_active = load_alerts(limit=100, active_only=True)
        if df_active.empty:
            st.success("✅ Nenhum alerta ativo no momento.")
        else:
            st.warning(f"**{len(df_active)} alerta(s) ativo(s)**")
            for _, row in df_active.iterrows():
                severity = row.get("severity", "INFO")
                color = {"INFO": "blue", "ATENCAO": "orange",
                         "ALERTA": "red", "EMERGENCIA": "violet"}.get(severity, "blue")
                with st.container():
                    cols = st.columns([1, 5, 2])
                    cols[0].markdown(
                        f":{color}[**{severity}**]")
                    cols[1].markdown(
                        f"**{row.get('alert_type','')}** — "
                        f"{row.get('location','')} {row.get('river','')}<br>"
                        f"<small>{row.get('message','')}</small>",
                        unsafe_allow_html=True,
                    )
                    ts_str = row["ts"].strftime("%d/%m %H:%M") if hasattr(row["ts"], "strftime") else str(row["ts"])
                    cols[2].caption(ts_str)

    with tab2:
        df_hist = load_alerts(limit=200, active_only=False)
        if df_hist.empty:
            st.info("Sem histórico de alertas.")
        else:
            # Distribuição por severidade
            sev_counts = df_hist["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severidade", "Qtd"]
            fig = px.bar(
                sev_counts, x="Severidade", y="Qtd",
                color="Severidade",
                color_discrete_map={
                    "INFO": "#38bdf8", "ATENCAO": "#f59e0b",
                    "ALERTA": "#ef4444", "EMERGENCIA": "#7c3aed",
                },
                title="Alertas por severidade",
            )
            fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                              font={"color": "#e2e8f0"}, height=280,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            disp = df_hist[["ts", "severity", "alert_type",
                            "location", "level_m", "resolved"]].copy()
            if not disp.empty and hasattr(disp["ts"].iloc[0], "strftime"):
                disp["ts"] = disp["ts"].dt.strftime("%d/%m %H:%M")
            disp.columns = ["Hora", "Severidade", "Tipo", "Local", "Nível (m)", "Resolvido"]
            st.dataframe(disp, use_container_width=True, hide_index=True)


def _page_system() -> None:
    """Página de status do sistema e métricas operacionais."""
    st.markdown("## ⚙️ Status do Sistema")

    stats = load_system_stats()
    db_exists = Path(_DB_PATH).exists()

    c1, c2, c3 = st.columns(3)
    c1.metric("Banco de dados", "✅ Online" if db_exists else "❌ Offline",
              f"{Path(_DB_PATH).stat().st_size / 1024 / 1024:.1f} MB" if db_exists else "")
    c2.metric("Estações ativas", stats.get("stations", 0))
    c3.metric("Leituras chuva", f"{stats.get('rain_readings', 0):,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Níveis de rios", f"{stats.get('river_levels', 0):,}")
    c5.metric("Previsões NWP",  f"{stats.get('forecasts', 0):,}")
    c6.metric("Alertas emitidos", stats.get("alerts_log", 0))

    st.markdown("---")
    st.markdown("### 🗄️ Banco de dados")
    st.code(f"Caminho: {_DB_PATH}", language="text")

    st.markdown("### 🔗 APIs integradas")
    apis = {
        "ANA HidroWeb":   "Coleta histórica de rios RS",
        "INMET":          "500 estações automáticas RS (requer IP brasileiro)",
        "Open-Meteo":     "Previsões NOAA/GFS + ECMWF 7 dias",
        "GPM IMERG NASA": "Precipitação global 30min",
        "GOES-16 AWS":    "Satélite infravermelho",
        "Telegram Bot":   "Alertas automáticos",
    }
    for api, descr in apis.items():
        st.markdown(f"- **{api}**: {descr}")

    st.markdown("### 🔄 Comandos de atualização")
    st.code(
        "# Coletar dados ANA\n"
        "python -m collectors.ana_collector\n\n"
        "# Calcular acumulados\n"
        "python -m processors.rain_accumulator\n\n"
        "# API REST\n"
        "python -m uvicorn api.main:app --port 8765 --app-dir src\n\n"
        "# Alertas Telegram\n"
        "python -m alerts.telegram_bot",
        language="bash",
    )


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point do dashboard."""
    # Auto-refresh
    if _AUTOREFRESH_OK:
        count = st_autorefresh(interval=_REFRESH_MS, key="main_refresh")
    else:
        count = 0

    # CSS
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🌧️ Monitor RS")
        st.markdown("Plataforma hidrometeorológica operacional para o **Rio Grande do Sul**.")
        st.markdown("---")

        page = st.radio(
            "Navegação",
            ["🗺️ Visão Geral", "🌊 Rios", "🌧️ Chuva",
             "🔮 Previsões", "🚨 Alertas", "⚙️ Sistema"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.caption(f"🕐 {now}")
        if _AUTOREFRESH_OK:
            st.caption(f"🔄 Auto-refresh: 10 min (#{count})")
        else:
            st.caption("💡 `pip install streamlit-autorefresh`")
            if st.button("🔄 Atualizar agora"):
                st.cache_data.clear()
                st.rerun()

        st.markdown("---")
        db_ok = Path(_DB_PATH).exists()
        st.caption(f"DB: {'✅' if db_ok else '❌'} {'encontrado' if db_ok else 'não encontrado'}")

    # Carregamento de dados comuns (com cache) — GitHub Raw → local fallback
    @st.cache_data(ttl=600)
    def _cached_river_status() -> pd.DataFrame:
        return load_river_status()

    @st.cache_data(ttl=600)
    def _cached_rain() -> tuple:
        return load_rain_accumulated()

    @st.cache_data(ttl=3600)
    def _cached_stations() -> pd.DataFrame:
        return load_stations()

    river_status_df     = _cached_river_status()
    rain_result         = _cached_rain()
    # load_rain_accumulated retorna (df, DataSource) se loader disponível
    if isinstance(rain_result, tuple):
        rain_df, rain_src = rain_result
    else:
        rain_df, rain_src = rain_result, None
    stations_df = _cached_stations()

    # Badge de fonte dos dados na sidebar
    with st.sidebar:
        st.markdown("---")
        if rain_src is not None:
            color  = rain_src.badge_color
            label  = rain_src.label
            st.markdown(
                f"<div style='font-size:0.78em;padding:4px 8px;border-radius:6px;"
                f"background:{color}22;border:1px solid {color};color:{color};"
                f"text-align:center'>{label}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("📦 Fonte: banco local")

    # Alertas ativos na sidebar
    df_act = load_alerts(limit=5, active_only=True)
    if not df_act.empty:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 🚨 {len(df_act)} alerta(s) ativo(s)")
            for _, a in df_act.iterrows():
                sev = a.get("severity", "")
                st.markdown(f"- **{sev}** {a.get('alert_type', '')}")

    # Roteamento
    if page.startswith("🗺️"):
        _page_overview(river_status_df, rain_df, stations_df)
    elif page.startswith("🌊"):
        _page_rivers(river_status_df)
    elif page.startswith("🌧️"):
        _page_rain(rain_df)
    elif page.startswith("🔮"):
        _page_forecasts()
    elif page.startswith("🚨"):
        _page_alerts()
    elif page.startswith("⚙️"):
        _page_system()


if __name__ == "__main__":
    main()
