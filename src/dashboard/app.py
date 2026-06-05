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

# folium + streamlit-folium
try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

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


def load_rain_accumulated() -> pd.DataFrame:
    """Carrega acumulados de chuva mais recentes por estação."""
    return _query("""
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


def load_forecasts(location: str = "Porto Alegre") -> pd.DataFrame:
    """Carrega previsões NWP para uma localidade."""
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

    # Anexar coordenadas dos NWP_POINTS
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
            fig.add_trace(go.Bar(
                name=label,
                x=top["station_name"].fillna(top["station_id"]),
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
    """Gráfico de previsão NWP: chuva + temperatura nas próximas 7 dias."""
    if df.empty:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Chuva prevista (mm/h)", "Temperatura (°C)"),
                        vertical_spacing=0.08)

    for model, grp in df.groupby("model_source"):
        color = {"openmeteo": "#38bdf8", "noaa": "#f59e0b",
                 "ecmwf": "#a78bfa"}.get(model, "#94a3b8")
        if "rain_mm" in grp.columns:
            fig.add_trace(go.Bar(
                x=grp["valid_ts"], y=grp["rain_mm"].clip(lower=0),
                name=f"Chuva {model}", marker_color=color, opacity=0.75,
            ), row=1, col=1)
        if "temperature" in grp.columns:
            fig.add_trace(go.Scatter(
                x=grp["valid_ts"], y=grp["temperature"],
                name=f"Temp {model}", mode="lines",
                line={"color": color, "width": 2},
            ), row=2, col=1)

    fig.update_layout(
        title=f"Previsão NWP — {location}",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0"},
        legend={"bgcolor": "#1e293b"},
        hovermode="x unified",
        height=450,
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )
    fig.update_xaxes(gridcolor="#334155", linecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", linecolor="#334155")
    return fig


def _build_map(stations_df: pd.DataFrame,
               rain_df: pd.DataFrame,
               river_status_df: pd.DataFrame) -> Optional["folium.Map"]:
    """Mapa Folium com estações, bolhas de chuva e segmentos de rio coloridos."""
    if not _FOLIUM_OK:
        return None

    m = folium.Map(
        location=[-29.8, -51.5],
        zoom_start=8,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # Camada de chuva (bolhas proporcionais a rain_24h)
    rain_layer = folium.FeatureGroup(name="Chuva 24h (bolhas)", show=True)
    if not rain_df.empty:
        for _, row in rain_df.iterrows():
            r24 = row.get("rain_24h", 0) or 0
            if r24 <= 0 or pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            color = ("#38bdf8" if r24 < 20 else
                     "#f59e0b" if r24 < 50 else
                     "#ef4444" if r24 < 100 else "#7c3aed")
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=max(4, min(30, r24 / 5)),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>{row.get('station_name', row['station_id'])}</b><br>"
                    f"Chuva 24h: {r24:.1f} mm<br>"
                    f"1h: {row.get('rain_1h', 0):.1f} | "
                    f"6h: {row.get('rain_6h', 0):.1f} | "
                    f"72h: {row.get('rain_72h', 0):.1f} mm",
                    max_width=200,
                ),
                tooltip=f"{row.get('station_name', row['station_id'])}: {r24:.1f} mm/24h",
            ).add_to(rain_layer)
    rain_layer.add_to(m)

    # Camada de rios (marcadores coloridos por status)
    rivers_layer = folium.FeatureGroup(name="Status dos Rios", show=True)
    if not river_status_df.empty:
        for _, row in river_status_df.iterrows():
            river = row.get("river", "")
            cfg = RIOS_COTAS.get(river)
            if cfg is None:
                continue
            status = row.get("status", "NORMAL")
            color = STATUS_COLORS.get(status, "#22c55e")
            level = row.get("level_m", 0) or 0
            pct   = row.get("pct_cota_atencao", 0) or 0
            folium.Marker(
                location=[cfg["lat"], cfg["lon"]],
                icon=folium.Icon(color={
                    "NORMAL": "green", "ATENCAO": "orange",
                    "ALERTA": "red",   "EMERGENCIA": "purple",
                }.get(status, "green"), icon="tint", prefix="fa"),
                popup=folium.Popup(
                    f"<b>Rio {river}</b><br>"
                    f"Nível: {level:.2f} m<br>"
                    f"Status: {status}<br>"
                    f"% cota atenção: {pct:.0f}%",
                    max_width=200,
                ),
                tooltip=f"Rio {river}: {level:.2f} m ({status})",
            ).add_to(rivers_layer)
    rivers_layer.add_to(m)

    # Camada de estações INMET
    inmet_layer = folium.FeatureGroup(name="Estações INMET", show=False)
    if not stations_df.empty:
        inmet = stations_df[stations_df["source"] == "INMET"]
        for _, row in inmet.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color="#94a3b8",
                fill=True,
                fill_color="#94a3b8",
                fill_opacity=0.5,
                tooltip=f"INMET: {row.get('name', row['station_id'])}",
            ).add_to(inmet_layer)
    inmet_layer.add_to(m)

    folium.LayerControl().add_to(m)
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
    """Página principal: cartões de status + mapa."""
    st.markdown("## 🗺️ Visão Geral")

    # --- Cartões de status dos rios ---
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

    # --- Mapa ---
    if _FOLIUM_OK:
        m = _build_map(stations_df, rain_df, river_status_df)
        if m is not None:
            with st.container():
                st_folium(m, use_container_width=True, height=520,
                          returned_objects=[])
    else:
        st.warning("Instale `streamlit-folium` para ver o mapa: `pip install streamlit-folium`")

    # --- Resumo de chuva ---
    if not rain_df.empty:
        st.markdown("### 🌧️ Maiores acumulados 24h")
        top5 = rain_df.nlargest(5, "rain_24h")[
            ["station_name", "municipality", "river",
             "rain_1h", "rain_6h", "rain_24h", "rain_72h"]
        ].rename(columns={
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
        disp = rain_df[["station_name", "municipality", "river",
                         "rain_1h", "rain_3h", "rain_6h",
                         "rain_24h", "rain_48h", "rain_72h"]].copy()
        disp.columns = ["Estação", "Município", "Rio",
                        "1h", "3h", "6h", "24h", "48h", "72h"]
        st.dataframe(disp, use_container_width=True, hide_index=True)


def _page_forecasts() -> None:
    """Página de previsões NWP com mapa interativo estilo Windy + gráfico série temporal."""
    st.markdown("## 🔮 Previsões Meteorológicas")

    # Dropdown com todos os 10 pontos NWP do config.yaml
    all_locations = [p["nome"] for p in NWP_POINTS]
    loc = st.selectbox("Localidade", all_locations, index=0)

    # Carrega resumo de todos os pontos (para o mapa) e série do ponto selecionado
    @st.cache_data(ttl=600)
    def _cached_summary() -> pd.DataFrame:
        return load_all_forecasts_summary()

    summary_df = _cached_summary()
    df_fc      = load_forecasts(loc)

    # ── Métricas de instabilidade (ponto selecionado) ──────────────────────
    if not df_fc.empty:
        latest = df_fc.iloc[0]
        cape   = latest.get("cape_j_kg")
        li     = latest.get("lifted_index")
        ki     = latest.get("k_index")
        t0     = df_fc["valid_ts"].iloc[0]
        rain_6h  = df_fc[df_fc["valid_ts"] <= t0 + timedelta(hours=6)]["rain_mm"].sum()
        rain_24h = df_fc[df_fc["valid_ts"] <= t0 + timedelta(hours=24)]["rain_mm"].sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CAPE (J/kg)",    f"{cape:.0f}"  if cape is not None else "N/D",
                  help="< 1000 fraco | 1000-2500 moderado | > 2500 severo")
        c2.metric("Lifted Index",   f"{li:.1f}"    if li   is not None else "N/D",
                  help="> 0 estável | -2 a 0 leve | < -6 severo")
        c3.metric("K-Index",        f"{ki:.0f}"    if ki   is not None else "N/D",
                  help="< 20 baixo | 20-30 moderado | > 35 alto")
        c4.metric("Chuva próx. 6h",  f"{rain_6h:.1f} mm"  if rain_6h  else "N/D")
        c5.metric("Chuva próx. 24h", f"{rain_24h:.1f} mm" if rain_24h else "N/D")
    else:
        st.info(f"Sem previsões para {loc}. Execute o coletor Open-Meteo/NOAA.")

    st.markdown("---")

    # ── Mapa Windy de previsão ─────────────────────────────────────────────
    st.markdown("### 🗺️ Chuva prevista 24h — grade RS")

    if _FOLIUM_OK:
        fc_map = _build_forecast_map(summary_df, loc)
        if fc_map is not None:
            st_folium(fc_map, use_container_width=True, height=500,
                      returned_objects=[])
        else:
            st.info("Mapa não disponível.")
    else:
        st.warning("Instale `streamlit-folium`: `pip install streamlit-folium`")

        # Fallback: tabela resumo quando folium não disponível
        if not summary_df.empty:
            disp_sum = summary_df[["location_name", "rain_6h",
                                   "rain_24h", "rain_48h", "temp_mean"]].copy()
            disp_sum.columns = ["Cidade", "6h (mm)", "24h (mm)", "48h (mm)", "Temp °C"]
            st.dataframe(disp_sum, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Gráfico série temporal (ponto selecionado) ────────────────────────
    if not df_fc.empty:
        st.markdown(f"### 📈 Série temporal — {loc}")
        fig = _forecast_chart(df_fc, loc)
        st.plotly_chart(fig, use_container_width=True)

        # Tabela resumida
        with st.expander("Ver dados tabulares"):
            disp = df_fc[["valid_ts", "rain_mm", "temperature",
                          "wind_speed", "cape_j_kg", "model_source"]].copy()
            disp["valid_ts"] = disp["valid_ts"].dt.strftime("%d/%m %H:%M")
            disp.columns = ["Data/Hora", "Chuva (mm)", "Temp (°C)",
                            "Vento (km/h)", "CAPE (J/kg)", "Modelo"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Resumo comparativo dos 10 pontos ─────────────────────────────────
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

    # Carregamento de dados comuns (com cache)
    @st.cache_data(ttl=600)
    def _cached_river_status() -> pd.DataFrame:
        return load_river_status()

    @st.cache_data(ttl=600)
    def _cached_rain() -> pd.DataFrame:
        return load_rain_accumulated()

    @st.cache_data(ttl=3600)
    def _cached_stations() -> pd.DataFrame:
        return load_stations()

    river_status_df = _cached_river_status()
    rain_df         = _cached_rain()
    stations_df     = _cached_stations()

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
