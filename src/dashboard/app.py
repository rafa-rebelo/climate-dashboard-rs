"""
Agente 4 — Especialista em Visualização + Designer de UX/UI (missão crítica)
Dashboard do Monitor Hidrometeorológico RS — 100% STATELESS.

Consome exclusivamente a API FastAPI v3 (Supabase + R2). Zero DuckDB, zero
leitura local. Pensado para deploy stateless (Streamlit Cloud) após o Render.

Seções:
  1. Mapa de Estações (REDEMET) — pin + painel lateral + série; camadas chuva/temp
  2. Mapa Climático em Camadas (Windy) — camadas alternáveis + timeline + legenda
  3. Dados Climáticos (Windy) — mapa por variável (obs+prev) + legenda por camada
  4. Sobre o Modelo ML — transparência didática (MAE, status, IC explicados)

Config: API_BASE_URL (env). Auto-refresh 10 min. Tema dark Slate. Mobile 768px.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_OK = True
except ImportError:
    _AUTOREFRESH_OK = False

st.set_page_config(
    page_title="Monitor Hidrometeorológico RS",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
API_BASE: str = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
_API_IS_LOCAL: bool = ("localhost" in API_BASE) or ("127.0.0.1" in API_BASE)
_TTL = 600
_REFRESH_MS = 10 * 60 * 1000
_RS_CENTER = {"lat": -30.0, "lon": -51.5}   # centro estilo Windy (link de referência)
_RS_ZOOM = 5.4

STATUS_COLORS: dict[str, str] = {
    "NORMAL":     "rgba(34,197,94,1)",
    "ATENCAO":    "rgba(245,158,11,1)",
    "ALERTA":     "rgba(239,68,68,1)",
    "EMERGENCIA": "rgba(124,58,237,1)",
}
STATUS_EMOJI = {"NORMAL": "🟢", "ATENCAO": "🟡", "ALERTA": "🔴", "EMERGENCIA": "🟣"}

# ── Escalas de cor padrão por variável (estilo Windy) ──────────────────────
SCALE_PRECIP = [
    [0.0, "rgba(120,124,140,0.35)"], [0.08, "rgba(56,189,248,0.85)"],
    [0.30, "rgba(37,99,235,0.95)"], [0.55, "rgba(34,197,94,0.95)"],
    [0.78, "rgba(245,158,11,0.97)"], [1.0, "rgba(220,38,38,1)"],
]
SCALE_TEMP = [
    [0.0, "rgba(49,54,149,1)"], [0.25, "rgba(69,170,220,1)"],
    [0.50, "rgba(34,197,94,1)"], [0.70, "rgba(245,200,66,1)"],
    [0.85, "rgba(245,130,48,1)"], [1.0, "rgba(220,38,38,1)"],
]
SCALE_WIND = [
    [0.0, "rgba(34,197,94,1)"], [0.4, "rgba(245,200,66,1)"],
    [0.7, "rgba(245,130,48,1)"], [0.9, "rgba(220,38,38,1)"], [1.0, "rgba(147,51,234,1)"],
]

# Legenda discreta (chips) por variável — conectada à camada selecionada
LEGEND_PRECIP = [("rgba(120,124,140,0.6)", "0"), ("rgba(56,189,248,0.9)", "<5"),
                 ("rgba(37,99,235,1)", "5–20"), ("rgba(34,197,94,1)", "20–40"),
                 ("rgba(245,158,11,1)", "40–60"), ("rgba(220,38,38,1)", ">60")]
LEGEND_TEMP = [("rgba(49,54,149,1)", "<10"), ("rgba(69,170,220,1)", "10–18"),
               ("rgba(34,197,94,1)", "18–24"), ("rgba(245,200,66,1)", "24–30"),
               ("rgba(245,130,48,1)", "30–35"), ("rgba(220,38,38,1)", ">35")]
LEGEND_WIND = [("rgba(34,197,94,1)", "<20"), ("rgba(245,200,66,1)", "20–40"),
               ("rgba(245,130,48,1)", "40–60"), ("rgba(220,38,38,1)", "60–80"),
               ("rgba(147,51,234,1)", ">80")]

MODEL_STATUS = {
    "aprovado":   ("rgba(34,197,94,1)",  "✅ Aprovado"),
    "calibrando": ("rgba(245,158,11,1)", "🟡 Em calibração"),
    "pendente":   ("rgba(148,163,184,1)", "⏳ Pendente"),
    "ausente":    ("rgba(100,116,139,1)", "— Sem modelo"),
}

DARK_CSS = """
<style>
.stApp { background-color: #0f172a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background-color: #0d1b2a; border-right: 1px solid #1e293b; }
div[data-testid="stMetric"] { background-color: #1e293b; border-radius: 10px; padding: 12px; }
.card { background: #1e293b; border-radius: 12px; padding: 14px 16px; margin: 6px 0;
        border-left: 4px solid #334155; color: #cbd5e1; font-size: 0.9em; }
.card b { color: #f8fafc; }
.muted { color: #64748b; font-size: 0.8em; }
.bar-wrap { background: #334155; border-radius: 4px; height: 6px; margin-top: 8px; overflow: hidden; }
.bar-fill { height: 6px; border-radius: 4px; transition: width 0.4s; }
.stTabs [data-baseweb="tab-list"] { background: #0d1b2a; border-bottom: 1px solid #1e293b; gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; font-size: 0.9em; padding: 6px 16px; }
.stTabs [aria-selected="true"] { background: #1e293b; color: #38bdf8; border-bottom: 2px solid #38bdf8; }
.warn-box { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.5);
            border-radius: 8px; padding: 10px 14px; color: #fcd34d; font-size: 0.85em; margin: 8px 0; }
.info-box { background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.4);
            border-radius: 8px; padding: 10px 14px; color: #bae6fd; font-size: 0.85em; margin: 8px 0; }
.legend-bar { background: rgba(13,27,42,0.92); border: 1px solid #334155; border-radius: 8px;
              padding: 7px 12px; margin: 0 0 -42px 12px; position: relative; z-index: 500;
              display: inline-block; font-size: 0.78em; color: #e2e8f0; }
.legend-chip { display: inline-flex; align-items: center; gap: 4px; margin-right: 10px; }
.legend-dot { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
@media (max-width: 768px) { .block-container { padding: 0.5rem; } }
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(15,23,42,1)",
    plot_bgcolor="rgba(30,41,59,1)",
    font={"color": "rgba(226,232,240,1)"},
    margin={"t": 40, "b": 30, "l": 50, "r": 20},
)


# ---------------------------------------------------------------------------
# Camada HTTP
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_TTL, show_spinner=False)
def _api_get_cached(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Busca crua na API v3. LEVANTA em erro de propósito.

    O st.cache_data só memoiza o retorno; quando esta função levanta, nada é
    cacheado. Assim o caminho de erro fica FORA do cache (ver api_get).
    """
    resp = httpx.get(f"{API_BASE}{path}", params=params or {}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET na API FastAPI v3: sucesso cacheado 10 min, erro NÃO cacheado.

    Cachear a falha faria o dashboard ficar "offline" por até 10 min mesmo
    depois de a API voltar — e o F5 do navegador não limpa o cache do Streamlit
    (ele vive no processo do servidor). Deixando o erro fora do cache, o próximo
    auto-refresh/rerun já reconecta assim que a API responde.

    Args:
        path:   Caminho do endpoint.
        params: Query params opcionais.

    Returns:
        Corpo JSON; em erro, dict com chave "_erro" (não cacheado).
    """
    url = f"{API_BASE}{path}"
    try:
        return _api_get_cached(path, params)
    except httpx.HTTPStatusError as exc:
        return {"_erro": f"HTTP {exc.response.status_code}", "_url": url}
    except httpx.HTTPError as exc:
        return {"_erro": str(exc), "_url": url}


def _ok(p: dict[str, Any]) -> bool:
    return "_erro" not in p


def _fmt(v: Any, suf: str = "", casas: int = 1, fb: str = "—") -> str:
    try:
        if v is None or pd.isna(v):
            return fb
        return f"{float(v):.{casas}f}{suf}"
    except (TypeError, ValueError):
        return fb


def _fmt_ts(iso: str | None, fmt: str = "%d/%m %H:%M") -> str:
    if not iso:
        return "—"
    try:
        return pd.to_datetime(iso).strftime(fmt)
    except (ValueError, TypeError):
        return str(iso)


def _legend_html(titulo: str, bins: list[tuple[str, str]], unidade: str) -> str:
    """Monta a barra de legenda (chips) que fica sobre o mapa, por camada."""
    chips = "".join(
        f"<span class='legend-chip'><span class='legend-dot' style='background:{c}'></span>{lab}</span>"
        for c, lab in bins
    )
    return (f"<div class='legend-bar'><b style='color:#f8fafc'>{titulo}</b> "
            f"<span class='muted'>({unidade})</span><br>{chips}</div>")


def _mapa_pontos(df: pd.DataFrame, value_col: str, scale: list, unidade: str,
                 cmin: float, cmax: float, hover_nome: str,
                 modo: str = "scatter", size: int = 16) -> go.Figure:
    """Mapa estilo Windy (dark) colorido por uma variável; colorbar = legenda na tela.

    Args:
        df:        DataFrame com colunas 'lat'/'lon' (ou latitude/longitude) + value_col.
        value_col: Coluna numérica a colorir.
        scale:     Colorscale plotly.
        unidade:   Rótulo da unidade (mm, °C, km/h).
        cmin/cmax: Limites da escala.
        hover_nome: Coluna de rótulo no hover.
        modo:      'scatter' (pontos) ou 'density' (heatmap raster, p/ grade densa).
        size:      Tamanho do marcador (modo scatter).

    Returns:
        Figura Plotly Scattermapbox/Densitymapbox.
    """
    lat = "lat" if "lat" in df.columns else "latitude"
    lon = "lon" if "lon" in df.columns else "longitude"
    if modo == "density":
        fig = go.Figure(go.Densitymapbox(
            lat=df[lat], lon=df[lon], z=df[value_col].fillna(0).clip(lower=0),
            radius=18, colorscale=scale, zmin=cmin, zmax=cmax,
            colorbar={"title": unidade, "thickness": 14, "len": 0.85,
                      "bgcolor": "rgba(13,27,42,0.6)"},
        ))
    else:
        fig = go.Figure(go.Scattermapbox(
            lat=df[lat], lon=df[lon], mode="markers",
            marker={"size": size, "opacity": 0.82,
                    "color": df[value_col].fillna(0), "colorscale": scale,
                    "cmin": cmin, "cmax": cmax,
                    "colorbar": {"title": unidade, "thickness": 14, "len": 0.85,
                                 "bgcolor": "rgba(13,27,42,0.6)"}},
            text=df[hover_nome] if hover_nome in df.columns else None,
            customdata=df[[value_col]].values,
            hovertemplate="<b>%{text}</b><br>%{customdata[0]:.1f} " + unidade + "<extra></extra>",
        ))
    fig.update_layout(
        mapbox={"style": "carto-darkmatter", "center": _RS_CENTER, "zoom": _RS_ZOOM},
        height=560, margin={"t": 0, "b": 0, "l": 0, "r": 0},
        paper_bgcolor="rgba(15,23,42,1)",
    )
    return fig


# ---------------------------------------------------------------------------
# SEÇÃO 1 — Mapa de Estações (REDEMET) com camadas chuva/temperatura
# ---------------------------------------------------------------------------

def secao_estacoes() -> None:
    """Mapa de estações colorido por camada selecionável; clique abre painel."""
    st.subheader("📍 Estações de Monitoramento — INMET/ANA")

    data = api_get("/api/v3/stations/readings")
    if not _ok(data):
        st.error(f"API indisponível para estações: {data['_erro']}")
        return
    df = pd.DataFrame(data.get("estacoes", []))
    if not df.empty:
        df = df[df["latitude"].notna() & df["longitude"].notna()]
    if df.empty:
        st.warning("Sem estações com coordenadas no momento.")
        return

    # Data do dado: a leitura mais recente entre as estações (sempre a atual)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    data_dado = ts.max()
    data_txt = data_dado.strftime("%d/%m/%Y %H:%M UTC") if pd.notna(data_dado) else "—"

    camadas = {
        "🌧️ Precipitação 24h": ("precip_24h", "mm", SCALE_PRECIP, LEGEND_PRECIP, 0.0,
                                 max(20.0, float(df["precip_24h"].fillna(0).max() or 20))),
        "🌡️ Temperatura": ("temperatura", "°C", SCALE_TEMP, LEGEND_TEMP,
                            float(df["temperatura"].dropna().min() or 0) if df["temperatura"].notna().any() else 0.0,
                            float(df["temperatura"].dropna().max() or 35) if df["temperatura"].notna().any() else 35.0),
    }
    c_sel, c_data = st.columns([3, 2])
    with c_sel:
        nome_cam = st.radio("Camada", list(camadas.keys()), horizontal=True, key="est_layer")
    with c_data:
        st.caption(f"📅 Dado mais recente: **{data_txt}**")

    col, unidade, scale, legenda, cmin, cmax = camadas[nome_cam]

    col_map, col_panel = st.columns([3, 2], gap="medium")
    with col_map:
        st.markdown(_legend_html(nome_cam, legenda, unidade), unsafe_allow_html=True)
        fig = _mapa_pontos(df, col, scale, unidade, cmin, cmax, "nome", modo="scatter", size=13)
        # rótulo de hover precisa do nome
        fig.data[0].text = df["nome"].fillna(df["station_id"])
        fig.data[0].customdata = df[[col, "station_id"]].values
        fig.data[0].hovertemplate = ("<b>%{text}</b><br>%{customdata[0]:.1f} " + unidade
                                     + "<extra></extra>")
        evento = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                                 key="map_estacoes")

    sel_id = None
    pontos = (evento.get("selection", {}) or {}).get("points", []) if evento else []
    if pontos and pontos[0].get("customdata"):
        cd = pontos[0]["customdata"]
        sel_id = str(cd[1]) if len(cd) > 1 else None

    with col_panel:
        nomes = df.sort_values("station_id")["station_id"].tolist()
        labels = {r.station_id: (r.nome or r.station_id) for r in df.itertuples()}
        if sel_id is None:
            sel_id = st.selectbox("Estação (clique no mapa ou selecione)", nomes,
                                  format_func=lambda s: labels.get(s, s), key="sel_estacao")
        else:
            st.caption(f"Selecionada no mapa: **{labels.get(sel_id, sel_id)}**")
        _painel_estacao(df, sel_id)


def _painel_estacao(df: pd.DataFrame, station_id: str) -> None:
    """Painel lateral de uma estação + série recente."""
    linha = df[df["station_id"] == station_id]
    if linha.empty:
        st.info("Estação sem leitura recente.")
        return
    r = linha.iloc[0]
    st.markdown(
        f"<div class='card'><b>{r.get('nome') or station_id}</b><br>"
        f"<span class='muted'>{r.get('municipio') or ''} · {station_id}</span><br>"
        f"🌡️ {_fmt(r.get('temperatura'), ' °C')} &nbsp; 💧 {_fmt(r.get('umidade'), ' %', 0)}<br>"
        f"🌧️ 1h {_fmt(r.get('precip_1h'), ' mm')} · 3h {_fmt(r.get('precip_3h'), ' mm')} · "
        f"6h {_fmt(r.get('precip_6h'), ' mm')} · 12h {_fmt(r.get('precip_12h'), ' mm')} · "
        f"24h {_fmt(r.get('precip_24h'), ' mm')}<br>"
        f"<span class='muted'>&nbsp;&nbsp;&nbsp;&nbsp;acum. 48h {_fmt(r.get('precip_48h'), ' mm')} · "
        f"72h {_fmt(r.get('precip_72h'), ' mm')} · 7d {_fmt(r.get('precip_7d'), ' mm')}</span><br>"
        f"<span class='muted'>🕐 leitura: {_fmt_ts(r.get('timestamp'))}</span></div>",
        unsafe_allow_html=True,
    )
    # days=3: cada Parquet do INMET já carrega ~30 dias de série, então uma
    # janela curta basta para recuperá-la; 3 dá folga se o run de hoje ainda
    # não gravou. A defasagem da fonte é sinalizada abaixo, não escondida.
    hist = api_get("/api/v3/analytics/history",
                   {"days": 3, "data_source": "inmet", "station_id": station_id})
    serie = _serie_estacao(hist, station_id)
    if serie.empty:
        st.caption("Sem série histórica recente para esta estação.")
        return
    fig = go.Figure()
    if "precip_1h" in serie:
        fig.add_trace(go.Bar(x=serie["ts"], y=serie["precip_1h"], name="Chuva 1h",
                             marker_color="rgba(56,189,248,0.75)", yaxis="y1"))
    if "temperatura" in serie:
        fig.add_trace(go.Scatter(x=serie["ts"], y=serie["temperatura"], name="Temp",
                                 mode="lines", line={"color": "rgba(251,146,60,1)", "width": 2},
                                 yaxis="y2"))
    fig.update_layout(**PLOTLY_LAYOUT, height=260, title="Série recente",
                      legend={"bgcolor": "rgba(30,41,59,0.5)", "orientation": "h", "y": 1.15},
                      yaxis={"title": "mm", "gridcolor": "rgba(51,65,85,1)"},
                      yaxis2={"title": "°C", "overlaying": "y", "side": "right", "showgrid": False},
                      xaxis={"gridcolor": "rgba(51,65,85,1)"})
    st.plotly_chart(fig, use_container_width=True)

    # Coerência com o lag da fonte: sinaliza quando a última leitura é antiga.
    # O INMET (ZIP público) tem defasagem estrutural (~semanas); o CEMADEN
    # (prefixo CEM_) é ~10 min. Assim o usuário não lê dado velho como atual.
    ult = serie["ts"].max()
    if pd.notna(ult):
        idade_h = (pd.Timestamp.now(tz="UTC") - ult).total_seconds() / 3600
        if idade_h > 24:
            st.caption(
                f"⏳ Última leitura há ~{idade_h / 24:.0f} dia(s). A fonte INMET "
                "(ZIP público) é defasada por natureza; para chuva quase em tempo "
                "real, consulte as estações **CEMADEN** (prefixo CEM_)."
            )


def _serie_estacao(hist: dict[str, Any], station_id: str) -> pd.DataFrame:
    if not _ok(hist):
        return pd.DataFrame()
    regs: list[dict] = []
    for ponto in hist.get("historico", []):
        regs.extend(ponto.get("dados", []))
    if not regs:
        return pd.DataFrame()
    df = pd.DataFrame(regs)
    if "station_id" in df.columns:
        df = df[df["station_id"].astype(str) == str(station_id)]
    tsc = next((c for c in ("timestamp", "ts", "date_utc") if c in df.columns), None)
    if df.empty or tsc is None:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df[tsc], errors="coerce", utc=True)
    # Dedup: o history concatena vários Parquets (1/run) da mesma partição-dia,
    # cada um com a série de 30 dias — sem isso o mesmo ts se repete e o gráfico
    # sobrepõe pontos idênticos.
    return (df.dropna(subset=["ts"])
            .drop_duplicates(subset="ts", keep="last")
            .sort_values("ts"))


# ---------------------------------------------------------------------------
# Helpers de previsão NWP (seções 2 e 3)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_TTL, show_spinner=False)
def _nwp_df() -> pd.DataFrame:
    """Carrega a previsão NWP (7 dias) como DataFrame normalizado."""
    fc = api_get("/api/v3/forecasts/weather", {"horas": 168})
    if not _ok(fc) or not fc.get("previsoes"):
        return pd.DataFrame()
    df = pd.DataFrame(fc["previsoes"])
    df["valid_ts"] = pd.to_datetime(df["valid_ts"], errors="coerce")
    return df.dropna(subset=["valid_ts", "lat", "lon"])


def _gpm_df() -> pd.DataFrame:
    data = api_get("/api/v3/weather/heatmap")
    if not _ok(data) or not data.get("pontos"):
        return pd.DataFrame()
    return pd.DataFrame(data["pontos"]).dropna(subset=["lat", "lon"])


# ---------------------------------------------------------------------------
# SEÇÃO 2 — Mapa Climático em Camadas (Windy)
# ---------------------------------------------------------------------------

def secao_mapa_climatico() -> None:
    """Mapa Windy-style: camada satélite/prevista + timeline + legenda por camada."""
    st.subheader("🌐 Mapa Climático em Camadas")
    st.markdown("<div class='muted'>Estilo Windy — selecione a camada; a legenda "
                "sobre o mapa muda junto.</div>", unsafe_allow_html=True)

    camada = st.radio("Camada", ["🛰️ Chuva (satélite agora)", "🌧️ Chuva prevista",
                                 "🌡️ Temperatura prevista", "💨 Vento previsto"],
                      horizontal=True, key="windy_layer")

    if camada.startswith("🛰️"):
        gpm = _gpm_df()
        if gpm.empty:
            st.warning("Sem grade GPM disponível na API.")
            return
        pmax = max(5.0, float(gpm["precip_mm"].fillna(0).quantile(0.97) or 5))
        st.markdown(_legend_html("Chuva satélite (GPM IMERG)", LEGEND_PRECIP, "mm"),
                    unsafe_allow_html=True)
        fig = _mapa_pontos(gpm, "precip_mm", SCALE_PRECIP, "mm", 0.0, pmax,
                           "", modo="density")
        st.plotly_chart(fig, use_container_width=True)
        tot = (gpm["precip_mm"].fillna(0) > 0).sum()
        st.caption(f"GPM IMERG · {len(gpm)} pontos · {tot} com chuva > 0")
        return

    df = _nwp_df()
    if df.empty:
        st.warning("Sem previsão NWP disponível na API (verifique se a API está online).")
        return

    horarios = sorted(df["valid_ts"].unique())
    idx = st.select_slider("🕐 Timeline da previsão (arraste)",
                           options=list(range(len(horarios))), value=0,
                           format_func=lambda i: pd.to_datetime(horarios[i]).strftime("%d/%m %Hh"),
                           key="windy_timeline")
    snap = df[df["valid_ts"] == horarios[idx]]

    cfg = {
        "🌧️ Chuva prevista": ("rain_mm", "mm", SCALE_PRECIP, LEGEND_PRECIP, 0.0,
                              max(10.0, float(df["rain_mm"].fillna(0).quantile(0.95) or 10))),
        "🌡️ Temperatura prevista": ("temperature", "°C", SCALE_TEMP, LEGEND_TEMP,
                                    float(df["temperature"].dropna().min() or 0),
                                    float(df["temperature"].dropna().max() or 35)),
        "💨 Vento previsto": ("wind_speed", "km/h", SCALE_WIND, LEGEND_WIND, 0.0,
                             max(40.0, float(df["wind_speed"].fillna(0).quantile(0.95) or 40))),
    }
    col, unidade, scale, legenda, cmin, cmax = cfg[camada]
    if col not in snap.columns:
        st.warning(f"Variável {col} ausente na previsão.")
        return
    st.markdown(_legend_html(camada, legenda, unidade), unsafe_allow_html=True)
    fig = _mapa_pontos(snap, col, scale, unidade, cmin, cmax, "location_name",
                       modo="scatter", size=30)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Previsão NWP (Open-Meteo) para {pd.to_datetime(horarios[idx]):%d/%m %Hh} UTC · "
               f"{len(snap)} pontos.")


# ---------------------------------------------------------------------------
# SEÇÃO 3 — Dados Climáticos (Windy) — observado + previsto por camada
# ---------------------------------------------------------------------------

def secao_clima() -> None:
    """Mapa Windy-style por variável (obs+prev) + legenda conectada + resumo."""
    st.subheader("📊 Dados Climáticos — Observado e Previsto")
    st.markdown("<div class='muted'>Selecione a variável; o mapa e a legenda mudam juntos. "
                "Observado = estações INMET; previsto = NWP Open-Meteo.</div>",
                unsafe_allow_html=True)

    camada = st.radio("Variável", ["🌧️ Precipitação acumulada 24h", "🌡️ Temperatura",
                                   "💨 Vento previsto", "☀️ Radiação solar"],
                      horizontal=True, key="clima_layer")

    est = api_get("/api/v3/stations/readings")
    df_e = pd.DataFrame(est.get("estacoes", [])) if _ok(est) else pd.DataFrame()
    if not df_e.empty:
        df_e = df_e[df_e["latitude"].notna() & df_e["longitude"].notna()]

    if camada.startswith("☀️"):
        st.markdown("<div class='info-box'>☀️ <b>Radiação solar</b> ainda não é coletada. "
                    "O Open-Meteo expõe <code>shortwave_radiation</code> — é um item de backlog "
                    "(~1h) no <code>noaa_collector.py</code>. Não exibimos camada para não "
                    "plotar dado falso.</div>", unsafe_allow_html=True)
        return

    if camada.startswith("🌧️"):
        if df_e.empty or "precip_24h" not in df_e:
            st.warning("Sem dados de chuva das estações.")
            return
        cmax = max(20.0, float(df_e["precip_24h"].fillna(0).max() or 20))
        st.markdown(_legend_html("Precipitação acumulada 24h", LEGEND_PRECIP, "mm"),
                    unsafe_allow_html=True)
        fig = _mapa_pontos(df_e.rename(columns={"latitude": "lat", "longitude": "lon"}),
                           "precip_24h", SCALE_PRECIP, "mm", 0.0, cmax, "nome", size=13)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Acumulado de 24h por estação INMET (observado).")
        return

    if camada.startswith("🌡️"):
        if df_e.empty or not df_e["temperatura"].notna().any():
            st.warning("Sem dados de temperatura das estações.")
            return
        cmin = float(df_e["temperatura"].dropna().min())
        cmax = float(df_e["temperatura"].dropna().max())
        st.markdown(_legend_html("Temperatura observada", LEGEND_TEMP, "°C"),
                    unsafe_allow_html=True)
        fig = _mapa_pontos(df_e.rename(columns={"latitude": "lat", "longitude": "lon"}),
                           "temperatura", SCALE_TEMP, "°C", cmin, cmax, "nome", size=13)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Temperatura atual por estação INMET (observado).")
        return

    # 💨 Vento previsto (NWP) — média das próximas 24h por ponto
    df = _nwp_df()
    if df.empty:
        st.warning("Sem previsão NWP disponível na API.")
        return
    prox = df[df["valid_ts"] <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=24)]
    if prox.empty:
        prox = df
    agg = (prox.groupby(["location_name", "lat", "lon"])["wind_speed"]
           .mean().reset_index())
    cmax = max(40.0, float(agg["wind_speed"].fillna(0).max() or 40))
    st.markdown(_legend_html("Vento previsto (próx. 24h)", LEGEND_WIND, "km/h"),
                unsafe_allow_html=True)
    fig = _mapa_pontos(agg, "wind_speed", SCALE_WIND, "km/h", 0.0, cmax,
                       "location_name", size=30)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Velocidade média do vento prevista para as próximas 24h (NWP). "
               "Direção não disponível na tabela de previsão.")


# ---------------------------------------------------------------------------
# SEÇÃO Bacias RS — Rede Hidrometeorológica da Defesa Civil RS (DCRS)
# ---------------------------------------------------------------------------

_DCRS_MAPA_OFICIAL = "https://redehidrometeorologica.defesacivil.rs.gov.br/Mapa"

# Variável → (coluna, unidade, colorscale, legenda, cmin_fixo, cmax_minimo)
_DCRS_VARS: dict[str, tuple] = {
    "🌊 Nível do rio":       ("rio_nivel",   "m",    SCALE_WIND,   LEGEND_WIND,   None, None),
    "🌧️ Chuva acumulada":    ("chuva",       "mm",   SCALE_PRECIP, LEGEND_PRECIP, 0.0,  10.0),
    "🌡️ Temperatura":        ("temperatura", "°C",   SCALE_TEMP,   LEGEND_TEMP,   None, None),
    "💧 Umidade":             ("umidade",     "%",    SCALE_PRECIP, LEGEND_PRECIP, 0.0,  100.0),
    "💨 Vento":               ("vento_vel",   "km/h", SCALE_WIND,   LEGEND_WIND,   0.0,  40.0),
    "🥵 Sensação térmica":    ("senstermica", "°C",   SCALE_TEMP,   LEGEND_TEMP,   None, None),
}
_DCRS_JANELAS = {"1h": "chuva_1h", "3h": "chuva_3h", "6h": "chuva_6h",
                 "12h": "chuva_12h", "24h": "chuva_24h", "48h": "chuva_48h",
                 "72h": "chuva_72h", "7d": "chuva_168h"}


def _dcrs_nivel_exibicao(v: Any) -> float | None:
    """Nível para exibição em metros, com heurística de unidade.

    A rede DCRS entrega rio_nivel em unidade mista por estação (lagoas ~0,3 m;
    serra ~860, aparenta cm). Valores >= 30 são tratados como cm e divididos
    por 100 — heurística SINALIZADA na interface, não silenciosa.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return round(f / 100.0, 2) if f >= 30.0 else round(f, 2)


def secao_bacias() -> None:
    """Aba Bacias RS: mapa por bacia da rede DCRS com seletor de variável."""
    st.subheader("🗺️ Bacias Hidrográficas RS — Defesa Civil (tempo real)")
    st.markdown(
        f"<div class='muted'>Rede Hidrometeorológica da Defesa Civil RS "
        f"(estações DCRS, cadência de minutos) · "
        f"<a href='{_DCRS_MAPA_OFICIAL}' target='_blank' "
        f"style='color:#38bdf8'>mapa oficial ↗</a></div>",
        unsafe_allow_html=True,
    )

    data = api_get("/api/v3/dcrs/stations")
    if not _ok(data):
        st.error(f"API indisponível para bacias DCRS: {data['_erro']}")
        return
    df = pd.DataFrame(data.get("estacoes", []))
    if df.empty:
        st.warning("Sem estações DCRS no momento (coletor recém-ativado?).")
        return
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    df["nivel_m"] = df["rio_nivel"].map(_dcrs_nivel_exibicao)

    # ── Seletores: Bacia + Variável + Janela de chuva ─────────────────────
    bacias = sorted(b for b in df["bacia"].dropna().unique() if b)
    c1, c2, c3 = st.columns([2.2, 1.6, 1.0])
    with c1:
        bacia_sel = st.selectbox("Bacia hidrográfica",
                                 ["🌎 Todas as bacias"] + bacias, key="dcrs_bacia")
    with c2:
        var_sel = st.selectbox("Variável", list(_DCRS_VARS.keys()), key="dcrs_var")
    janela_lbl = "24h"
    with c3:
        if var_sel.startswith("🌧️"):
            janela_lbl = st.selectbox("Janela", list(_DCRS_JANELAS.keys()),
                                      index=4, key="dcrs_janela")
        else:
            st.caption("")

    sel = df if bacia_sel.startswith("🌎") else df[df["bacia"] == bacia_sel]
    if sel.empty:
        st.warning("Sem estações nesta bacia.")
        return

    col, unidade, scale, legenda, cmin, cmax_min = _DCRS_VARS[var_sel]
    if col == "chuva":
        col = _DCRS_JANELAS[janela_lbl]
    elif col == "rio_nivel":
        col = "nivel_m"

    vals = pd.to_numeric(sel[col], errors="coerce")
    plot = sel[vals.notna()].copy()
    if plot.empty:
        st.warning(f"Nenhuma estação da seleção mede {var_sel}.")
        return
    vmin = float(vals.min()) if cmin is None else cmin
    vmax = max(float(vals.max()), (cmax_min or float(vals.max()) or 1.0))
    if vmin == vmax:
        vmax = vmin + 1.0

    # Centro/zoom: aproxima quando uma bacia específica é escolhida.
    centro = {"lat": float(plot["latitude"].mean()),
              "lon": float(plot["longitude"].mean())}
    zoom = _RS_ZOOM if bacia_sel.startswith("🌎") else 7.2

    titulo = (f"{var_sel} — {janela_lbl}" if var_sel.startswith("🌧️") else var_sel)
    st.markdown(_legend_html(titulo, legenda, unidade), unsafe_allow_html=True)
    fig = _mapa_pontos(
        plot.rename(columns={"latitude": "lat", "longitude": "lon"}),
        col, scale, unidade, vmin, vmax, "nome", modo="scatter", size=14,
    )
    fig.update_layout(mapbox={"style": "carto-darkmatter",
                              "center": centro, "zoom": zoom})
    fig.data[0].customdata = plot[[col, "codigo"]].values
    evento = st.plotly_chart(fig, use_container_width=True,
                             on_select="rerun", key="map_dcrs")

    if var_sel.startswith("🌊"):
        st.markdown(
            "<div class='info-box'>ℹ️ O nível vem na unidade bruta da rede "
            "(mista por estação). Valores ≥ 30 são exibidos como cm→m "
            "(heurística sinalizada); confirme na estação antes de decisões "
            "operacionais.</div>", unsafe_allow_html=True)

    ts = pd.to_datetime(plot["timestamp"], errors="coerce", utc=True)
    st.caption(
        f"{len(plot)} estações · leitura mais recente "
        f"{ts.max():%d/%m %H:%M} UTC · fonte: Defesa Civil RS"
    )

    # ── Painel da estação (clique ou seleção) ─────────────────────────────
    sel_cod = None
    pontos = (evento.get("selection", {}) or {}).get("points", []) if evento else []
    if pontos and pontos[0].get("customdata") is not None:
        cd = pontos[0]["customdata"]
        sel_cod = str(cd[1]) if len(cd) > 1 else None

    col_e, col_t = st.columns([2, 3], gap="medium")
    with col_e:
        codigos = plot.sort_values("nome")["codigo"].tolist()
        labels = {r.codigo: f"{r.nome} ({r.codigo})" for r in plot.itertuples()}
        if sel_cod is None:
            sel_cod = st.selectbox("Estação (clique no mapa ou selecione)",
                                   codigos, format_func=lambda c: labels.get(c, c),
                                   key="dcrs_estacao")
        else:
            st.caption(f"Selecionada no mapa: **{labels.get(sel_cod, sel_cod)}**")
        _painel_dcrs(df, sel_cod)

    with col_t:
        st.markdown("**Resumo da seleção — maiores chuvas 24h**")
        top = (sel.assign(chuva_24h=pd.to_numeric(sel["chuva_24h"], errors="coerce"))
               .dropna(subset=["chuva_24h"])
               .nlargest(8, "chuva_24h")
               [["nome", "bacia", "chuva_24h", "chuva_72h", "nivel_m"]])
        if top.empty:
            st.caption("Sem dados de chuva na seleção.")
        else:
            st.dataframe(
                top.rename(columns={"nome": "Estação", "bacia": "Bacia",
                                    "chuva_24h": "24h (mm)", "chuva_72h": "72h (mm)",
                                    "nivel_m": "Nível (m)"}),
                use_container_width=True, hide_index=True, height=300,
            )


def _painel_dcrs(df: pd.DataFrame, codigo: str) -> None:
    """Card de uma estação DCRS com todas as métricas + cotas se houver."""
    linha = df[df["codigo"] == codigo]
    if linha.empty:
        st.info("Estação sem leitura recente.")
        return
    r = linha.iloc[0]
    nivel = r.get("nivel_m")
    bruto = r.get("rio_nivel")
    tend = r.get("rio_tendencia")
    seta = "↑" if (tend or 0) > 0 else "↓" if (tend or 0) < 0 else "→"
    cotas = ""
    if pd.notna(r.get("cota_alerta")):
        cotas = (f"<br><span class='muted'>cotas: atenção "
                 f"{_fmt(r.get('cota_atencao'), 'm')} · alerta "
                 f"{_fmt(r.get('cota_alerta'), 'm')} · emerg. "
                 f"{_fmt(r.get('cota_emergencia'), 'm')}</span>")
    st.markdown(
        f"<div class='card'><b>{r.get('nome') or codigo}</b><br>"
        f"<span class='muted'>{r.get('bacia') or ''} · {codigo}</span><br>"
        f"🌊 nível <b>{_fmt(nivel, ' m', 2)}</b> {seta} "
        f"<span class='muted'>(bruto {_fmt(bruto, '', 1)})</span>{cotas}<br>"
        f"🌧️ 1h {_fmt(r.get('chuva_1h'), '')} · 24h {_fmt(r.get('chuva_24h'), '')} · "
        f"72h {_fmt(r.get('chuva_72h'), '')} · 7d {_fmt(r.get('chuva_168h'), '')} mm<br>"
        f"🌡️ {_fmt(r.get('temperatura'), ' °C')} · 💧 {_fmt(r.get('umidade'), ' %', 0)} · "
        f"💨 {_fmt(r.get('vento_vel'), ' km/h', 0)}<br>"
        f"<span class='muted'>🕐 leitura: {_fmt_ts(r.get('timestamp'))} UTC</span></div>",
        unsafe_allow_html=True,
    )

    # Mini-gráfico Observado × Previsão LSTM quando a bacia tem rio com modelo.
    rio_modelo = _BACIA_RIO_MODELO.get(str(r.get("bacia") or "").strip())
    if rio_modelo and pd.notna(r.get("rio_nivel")):
        fc = api_get("/api/v3/forecasts/rivers", {"rio_id": rio_modelo})
        validas = [p for p in (fc.get("previsoes", []) if _ok(fc) else [])
                   if p.get("status") == "ok" and p.get("nivel_previsto_m") is not None]
        obs = _serie_dcrs(codigo, dias=7)
        if validas or not obs.empty:
            _grafico_dcrs_ml(
                obs, validas,
                f"Observado ({codigo}) × previsão LSTM do rio {rio_modelo.title()}",
            )


# ---------------------------------------------------------------------------
# SEÇÃO 4 — Sobre o Modelo ML (transparência didática)
# ---------------------------------------------------------------------------

def secao_modelo() -> None:
    """Painel de transparência explicado de forma intuitiva."""
    st.subheader("🧠 Sobre o Modelo de Previsão (LSTM por bacia)")

    st.markdown(
        "<div class='info-box'>"
        "<b>O que é?</b> Uma rede neural <b>LSTM</b> treinada por rio que aprende, "
        "com ~25 anos de histórico (nível ANA + chuva/temperatura INMET), a prever o "
        "<b>nível do rio</b> nos próximos <b>1, 2, 3 e 6 dias</b>.<br>"
        "<b>Como ler os cards:</b> cada modelo traz a qualidade medida no teste e quando "
        "rodou pela última vez. Quanto menor o erro, melhor."
        "</div>", unsafe_allow_html=True)

    # Glossário curto
    with st.expander("📖 O que significam os termos?"):
        st.markdown(
            "- **MAE 24h** — *Erro Médio Absoluto* na previsão de 1 dia: em média, quantos "
            "metros a previsão erra para mais ou para menos. **Menor = melhor.**\n"
            "- **nMAE** — o mesmo erro em **% da faixa do rio** (do mínimo ao máximo histórico). "
            "Permite comparar rios de portes diferentes; homologamos abaixo de **5%**.\n"
            "- **Status Aprovado** — passou no critério de homologação (nMAE < 5%).\n"
            "- **IC 95% (intervalo de confiança)** — a faixa sombreada nos gráficos de rio. "
            "Hoje está **em calibração** (mais estreita que o ideal), então é referência, "
            "não certeza absoluta.\n"
            "- **Última inferência** — quando o sistema gerou a previsão mais recente "
            "(roda a cada ciclo do cron, ~10 min)."
        )

    data = api_get("/api/v3/forecasts/model-info")
    if not _ok(data):
        st.error(f"API indisponível para model-info: {data['_erro']}")
        return
    modelos = data.get("modelos", [])
    if not modelos:
        st.warning("Sem metadados de modelo disponíveis.")
        return

    # Resumo geral
    n_ok = sum(1 for m in modelos if m.get("status") in ("aprovado", "calibrando"))
    maes = [m["mae_24h_m"] for m in modelos if m.get("mae_24h_m") is not None]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rios com modelo ativo", f"{n_ok}/{len(modelos)}")
    c2.metric("Melhor MAE (24h)", f"{min(maes):.3f} m" if maes else "—")
    c3.metric("Horizontes previstos", "1 · 2 · 3 · 6 dias")

    aviso = next((m.get("aviso_ic") for m in modelos if m.get("aviso_ic")), "")
    if aviso:
        st.markdown(f"<div class='warn-box'>⚠️ <b>Intervalo de confiança em calibração.</b> "
                    f"{aviso}</div>", unsafe_allow_html=True)

    for i in range(0, len(modelos), 2):
        cols = st.columns(2, gap="medium")
        for col, m in zip(cols, modelos[i:i + 2]):
            cor, label = MODEL_STATUS.get(m.get("status", "ausente"), MODEL_STATUS["ausente"])
            mae = m.get("mae_24h_m")
            nmae = m.get("nmae_pct")
            qualidade = ("excelente" if (nmae is not None and nmae < 3)
                         else "boa" if (nmae is not None and nmae < 5)
                         else "homologada" if mae is not None else "—")
            per = m.get("periodo_treino") or {}
            periodo_txt = (f"<br>📅 Dados de treino: <b>{per.get('inicio', '?')} → "
                           f"{per.get('fim', '?')}</b> "
                           f"({per.get('dias_com_dado', 0):,} dias)".replace(",", ".")
                           if per else "")
            ressalva_txt = (f"<br><span style='color:#fcd34d'>⚠️ "
                            f"{m['ressalva']}</span>" if m.get("ressalva") else "")
            with col:
                st.markdown(
                    f"<div class='card' style='border-left-color:{cor}'>"
                    f"<b style='font-size:1.05em'>{m['rio_id'].title()}</b> "
                    f"<span style='color:{cor}'>{label}</span><br>"
                    f"📏 Erro típico (1 dia): <b>{_fmt(mae, ' m', 3)}</b>"
                    + (f" &nbsp;·&nbsp; {nmae:.1f}% da faixa do rio" if nmae is not None else "")
                    + f"<br>🎯 Qualidade: <b>{qualidade}</b>"
                    + periodo_txt + ressalva_txt +
                    f"<br><span class='muted'>treinado em {_fmt_ts(m.get('treinado_em'), '%d/%m/%Y')} · "
                    f"última previsão {_fmt_ts(m.get('ultima_inferencia'))}</span></div>",
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Rios monitorados via DCRS (Observado Defesa Civil + Previsão LSTM/ANA)
# ---------------------------------------------------------------------------

# Rio com modelo LSTM cuja régua ANA MORREU (observado só via DCRS).
# Caí/Ibicuí/Ijuí saíram daqui em 05/07: telemetria ANA ativa na própria
# régua de treino → entraram no pipeline clássico (RIOS_RS → live_river_levels)
# com cards padronizados e cotas provisórias por percentil. No Gravataí as
# RÉGUAS SÃO DISTINTAS: previsão na régua ANA × observado na régua DCRS —
# gráfico de eixo duplo, comparar TENDÊNCIA.
_RIOS_DCRS: dict[str, dict[str, str]] = {
    "gravatai": {"nome": "Gravataí", "codigo": "DCRS-00120", "estacao": "Gravataí"},
    # Pardo re-homologado em 05/07 (retreino seq_len=96, nMAE 5,0%): régua
    # ANA de Candelária instável desde 2024 → observado via DCRS Candelária
    # (mesmo município da régua de treino).
    "pardo":    {"nome": "Pardo",    "codigo": "DCRS-00124", "estacao": "Candelária"},
}

# Bacia DCRS → rio com modelo LSTM (mini-gráfico no painel das Bacias RS)
_BACIA_RIO_MODELO: dict[str, str] = {
    "RS - Rio Caí":          "cai",
    "RS - Rio Gravataí":     "gravatai",
    "RS - Rio Ibicuí":       "ibicui",
    "RS - Rio Ijuí":         "ijui",
    "RS - Sinos":            "sinos",
    "RS - Rio Camaquã":      "camaqua",
    "RS - Rio Taquari-Antas": "taquari",
    "RS - Lago Guaíba":      "guaiba",
    "RS - Baixo Jacuí":      "jacui",
    "RS - Rio Pardo":        "pardo",
}

_AVISO_REGUA = ("Réguas distintas: observado na régua DCRS (Defesa Civil) e "
                "previsão na régua ANA do treino — compare tendências, não "
                "valores absolutos.")


def _serie_dcrs(codigo: str, dias: int = 7) -> pd.DataFrame:
    """Série horária observada de uma estação DCRS via /api/v3/dcrs/history."""
    h = api_get("/api/v3/dcrs/history", {"codigo": codigo, "dias": dias})
    if not _ok(h) or not h.get("serie"):
        return pd.DataFrame()
    df = pd.DataFrame(h["serie"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["nivel"] = df["rio_nivel"].map(_dcrs_nivel_exibicao)
    return df.dropna(subset=["ts", "nivel"]).sort_values("ts")


def _grafico_dcrs_ml(obs: pd.DataFrame, previsoes: list[dict], titulo: str) -> None:
    """Gráfico eixo-duplo: observado DCRS (y1) × previsão LSTM régua ANA (y2)."""
    fig = go.Figure()
    if not obs.empty:
        fig.add_trace(go.Scatter(x=obs["ts"], y=obs["nivel"], name="Observado (DCRS)",
                                 mode="lines+markers", marker={"size": 5},
                                 line={"color": "rgba(56,189,248,1)", "width": 2},
                                 yaxis="y1"))
    if previsoes:
        prev = sorted(previsoes, key=lambda p: p["horizonte_h"])
        xs = [pd.to_datetime(p["valid_ts"]) for p in prev]
        ys = [p["nivel_previsto_m"] for p in prev]
        lo = [p.get("ic_inferior_m") for p in prev]
        hi = [p.get("ic_superior_m") for p in prev]
        if all(v is not None for v in lo + hi):
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself",
                                     fillcolor="rgba(167,139,250,0.18)",
                                     line={"color": "rgba(0,0,0,0)"}, name="IC 95%",
                                     hoverinfo="skip", yaxis="y2"))
        fig.add_trace(go.Scatter(x=xs, y=ys, name="Previsão LSTM (régua ANA)",
                                 mode="lines+markers",
                                 line={"color": "rgba(167,139,250,1)", "width": 2, "dash": "dot"},
                                 yaxis="y2"))
    fig.update_layout(**PLOTLY_LAYOUT, height=260, title=titulo,
                      legend={"bgcolor": "rgba(30,41,59,0.5)", "orientation": "h", "y": 1.18},
                      xaxis={"gridcolor": "rgba(51,65,85,1)"},
                      yaxis={"title": "DCRS (m)", "gridcolor": "rgba(51,65,85,1)"},
                      yaxis2={"title": "ANA (m)", "overlaying": "y", "side": "right",
                              "showgrid": False})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"⚖️ {_AVISO_REGUA}")


def _card_rio_dcrs(rio_id: str, cfg: dict[str, str], dcrs_df: pd.DataFrame,
                   previsoes: list[dict]) -> None:
    """Card de rio novo: observado DCRS + previsão LSTM, sem cota oficial."""
    est = dcrs_df[dcrs_df["codigo"] == cfg["codigo"]] if not dcrs_df.empty else pd.DataFrame()
    nivel = tend = ts_txt = None
    if not est.empty:
        r = est.iloc[0]
        nivel = _dcrs_nivel_exibicao(r.get("rio_nivel"))
        tend = r.get("rio_tendencia")
        ts_txt = _fmt_ts(r.get("timestamp"))
    seta = "↑" if (tend or 0) > 0 else "↓" if (tend or 0) < 0 else "→"
    cor = "rgba(148,163,184,1)"   # slate — sem cota oficial, sem status de alerta
    st.markdown(
        f"<div class='card' style='border-left-color:{cor}'>"
        f"<b>{cfg['nome']}</b> 🆕 <span style='color:{cor}'>MONITORADO (DCRS)</span><br>"
        f"📏 <b>{_fmt(nivel, ' m', 2)}</b> {seta} · estação {cfg['estacao']} "
        f"({cfg['codigo']})<br>"
        f"<span class='muted'>cotas oficiais de alerta pendentes (Agente 5) — "
        f"sem classificação de status · leitura {ts_txt or '—'}</span></div>",
        unsafe_allow_html=True,
    )
    validas = [p for p in previsoes
               if p.get("status") == "ok" and p.get("nivel_previsto_m") is not None]
    obs = _serie_dcrs(cfg["codigo"])
    if not validas and obs.empty:
        st.caption("Sem série DCRS acumulada ainda (histórico iniciou 04/07/2026).")
        return
    _grafico_dcrs_ml(obs, validas, f"{cfg['nome']} — observado DCRS × previsão LSTM")


# ---------------------------------------------------------------------------
# SEÇÃO Rios (mantida) — Observado + Previsão LSTM
# ---------------------------------------------------------------------------

def secao_rios() -> None:
    """Cards por rio: nível atual, status e previsão LSTM com IC 95%."""
    st.subheader("🌊 Rios Monitorados — Observado + Previsão LSTM")
    rios = api_get("/api/v3/rivers/status")
    if not _ok(rios):
        st.error(f"API indisponível para rios: {rios['_erro']}")
        return
    lista = rios.get("rios", [])
    if not lista:
        st.warning("Sem dados de rios no momento.")
        return
    fc = api_get("/api/v3/forecasts/rivers")
    por_rio: dict[str, list[dict]] = {}
    if _ok(fc):
        for p in fc.get("previsoes", []):
            por_rio.setdefault(str(p["rio_id"]), []).append(p)
    hist = api_get("/api/v3/analytics/history", {"days": 7, "data_source": "ana"})
    for i in range(0, len(lista), 2):
        cols = st.columns(2, gap="medium")
        for col, rio in zip(cols, lista[i:i + 2]):
            with col:
                _card_rio(rio, por_rio.get(str(rio["rio_id"]), []), hist)

    # ── Gravataí: régua ANA perdida na enchente de mai/24 — observado DCRS ─
    st.markdown("---")
    st.markdown("#### Gravataí e Pardo — observado Defesa Civil (réguas ANA perdidas/instáveis desde 2024)")
    st.markdown(f"<div class='muted'>⚖️ {_AVISO_REGUA}</div>", unsafe_allow_html=True)
    dcrs = api_get("/api/v3/dcrs/stations")
    dcrs_df = pd.DataFrame(dcrs.get("estacoes", [])) if _ok(dcrs) else pd.DataFrame()
    novos = list(_RIOS_DCRS.items())
    for i in range(0, len(novos), 2):
        cols = st.columns(2, gap="medium")
        for col, (rio_id, cfg) in zip(cols, novos[i:i + 2]):
            with col:
                _card_rio_dcrs(rio_id, cfg, dcrs_df, por_rio.get(rio_id, []))


def _card_rio(rio: dict[str, Any], previsoes: list[dict], hist: dict[str, Any]) -> None:
    status = str(rio.get("status") or "NORMAL").upper()
    cor = STATUS_COLORS.get(status, STATUS_COLORS["NORMAL"])
    pct = rio.get("percentual_cota") or 0.0
    st.markdown(
        f"<div class='card' style='border-left-color:{cor}'>"
        f"<b>{rio.get('rio_nome') or rio['rio_id']}</b> "
        f"{STATUS_EMOJI.get(status, '⚪')} <span style='color:{cor}'>{status}</span><br>"
        f"📏 <b>{_fmt(rio.get('nivel_atual_m'), ' m', 2)}</b> · "
        f"{_fmt(pct, ' %', 0)} da cota atenção<br>"
        f"<span class='muted'>atenção {_fmt(rio.get('cota_atencao_m'), 'm')} · "
        f"alerta {_fmt(rio.get('cota_alerta_m'), 'm')} · "
        f"emerg. {_fmt(rio.get('cota_emergencia_m'), 'm')}"
        + (f" · máx. hist. {_fmt(rio.get('cota_max_hist_m'), 'm')}"
           if rio.get('cota_max_hist_m') else "") + "</span>"
        f"<div class='bar-wrap'><div class='bar-fill' "
        f"style='width:{min(float(pct), 100):.0f}%;background:{cor}'></div></div></div>",
        unsafe_allow_html=True,
    )
    validas = [p for p in previsoes
               if p.get("status") == "ok" and p.get("nivel_previsto_m") is not None]
    obs = _serie_rio(hist, str(rio["rio_id"]))
    if not validas:
        st.markdown("<div class='warn-box'>🔧 Previsão em calibração — modelo deste rio "
                    "ainda não disponível.</div>", unsafe_allow_html=True)
        if not obs.empty:
            _grafico_rio(obs, [], rio)
        return
    _grafico_rio(obs, validas, rio)


def _slug_rio(nome: Any) -> str:
    """Normaliza nome de rio para slug ASCII ('Guaíba' → 'guaiba')."""
    import unicodedata
    return (unicodedata.normalize("NFD", str(nome))
            .encode("ascii", "ignore").decode().lower().strip())


def _serie_rio(hist: dict[str, Any], rio_id: str) -> pd.DataFrame:
    """Série diária de nível observado do rio, robusta ao schema do R2.

    O Parquet de live_river_levels no R2 pode vir com o schema de status
    (rio_nome/current_level_m/updated_at) ou de série (rio_nome/nivel_m/ts),
    e usa rio_nome (com acento), não rio_id. Aqui filtramos por rio_id direto
    ou por slug(rio_nome), aceitamos os aliases de coluna e agregamos 1 ponto
    por dia (mediana das estações) para uma linha "Observado" limpa.
    """
    if not _ok(hist):
        return pd.DataFrame()
    regs: list[dict] = []
    for ponto in hist.get("historico", []):
        regs.extend(ponto.get("dados", []))
    if not regs:
        return pd.DataFrame()
    df = pd.DataFrame(regs)
    if "rio_id" in df.columns:
        df = df[df["rio_id"].astype(str) == str(rio_id)]
    elif "rio_nome" in df.columns:
        df = df[df["rio_nome"].map(_slug_rio) == str(rio_id)]
    nivel = next((c for c in ("nivel_atual_m", "current_level_m", "nivel_m", "level_m")
                  if c in df.columns), None)
    tsc = next((c for c in ("timestamp", "ts", "updated_at", "data_hora_medicao")
                if c in df.columns), None)
    if df.empty or nivel is None or tsc is None:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df[tsc], utc=True, errors="coerce")
    df["nivel"] = pd.to_numeric(df[nivel], errors="coerce")
    df = df.dropna(subset=["ts", "nivel"])
    if df.empty:
        return pd.DataFrame()
    # 1 ponto/dia (mediana entre estações do rio) → série "Observado" limpa.
    df["_d"] = df["ts"].dt.date
    g = (df.sort_values("ts").groupby("_d")
         .agg(ts=("ts", "last"), nivel=("nivel", "median"))
         .reset_index(drop=True))
    return g


def _grafico_rio(obs: pd.DataFrame, previsoes: list[dict], rio: dict[str, Any]) -> None:
    fig = go.Figure()
    if not obs.empty:
        # lines+markers: com 1 ponto só (histórico recém-nascido dos rios
        # novos), mode="lines" não renderiza nada — o marker garante presença.
        fig.add_trace(go.Scatter(x=obs["ts"], y=obs["nivel"], name="Observado",
                                 mode="lines+markers",
                                 marker={"size": 6},
                                 line={"color": "rgba(56,189,248,1)", "width": 2}))
    if previsoes:
        previsoes = sorted(previsoes, key=lambda p: p["horizonte_h"])
        base_ts = obs["ts"].iloc[-1] if not obs.empty else pd.Timestamp.now(tz="UTC")
        base_y = obs["nivel"].iloc[-1] if not obs.empty else previsoes[0]["nivel_previsto_m"]
        xs = [base_ts] + [pd.to_datetime(p["valid_ts"]) for p in previsoes]
        ys = [base_y] + [p["nivel_previsto_m"] for p in previsoes]
        lo = [base_y] + [p.get("ic_inferior_m") for p in previsoes]
        hi = [base_y] + [p.get("ic_superior_m") for p in previsoes]
        if all(v is not None for v in lo + hi):
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself",
                                     fillcolor="rgba(167,139,250,0.18)",
                                     line={"color": "rgba(0,0,0,0)"}, name="IC 95%",
                                     hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, name="Previsão LSTM", mode="lines+markers",
                                 line={"color": "rgba(167,139,250,1)", "width": 2, "dash": "dot"}))
    for chave, label, cor in [("cota_atencao_m", "Atenção", "rgba(245,158,11,0.8)"),
                              ("cota_alerta_m", "Alerta", "rgba(239,68,68,0.8)"),
                              ("cota_emergencia_m", "Emerg.", "rgba(124,58,237,0.8)"),
                              ("cota_max_hist_m", "Máx. hist.", "rgba(148,163,184,0.55)")]:
        if rio.get(chave):
            fig.add_hline(y=float(rio[chave]), line_dash="dash", line_color=cor,
                          annotation_text=label, annotation_font_color=cor)
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                      legend={"bgcolor": "rgba(30,41,59,0.5)", "orientation": "h", "y": 1.15},
                      xaxis={"gridcolor": "rgba(51,65,85,1)"},
                      yaxis={"title": "Nível (m)", "gridcolor": "rgba(51,65,85,1)"})
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar + main
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    st.sidebar.title("🌧️ Monitor RS")
    st.sidebar.caption("100% stateless · dados via FastAPI v3")
    health = api_get("/health")
    if _ok(health):
        sup = health.get("supabase", {}).get("status", "?")
        r2 = health.get("r2", {}).get("status", "?")
        st.sidebar.success(f"API online · Supabase: {sup} · R2: {r2}")
    elif _API_IS_LOCAL:
        # Causa nº 1 de tela vazia: dashboard publicado (ou local) sem a API de
        # pé em localhost:8000. Mensagem acionável em vez de só "offline".
        st.sidebar.error(
            f"API offline em **{API_BASE}**.\n\n"
            "Você está apontando para *localhost*. Para dados reais, defina a env "
            "**API_BASE_URL** com a URL da API publicada (ex.: Render).\n\n"
            "Para rodar local, suba a API antes:\n"
            "`uvicorn main:app --port 8000` em `src/api/`."
        )
    else:
        st.sidebar.error(
            f"API offline — {health.get('_erro', '')}\n\n"
            f"Verifique se **{API_BASE}** está no ar."
        )
    st.sidebar.caption(f"Endpoint: {API_BASE}")
    st.sidebar.caption(f"Atualizado: {datetime.now(timezone.utc):%d/%m %H:%M} UTC")


def main() -> None:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    if _AUTOREFRESH_OK:
        st_autorefresh(interval=_REFRESH_MS, key="auto_refresh")
    _sidebar()
    st.title("Monitor Hidrometeorológico — Rio Grande do Sul")

    aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs([
        "📍 Estações", "🌐 Mapa Climático", "🌊 Rios + ML", "🗺️ Bacias RS",
        "📊 Clima", "🧠 Modelo ML",
    ])
    with aba1:
        secao_estacoes()
    with aba2:
        secao_mapa_climatico()
    with aba3:
        secao_rios()
    with aba4:
        secao_bacias()
    with aba5:
        secao_clima()
    with aba6:
        secao_modelo()


main()
