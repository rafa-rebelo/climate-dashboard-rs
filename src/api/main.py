"""
Agente 3 — Engenheiro de Software / API
API REST FastAPI v3 — Monitor Hidrometeorológico RS

Endpoints:
  GET  /health                       — status Supabase + R2
  GET  /api/v3/rivers/status         — níveis e alertas (live_river_levels)
  GET  /api/v3/weather/heatmap       — grade GPM precip (live_gpm_precip) GZip
  GET  /api/v3/stations/readings     — leituras INMET (live_rain_readings)
  GET  /api/v3/analytics/history     — histórico R2 por período (boto3 Parquet)
  GET  /api/v3/analytics/trend       — tendência diária + regressão 7d (R2)
  GET  /api/v3/forecasts/rivers      — previsões LSTM (river_ai_forecasts)
  GET  /api/v3/forecasts/weather     — timeline NWP por ponto/hora (forecasts)
  GET  /api/v3/forecasts/model-info  — metadados dos modelos LSTM (transparência)

Deploy: Render Free Tier — PORT env var.
Banco:  Supabase PostgreSQL via SUPABASE_DATABASE_URL_POOLER (Session Pooler IPv4).
Store:  Cloudflare R2 — historico/{fonte}/ano={Y}/mes={MM}/dia={DD}/{fonte}_{ts}UTC.parquet
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import boto3
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Constantes e pool de conexão
# ---------------------------------------------------------------------------

_PG_URL: str = (
    os.getenv("SUPABASE_DATABASE_URL_POOLER")
    or os.getenv("SUPABASE_DATABASE_URL")
    or ""
)

_PG_POOL: psycopg2.pool.SimpleConnectionPool | None = None

_STATUS_COLORS: dict[str, str] = {
    "NORMAL":     "rgba(40,167,69,1.0)",
    "ATENCAO":    "rgba(255,193,7,1.0)",
    "ALERTA":     "rgba(253,126,20,1.0)",
    "EMERGENCIA": "rgba(220,53,69,1.0)",
}

# TTL cache: {chave: (payload, monotonic_ts)}
_CACHE: dict[str, tuple[Any, float]] = {}


def _cache_get(key: str, ttl_s: int) -> Any | None:
    """Retorna payload em cache se ainda válido, ou None."""
    entry = _CACHE.get(key)
    if entry and (time.monotonic() - entry[1]) < ttl_s:
        return entry[0]
    return None


def _cache_set(key: str, payload: Any) -> None:
    """Armazena payload no cache com timestamp atual."""
    _CACHE[key] = (payload, time.monotonic())


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    """Inicializa ou retorna o pool de conexão PostgreSQL.

    Returns:
        Pool com 1-5 conexões ao Supabase.

    Raises:
        RuntimeError: Se SUPABASE_DATABASE_URL* não estiver configurado.
    """
    global _PG_POOL
    if _PG_POOL is None:
        if not _PG_URL:
            raise RuntimeError("SUPABASE_DATABASE_URL_POOLER não configurado.")
        _PG_POOL = psycopg2.pool.SimpleConnectionPool(
            1, 5, _PG_URL, connect_timeout=30
        )
    return _PG_POOL


def _pg_query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Executa SELECT no Supabase e retorna lista de dicionários.

    Args:
        sql:    Query SQL com placeholders %s.
        params: Parâmetros para a query.

    Returns:
        Lista de dicts com os resultados.

    Raises:
        HTTPException 503: Se a conexão falhar.
        HTTPException 500: Em erros de banco inesperados.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    except psycopg2.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"Supabase indisponível: {exc}") from exc
    except psycopg2.Error as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        pool.putconn(conn)


def _s3_client() -> Any:
    """Cria cliente boto3 apontado para o Cloudflare R2.

    Returns:
        boto3 S3 client configurado para R2.
    """
    return boto3.client(
        "s3",
        endpoint_url         = os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id    = os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key= os.getenv("R2_SECRET_ACCESS_KEY"),
        config               = BotoConfig(connect_timeout=10, read_timeout=60),
    )


def _safe_float(val: Any) -> float | None:
    """Converte para float ou None se NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _iso(val: Any) -> str | None:
    """Converte datetime/str para ISO 8601 ou None."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Monitor Hidrometeorológico RS",
    description = (
        "Dados hidrometeorológicos em tempo real do Rio Grande do Sul. "
        "Rios: Sinos, Taquari, Jacuí, Guaíba, Camaquã, Lagoa dos Patos."
    ),
    version  = "3.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["GET"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------

class RioItem(BaseModel):
    rio_id:            str
    rio_nome:          str
    nivel_atual_m:     float | None
    cota_atencao_m:    float | None
    cota_alerta_m:     float | None
    cota_emergencia_m: float | None
    percentual_cota:   float | None
    status:            str
    status_color:      str
    timestamp_leitura: str | None


class RiversStatusResponse(BaseModel):
    timestamp:  str
    total_rios: int
    rios:       list[RioItem]


class HeatmapPoint(BaseModel):
    lat:       float
    lon:       float
    precip_mm: float | None


class HeatmapResponse(BaseModel):
    timestamp:    str
    total_pontos: int
    pontos:       list[HeatmapPoint]


class StationReading(BaseModel):
    station_id:   str
    nome:         str | None
    municipio:    str | None
    latitude:     float | None
    longitude:    float | None
    precip_1h:    float | None
    precip_3h:    float | None
    precip_6h:    float | None
    precip_12h:   float | None
    precip_24h:   float | None
    precip_48h:   float | None
    precip_72h:   float | None
    precip_7d:    float | None
    temperatura:  float | None
    umidade:      float | None
    timestamp:    str | None


class StationsResponse(BaseModel):
    timestamp:       str
    total_estacoes:  int
    estacoes:        list[StationReading]


class HistoryPoint(BaseModel):
    data:       str
    fonte:      str
    registros:  int
    colunas:    list[str]
    dados:      list[dict[str, Any]]


class HistoryResponse(BaseModel):
    timestamp:    str
    data_source:  str
    dias:         int
    total_arquivos: int
    total_registros: int
    historico:    list[HistoryPoint]


class RiverForecast(BaseModel):
    rio_id:           str
    horizonte_h:      int
    horizonte_dias:   float | None
    valid_ts:         str | None
    nivel_previsto_m: float | None
    ic_inferior_m:    float | None
    ic_superior_m:    float | None
    modelo_versao:    str | None
    status:           str
    gerado_em:        str | None


class ForecastsResponse(BaseModel):
    timestamp:  str
    fase:       str
    total:      int
    previsoes:  list[RiverForecast]


class WeatherForecastPoint(BaseModel):
    location_name: str
    lat:           float | None
    lon:           float | None
    valid_ts:      str
    rain_mm:       float | None
    temperature:   float | None
    wind_speed:    float | None
    cape_j_kg:     float | None
    model_source:  str | None


class WeatherForecastResponse(BaseModel):
    timestamp:     str
    horas:         int
    total_pontos:  int
    localidades:   list[str]
    previsoes:     list[WeatherForecastPoint]


class ModelInfoItem(BaseModel):
    rio_id:        str
    status:        str          # aprovado | calibrando | pendente | ausente
    modelo_versao: str | None
    mae_24h_m:     float | None
    nmae_pct:      float | None
    treinado_em:   str | None
    ultima_inferencia: str | None
    horizontes_dias:   list[int]
    aviso_ic:      str
    periodo_treino:    dict[str, Any] | None = None   # {inicio, fim, dias_com_dado}
    ressalva:          str | None = None              # ressalva de fonte (Agente 5)


class ModelInfoResponse(BaseModel):
    timestamp: str
    total:     int
    modelos:   list[ModelInfoItem]


class DcrsStation(BaseModel):
    codigo:           str
    nome:             str | None
    bacia:            str | None
    latitude:         float | None
    longitude:        float | None
    rio_nome:         str | None
    rio_nivel:        float | None
    rio_vazao:        float | None
    rio_tendencia:    float | None
    cota_atencao:     float | None
    cota_alerta:      float | None
    cota_emergencia:  float | None
    chuva_1h:         float | None
    chuva_3h:         float | None
    chuva_6h:         float | None
    chuva_12h:        float | None
    chuva_24h:        float | None
    chuva_48h:        float | None
    chuva_72h:        float | None
    chuva_168h:       float | None
    temperatura:      float | None
    umidade:          float | None
    vento_vel:        float | None
    vento_dir:        float | None
    pressao:          float | None
    senstermica:      float | None
    radiacao:         float | None
    timestamp:        str | None


class DcrsStationsResponse(BaseModel):
    timestamp:      str
    fonte:          str
    total_estacoes: int
    bacias:         list[str]
    estacoes:       list[DcrsStation]


class DcrsBaciaItem(BaseModel):
    bacia:      str
    estacoes:   int
    com_rio:    int
    chuva_24h_max: float | None


class DcrsBaciasResponse(BaseModel):
    timestamp: str
    total:     int
    bacias:    list[DcrsBaciaItem]


class TrendDay(BaseModel):
    data:        str
    precip_mm:   float | None
    nivel_m:     float | None


class TrendResponse(BaseModel):
    timestamp:    str
    id:           str
    tipo:         str          # "rio" | "estacao"
    dias:         int
    n_dias_dados: int
    tendencia:    str          # SUBINDO | ESTAVEL | DESCENDO | INDEFINIDA
    inclinacao:   float | None  # slope da regressão (unidade/dia)
    metrica_tendencia: str      # "nivel_m" | "precip_mm"
    serie:        list[TrendDay]


# ---------------------------------------------------------------------------
# Endpoint 6 — GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"], summary="Health check Supabase + R2")
async def health() -> JSONResponse:
    """Verifica conectividade com Supabase e Cloudflare R2.

    Returns:
        JSON com status de cada serviço e latências.
    """
    checks: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0.0",
    }

    # Supabase
    t0 = time.monotonic()
    try:
        rows = _pg_query("SELECT COUNT(*) AS n FROM live_river_levels")
        checks["supabase"] = {
            "status": "ok",
            "live_river_levels": rows[0]["n"] if rows else 0,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }
    except HTTPException as exc:
        checks["supabase"] = {"status": "error", "detail": exc.detail}

    # R2
    t0 = time.monotonic()
    try:
        s3  = _s3_client()
        bucket = os.getenv("R2_BUCKET_NAME", "")
        s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
        checks["r2"] = {
            "status": "ok",
            "bucket": bucket,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }
    except Exception as exc:  # noqa: BLE001
        checks["r2"] = {"status": "error", "detail": str(exc)}

    http_status = 200 if checks.get("supabase", {}).get("status") == "ok" else 503
    return JSONResponse(content=checks, status_code=http_status)


# ---------------------------------------------------------------------------
# Endpoint 1 — GET /api/v3/rivers/status
# ---------------------------------------------------------------------------

@app.get(
    "/api/v3/rivers/status",
    tags           = ["Rios"],
    response_model = RiversStatusResponse,
    summary        = "Status atual e nível dos rios monitorados",
)
async def rivers_status() -> RiversStatusResponse:
    """Retorna nível atual, cotas e status de todos os rios.

    Dados de ``live_river_levels``. Cache 5 min.

    Returns:
        RiversStatusResponse com lista completa de rios.
    """
    cached = _cache_get("rivers_status", ttl_s=300)
    if cached:
        return cached

    rows = _pg_query("""
        SELECT rio_id, rio_nome, nivel_atual_m,
               cota_atencao_m, cota_alerta_m, cota_emergencia_m,
               percentual_cota, status, "timestamp"
        FROM   live_river_levels
        ORDER  BY rio_id
    """)

    rios: list[RioItem] = []
    for r in rows:
        status_str = str(r.get("status") or "NORMAL").upper()
        rios.append(RioItem(
            rio_id            = str(r["rio_id"]),
            rio_nome          = str(r.get("rio_nome") or r["rio_id"]),
            nivel_atual_m     = _safe_float(r.get("nivel_atual_m")),
            cota_atencao_m    = _safe_float(r.get("cota_atencao_m")),
            cota_alerta_m     = _safe_float(r.get("cota_alerta_m")),
            cota_emergencia_m = _safe_float(r.get("cota_emergencia_m")),
            percentual_cota   = _safe_float(r.get("percentual_cota")),
            status            = status_str,
            status_color      = _STATUS_COLORS.get(status_str, _STATUS_COLORS["NORMAL"]),
            timestamp_leitura = _iso(r.get("timestamp")),
        ))

    resp = RiversStatusResponse(
        timestamp  = datetime.now(timezone.utc).isoformat(),
        total_rios = len(rios),
        rios       = rios,
    )
    _cache_set("rivers_status", resp)
    return resp


# ---------------------------------------------------------------------------
# Endpoint 2 — GET /api/v3/weather/heatmap
# ---------------------------------------------------------------------------

@app.get(
    "/api/v3/weather/heatmap",
    tags    = ["Precipitação"],
    summary = "Grade de precipitação GPM para Folium/Leaflet (GZip)",
)
async def weather_heatmap() -> JSONResponse:
    """Retorna todos os pontos GPM do RS com precip_mm.

    Dados de ``live_gpm_precip``. Cache 10 min. Comprimido via GZip.

    Returns:
        JSON com timestamp, total_pontos e lista de {lat, lon, precip_mm}.
    """
    cached = _cache_get("weather_heatmap", ttl_s=600)
    if cached:
        return JSONResponse(content=cached)

    rows = _pg_query("""
        SELECT latitude AS lat, longitude AS lon, precip_mm
        FROM   live_gpm_precip
        ORDER  BY lat, lon
    """)

    pontos = [
        {
            "lat":       float(r["lat"]),
            "lon":       float(r["lon"]),
            "precip_mm": _safe_float(r.get("precip_mm")),
        }
        for r in rows
    ]

    payload: dict[str, Any] = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "total_pontos": len(pontos),
        "pontos":       pontos,
    }
    _cache_set("weather_heatmap", payload)
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Endpoint 3 — GET /api/v3/stations/readings
# ---------------------------------------------------------------------------

@app.get(
    "/api/v3/stations/readings",
    tags           = ["Estações"],
    response_model = StationsResponse,
    summary        = "Última leitura de cada estação INMET",
)
async def stations_readings(
    station_id: str | None = Query(None, description="Filtro por station_id"),
) -> StationsResponse:
    """Retorna a leitura mais recente de cada estação em live_rain_readings.

    Faz LEFT JOIN com ``stations`` para enriquecer com nome e município.
    Cache 5 min.

    Args:
        station_id: Código da estação para filtrar (opcional).

    Returns:
        StationsResponse com lista de estações e leituras.
    """
    cache_key = f"stations_readings:{station_id or 'all'}"
    cached = _cache_get(cache_key, ttl_s=300)
    if cached:
        return cached

    where  = "WHERE r.station_id = %s" if station_id else ""
    params: tuple[Any, ...] = (station_id,) if station_id else ()

    rows = _pg_query(f"""
        SELECT r.station_id,
               s.nome,
               s.municipio,
               s.latitude,
               s.longitude,
               r.precip_1h,
               r.precip_3h,
               r.precip_6h,
               r.precip_12h,
               r.precip_24h,
               r.precip_48h,
               r.precip_72h,
               r.precip_7d,
               r.temperatura,
               r.umidade,
               r."timestamp"
        FROM   live_rain_readings r
        LEFT JOIN stations s ON s.station_id = r.station_id
        {where}
        ORDER  BY r.station_id
    """, params)

    estacoes = [
        StationReading(
            station_id  = str(r["station_id"]),
            nome        = r.get("nome"),
            municipio   = r.get("municipio"),
            latitude    = _safe_float(r.get("latitude")),
            longitude   = _safe_float(r.get("longitude")),
            precip_1h   = _safe_float(r.get("precip_1h")),
            precip_3h   = _safe_float(r.get("precip_3h")),
            precip_6h   = _safe_float(r.get("precip_6h")),
            precip_12h  = _safe_float(r.get("precip_12h")),
            precip_24h  = _safe_float(r.get("precip_24h")),
            precip_48h  = _safe_float(r.get("precip_48h")),
            precip_72h  = _safe_float(r.get("precip_72h")),
            precip_7d   = _safe_float(r.get("precip_7d")),
            temperatura = _safe_float(r.get("temperatura")),
            umidade     = _safe_float(r.get("umidade")),
            timestamp   = _iso(r.get("timestamp")),
        )
        for r in rows
    ]

    resp = StationsResponse(
        timestamp      = datetime.now(timezone.utc).isoformat(),
        total_estacoes = len(estacoes),
        estacoes       = estacoes,
    )
    _cache_set(cache_key, resp)
    return resp


# ---------------------------------------------------------------------------
# Endpoint 4 — GET /api/v3/analytics/history
# ---------------------------------------------------------------------------

@app.get(
    "/api/v3/analytics/history",
    tags    = ["Analítica"],
    summary = "Histórico de dados do R2 por período",
)
async def analytics_history(
    days: int = Query(
        30, ge=1, le=90,
        description="Quantidade de dias para trás (1–90)",
    ),
    station_id: str | None = Query(
        None, description="Filtro por station_id (gpm e inmet)"
    ),
    data_source: Literal["gpm", "inmet", "ana"] = Query(
        "gpm", description="Fonte de dados: gpm | inmet | ana"
    ),
) -> HistoryResponse:
    """Baixa Parquets do R2 para o período e retorna série histórica.

    Percorre ``historico/{data_source}/ano={Y}/mes={MM}/dia={DD}/``
    para cada dia no intervalo, baixa via boto3 e agrega os DataFrames.

    Args:
        days:        Número de dias para trás (1–90).
        station_id:  Filtro opcional por estação.
        data_source: Fonte (gpm | inmet | ana).

    Returns:
        HistoryResponse com lista de HistoryPoint por dia.
    """
    cache_key = f"history:{data_source}:{days}:{station_id or 'all'}"
    cached = _cache_get(cache_key, ttl_s=600)
    if cached:
        return cached

    fonte_map = {
        "gpm":   "live_gpm_precip",
        "inmet": "live_rain_readings",
        "ana":   "live_river_levels",
    }
    fonte = fonte_map[data_source]
    bucket = os.getenv("R2_BUCKET_NAME", "")

    today   = date.today()
    dates   = [today - timedelta(days=i) for i in range(days)]

    try:
        s3 = _s3_client()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"R2 indisponível: {exc}") from exc

    historico: list[HistoryPoint] = []
    total_registros = 0

    for d in dates:
        prefix = (
            f"historico/{fonte}"
            f"/ano={d.year}"
            f"/mes={d.month:02d}"
            f"/dia={d.day:02d}/"
        )
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        except ClientError as exc:
            logger.warning(f"R2 list falhou para {prefix}: {exc}")
            continue

        contents = resp.get("Contents", [])
        if not contents:
            continue

        # 1 Parquet por dia (o mais recente) basta: as fontes de SNAPSHOT (ana)
        # gravam o estado a cada run e as de SÉRIE (inmet) já carregam ~30 dias
        # por arquivo. Baixar os ~72 arquivos/dia de cada partição deixava a
        # resposta lenta (>30s → timeout no dashboard, sumindo a série
        # "Observado") e chegava a travar o event loop. O dedup a jusante
        # (dashboard) resolve sobreposições de série.
        contents = [max(contents, key=lambda o: o["LastModified"])]

        frames: list[pd.DataFrame] = []
        for obj in contents:
            try:
                raw = s3.get_object(Bucket=bucket, Key=obj["Key"])
                df  = pd.read_parquet(io.BytesIO(raw["Body"].read()))
                if station_id and "station_id" in df.columns:
                    df = df[df["station_id"] == station_id]
                frames.append(df)
            except ClientError as exc:
                logger.warning(f"R2 get falhou para {obj['Key']}: {exc}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Parquet parse falhou para {obj['Key']}: {exc}")

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)

        # Converte timestamps para string (JSON-safe)
        for col in combined.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            combined[col] = combined[col].astype(str)

        dados = combined.where(combined.notna(), other=None).to_dict(orient="records")
        total_registros += len(dados)

        historico.append(HistoryPoint(
            data      = d.isoformat(),
            fonte     = fonte,
            registros = len(dados),
            colunas   = list(combined.columns),
            dados     = dados,
        ))

    resp_obj = HistoryResponse(
        timestamp        = datetime.now(timezone.utc).isoformat(),
        data_source      = data_source,
        dias             = days,
        total_arquivos   = sum(1 for h in historico),
        total_registros  = total_registros,
        historico        = historico,
    )
    _cache_set(cache_key, resp_obj)
    return resp_obj


# ---------------------------------------------------------------------------
# Endpoint 5 — GET /api/v3/forecasts/rivers
# ---------------------------------------------------------------------------

@app.get(
    "/api/v3/forecasts/rivers",
    tags           = ["Previsões ML"],
    response_model = ForecastsResponse,
    summary        = "Previsões LSTM de nível dos rios (river_ai_forecasts)",
)
async def forecasts_rivers(
    rio_id: str | None = Query(None, description="Filtro por rio_id (ex.: guaiba)"),
    horizonte_h: int | None = Query(
        None, ge=1, le=240, description="Horizonte máximo em horas (vazio = todos)"
    ),
) -> ForecastsResponse:
    """Retorna previsões LSTM por bacia hidrográfica de ``river_ai_forecasts``.

    Snapshot vivo (1 linha por rio × horizonte) gravado pela inferência no CI.
    Linhas com ``status='pendente'`` indicam rio sem modelo treinado/calibrado
    — o frontend deve exibir "previsão em calibração" e nunca um dado falso.

    Args:
        rio_id:      Filtro por rio (ex.: guaiba, sinos, camaqua).
        horizonte_h: Horizonte máximo em horas (None = todos: 24/48/72/144).

    Returns:
        ForecastsResponse com lista de previsões (inclui linhas 'pendente').
    """
    cache_key = f"forecasts:{rio_id or 'all'}:{horizonte_h or 'all'}"
    cached = _cache_get(cache_key, ttl_s=300)
    if cached:
        return cached

    try:
        where_parts: list[str] = []
        params: list[Any] = []
        if rio_id:
            where_parts.append("rio_id = %s")
            params.append(rio_id)
        if horizonte_h:
            where_parts.append("horizonte_h <= %s")
            params.append(horizonte_h)
        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        rows = _pg_query(f"""
            SELECT rio_id, horizonte_h, valid_ts,
                   nivel_previsto_m, ic_inferior_m, ic_superior_m,
                   modelo_versao, status, gerado_em
            FROM   river_ai_forecasts
            {where}
            ORDER  BY rio_id, horizonte_h
        """, tuple(params))
    except HTTPException:
        rows = []

    previsoes = [
        RiverForecast(
            rio_id           = str(r["rio_id"]),
            horizonte_h      = int(r["horizonte_h"]),
            horizonte_dias   = round(int(r["horizonte_h"]) / 24, 1) if r.get("horizonte_h") else None,
            valid_ts         = _iso(r.get("valid_ts")),
            nivel_previsto_m = _safe_float(r.get("nivel_previsto_m")),
            ic_inferior_m    = _safe_float(r.get("ic_inferior_m")),
            ic_superior_m    = _safe_float(r.get("ic_superior_m")),
            modelo_versao    = r.get("modelo_versao"),
            status           = str(r.get("status") or "ok"),
            gerado_em        = _iso(r.get("gerado_em")),
        )
        for r in rows
    ]

    com_modelo = any(p.status == "ok" for p in previsoes)
    resp = ForecastsResponse(
        timestamp = datetime.now(timezone.utc).isoformat(),
        fase      = "LSTM ativo" if com_modelo else "modelos em calibração",
        total     = len(previsoes),
        previsoes = previsoes,
    )
    _cache_set(cache_key, resp)
    return resp


# ---------------------------------------------------------------------------
# Endpoint 7 — GET /api/v3/forecasts/weather  (timeline NWP estilo Windy)
# ---------------------------------------------------------------------------

@app.get(
    "/api/v3/forecasts/weather",
    tags           = ["Previsões ML"],
    response_model = WeatherForecastResponse,
    summary        = "Previsão NWP por ponto e horário (timeline Windy)",
)
async def forecasts_weather(
    horas: int = Query(168, ge=6, le=168, description="Horizonte em horas (até 7 dias)"),
    location: str | None = Query(None, description="Filtro por localidade"),
) -> WeatherForecastResponse:
    """Retorna a série de previsão NWP (chuva/temp/vento/CAPE) por ponto e hora.

    Alimenta a timeline arrastável e as camadas alternáveis do mapa Windy-style.
    Dados de ``forecasts`` (Open-Meteo, 10 pontos RS). Cache 10 min.

    Args:
        horas:    Janela futura em horas a partir de agora (6–168).
        location: Filtro opcional por localidade (location_name).

    Returns:
        WeatherForecastResponse com pontos {location, lat, lon, valid_ts, ...}.
    """
    cache_key = f"fc_weather:{horas}:{location or 'all'}"
    cached = _cache_get(cache_key, ttl_s=600)
    if cached:
        return cached

    limite = datetime.now(timezone.utc) + timedelta(hours=horas)
    where_parts = ["valid_ts >= NOW()", "valid_ts <= %s"]
    params: list[Any] = [limite]
    if location:
        where_parts.append("location_name = %s")
        params.append(location)
    where = " AND ".join(where_parts)

    try:
        rows = _pg_query(f"""
            SELECT location_name, lat, lon, valid_ts,
                   rain_mm, temperature, wind_speed, cape_j_kg, model_source
            FROM   forecasts
            WHERE  {where}
            ORDER  BY valid_ts, location_name
        """, tuple(params))
    except HTTPException:
        rows = []

    pontos = [
        WeatherForecastPoint(
            location_name = str(r["location_name"]),
            lat           = _safe_float(r.get("lat")),
            lon           = _safe_float(r.get("lon")),
            valid_ts      = _iso(r.get("valid_ts")) or "",
            rain_mm       = _safe_float(r.get("rain_mm")),
            temperature   = _safe_float(r.get("temperature")),
            wind_speed    = _safe_float(r.get("wind_speed")),
            cape_j_kg     = _safe_float(r.get("cape_j_kg")),
            model_source  = r.get("model_source"),
        )
        for r in rows
    ]
    locs = sorted({p.location_name for p in pontos})

    resp = WeatherForecastResponse(
        timestamp    = datetime.now(timezone.utc).isoformat(),
        horas        = horas,
        total_pontos = len(pontos),
        localidades  = locs,
        previsoes    = pontos,
    )
    _cache_set(cache_key, resp)
    return resp


# ---------------------------------------------------------------------------
# Endpoint 8 — GET /api/v3/forecasts/model-info  (transparência ML)
# ---------------------------------------------------------------------------

_IC_AVISO = (
    "Intervalos de confiança em calibração — cobertura empírica atual "
    "16–37%; não interpretar a faixa como certeza absoluta."
)


@app.get(
    "/api/v3/forecasts/model-info",
    tags           = ["Previsões ML"],
    response_model = ModelInfoResponse,
    summary        = "Metadados dos modelos LSTM (MAE, treino, status)",
)
async def forecasts_model_info() -> ModelInfoResponse:
    """Retorna metadados de transparência por modelo LSTM.

    Combina: (1) sidecar leve ``models/{rio}_meta.json`` no R2 (MAE, nMAE,
    data de treino — escrito pelo save_model) e (2) status/última inferência
    de ``river_ai_forecasts`` no Supabase. Não carrega torch.

    Status derivado: ``aprovado`` (modelo homologado e inferência ok),
    ``calibrando`` (homologado mas IC sob revisão), ``pendente`` (sem
    inferência válida), ``ausente`` (sem modelo no R2).

    Returns:
        ModelInfoResponse com um item por rio conhecido.
    """
    import json

    cached = _cache_get("model_info", ttl_s=600)
    if cached:
        return cached

    rios = ["guaiba", "jacui", "taquari", "sinos", "camaqua",
            "cai", "ibicui", "ijui", "gravatai", "pardo"]

    # Última inferência por rio (status real do snapshot).
    infer: dict[str, dict[str, Any]] = {}
    try:
        for r in _pg_query("""
            SELECT rio_id, MAX(gerado_em) AS ultima,
                   BOOL_OR(status = 'ok') AS tem_ok
            FROM   river_ai_forecasts
            GROUP  BY rio_id
        """):
            infer[str(r["rio_id"])] = r
    except HTTPException:
        infer = {}

    bucket = os.getenv("R2_BUCKET_NAME", "")
    try:
        s3 = _s3_client()
    except Exception:  # noqa: BLE001
        s3 = None

    modelos: list[ModelInfoItem] = []
    for rio in rios:
        meta: dict[str, Any] = {}
        if s3 and bucket:
            try:
                body = s3.get_object(Bucket=bucket, Key=f"models/{rio}_meta.json")["Body"].read()
                meta = json.loads(body)
            except Exception:  # noqa: BLE001 — sidecar ausente é esperado
                meta = {}

        inf = infer.get(rio, {})
        tem_ok = bool(inf.get("tem_ok"))
        homolog = str(meta.get("status", "")).upper() == "APROVADO"

        if not meta and not inf:
            status = "ausente"
        elif tem_ok and homolog:
            status = "calibrando"   # homologado por MAE, mas IC ainda em calibração
        elif tem_ok:
            status = "aprovado"
        else:
            status = "pendente"

        modelos.append(ModelInfoItem(
            rio_id            = rio,
            status            = status,
            modelo_versao     = meta.get("version"),
            mae_24h_m         = _safe_float((meta.get("metrics") or {}).get("mae_24h_m")),
            nmae_pct          = _safe_float((meta.get("metrics") or {}).get("nmae_pct")),
            treinado_em       = meta.get("salvo_em"),
            ultima_inferencia = _iso(inf.get("ultima")),
            horizontes_dias   = meta.get("horizontes_dias") or [1, 2, 3, 6],
            aviso_ic          = _IC_AVISO,
            periodo_treino    = meta.get("periodo_treino"),
            ressalva          = meta.get("ressalva"),
        ))

    resp = ModelInfoResponse(
        timestamp = datetime.now(timezone.utc).isoformat(),
        total     = len(modelos),
        modelos   = modelos,
    )
    _cache_set("model_info", resp)
    return resp


# ---------------------------------------------------------------------------
# Endpoint 9 — GET /api/v3/analytics/trend  (tendência histórica do R2)
# ---------------------------------------------------------------------------

def _read_r2_one_per_day(fonte: str, days: int) -> pd.DataFrame:
    """Lê 1 Parquet por dia (o mais recente de cada partição) — leve.

    Para fontes de SNAPSHOT (ex.: live_river_levels grava o estado a cada run),
    basta um arquivo por dia para montar a série diária. Evita baixar as ~144
    partições/dia do cron, mantendo a resposta < 2s.

    Args:
        fonte: nome da fonte (partição Hive em historico/{fonte}/...).
        days:  janela em dias para trás.

    Returns:
        DataFrame concatenado (1 snapshot/dia), possivelmente vazio.
    """
    from concurrent.futures import ThreadPoolExecutor

    bucket = os.getenv("R2_BUCKET_NAME", "")
    s3 = _s3_client()
    hoje = date.today()

    def _um_dia(i: int) -> pd.DataFrame | None:
        d = hoje - timedelta(days=i)
        prefix = f"historico/{fonte}/ano={d.year}/mes={d.month:02d}/dia={d.day:02d}/"
        try:
            objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
            if not objs:
                return None
            key = max(objs, key=lambda o: o["LastModified"])["Key"]
            raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            return pd.read_parquet(io.BytesIO(raw))
        except (ClientError, ValueError, OSError) as exc:
            logger.warning(f"trend: dia {d} ignorado ({exc})")
            return None

    # Paraleliza os round-trips ao R2 (latência domina) — mantém < 2s.
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = [f for f in ex.map(_um_dia, range(days)) if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_r2_latest(fonte: str, days_back: int = 7) -> pd.DataFrame:
    """Lê o Parquet mais recente de uma fonte (1 arquivo).

    Para fontes que gravam SÉRIE (ex.: live_rain_readings carrega ~30 dias de
    rain_1h_mm por arquivo), um único Parquet já cobre a janela — leitura O(1).

    Args:
        fonte:     nome da fonte.
        days_back: dias a retroceder procurando a partição mais recente.

    Returns:
        DataFrame do Parquet mais novo, ou vazio.
    """
    bucket = os.getenv("R2_BUCKET_NAME", "")
    s3 = _s3_client()
    now = datetime.now(timezone.utc)
    for i in range(days_back + 1):
        d = now - timedelta(days=i)
        prefix = f"historico/{fonte}/ano={d.year}/mes={d.month:02d}/dia={d.day:02d}/"
        try:
            objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
        except ClientError:
            continue
        if not objs:
            continue
        key = max(objs, key=lambda o: o["LastModified"])["Key"]
        try:
            raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            return pd.read_parquet(io.BytesIO(raw))
        except (ClientError, ValueError, OSError) as exc:
            logger.warning(f"trend: parquet {key} ignorado ({exc})")
            return pd.DataFrame()
    return pd.DataFrame()


def _classifica_tendencia(dias: list[str], valores: list[float],
                          deadband: float) -> tuple[str, float | None]:
    """Regressão linear simples (mín. quadrados) nos últimos 7 pontos.

    Args:
        dias:     Datas ISO ordenadas.
        valores:  Valores diários alinhados a ``dias``.
        deadband: Faixa morta da inclinação (unidade/dia) p/ classificar ESTAVEL.

    Returns:
        (tendencia, inclinacao) — tendência em {SUBINDO, ESTAVEL, DESCENDO,
        INDEFINIDA}; inclinacao em unidade/dia (None se dados insuficientes).
    """
    import numpy as np

    pares = [(i, v) for i, v in enumerate(valores) if v is not None and not math.isnan(v)]
    if len(pares) < 3:
        return "INDEFINIDA", None
    janela = pares[-7:]
    xs = np.array([p[0] for p in janela], dtype=float)
    ys = np.array([p[1] for p in janela], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    if slope > deadband:
        return "SUBINDO", round(slope, 4)
    if slope < -deadband:
        return "DESCENDO", round(slope, 4)
    return "ESTAVEL", round(slope, 4)


@app.get(
    "/api/v3/analytics/trend",
    tags           = ["Analítica"],
    response_model = TrendResponse,
    summary        = "Tendência histórica de um rio ou estação (R2)",
)
async def analytics_trend(
    rio_ou_estacao: str = Query(..., description="rio_id (ex.: guaiba) ou station_id (ex.: A801)"),
    dias: int = Query(14, ge=3, le=90, description="Janela histórica em dias (3–90)"),
) -> TrendResponse:
    """Série diária agregada + indicador de tendência (regressão 7 dias).

    Detecta automaticamente se o id é um rio (``live_river_levels``) ou uma
    estação (``live_rain_readings``); lê os Parquets do R2 no período, agrega
    por dia (nível médio do rio / precipitação diária da estação) e classifica
    a tendência por regressão linear simples nos últimos 7 dias. Cache 1h.

    Args:
        rio_ou_estacao: rio_id ou station_id.
        dias:           Janela histórica (3–90).

    Returns:
        TrendResponse com série diária e tendência.
    """
    ident = rio_ou_estacao.strip()
    cache_key = f"trend:{ident}:{dias}"
    cached = _cache_get(cache_key, ttl_s=3600)
    if cached:
        return cached

    # Detecta tipo: existe como rio? (e recupera rio_nome p/ casar com o Parquet,
    # cujo identificador é rio_nome/station_code, não rio_id)
    rio_nome: str | None = None
    try:
        row = _pg_query(
            "SELECT rio_nome FROM live_river_levels WHERE rio_id = %s LIMIT 1", (ident,)
        )
        if row:
            rio_nome = row[0].get("rio_nome")
    except HTTPException:
        row = []
    is_rio = bool(row)
    tipo = "rio" if is_rio else "estacao"

    serie: list[TrendDay] = []
    metrica = "nivel_m" if is_rio else "precip_mm"

    if is_rio:
        # Snapshots por run → 1 arquivo/dia basta para a série diária de nível.
        df = _read_r2_one_per_day("live_river_levels", dias)
        if not df.empty:
            if "rio_id" in df.columns:
                df = df[df["rio_id"].astype(str) == ident].copy()
            elif "rio_nome" in df.columns and rio_nome:
                df = df[df["rio_nome"].astype(str) == rio_nome].copy()
            else:
                df = df.iloc[0:0].copy()
            nivel = next((c for c in ("current_level_m", "nivel_atual_m", "nivel_m", "level_m")
                          if c in df.columns), None)
            tsc = next((c for c in ("updated_at", "timestamp", "ts") if c in df.columns), None)
            if nivel and tsc and not df.empty:
                df["d"] = pd.to_datetime(df[tsc], errors="coerce").dt.date
                df["v"] = pd.to_numeric(df[nivel], errors="coerce")
                g = df.dropna(subset=["d", "v"]).groupby("d")["v"].mean().sort_index()
                serie = [TrendDay(data=str(d), nivel_m=round(float(v), 3), precip_mm=None)
                         for d, v in g.items()]
        deadband = 0.03   # m/dia
    else:
        # live_rain_readings grava ~30 dias de série por arquivo → 1 arquivo cobre a janela.
        df = _read_r2_latest("live_rain_readings")
        if not df.empty and "station_id" in df.columns:
            df = df[df["station_id"].astype(str) == ident].copy()
            rain = next((c for c in ("rain_1h_mm", "precip_1h", "rain_mm") if c in df.columns), None)
            tsc = next((c for c in ("ts", "timestamp") if c in df.columns), None)
            if rain and tsc and not df.empty:
                df["_ts"] = pd.to_datetime(df[tsc], utc=True, errors="coerce")
                df = df.dropna(subset=["_ts"])
                # Ancora no último dado disponível (robusto ao lag do INMET),
                # não em now() — mesma filosofia do rain_accumulator.
                if not df.empty:
                    corte = df["_ts"].max() - pd.Timedelta(days=dias)
                    df = df[df["_ts"] >= corte]
                    df["d"] = df["_ts"].dt.date
                    df["v"] = pd.to_numeric(df[rain], errors="coerce")
                    g = df.dropna(subset=["d", "v"]).groupby("d")["v"].sum().sort_index()
                    serie = [TrendDay(data=str(d), precip_mm=round(float(v), 2), nivel_m=None)
                             for d, v in g.items()]
        deadband = 2.0    # mm/dia

    valores = [(s.nivel_m if is_rio else s.precip_mm) for s in serie]
    tendencia, inclinacao = _classifica_tendencia(
        [s.data for s in serie], valores, deadband
    )

    resp = TrendResponse(
        timestamp        = datetime.now(timezone.utc).isoformat(),
        id               = ident,
        tipo             = tipo,
        dias             = dias,
        n_dias_dados     = len(serie),
        tendencia        = tendencia,
        inclinacao       = inclinacao,
        metrica_tendencia = metrica,
        serie            = serie,
    )
    _cache_set(cache_key, resp)
    return resp


# ---------------------------------------------------------------------------
# Endpoints 10/11 — Rede Hidrometeorológica Defesa Civil RS (DCRS)
# ---------------------------------------------------------------------------

@app.get(
    "/api/v3/dcrs/bacias",
    tags           = ["Bacias RS (Defesa Civil)"],
    response_model = DcrsBaciasResponse,
    summary        = "Bacias hidrográficas com estações DCRS (contagens)",
)
async def dcrs_bacias() -> DcrsBaciasResponse:
    """Lista as bacias da rede DCRS com nº de estações e chuva 24h máxima.

    Alimenta o selectbox de bacias da aba "Bacias RS". Cache 5 min.

    Returns:
        DcrsBaciasResponse ordenada por nº de estações (desc).
    """
    cached = _cache_get("dcrs_bacias", ttl_s=300)
    if cached:
        return cached

    rows = _pg_query("""
        SELECT bacia,
               COUNT(*)                              AS estacoes,
               COUNT(rio_nivel)                      AS com_rio,
               MAX(chuva_24h)                        AS chuva_24h_max
        FROM   live_dcrs_stations
        WHERE  bacia IS NOT NULL AND bacia <> ''
        GROUP  BY bacia
        ORDER  BY estacoes DESC, bacia
    """)
    bacias = [
        DcrsBaciaItem(
            bacia         = str(r["bacia"]),
            estacoes      = int(r["estacoes"]),
            com_rio       = int(r["com_rio"]),
            chuva_24h_max = _safe_float(r.get("chuva_24h_max")),
        )
        for r in rows
    ]
    resp = DcrsBaciasResponse(
        timestamp = datetime.now(timezone.utc).isoformat(),
        total     = len(bacias),
        bacias    = bacias,
    )
    _cache_set("dcrs_bacias", resp)
    return resp


@app.get(
    "/api/v3/dcrs/stations",
    tags           = ["Bacias RS (Defesa Civil)"],
    response_model = DcrsStationsResponse,
    summary        = "Estações DCRS (nível de rio + chuva + met) por bacia",
)
async def dcrs_stations(
    bacia: str | None = Query(None, description="Filtro por bacia (nome exato)"),
) -> DcrsStationsResponse:
    """Snapshot das estações da Defesa Civil RS (live_dcrs_stations).

    IMPORTANTE: ``rio_nivel`` está na unidade BRUTA da rede (mista por
    estação — lagoas em metros, serra aparenta cm). O frontend aplica
    heurística de exibição sinalizada. Cache 5 min.

    Args:
        bacia: Nome exato da bacia para filtrar (None = todas).

    Returns:
        DcrsStationsResponse com estações e lista de bacias presentes.
    """
    cache_key = f"dcrs_stations:{bacia or 'all'}"
    cached = _cache_get(cache_key, ttl_s=300)
    if cached:
        return cached

    where  = "WHERE bacia = %s" if bacia else ""
    params: tuple[Any, ...] = (bacia,) if bacia else ()
    rows = _pg_query(f"""
        SELECT codigo, nome, bacia, latitude, longitude,
               rio_nome, rio_nivel, rio_vazao, rio_tendencia,
               cota_atencao, cota_alerta, cota_emergencia,
               chuva_1h, chuva_3h, chuva_6h, chuva_12h, chuva_24h,
               chuva_48h, chuva_72h, chuva_168h,
               temperatura, umidade, vento_vel, vento_dir,
               pressao, senstermica, radiacao, "timestamp"
        FROM   live_dcrs_stations
        {where}
        ORDER  BY bacia, codigo
    """, params)

    estacoes = [
        DcrsStation(
            codigo          = str(r["codigo"]),
            nome            = r.get("nome"),
            bacia           = r.get("bacia"),
            latitude        = _safe_float(r.get("latitude")),
            longitude       = _safe_float(r.get("longitude")),
            rio_nome        = r.get("rio_nome"),
            rio_nivel       = _safe_float(r.get("rio_nivel")),
            rio_vazao       = _safe_float(r.get("rio_vazao")),
            rio_tendencia   = _safe_float(r.get("rio_tendencia")),
            cota_atencao    = _safe_float(r.get("cota_atencao")),
            cota_alerta     = _safe_float(r.get("cota_alerta")),
            cota_emergencia = _safe_float(r.get("cota_emergencia")),
            chuva_1h        = _safe_float(r.get("chuva_1h")),
            chuva_3h        = _safe_float(r.get("chuva_3h")),
            chuva_6h        = _safe_float(r.get("chuva_6h")),
            chuva_12h       = _safe_float(r.get("chuva_12h")),
            chuva_24h       = _safe_float(r.get("chuva_24h")),
            chuva_48h       = _safe_float(r.get("chuva_48h")),
            chuva_72h       = _safe_float(r.get("chuva_72h")),
            chuva_168h      = _safe_float(r.get("chuva_168h")),
            temperatura     = _safe_float(r.get("temperatura")),
            umidade         = _safe_float(r.get("umidade")),
            vento_vel       = _safe_float(r.get("vento_vel")),
            vento_dir       = _safe_float(r.get("vento_dir")),
            pressao         = _safe_float(r.get("pressao")),
            senstermica     = _safe_float(r.get("senstermica")),
            radiacao        = _safe_float(r.get("radiacao")),
            timestamp       = _iso(r.get("timestamp")),
        )
        for r in rows
    ]
    resp = DcrsStationsResponse(
        timestamp      = datetime.now(timezone.utc).isoformat(),
        fonte          = "Rede Hidrometeorológica Defesa Civil RS",
        total_estacoes = len(estacoes),
        bacias         = sorted({e.bacia for e in estacoes if e.bacia}),
        estacoes       = estacoes,
    )
    _cache_set(cache_key, resp)
    return resp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    logger.info(f"Iniciando Monitor Hidrometeorológico RS API na porta {port}...")
    uvicorn.run(
        "main:app",
        host      = "0.0.0.0",
        port      = port,
        reload    = False,
        workers   = 1,
        log_level = "info",
    )
