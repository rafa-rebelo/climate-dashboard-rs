"""
Agente 3 — Engenheiro de Software / API
API REST FastAPI v3 — Monitor Hidrometeorológico RS

Endpoints:
  GET  /health                       — status Supabase + R2
  GET  /api/v3/rivers/status         — níveis e alertas (live_river_levels)
  GET  /api/v3/weather/heatmap       — grade GPM precip (live_gpm_precip) GZip
  GET  /api/v3/stations/readings     — leituras INMET (live_rain_readings)
  GET  /api/v3/analytics/history     — histórico R2 por período (boto3 Parquet)
  GET  /api/v3/forecasts/rivers      — previsões ML (river_ai_forecasts — Fase 2)

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
    precip_1h:    float | None
    precip_6h:    float | None
    precip_24h:   float | None
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
    rio_id:         str
    rio_nome:       str | None
    horizonte_h:    int
    nivel_previsto_m: float | None
    intervalo_inf_m:  float | None
    intervalo_sup_m:  float | None
    confianca_pct:    float | None
    modelo:         str | None
    gerado_em:      str | None


class ForecastsResponse(BaseModel):
    timestamp:  str
    fase:       str
    total:      int
    previsoes:  list[RiverForecast]


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
               r.precip_1h,
               r.precip_6h,
               r.precip_24h,
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
            precip_1h   = _safe_float(r.get("precip_1h")),
            precip_6h   = _safe_float(r.get("precip_6h")),
            precip_24h  = _safe_float(r.get("precip_24h")),
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
    summary        = "Previsões ML de nível dos rios (Fase 2)",
)
async def forecasts_rivers(
    rio_id: str | None = Query(None, description="Filtro por rio_id"),
    horizonte_h: int   = Query(24, ge=6, le=72, description="Horizonte em horas"),
) -> ForecastsResponse:
    """Retorna previsões LSTM por bacia hidrográfica.

    Tabela ``river_ai_forecasts`` — disponível na Fase 2 (modelos ML).
    Retorna lista vazia enquanto os modelos estão em treinamento.

    Args:
        rio_id:      Filtro por rio (ex.: guaiba, sinos).
        horizonte_h: Horizonte máximo de previsão em horas.

    Returns:
        ForecastsResponse com lista de previsões (vazia na Fase 1).
    """
    cache_key = f"forecasts:{rio_id or 'all'}:{horizonte_h}"
    cached = _cache_get(cache_key, ttl_s=300)
    if cached:
        return cached

    try:
        where_parts = []
        params: list[Any] = []
        if rio_id:
            where_parts.append("rio_id = %s")
            params.append(rio_id)
        if horizonte_h:
            where_parts.append("horizonte_h <= %s")
            params.append(horizonte_h)
        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        rows = _pg_query(f"""
            SELECT rio_id, rio_nome, horizonte_h,
                   nivel_previsto_m, intervalo_inf_m, intervalo_sup_m,
                   confianca_pct, modelo, gerado_em
            FROM   river_ai_forecasts
            {where}
            ORDER  BY rio_id, horizonte_h
        """, tuple(params))
    except HTTPException:
        # Tabela pode não existir na Fase 1
        rows = []

    previsoes = [
        RiverForecast(
            rio_id              = str(r["rio_id"]),
            rio_nome            = r.get("rio_nome"),
            horizonte_h         = int(r["horizonte_h"]),
            nivel_previsto_m    = _safe_float(r.get("nivel_previsto_m")),
            intervalo_inf_m     = _safe_float(r.get("intervalo_inf_m")),
            intervalo_sup_m     = _safe_float(r.get("intervalo_sup_m")),
            confianca_pct       = _safe_float(r.get("confianca_pct")),
            modelo              = r.get("modelo"),
            gerado_em           = _iso(r.get("gerado_em")),
        )
        for r in rows
    ]

    resp = ForecastsResponse(
        timestamp = datetime.now(timezone.utc).isoformat(),
        fase      = "Fase 1 — modelos em treinamento" if not previsoes else "Fase 2",
        total     = len(previsoes),
        previsoes = previsoes,
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
