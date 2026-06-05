"""
Agente 3 — Engenheiro de Software
API REST FastAPI para o Monitor Hidrometeorológico RS.

Endpoints:
  GET  /                         — health check
  GET  /v1/rivers/status         — status atual de todos os rios
  GET  /v1/rivers/{river}/levels — série temporal de nível por rio
  GET  /v1/rain/accumulated      — acumulados de chuva por estação
  GET  /v1/forecasts             — previsões NWP próximas 7 dias
  GET  /v1/alerts                — alertas ativos não resolvidos
  POST /v1/alerts/{id}/resolve   — marca alerta como resolvido
  GET  /v1/stats                 — estatísticas do banco

Segurança: Bearer JWT via header Authorization.
Rate limit: 60 req/min por IP (produção) ou 600 (dev).
Deploy: Railway free tier — porta $PORT ou 8000.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR      = _PROJECT_ROOT / "src"
for _p in (str(_SRC_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from database.db_manager import ClimateDB  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração JWT e rate limiting
# ---------------------------------------------------------------------------

_JWT_SECRET    = os.getenv("JWT_SECRET", "monitor-hidro-rs-dev-secret-change-in-prod")
_JWT_ALGORITHM = "HS256"
_JWT_EXP_H     = int(os.getenv("JWT_EXP_HOURS", "24"))
_API_KEY       = os.getenv("API_KEY", "dev-key-monitor-rs")  # para gerar tokens

_limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Monitor Hidrometeorológico RS",
    description = (
        "API REST para dados hidrometeorológicos em tempo real do "
        "Rio Grande do Sul. Cobre rios Sinos, Taquari, Jacuí, Guaíba, "
        "Camaquã e Lagoa dos Patos."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Authorization", "Content-Type"],
)

_security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Dependências: autenticação e banco
# ---------------------------------------------------------------------------

def _verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict[str, Any]:
    """Verifica e decodifica o Bearer JWT.

    Args:
        credentials: Header Authorization do request.

    Returns:
        Payload do JWT como dicionário.

    Raises:
        HTTPException 401: Se token ausente, expirado ou inválido.
    """
    if credentials is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Token JWT obrigatório.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    try:
        payload: dict[str, Any] = jwt.decode(
            credentials.credentials, _JWT_SECRET, algorithms=[_JWT_ALGORITHM]
        )
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = f"Token inválido: {exc}",
            headers     = {"WWW-Authenticate": "Bearer"},
        ) from exc


def _get_db() -> ClimateDB:
    """Abre uma conexão DuckDB para o request.

    Returns:
        Instância ClimateDB conectada.
    """
    return ClimateDB()


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------

class TokenRequest(BaseModel):
    api_key: str = Field(..., description="Chave de API para obter JWT")


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int


class RiverStatusItem(BaseModel):
    river:             str
    segment:           str
    level_m:           float | None
    status:            str
    trend:             str | None
    pct_cota_alerta:   float | None
    municipalities_risk: list[str] | None
    updated_at:        datetime | None


class RainAccumItem(BaseModel):
    station_id: str
    date:       str
    rain_1h:    float
    rain_3h:    float
    rain_6h:    float
    rain_24h:   float
    rain_72h:   float
    rain_7d:    float | None
    updated_at: datetime | None


class ForecastItem(BaseModel):
    location_name:  str
    valid_ts:       datetime
    rain_mm:        float | None
    temperature:    float | None
    wind_speed:     float | None
    cape_j_kg:      float | None
    lifted_index:   float | None
    model_source:   str


class AlertItem(BaseModel):
    id:           int
    ts:           datetime
    alert_type:   str
    severity:     str
    location:     str | None
    river:        str | None
    message:      str
    level_m:      float | None
    threshold_m:  float | None
    telegram_sent: bool


class StatsResponse(BaseModel):
    tables:         dict[str, int]
    db_size_mb:     float
    oldest_reading: str
    newest_reading: str
    generated_at:   datetime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
@_limiter.limit("120/minute")
async def health(request: Request) -> dict[str, str]:
    """Health check — retorna status e versão."""
    return {
        "status":     "ok",
        "service":    "Monitor Hidrometeorológico RS",
        "version":    "1.0.0",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/token", tags=["Auth"], response_model=TokenResponse)
@_limiter.limit("10/minute")
async def get_token(request: Request, body: TokenRequest) -> TokenResponse:
    """Gera um JWT Bearer token a partir de uma API key.

    A API key é configurada via variável de ambiente `API_KEY`.

    Args:
        body: Payload com api_key.

    Returns:
        TokenResponse com access_token e validade.

    Raises:
        HTTPException 401: Se a API key for inválida.
    """
    if body.api_key != _API_KEY:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "API key inválida.",
        )
    exp     = datetime.now(timezone.utc) + timedelta(hours=_JWT_EXP_H)
    payload = {"sub": "api_client", "exp": exp}
    token   = jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)
    return TokenResponse(
        access_token = token,
        expires_in   = _JWT_EXP_H * 3600,
    )


@app.get(
    "/v1/rivers/status",
    tags     = ["Rios"],
    response_model = list[RiverStatusItem],
    summary  = "Status atual de todos os rios monitorados",
)
@_limiter.limit("60/minute")
async def rivers_status(
    request: Request,
    river:   str | None = Query(None, description="Filtro por nome do rio"),
    _token:  dict       = Depends(_verify_token),
    db:      ClimateDB  = Depends(_get_db),
) -> list[RiverStatusItem]:
    """Retorna o status operacional atual de cada segmento monitorado.

    Args:
        river: Nome do rio para filtrar (opcional).

    Returns:
        Lista de RiverStatusItem com nível, status, tendência e municípios.
    """
    try:
        df = db.get_river_status(river=river)
    finally:
        db.close()

    items: list[RiverStatusItem] = []
    for _, row in df.iterrows():
        mun = row.get("municipalities_risk")
        if isinstance(mun, list):
            pass
        elif mun is not None:
            mun = list(mun)
        else:
            mun = []
        items.append(RiverStatusItem(
            river             = str(row["river"]),
            segment           = str(row["segment"]),
            level_m           = _safe_float(row.get("level_m")),
            status            = str(row.get("status", "NORMAL")),
            trend             = str(row.get("trend", "ESTAVEL")),
            pct_cota_alerta   = _safe_float(row.get("pct_cota_alerta")),
            municipalities_risk = mun,
            updated_at        = _safe_dt(row.get("updated_at")),
        ))
    return items


@app.get(
    "/v1/rivers/{river}/levels",
    tags    = ["Rios"],
    summary = "Série temporal de nível por rio",
)
@_limiter.limit("30/minute")
async def river_levels(
    request:  Request,
    river:    str,
    hours:    int       = Query(72, ge=1, le=720, description="Horas para trás"),
    _token:   dict      = Depends(_verify_token),
    db:       ClimateDB = Depends(_get_db),
) -> JSONResponse:
    """Retorna série temporal de nível de um rio específico.

    Args:
        river: Nome do rio (ex.: Guaíba, Taquari).
        hours: Janela de tempo em horas (1–720).

    Returns:
        JSON com lista de {ts, station_id, level_m, flow_cms}.
    """
    try:
        df = db._con.execute("""
            SELECT ts, station_id, level_m, flow_cms
            FROM   river_levels
            WHERE  river = ?
              AND  ts >= now() - INTERVAL ? || ' hours'
            ORDER BY ts DESC
        """, [river, str(hours)]).df()
    except duckdb.Error as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()

    if df.empty:
        return JSONResponse(content={"river": river, "data": []})

    df["ts"] = df["ts"].astype(str)
    return JSONResponse(content={
        "river": river,
        "hours": hours,
        "count": len(df),
        "data":  df.to_dict(orient="records"),
    })


@app.get(
    "/v1/rain/accumulated",
    tags           = ["Chuva"],
    response_model = list[RainAccumItem],
    summary        = "Acumulados de chuva por estação",
)
@_limiter.limit("60/minute")
async def rain_accumulated(
    request:    Request,
    station_id: str | None = Query(None, description="Filtro por estação"),
    _token:     dict       = Depends(_verify_token),
    db:         ClimateDB  = Depends(_get_db),
) -> list[RainAccumItem]:
    """Retorna o acumulado de chuva mais recente por estação.

    Args:
        station_id: Código da estação para filtrar (opcional).

    Returns:
        Lista de RainAccumItem com acumulados 1h/3h/6h/24h/72h/7d.
    """
    try:
        where = "WHERE station_id = ?" if station_id else ""
        params = [station_id] if station_id else []
        df = db._con.execute(f"""
            SELECT station_id, date, rain_1h, rain_3h, rain_6h, rain_12h,
                   rain_24h, rain_48h, rain_72h,
                   TRY_CAST(rain_7d AS DOUBLE) AS rain_7d,
                   updated_at
            FROM   rain_accumulated
            {where}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY date DESC) = 1
            ORDER BY station_id
        """, params).df()
    except duckdb.Error as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()

    items: list[RainAccumItem] = []
    for _, row in df.iterrows():
        items.append(RainAccumItem(
            station_id = str(row["station_id"]),
            date       = str(row["date"]),
            rain_1h    = float(row.get("rain_1h")  or 0),
            rain_3h    = float(row.get("rain_3h")  or 0),
            rain_6h    = float(row.get("rain_6h")  or 0),
            rain_24h   = float(row.get("rain_24h") or 0),
            rain_72h   = float(row.get("rain_72h") or 0),
            rain_7d    = _safe_float(row.get("rain_7d")),
            updated_at = _safe_dt(row.get("updated_at")),
        ))
    return items


@app.get(
    "/v1/forecasts",
    tags           = ["Previsões"],
    response_model = list[ForecastItem],
    summary        = "Previsões NWP próximas 7 dias",
)
@_limiter.limit("30/minute")
async def forecasts(
    request:  Request,
    location: str | None = Query(None, description="Filtro por localidade"),
    hours:    int        = Query(48,  ge=1, le=168, description="Horas à frente"),
    _token:   dict       = Depends(_verify_token),
    db:       ClimateDB  = Depends(_get_db),
) -> list[ForecastItem]:
    """Retorna previsões meteorológicas NWP (Open-Meteo/NOAA).

    Args:
        location: Nome da localidade para filtrar (ex.: Porto Alegre).
        hours: Horizonte de previsão em horas (1–168).

    Returns:
        Lista de ForecastItem com variáveis meteorológicas por hora.
    """
    try:
        where  = "AND location_name = ?" if location else ""
        params: list[Any] = [str(hours)]
        if location:
            params.append(location)
        df = db._con.execute(f"""
            SELECT location_name, valid_ts, rain_mm, temperature,
                   wind_speed, cape_j_kg, lifted_index, model_source
            FROM   forecasts
            WHERE  valid_ts BETWEEN now() AND now() + INTERVAL ? || ' hours'
              {where}
            ORDER BY location_name, valid_ts
            LIMIT  2016
        """, params).df()
    except duckdb.Error as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()

    items: list[ForecastItem] = []
    for _, row in df.iterrows():
        items.append(ForecastItem(
            location_name = str(row["location_name"]),
            valid_ts      = row["valid_ts"],
            rain_mm       = _safe_float(row.get("rain_mm")),
            temperature   = _safe_float(row.get("temperature")),
            wind_speed    = _safe_float(row.get("wind_speed")),
            cape_j_kg     = _safe_float(row.get("cape_j_kg")),
            lifted_index  = _safe_float(row.get("lifted_index")),
            model_source  = str(row.get("model_source", "")),
        ))
    return items


@app.get(
    "/v1/alerts",
    tags           = ["Alertas"],
    response_model = list[AlertItem],
    summary        = "Alertas ativos não resolvidos",
)
@_limiter.limit("60/minute")
async def active_alerts(
    request:  Request,
    severity: str | None = Query(
        None,
        description="Filtro por severidade: INFO | ATENCAO | ALERTA | EMERGENCIA",
    ),
    _token:  dict       = Depends(_verify_token),
    db:      ClimateDB  = Depends(_get_db),
) -> list[AlertItem]:
    """Retorna os alertas ativos (não resolvidos).

    Args:
        severity: Filtro opcional por nível de severidade.

    Returns:
        Lista de AlertItem ordenada por data decrescente.
    """
    try:
        df = db.get_active_alerts(severity=severity)
    finally:
        db.close()

    items: list[AlertItem] = []
    for _, row in df.iterrows():
        items.append(AlertItem(
            id            = int(row["id"]),
            ts            = row["ts"],
            alert_type    = str(row["alert_type"]),
            severity      = str(row["severity"]),
            location      = row.get("location"),
            river         = row.get("river"),
            message       = str(row["message"]),
            level_m       = _safe_float(row.get("level_m")),
            threshold_m   = _safe_float(row.get("threshold_m")),
            telegram_sent = bool(row.get("telegram_sent", False)),
        ))
    return items


@app.post(
    "/v1/alerts/{alert_id}/resolve",
    tags    = ["Alertas"],
    summary = "Marca um alerta como resolvido",
)
@_limiter.limit("30/minute")
async def resolve_alert(
    request:  Request,
    alert_id: int,
    _token:   dict      = Depends(_verify_token),
    db:       ClimateDB = Depends(_get_db),
) -> dict[str, Any]:
    """Marca um alerta como resolvido com timestamp atual.

    Args:
        alert_id: ID do alerta na tabela alerts_log.

    Returns:
        Confirmação com id e resolved_at.

    Raises:
        HTTPException 404: Se o alerta não for encontrado.
    """
    try:
        result = db._con.execute("""
            UPDATE alerts_log
            SET    resolved    = TRUE,
                   resolved_at = now()
            WHERE  id          = ?
        """, [alert_id])
        db._con.commit()
        rows = result.rowcount if result.rowcount >= 0 else 1
    except duckdb.Error as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()

    if rows == 0:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"Alerta {alert_id} não encontrado.",
        )
    return {
        "id":          alert_id,
        "resolved":    True,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/v1/stats",
    tags           = ["Sistema"],
    response_model = StatsResponse,
    summary        = "Estatísticas do banco de dados",
)
@_limiter.limit("10/minute")
async def db_stats(
    request: Request,
    _token:  dict      = Depends(_verify_token),
    db:      ClimateDB = Depends(_get_db),
) -> StatsResponse:
    """Retorna contagem de registros em cada tabela e tamanho do banco.

    Returns:
        StatsResponse com tabelas, tamanho em MB e timestamps extremos.
    """
    try:
        stats = db.get_db_stats()
    finally:
        db.close()

    return StatsResponse(
        tables         = stats["tables"],
        db_size_mb     = stats["db_size_mb"],
        oldest_reading = stats.get("oldest_reading", "N/A"),
        newest_reading = stats.get("newest_reading", "N/A"),
        generated_at   = datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Converte valor para float, retornando None em caso de falha.

    Args:
        val: Qualquer valor (float, int, str, None, NaN).

    Returns:
        float ou None.
    """
    if val is None:
        return None
    try:
        f = float(val)
        import math
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_dt(val: Any) -> datetime | None:
    """Converte valor para datetime, retornando None em caso de falha.

    Args:
        val: Timestamp ou string de data.

    Returns:
        datetime ou None.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Entry point para uvicorn
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    logger.info(f"Iniciando API na porta {port}...")
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = port,
        reload  = False,
        workers = 1,
        log_level = "info",
    )
