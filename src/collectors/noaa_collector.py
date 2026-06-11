"""
Agente 1 — Arquiteto de Dados
Coletor de previsões NWP via Open-Meteo API (NOAA GFS + ECMWF, sem autenticação).
Coleta dados horários e diários para os 10 pontos de grade do RS definidos em
config.yaml, aplica retry com tenacity, persiste em Parquet e faz upsert no DuckDB.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import niquests
import numpy as np
import openmeteo_requests
import pandas as pd
import yaml
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Garante que src/ está no path quando executado como script standalone
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

try:
    from database.hybrid_writer import HybridWriter
    _HW_OK = True
except ImportError:
    HybridWriter = None   # type: ignore[assignment,misc]
    _HW_OK = False
    logger.warning("HybridWriter não disponível — noaa_collector sem persistência híbrida.")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
_CONFIG_PATH   = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_CACHE_DIR     = Path(__file__).resolve().parents[2] / ".cache"
_PARQUET_DIR   = Path(__file__).resolve().parents[2] / "data" / "forecasts"
_CACHE_TTL_S   = 600   # 10 minutos

# Variáveis horárias solicitadas (a ordem define o índice em Variables(i))
_HOURLY_VARS: list[str] = [
    "precipitation",            # 0  → rain_mm
    "precipitation_probability",# 1
    "cape",                     # 2  → cape_j_kg
    "lifted_index",             # 3
    "k_index",                  # 4
    "cloud_cover",              # 5
    "wind_speed_10m",           # 6  → wind_speed
    "wind_direction_10m",       # 7
    "temperature_2m",           # 8  → temperature
    "relative_humidity_2m",     # 9
    "surface_pressure",         # 10
]

# Variáveis diárias solicitadas
_DAILY_VARS: list[str] = [
    "precipitation_sum",              # 0
    "precipitation_hours",            # 1
    "precipitation_probability_max",  # 2
    "temperature_2m_max",             # 3
    "temperature_2m_min",             # 4
]


# ---------------------------------------------------------------------------
# Carregamento de configuração
# ---------------------------------------------------------------------------

def _load_grade_points() -> list[dict[str, Any]]:
    """Carrega os pontos de grade NWP do config.yaml.

    Returns:
        Lista de dicts com chaves nome, lat, lon.

    Raises:
        FileNotFoundError: Se config.yaml não for encontrado.
        KeyError: Se a chave grade_nwp não existir no config.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml não encontrado: {_CONFIG_PATH}")
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    points: list[dict[str, Any]] = cfg["grade_nwp"]
    logger.info(f"Pontos de grade carregados: {len(points)} locais.")
    return points


# ---------------------------------------------------------------------------
# Cliente Open-Meteo com cache e retry
# ---------------------------------------------------------------------------

def _parquet_is_fresh(path: Path, ttl_s: int = _CACHE_TTL_S) -> bool:
    """Verifica se um arquivo Parquet existe e foi modificado há menos de ttl_s segundos.

    Args:
        path: Caminho do arquivo Parquet.
        ttl_s: TTL em segundos para considerar o cache válido.

    Returns:
        True se o arquivo existir e for mais recente que ttl_s segundos.
    """
    if not path.exists():
        return False
    age = datetime.now(tz=timezone.utc).timestamp() - path.stat().st_mtime
    return age < ttl_s


def _build_client() -> openmeteo_requests.Client:
    """Cria o cliente Open-Meteo com sessão HTTP padrão (retry via tenacity).

    Returns:
        openmeteo_requests.Client pronto para uso.
    """
    return openmeteo_requests.Client(session=niquests.Session())


@retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_all_points(
    client: openmeteo_requests.Client,
    lats: list[float],
    lons: list[float],
) -> list[Any]:
    """Faz a requisição à API Open-Meteo para todos os pontos de uma vez.

    A API aceita listas de lat/lon e retorna uma resposta por ponto.
    Decorado com retry exponencial (4s → 60s, máx. 5 tentativas).

    Args:
        client: Cliente openmeteo_requests com cache.
        lats: Lista de latitudes dos pontos de grade.
        lons: Lista de longitudes dos pontos de grade.

    Returns:
        Lista de WeatherApiResponse, uma por ponto.

    Raises:
        Exception: Após 5 tentativas sem sucesso (reraise=True).
    """
    params: dict[str, Any] = {
        "latitude":    lats,
        "longitude":   lons,
        "hourly":      _HOURLY_VARS,
        "daily":       _DAILY_VARS,
        "forecast_days": 7,
        "timezone":    "America/Sao_Paulo",
        "wind_speed_unit": "ms",   # m/s (padrão do dashboard)
    }
    logger.debug(f"Requisitando Open-Meteo para {len(lats)} pontos...")
    return client.weather_api(_OPENMETEO_URL, params=params)


# ---------------------------------------------------------------------------
# Parsing das respostas
# ---------------------------------------------------------------------------

def _parse_hourly(response: Any, point_name: str, lat: float, lon: float) -> pd.DataFrame:
    """Converte a seção horária de uma WeatherApiResponse em DataFrame.

    Args:
        response: WeatherApiResponse retornado pela API.
        point_name: Nome do ponto de grade (ex.: "Porto Alegre").
        lat: Latitude do ponto.
        lon: Longitude do ponto.

    Returns:
        DataFrame com colunas compatíveis com HybridWriter.write_forecasts().
    """
    h = response.Hourly()
    times = pd.date_range(
        start=pd.Timestamp(h.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(h.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left",
    )

    def _var(idx: int) -> np.ndarray:
        return h.Variables(idx).ValuesAsNumpy()  # type: ignore[no-any-return]

    df = pd.DataFrame({
        "location_name":            point_name,
        "lat":                      lat,
        "lon":                      lon,
        "forecast_ts":              datetime.now(tz=timezone.utc),
        "valid_ts":                 times,
        "rain_mm":                  _var(0),
        "precipitation_prob":       _var(1),
        "cape_j_kg":                _var(2),
        "lifted_index":             _var(3),
        "k_index":                  _var(4),
        "cloud_cover":              _var(5),
        "wind_speed":               _var(6),
        "wind_dir":                 _var(7),
        "temperature":              _var(8),
        "humidity":                 _var(9),
        "pressure_hpa":             _var(10),
        "model_source":             "openmeteo",
    })
    return df


def _parse_daily(response: Any, point_name: str, lat: float, lon: float) -> pd.DataFrame:
    """Converte a seção diária de uma WeatherApiResponse em DataFrame.

    Args:
        response: WeatherApiResponse retornado pela API.
        point_name: Nome do ponto de grade.
        lat: Latitude do ponto.
        lon: Longitude do ponto.

    Returns:
        DataFrame com um registro por dia para o ponto informado.
    """
    d = response.Daily()
    dates = pd.date_range(
        start=pd.Timestamp(d.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(d.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=d.Interval()),
        inclusive="left",
    )

    def _var(idx: int) -> np.ndarray:
        return d.Variables(idx).ValuesAsNumpy()  # type: ignore[no-any-return]

    df = pd.DataFrame({
        "location_name":              point_name,
        "lat":                        lat,
        "lon":                        lon,
        "date":                       dates.date,
        "precipitation_sum_mm":       _var(0),
        "precipitation_hours":        _var(1),
        "precipitation_prob_max":     _var(2),
        "temperature_max":            _var(3),
        "temperature_min":            _var(4),
    })
    return df


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def collect_nwp_forecasts(db: Optional[Any] = None) -> dict[str, Any]:
    """Coleta previsões NWP para todos os pontos de grade do RS.

    Fluxo:
    1. Carrega pontos de grade do config.yaml
    2. Faz requisição única à Open-Meteo com retry exponencial
    3. Parseia respostas em DataFrames horário e diário
    4. Persiste via HybridWriter (Fluxo A: Supabase PG, Fluxo B: Cloudflare R2)

    Args:
        db: Ignorado — mantido apenas para compatibilidade com chamadores antigos.

    Returns:
        Dicionário com chaves: points_collected (int), hourly_rows (int),
        daily_rows (int), pg_rows (int), pg_ok (bool), r2_ok (bool),
        r2_key (str), duration_s (float).

    Raises:
        FileNotFoundError: Se config.yaml não for encontrado.
        RetryError: Se a API falhar após todas as tentativas de retry.
    """
    t_start = datetime.now(tz=timezone.utc)
    logger.info("=== NOAACollector (Open-Meteo) — iniciando coleta ===")

    # Cache TTL: reutiliza o Parquet se foi gerado há menos de _CACHE_TTL_S segundos
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path_hourly = _PARQUET_DIR / "nwp_7days.parquet"
    path_daily  = _PARQUET_DIR / "nwp_7days_daily.parquet"

    if _parquet_is_fresh(path_hourly, _CACHE_TTL_S) and _parquet_is_fresh(path_daily, _CACHE_TTL_S):
        logger.info(f"Cache válido (TTL={_CACHE_TTL_S}s) — ignorando requisição à API.")
        df_h = pd.read_parquet(path_hourly)
        df_d = pd.read_parquet(path_daily)
        return {
            "points_collected": df_h["location_name"].nunique(),
            "hourly_rows":      len(df_h),
            "daily_rows":       len(df_d),
            "parquet_hourly":   str(path_hourly),
            "parquet_daily":    str(path_daily),
            "duration_s":       0.0,
            "source":           "cache",
        }

    points = _load_grade_points()
    lats   = [float(p["lat"]) for p in points]
    lons   = [float(p["lon"]) for p in points]
    names  = [str(p["nome"]) for p in points]

    client = _build_client()

    try:
        responses = _fetch_all_points(client, lats, lons)
    except RetryError as exc:
        logger.error(f"Open-Meteo falhou após todas as tentativas: {exc}")
        raise

    hourly_frames: list[pd.DataFrame] = []
    daily_frames:  list[pd.DataFrame] = []

    for resp, name, lat, lon in zip(responses, names, lats, lons):
        try:
            hourly_frames.append(_parse_hourly(resp, name, lat, lon))
            daily_frames.append(_parse_daily(resp, name, lat, lon))
            logger.debug(f"  Ponto processado: {name} ({lat}, {lon})")
        except (AttributeError, IndexError) as exc:
            logger.warning(f"  Erro ao parsear ponto {name}: {exc} — ignorado.")

    if not hourly_frames:
        logger.error("Nenhum ponto coletado com sucesso.")
        return {"points_collected": 0, "hourly_rows": 0, "daily_rows": 0}

    df_hourly = pd.concat(hourly_frames, ignore_index=True)
    df_daily  = pd.concat(daily_frames,  ignore_index=True)

    # Persistência híbrida: Fluxo A (Supabase PG) + Fluxo B (Cloudflare R2)
    pg_rows, pg_ok, r2_ok, r2_key = 0, False, False, ""
    if _HW_OK:
        writer = HybridWriter()
        res = writer.write_forecasts(df_hourly, df_daily, path_hourly, path_daily)
        pg_rows, pg_ok, r2_ok, r2_key = res.pg_rows, res.pg_ok, res.r2_ok, res.r2_key
        if not res.success:
            logger.error("write_forecasts: falha total (PG + R2). Verifique variáveis de ambiente.")
    else:
        logger.warning("HybridWriter indisponível — previsões não persistidas.")

    duration = (datetime.now(tz=timezone.utc) - t_start).total_seconds()
    logger.info(f"=== Coleta concluída em {duration:.1f}s ===")

    return {
        "points_collected": len(responses),
        "hourly_rows":      len(df_hourly),
        "daily_rows":       len(df_daily),
        "pg_rows":          pg_rows,
        "pg_ok":            pg_ok,
        "r2_ok":            r2_ok,
        "r2_key":           r2_key,
        "duration_s":       round(duration, 2),
    }


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="DEBUG",
    )

    result = collect_nwp_forecasts()

    print("\n--- Resultado -----------------------------------")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("-------------------------------------------------")
