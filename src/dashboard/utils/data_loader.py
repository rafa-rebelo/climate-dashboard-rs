"""
Agente 4 — Especialista em Visualização
Carregador de dados para o dashboard — GitHub Raw primeiro, local como fallback.

Prioridade:
  1. GitHub Raw (climate-data-rs) — dados atualizados pelo Actions a cada 10min
  2. Arquivo local em data/ — fallback quando GitHub indisponível

Retorna sempre um DataSource com metadata de origem e timestamp.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# Importa niquests se disponível, cai em requests como fallback
try:
    import niquests as requests  # type: ignore
except ImportError:
    import requests  # type: ignore

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
GITHUB_RAW = "https://raw.githubusercontent.com/rafa-rebelo/climate-data-rs/main"

_URLS: dict[str, str] = {
    "accumulated_rain": f"{GITHUB_RAW}/data/processed/accumulated_rain.parquet",
    "river_status":     f"{GITHUB_RAW}/data/processed/river_status.parquet",
    "nwp_7days":        f"{GITHUB_RAW}/data/forecasts/nwp_7days.parquet",
    "climate_duckdb":   f"{GITHUB_RAW}/data/climate.duckdb",
    "progress":         f"{GITHUB_RAW}/data/processed/progress.json",
    "gpm_precip":       f"{GITHUB_RAW}/data/processed/gpm_precip_1d.parquet",
    "inmet_historico":  f"{GITHUB_RAW}/data/processed/inmet_historico_rs.parquet",
    "inmet_stations":   f"{GITHUB_RAW}/data/processed/inmet_stations_rs.parquet",
}

_ROOT = Path(__file__).resolve().parents[3]
_LOCAL: dict[str, Path] = {
    "accumulated_rain": _ROOT / "data" / "processed" / "accumulated_rain.parquet",
    "river_status":     _ROOT / "data" / "processed" / "river_status.parquet",
    "nwp_7days":        _ROOT / "data" / "forecasts"  / "nwp_7days.parquet",
    "climate_duckdb":   _ROOT / "data" / "climate.duckdb",
    "progress":         _ROOT / "data" / "processed"  / "progress.json",
    "gpm_precip":       _ROOT / "data" / "processed" / "gpm_precip_1d.parquet",
    "inmet_historico":  _ROOT / "data" / "processed" / "inmet_historico_rs.parquet",
    "inmet_stations":   _ROOT / "data" / "processed" / "inmet_stations_rs.parquet",
}

_TIMEOUT = 8   # segundos — não bloqueia o dashboard por muito tempo


# ---------------------------------------------------------------------------
# DataSource — metadados de origem retornados junto com o dado
# ---------------------------------------------------------------------------

@dataclass
class DataSource:
    """Metadados de origem de um dado carregado pelo data_loader.

    Attributes:
        source: "github" | "local" | "unavailable".
        url: URL ou caminho de onde o dado foi lido.
        fetched_at: Timestamp UTC do momento do fetch.
        size_kb: Tamanho em KB do payload recebido.
        error: Mensagem de erro se source == "unavailable".
    """
    source:     str
    url:        str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_kb:    float = 0.0
    error:      Optional[str] = None

    @property
    def label(self) -> str:
        """Rótulo compacto para exibição no dashboard."""
        ts = self.fetched_at.strftime("%H:%M UTC")
        if self.source == "github":
            return f"☁️ GitHub · {ts}"
        if self.source == "local":
            return f"💾 Local · {ts}"
        return f"⚠️ Indisponível · {ts}"

    @property
    def badge_color(self) -> str:
        """Cor CSS para o badge de fonte."""
        return {"github": "#22c55e", "local": "#f59e0b",
                "unavailable": "#ef4444"}.get(self.source, "#94a3b8")


_UNAVAILABLE = DataSource(source="unavailable", url="", error="sem dados")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _get_bytes(url: str) -> Optional[bytes]:
    """
    Baixa bytes de uma URL com timeout.

    Args:
        url: URL a buscar.

    Returns:
        Bytes do payload ou None em caso de erro.
    """
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.content
        logger.debug(f"GitHub retornou {resp.status_code} para {url}")
        return None
    except Exception as exc:
        logger.debug(f"GitHub indisponível ({url}): {exc}")
        return None


def _parquet_from_bytes(data: bytes) -> Optional[pd.DataFrame]:
    """
    Converte bytes em DataFrame Parquet.

    Args:
        data: Bytes de um arquivo Parquet válido.

    Returns:
        DataFrame ou None se falhar.
    """
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception as exc:
        logger.warning(f"Falha ao parsear Parquet: {exc}")
        return None


def _parquet_local(key: str) -> Optional[pd.DataFrame]:
    """
    Lê Parquet local pelo nome da chave.

    Args:
        key: Chave em _LOCAL (ex: "accumulated_rain").

    Returns:
        DataFrame ou None se arquivo não existir.
    """
    path = _LOCAL.get(key)
    if path and path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"Falha ao ler Parquet local {path}: {exc}")
    return None


# ---------------------------------------------------------------------------
# API pública — fetch com fallback
# ---------------------------------------------------------------------------

def fetch_parquet(key: str) -> tuple[pd.DataFrame, DataSource]:
    """
    Carrega um Parquet do GitHub Raw com fallback para arquivo local.

    Args:
        key: Nome do artefato — "accumulated_rain", "river_status" ou "nwp_7days".

    Returns:
        Tupla (DataFrame, DataSource). DataFrame pode ser vazio se tudo falhar.
    """
    url   = _URLS.get(key, "")
    local = _LOCAL.get(key)

    # Tenta GitHub primeiro
    data = _get_bytes(url) if url else None
    if data:
        df = _parquet_from_bytes(data)
        if df is not None and not df.empty:
            src = DataSource(source="github", url=url,
                             size_kb=round(len(data) / 1024, 1))
            logger.info(f"[data_loader] {key} ← GitHub ({src.size_kb} KB)")
            return df, src

    # Fallback local
    df_local = _parquet_local(key)
    if df_local is not None and not df_local.empty:
        src = DataSource(source="local", url=str(local),
                         size_kb=round(local.stat().st_size / 1024, 1)
                         if local and local.exists() else 0.0)
        logger.info(f"[data_loader] {key} ← local ({src.size_kb} KB)")
        return df_local, src

    logger.warning(f"[data_loader] {key}: GitHub e local indisponíveis")
    return pd.DataFrame(), DataSource(source="unavailable", url=url,
                                      error="GitHub e arquivo local indisponíveis")


def fetch_accumulated_rain() -> tuple[pd.DataFrame, DataSource]:
    """
    Carrega acumulados de chuva por estação (GitHub → local).

    Returns:
        (DataFrame com colunas: station_id, date, rain_*, DataSource)
    """
    return fetch_parquet("accumulated_rain")


def fetch_river_status() -> tuple[pd.DataFrame, DataSource]:
    """
    Carrega status atual dos rios (GitHub → local).

    Returns:
        (DataFrame com colunas: river, segment, level_m, status, trend, DataSource)
    """
    return fetch_parquet("river_status")


def fetch_nwp_forecasts() -> tuple[pd.DataFrame, DataSource]:
    """
    Carrega previsões NWP completas para os 10 pontos RS (GitHub → local).

    Returns:
        (DataFrame com colunas: location_name, valid_ts, rain_mm, temperature,
         wind_speed, cape_j_kg, lifted_index, k_index, model_source, DataSource)
    """
    return fetch_parquet("nwp_7days")


def fetch_duckdb() -> tuple[Optional[Path], DataSource]:
    """
    Baixa o arquivo climate.duckdb do GitHub para um arquivo temporário.

    Usado como fallback quando o banco local não existe (ex.: Streamlit Cloud).
    O arquivo temporário persiste enquanto o processo Python estiver rodando.

    Returns:
        (Path para o .duckdb temporário ou None, DataSource)
    """
    url = _URLS["climate_duckdb"]

    # Banco local disponível — usa diretamente
    local = _LOCAL["climate_duckdb"]
    if local.exists():
        src = DataSource(source="local", url=str(local),
                         size_kb=round(local.stat().st_size / 1024, 1))
        return local, src

    # Tenta baixar do GitHub
    logger.info(f"[data_loader] Baixando climate.duckdb do GitHub…")
    data = _get_bytes(url)
    if data:
        tmp = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        src = DataSource(source="github", url=url,
                         size_kb=round(len(data) / 1024, 1))
        logger.info(f"[data_loader] DuckDB ← GitHub ({src.size_kb} KB) → {tmp.name}")
        return Path(tmp.name), src

    return None, DataSource(source="unavailable", url=url,
                            error="DuckDB não encontrado")


def fetch_progress() -> tuple[dict, DataSource]:
    """
    Carrega o progress.json do GitHub (status do pipeline).

    Returns:
        (dict com metadados do pipeline, DataSource)
    """
    import json
    url   = _URLS["progress"]
    local = _LOCAL["progress"]

    data = _get_bytes(url)
    if data:
        try:
            return json.loads(data.decode("utf-8")), \
                   DataSource(source="github", url=url,
                              size_kb=round(len(data) / 1024, 1))
        except Exception:
            pass

    if local.exists():
        try:
            return json.loads(local.read_text("utf-8")), \
                   DataSource(source="local", url=str(local))
        except Exception:
            pass

    return {}, DataSource(source="unavailable", url=url,
                          error="progress.json indisponível")


def fetch_gpm_precip() -> tuple[pd.DataFrame, DataSource]:
    """
    Carrega precipitação satélite (GPM IMERG / CHIRPS) do GitHub → local.

    Returns:
        (DataFrame com colunas: lat, lon, precip_mm, timestamp, source, DataSource)
    """
    return fetch_parquet("gpm_precip")


def fetch_inmet_historico() -> tuple[pd.DataFrame, DataSource]:
    """
    Carrega histórico horário INMET RS (últimos 30 dias) do GitHub → local.

    Returns:
        (DataFrame com colunas: station_id, ts, precip_mm, temp_c, ..., DataSource)
    """
    return fetch_parquet("inmet_historico")


def fetch_inmet_stations() -> tuple[pd.DataFrame, DataSource]:
    """
    Carrega inventário de estações INMET RS (98 estações) do GitHub → local.

    Returns:
        (DataFrame com colunas: station_id, name, lat, lon, municipality, DataSource)
    """
    return fetch_parquet("inmet_stations")
