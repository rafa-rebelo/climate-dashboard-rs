"""
Agente 1 — Arquiteto de Dados
Coletor de precipitação por satélite para o RS.

Fontes em ordem de prioridade:
  1. NASA GPM IMERG Early Run (3IMERGHHE) via NASA CMR + Earthdata auth
     → HDF5, ~30min delay, 0.1° global
  2. CHIRPS v2.0 (Climate Hazards Group) — SEM autenticação
     → GeoTIFF.gz público, 0.05° global, ~5-7 semanas lag

Saída comum:
  data/processed/gpm_precip_1d.parquet
  Colunas: lat, lon, precip_mm, timestamp, source

DuckDB: tabela gpm_precip (upsert por lat/lon/timestamp)
"""

from __future__ import annotations

import gzip
import io
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

try:
    import niquests as requests  # type: ignore
except ImportError:
    import requests  # type: ignore

# rasterio — processa GeoTIFF (CHIRPS) e HDF5 window
try:
    import rasterio
    from rasterio.io import MemoryFile
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False
    logger.warning("rasterio não disponível — pip install rasterio")

# h5py — lê HDF5 do GPM IMERG (opcional)
try:
    import h5py
    _H5PY_OK = True
except ImportError:
    _H5PY_OK = False

# netCDF4 — lê HDF5/NC do GPM como fallback de h5py
try:
    import netCDF4  # type: ignore
    _NC4_OK = True
except ImportError:
    _NC4_OK = False

# ---------------------------------------------------------------------------
# Caminhos e constantes
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).resolve().parents[2]
_DB_PATH = Path(os.getenv("DB_PATH", str(_ROOT / "data" / "climate.duckdb")))
_OUT_DIR = _ROOT / "data" / "processed"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_PARQUET = _OUT_DIR / "gpm_precip_1d.parquet"

# Bounding box RS (com margem 0.5°)
_RS_LAT_MIN: float = -34.0
_RS_LAT_MAX: float = -27.0
_RS_LON_MIN: float = -57.0
_RS_LON_MAX: float = -49.0

_TIMEOUT      = 30     # segundos por request
_CHUNK_BYTES  = 512 * 1024  # 512 KB

# ── NASA GPM IMERG ────────────────────────────────────────────────────────────
_NASA_USER  = os.getenv("NASA_USER", "")
_NASA_PASS  = os.getenv("NASA_PASS", "")
_GPM_AUTH   = (_NASA_USER, _NASA_PASS) if _NASA_USER and _NASA_PASS else None

# NASA CMR — busca de granules (sem autenticação)
_CMR_URL           = "https://cmr.earthdata.nasa.gov/search/granules.json"
_GPM_HALF_HOURLY   = "GPM_3IMERGHHE"   # Early Run half-hourly (0.1°)
_GPM_DAILY_LATE    = "GPM_3IMERGDL"    # Daily Late Run (acumulado diário)

# ── CHIRPS v2.0 ───────────────────────────────────────────────────────────────
_CHIRPS_BASE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
)


# ===========================================================================
# Fonte 1 — NASA GPM IMERG (CMR search + Earthdata auth download)
# ===========================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=3, max=30),
    reraise=True,
)
def _cmr_search_gpm(days_back: int = 1,
                    product: str = _GPM_HALF_HOURLY) -> list[dict]:
    """
    Busca granules GPM IMERG no NASA CMR (sem autenticação).

    Args:
        days_back: Janela de busca em dias (atrás de hoje).
        product: Short name do produto CMR.

    Returns:
        Lista de dicts com 'title' e 'links' (pode ser vazia).

    Raises:
        RuntimeError: HTTP != 200 (relançado por tenacity).
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    params = {
        "short_name":    product,
        "temporal[]":    f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')},"
                         f"{end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "bounding_box":  f"{_RS_LON_MIN},{_RS_LAT_MIN},{_RS_LON_MAX},{_RS_LAT_MAX}",
        "sort_key":      "-start_date",
        "page_size":     5,
    }
    resp = requests.get(
        _CMR_URL, params=params,
        headers={"Accept": "application/json"},
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"CMR HTTP {resp.status_code}")

    entries = resp.json().get("feed", {}).get("entry", [])
    logger.debug(f"CMR {product}: {len(entries)} granules encontrados")
    return entries


def _gpm_extract_download_url(entry: dict) -> Optional[str]:
    """
    Extrai URL de download de um entry CMR (HTTPS, não S3).

    Args:
        entry: Elemento da lista 'feed.entry' do CMR.

    Returns:
        URL HTTPS do HDF5 ou None se não encontrada.
    """
    for link in entry.get("links", []):
        href = link.get("href", "")
        # Preferencia: URL HTTPS direto ao arquivo HDF5
        if (href.startswith("https://") and
                ".HDF5" in href and
                "earthdata.nasa.gov" in href and
                "s3credentials" not in href):
            return href
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    reraise=True,
)
def _download_bytes(url: str, auth: Optional[tuple] = None) -> bytes:
    """
    Baixa arquivo por streaming com suporte a auth e redirects.

    Args:
        url: URL do arquivo.
        auth: Tupla (user, password) ou None.

    Returns:
        Bytes completos do arquivo.

    Raises:
        RuntimeError: HTTP != 200 após todos os redirects.
        requests.RequestException: Erro de rede (retry automático).
    """
    kwargs: dict = {"timeout": _TIMEOUT, "stream": True, "allow_redirects": True}
    if auth:
        kwargs["auth"] = auth

    resp = requests.get(url, **kwargs)
    if resp.status_code != 200:
        raise RuntimeError(f"Download HTTP {resp.status_code}: {url[:80]}")

    buf   = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
        if chunk:
            buf.write(chunk)
            total += len(chunk)
    logger.debug(f"  Baixado: {total / 1024:.0f} KB de {url[:60]}...")
    return buf.getvalue()


def _process_gpm_hdf5(hdf5_bytes: bytes, timestamp: datetime) -> pd.DataFrame:
    """
    Extrai precipitação RS do HDF5 GPM IMERG (Early Run half-hourly).

    Produto: /Grid/precipitationCal (mm/hr) → converte para mm acumulado 30min.

    Args:
        hdf5_bytes: Bytes do arquivo HDF5 GPM.
        timestamp: Timestamp de referência (UTC).

    Returns:
        DataFrame com colunas lat, lon, precip_mm, timestamp, source.
    """
    if not _H5PY_OK and not _NC4_OK:
        logger.error("Nem h5py nem netCDF4 disponíveis para ler HDF5 do GPM")
        return pd.DataFrame()

    try:
        if _H5PY_OK:
            with h5py.File(io.BytesIO(hdf5_bytes), "r") as f:
                lats = f["Grid"]["lat"][:]          # (1800,) ou shape var
                lons = f["Grid"]["lon"][:]          # (3600,)
                # precipitationCal shape: (time, lat, lon)
                precip_rate = f["Grid"]["precipitationCal"][0, :, :]  # mm/hr
        else:
            # netCDF4 lê HDF5 via driver
            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(suffix=".HDF5", delete=False) as tmp:
                tmp.write(hdf5_bytes)
                tmp_path = tmp.name
            try:
                ds = netCDF4.Dataset(tmp_path, "r")
                lats = ds.variables["lat"][:]
                lons = ds.variables["lon"][:]
                precip_rate = ds.variables["precipitationCal"][0, :, :]
                ds.close()
            finally:
                _os.unlink(tmp_path)

        # Corta bbox RS
        lat_mask = (lats >= _RS_LAT_MIN) & (lats <= _RS_LAT_MAX)
        lon_mask = (lons >= _RS_LON_MIN) & (lons <= _RS_LON_MAX)

        lat_rs   = lats[lat_mask]
        lon_rs   = lons[lon_mask]
        pr_rs    = precip_rate[np.ix_(lat_mask, lon_mask)]

        # mm/hr → mm em 30min (produto half-hourly = 0.5h)
        precip_mm = pr_rs * 0.5

        # Grade flat
        lon_grid, lat_grid = np.meshgrid(lon_rs, lat_rs)
        mask_valid = (precip_mm >= 0) & ~np.isnan(precip_mm)

        df = pd.DataFrame({
            "lat":       np.round(lat_grid[mask_valid], 4),
            "lon":       np.round(lon_grid[mask_valid], 4),
            "precip_mm": np.round(precip_mm[mask_valid].astype("f4"), 3),
            "timestamp": timestamp,
            "source":    "GPM_IMERG_EARLY",
        })
        logger.success(f"GPM HDF5: {len(df):,} pontos RS | "
                       f"max={df['precip_mm'].max():.2f} mm/30min")
        return df

    except Exception as exc:
        logger.error(f"GPM HDF5 parse falhou: {exc}")
        return pd.DataFrame()


def collect_gpm_imerg(days_back: int = 1) -> pd.DataFrame:
    """
    Pipeline GPM IMERG: CMR search → download HDF5 → extrai RS.

    Args:
        days_back: Janela de busca em dias.

    Returns:
        DataFrame de precipitação RS ou vazio se falhou / sem auth.
    """
    if not _GPM_AUTH:
        logger.info("GPM IMERG: NASA_USER/NASA_PASS não configurados — pulando")
        return pd.DataFrame()

    try:
        entries = _cmr_search_gpm(days_back=days_back)
    except RetryError as exc:
        logger.warning(f"GPM CMR search falhou: {exc}")
        return pd.DataFrame()

    if not entries:
        logger.warning("GPM IMERG: nenhum granule encontrado no CMR")
        return pd.DataFrame()

    entry   = entries[0]
    dl_url  = _gpm_extract_download_url(entry)
    if not dl_url:
        logger.warning(f"GPM IMERG: URL de download não encontrada em {entry.get('title','?')}")
        return pd.DataFrame()

    logger.info(f"GPM IMERG: baixando {entry.get('title','?')[:60]}...")
    try:
        hdf5_bytes = _download_bytes(dl_url, auth=_GPM_AUTH)
    except RetryError as exc:
        logger.warning(f"GPM IMERG: download falhou: {exc}")
        return pd.DataFrame()

    # Extrai timestamp do título (ex: 3B-HHR-E...20260606-S153000...)
    title = entry.get("title", "")
    try:
        import re
        m = re.search(r"(\d{8})-S(\d{6})", title)
        if m:
            ts_str = m.group(1) + m.group(2)
            timestamp = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        else:
            timestamp = datetime.now(timezone.utc)
    except (ValueError, AttributeError):
        timestamp = datetime.now(timezone.utc)

    return _process_gpm_hdf5(hdf5_bytes, timestamp)


# ===========================================================================
# Fonte 2 — CHIRPS v2.0 (público, sem autenticação, GeoTIFF.gz)
# ===========================================================================

def _chirps_latest_url() -> Optional[tuple[str, date]]:
    """
    Encontra o GeoTIFF.gz CHIRPS mais recente disponível (HEAD requests).

    Começa de 5 dias atrás e vai até 60 dias buscando a data mais próxima
    do presente que retorne HTTP 200.

    Returns:
        Tupla (url_completa, date_ref) ou None se nenhuma encontrada.
    """
    for days_ago in range(5, 61):
        d = date.today() - timedelta(days=days_ago)
        fname = f"chirps-v2.0.{d.year}.{d.month:02d}.{d.day:02d}.tif.gz"
        url   = f"{_CHIRPS_BASE}/{d.year}/{fname}"
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                logger.debug(f"CHIRPS disponível: {fname} ({days_ago} dias atrás)")
                return url, d
        except requests.RequestException:
            continue
    return None


def _process_chirps_geotiff(tif_gz_bytes: bytes,
                             ref_date: date) -> pd.DataFrame:
    """
    Descomprime e processa GeoTIFF.gz do CHIRPS: recorta bbox RS.

    CHIRPS valores já estão em mm/dia (sem fator de escala adicional).
    nodata = -9999.0.

    Args:
        tif_gz_bytes: Bytes do arquivo .tif.gz comprimido.
        ref_date: Data de referência do produto.

    Returns:
        DataFrame com colunas lat, lon, precip_mm, timestamp, source.
    """
    if not _RASTERIO_OK:
        logger.error("rasterio não disponível para processar CHIRPS GeoTIFF")
        return pd.DataFrame()

    try:
        tif_bytes = gzip.decompress(tif_gz_bytes)
        logger.debug(f"CHIRPS descomprimido: {len(tif_bytes) / 1024:.0f} KB")

        with MemoryFile(tif_bytes) as mf:
            with mf.open() as ds:
                band   = ds.read(1).astype(np.float32)
                tfm    = ds.transform
                nodata = ds.nodata
                h, w   = band.shape

        rows_i, cols_i = np.mgrid[0:h, 0:w]
        xs, ys = rasterio.transform.xy(tfm, rows_i.ravel(), cols_i.ravel())
        lons   = np.array(xs, dtype=np.float32)
        lats   = np.array(ys, dtype=np.float32)
        vals   = band.ravel()

        bbox_mask   = ((lats >= _RS_LAT_MIN) & (lats <= _RS_LAT_MAX) &
                       (lons >= _RS_LON_MIN) & (lons <= _RS_LON_MAX))
        nodata_val  = float(nodata) if nodata is not None else -9999.0
        valid_mask  = bbox_mask & (vals != nodata_val) & (vals >= 0)

        timestamp = datetime.combine(ref_date, datetime.min.time(),
                                     tzinfo=timezone.utc)
        df = pd.DataFrame({
            "lat":       np.round(lats[valid_mask], 4),
            "lon":       np.round(lons[valid_mask], 4),
            "precip_mm": np.round(vals[valid_mask], 2).astype(np.float32),
            "timestamp": timestamp,
            "source":    "CHIRPS_v2",
        })
        logger.success(
            f"CHIRPS: {len(df):,} pontos RS | "
            f"max={df['precip_mm'].max():.2f} mm/dia | "
            f"data={ref_date.isoformat()}"
        )
        return df

    except Exception as exc:
        logger.error(f"CHIRPS processamento falhou: {exc}")
        return pd.DataFrame()


def collect_chirps() -> pd.DataFrame:
    """
    Pipeline CHIRPS: encontra GeoTIFF.gz mais recente → baixa → extrai RS.

    Sem autenticação. Funciona de qualquer IP (incluindo GitHub Actions).

    Returns:
        DataFrame de precipitação RS ou vazio se falhou.
    """
    result = _chirps_latest_url()
    if result is None:
        logger.warning("CHIRPS: nenhum arquivo disponível nos últimos 60 dias")
        return pd.DataFrame()

    url, ref_date = result
    logger.info(f"CHIRPS: baixando {url.split('/')[-1]} ({(date.today()-ref_date).days} dias atrás)...")
    try:
        gz_bytes = _download_bytes(url)
    except RetryError as exc:
        logger.warning(f"CHIRPS download falhou: {exc}")
        return pd.DataFrame()

    return _process_chirps_geotiff(gz_bytes, ref_date)


# ===========================================================================
# Persistência — Parquet + DuckDB
# ===========================================================================

def _save_parquet(df: pd.DataFrame) -> None:
    """
    Salva DataFrame em data/processed/gpm_precip_1d.parquet.

    Args:
        df: DataFrame com colunas lat, lon, precip_mm, timestamp, source.
    """
    try:
        df.to_parquet(_PARQUET, index=False, compression="snappy")
        logger.info(
            f"Parquet salvo: {_PARQUET.name} "
            f"({len(df):,} linhas, {_PARQUET.stat().st_size / 1024:.0f} KB)"
        )
    except OSError as exc:
        logger.error(f"Erro ao salvar Parquet: {exc}")


def _upsert_duckdb(df: pd.DataFrame) -> int:
    """
    Cria tabela gpm_precip (se necessário) e faz upsert dos registros.

    Args:
        df: DataFrame com colunas lat, lon, precip_mm, timestamp, source.

    Returns:
        Número de linhas inseridas.
    """
    if df.empty:
        return 0
    try:
        conn = duckdb.connect(str(_DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gpm_precip (
                lat        DOUBLE       NOT NULL,
                lon        DOUBLE       NOT NULL,
                precip_mm  FLOAT,
                timestamp  TIMESTAMPTZ  NOT NULL,
                source     VARCHAR      DEFAULT 'GPM_IMERG_EARLY',
                PRIMARY KEY (lat, lon, timestamp)
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO gpm_precip "
            "SELECT lat, lon, precip_mm, timestamp, source FROM df"
        )
        total = conn.execute("SELECT COUNT(*) FROM gpm_precip").fetchone()[0]
        conn.close()
        logger.success(f"DuckDB gpm_precip: {total:,} registros totais")
        return len(df)
    except duckdb.IOException as exc:
        logger.warning(f"DuckDB bloqueado — apenas Parquet salvo: {exc}")
        return 0
    except Exception as exc:
        logger.error(f"DuckDB erro: {exc}")
        return 0


# ===========================================================================
# Orquestrador principal
# ===========================================================================

def collect_gpm(days_back: int = 1) -> dict[str, int]:
    """
    Executa pipeline de precipitação por satélite para o RS.

    Ordem: GPM IMERG (se NASA auth configurada) → CHIRPS (fallback público).

    Args:
        days_back: Janela de busca GPM em dias (não afeta CHIRPS).

    Returns:
        Dicionário ``{"points": int, "upserted": int, "source": str}``.
    """
    # 1. Tenta GPM IMERG com auth NASA
    df = collect_gpm_imerg(days_back=days_back)

    # 2. Fallback: CHIRPS público
    if df.empty:
        logger.info("Usando fallback CHIRPS v2.0 (público, sem auth)...")
        df = collect_chirps()

    if df.empty:
        logger.error("Nenhuma fonte de precipitação por satélite retornou dados")
        return {"points": 0, "upserted": 0, "source": "none"}

    source = df["source"].iloc[0]

    # 3. Salvar
    _save_parquet(df)
    upserted = _upsert_duckdb(df)

    return {"points": len(df), "upserted": upserted, "source": source}


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
        level="DEBUG",
    )

    parser = argparse.ArgumentParser(
        description="Coletor precipitação satélite RS — GPM IMERG / CHIRPS"
    )
    parser.add_argument(
        "--dias", type=int, default=1,
        help="Janela de busca GPM IMERG em dias (padrão 1)",
    )
    parser.add_argument(
        "--fonte",
        choices=["auto", "gpm", "chirps"],
        default="auto",
        help="Fonte de dados: auto (padrão) | gpm | chirps",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Precipitação Satélite RS")
    logger.info(f"  DB_PATH : {_DB_PATH}")
    logger.info(f"  NASA    : {'auth OK' if _GPM_AUTH else 'sem credenciais (usará CHIRPS)'}")
    logger.info(f"  rasterio: {'OK' if _RASTERIO_OK else 'FALTANDO'}")
    logger.info("=" * 60)

    if args.fonte == "gpm":
        df = collect_gpm_imerg(days_back=args.dias)
        if not df.empty:
            _save_parquet(df)
            _upsert_duckdb(df)
        result = {"points": len(df), "source": "GPM_IMERG_EARLY"}
    elif args.fonte == "chirps":
        df = collect_chirps()
        if not df.empty:
            _save_parquet(df)
            _upsert_duckdb(df)
        result = {"points": len(df), "source": "CHIRPS_v2"}
    else:
        result = collect_gpm(days_back=args.dias)

    logger.info(f"Resultado : {result['points']:,} pontos RS coletados")
    logger.info(f"Fonte     : {result.get('source', '?')}")
    logger.info(f"Parquet   : {_PARQUET}")
