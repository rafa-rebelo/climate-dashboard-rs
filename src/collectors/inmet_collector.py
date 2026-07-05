"""
Agente 1 — Arquiteto de Dados
Coletor INMET — estações meteorológicas automáticas do RS.

Coleta via API pública apitempo.inmet.gov.br (sem token):
  - Inventário das ~98 estações automáticas RS (/estacoes/T)
  - Dados horários por estação (/estacao/dados/{data}/{CD}) — requer IP BR
  - Dados históricos via ZIP público (sem restrição de IP):
    https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip

Formato CSV INMET histórico:
  Linhas 0-7 : metadados (REGIAO, UF, ESTACAO, CODIGO, LAT, LON, ALT, FUNDACAO)
  Linha  8   : cabeçalho das colunas (sep=';')
  Linha  9+  : dados horários (encoding=latin-1)
  Timestamp  : Data (YYYY-MM-DD) + Hora UTC (HHMM UTC)
  Colunas RS : PRECIPITACAO, PRESSAO, TEMPERATURA_AR, UMIDADE, VENTO_DIR, VENTO_VEL
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Garante que src/ está no sys.path quando executado como script standalone
_SRC_DIR = Path(__file__).resolve().parent.parent  # src/collectors/.. = src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


import pandas as pd
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import niquests as niquests  # type: ignore
except ImportError:
    import requests as niquests  # type: ignore

try:
    from database.hybrid_writer import HybridWriter as _HybridWriter
    _HW_OK = True
except ImportError:
    _HW_OK = False

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_BASE_URL    = "https://apitempo.inmet.gov.br"
_HIST_BASE   = "https://portal.inmet.gov.br/uploads/dadoshistoricos"
_ROOT        = Path(__file__).resolve().parents[2]
_PARQUET_DIR = _ROOT / "data" / "processed"
_RAW_DIR     = _ROOT / "data" / "raw"
_OUT_PARQUET = _PARQUET_DIR / "inmet_hourly.parquet"
_INV_PARQUET = _PARQUET_DIR / "inmet_stations_rs.parquet"
_HIST_PARQUET = _PARQUET_DIR / "inmet_historico_rs.parquet"
_CACHE_TTL_S = 600   # 10 minutos
# Inventário oficial INMET RS (situação Operante/Pane) — fonte: portal INMET,
# atualizado 03/07/2026. Usado para marcar ativa=false nas estações em Pane
# (o ZIP histórico segue publicando os CSVs de estações quebradas).
_INV_OFICIAL_CSV = _ROOT / "config" / "inmet_stations_rs.csv"


def _aplicar_situacao_oficial(df_stations: "pd.DataFrame") -> "pd.DataFrame":
    """Marca active=False nas estações em Pane conforme o inventário oficial.

    Faz merge por station_id com config/inmet_stations_rs.csv (situação
    Operante/Pane do portal INMET). Estações fora do inventário mantêm
    active=True (default do write_stations). Best-effort: sem o CSV, o
    DataFrame volta intocado.

    Args:
        df_stations: Inventário montado a partir do ZIP (station_id, name, …).

    Returns:
        Mesmo DataFrame com a coluna active preenchida pela situação oficial.
    """
    if df_stations.empty or not _INV_OFICIAL_CSV.exists():
        return df_stations
    try:
        inv = pd.read_csv(_INV_OFICIAL_CSV, dtype={"station_id": str})
    except (OSError, ValueError) as exc:
        logger.warning(f"  Inventário oficial ilegível ({exc}) — situação ignorada.")
        return df_stations
    situacao = dict(zip(inv["station_id"].str.strip(), inv["situacao"].str.strip()))
    df = df_stations.copy()
    df["active"] = df["station_id"].astype(str).map(
        lambda s: situacao.get(s, "Operante") != "Pane"
    )
    n_pane = int((~df["active"]).sum())
    if n_pane:
        logger.info(f"  Situação oficial INMET: {n_pane} estações em Pane → ativa=false")
    return df


# ---------------------------------------------------------------------------
# Utilitários de conversão (módulo-level para uso em todas as funções)
# ---------------------------------------------------------------------------

from utils.comum import safe_float as _safe_float  # noqa: E402


# Colunas da API → padrão interno
# Confirmados em /estacoes/T e /estacao/dados/{CD_ESTACAO}
_COL_MAP: dict[str, str] = {
    # inventário /estacoes/T — CD_MUNICIPIO não existe neste endpoint
    "CD_ESTACAO":  "station_id",
    "DC_NOME":     "name",
    "SG_ESTADO":   "state",
    "VL_LATITUDE": "lat",
    "VL_LONGITUDE":"lon",
    "VL_ALTITUDE": "elevation_m",
    # dados horários /estacao/dados/{CD_ESTACAO} — campos confirmados
    "DT_MEDICAO":  "date",
    "HR_MEDICAO":  "hour_utc",
    "TEM_INS":     "temperature",
    "UMD_INS":     "humidity",
    "PRE_INS":     "pressure_hpa",
    "VEN_VEL":     "wind_speed",
    "VEN_DIR":     "wind_dir",
    "CHUVA":       "rain_1h_mm",
    # aliases encontrados em versões anteriores da API
    "TEMP_INS":    "temperature",
    "UMID_INS":    "humidity",
    "PRES_INS":    "pressure_hpa",
    "VENTO_VEL":   "wind_speed",
    "VENTO_DIR":   "wind_dir",
    "TEMP_MAX":    "temp_max",
    "TEMP_MIN":    "temp_min",
}

# Colunas numéricas a converter (a API retorna strings com vírgula decimal)
_NUMERIC_COLS: list[str] = [
    "temperature", "temp_max", "temp_min",
    "humidity", "humidity_max", "humidity_min",
    "pressure_hpa", "wind_speed", "wind_dir",
    "rain_1h_mm", "lat", "lon", "elevation_m",
]


# ---------------------------------------------------------------------------
# Cliente HTTP
# ---------------------------------------------------------------------------

class INMETClient:
    """Cliente para a API pública INMET (apitempo.inmet.gov.br).

    Usa niquests.Session para compatibilidade com urllib3-future e tenacity
    para retry com backoff exponencial.
    """

    def __init__(self) -> None:
        self.session = niquests.Session()
        self.session.headers.update({"Accept": "application/json"})

    @retry(
        retry=retry_if_exception_type((niquests.exceptions.RequestException,)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: dict | None = None) -> Any:
        """Executa GET com retry exponencial.

        Args:
            path: Caminho relativo à base URL (sem barra inicial).
            params: Parâmetros de query opcionais.

        Returns:
            JSON desserializado (list ou dict). None para 204 No Content.

        Raises:
            niquests.exceptions.HTTPError: Se o status HTTP for 4xx/5xx.
            niquests.exceptions.RequestException: Após esgotar tentativas.

        Note:
            O endpoint de dados horários retorna 204 quando acessado fora
            da rede INMET/Brasil. Nesse caso retorna None sem erro.
        """
        url = f"{_BASE_URL}/{path}"
        resp = self.session.get(url, params=params, timeout=15)
        # 204 No Content — sem dados disponíveis (fora da rede INMET ou sem token)
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return resp.json()

    def listar_estacoes_rs(self) -> pd.DataFrame:
        """Lista estações automáticas do RS — endpoint GET /estacoes/T.

        Filtra por SG_ESTADO == 'RS' e exclui estações com CD_SITUACAO == 'Pane'.

        Returns:
            DataFrame com colunas: station_id, name, lat, lon, elevation_m,
            state, active.

        Raises:
            niquests.exceptions.RequestException: Se a API falhar após retries.
        """
        logger.info("INMET: listando estacoes automaticas RS...")
        data = self._get("estacoes/T")
        df_all = pd.DataFrame(data)

        df = df_all[df_all["SG_ESTADO"] == "RS"].copy()

        df = df.rename(columns={
            "CD_ESTACAO":  "station_id",
            "DC_NOME":     "name",
            "SG_ESTADO":   "state",
            "VL_LATITUDE": "lat",
            "VL_LONGITUDE":"lon",
            "VL_ALTITUDE": "elevation_m",
            "CD_SITUACAO": "situacao",
        })

        for col in ["lat", "lon", "elevation_m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )

        df["active"] = df.get("situacao", pd.Series(dtype=str)).str.strip() != "Pane"

        operantes = df["active"].sum()
        logger.success(f"INMET: {len(df)} estacoes RS ({operantes} operantes, {len(df)-operantes} em pane).")
        return df

    def dados_estacao(self, station_id: str) -> pd.DataFrame:
        """Retorna dados da última hora de uma estação.

        Endpoint: GET /estacao/dados/{data_hoje}/{CD_ESTACAO}
        Sem token necessário. Retorna a leitura mais recente do dia.

        Args:
            station_id: Código da estação (ex.: 'B828').

        Returns:
            DataFrame com leituras horárias. Vazio se não houver dados (204).

        Raises:
            niquests.exceptions.RequestException: Após retries esgotados.
        """
        hoje = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            data = self._get(f"estacao/dados/{hoje}/{station_id}")
        except niquests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (404, 204):
                return pd.DataFrame()
            raise
        if data is None or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
        return df

    def dados_periodo(
        self,
        station_id: str,
        date_ini: str,
        date_fim: str,
    ) -> pd.DataFrame:
        """Retorna dados horários de uma estação em um período.

        Endpoint confirmado: GET /estacao/T/{data_ini}/{data_fim}/{CD_ESTACAO}
        Formato das datas: YYYY-MM-DD

        Args:
            station_id: Código da estação (ex.: 'B828').
            date_ini: Data início 'YYYY-MM-DD'.
            date_fim: Data fim 'YYYY-MM-DD'.

        Returns:
            DataFrame com leituras horárias do período. Vazio se sem dados.

        Raises:
            niquests.exceptions.RequestException: Após retries esgotados.
        """
        try:
            data = self._get(f"estacao/T/{date_ini}/{date_fim}/{station_id}")
        except niquests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (404, 204):
                return pd.DataFrame()
            raise
        if data is None or not data:
            return pd.DataFrame()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])


# ---------------------------------------------------------------------------
# Helpers de transformação
# ---------------------------------------------------------------------------

def _normalizar_df(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Padroniza colunas, converte tipos e constrói timestamp UTC.

    Args:
        df: DataFrame bruto retornado pela API INMET.
        station_id: Código da estação para preencher coluna faltante.

    Returns:
        DataFrame normalizado com colunas internas do sistema.
    """
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})

    # Constrói timestamp a partir de date + hour
    if "date" in df.columns and "hour_utc" in df.columns:
        df["hour_utc"] = df["hour_utc"].astype(str).str.zfill(4)
        df["ts"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["hour_utc"].str[:2] + ":00",
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
    elif "DT_MEDICAO" in df.columns:
        df["ts"] = pd.to_datetime(df["DT_MEDICAO"], errors="coerce")

    # Converte numéricos — API retorna strings com vírgula decimal
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".").str.strip(),
                errors="coerce",
            )

    if "station_id" not in df.columns:
        df["station_id"] = station_id

    df["source"] = "INMET"
    return df


def _parquet_is_fresh(path: Path, ttl_s: int = _CACHE_TTL_S) -> bool:
    """Verifica se um Parquet existe e é mais recente que ttl_s segundos.

    Args:
        path: Caminho do arquivo.
        ttl_s: TTL em segundos.

    Returns:
        True se o arquivo existir e estiver dentro do TTL.
    """
    if not path.exists():
        return False
    age = datetime.now(tz=timezone.utc).timestamp() - path.stat().st_mtime
    return age < ttl_s


# ---------------------------------------------------------------------------
# Funções de coleta
# ---------------------------------------------------------------------------

def coletar_inventario_rs(
    client: INMETClient,
    force: bool = False,
) -> pd.DataFrame:
    """Coleta e persiste o inventário de estações automáticas do RS.

    Usa cache em Parquet (TTL = 24h) para evitar chamadas repetidas.

    Args:
        client: INMETClient inicializado.
        force: Se True, ignora cache e reconecta à API.

    Returns:
        DataFrame com metadados das estações RS.

    Raises:
        niquests.exceptions.RequestException: Se a API falhar após retries.
    """
    cache_ttl = 86_400  # 24h — inventário muda raramente
    if not force and _parquet_is_fresh(_INV_PARQUET, cache_ttl):
        logger.info(f"Inventario INMET RS: cache válido ({_INV_PARQUET.name}).")
        return pd.read_parquet(_INV_PARQUET)

    df = client.listar_estacoes_rs()
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_INV_PARQUET, index=False, engine="pyarrow")
    logger.info(f"Inventario salvo: {_INV_PARQUET} ({len(df)} estacoes).")
    return df


def coletar_dados_rs(
    client: INMETClient,
    inventario: pd.DataFrame,
    dias_back: int = 1,
    delay_s: float = 0.3,
    max_estacoes: int | None = None,
) -> pd.DataFrame:
    """Coleta dados horários de todas as estações RS.

    Usa GET /estacao/dados/{CD_ESTACAO} (última hora, sem token) para coleta
    em tempo real, ou GET /estacao/T/{ini}/{fim}/{CD_ESTACAO} para histórico
    quando dias_back > 0.

    Salva leituras normalizadas em data/processed/inmet_hourly.parquet.

    Args:
        client: INMETClient inicializado.
        inventario: DataFrame do inventário RS (precisa de station_id).
        dias_back: 0 = apenas última hora; >0 = histórico dos últimos N dias.
        delay_s: Pausa entre chamadas para não sobrecarregar a API (segundos).
        max_estacoes: Limita o número de estações para testes. None = todas.

    Returns:
        DataFrame consolidado com todas as leituras coletadas.

    Raises:
        OSError: Se não for possível criar o diretório de saída.
    """
    agora   = datetime.now(timezone.utc).replace(tzinfo=None)
    ini     = (agora - timedelta(days=max(dias_back, 1))).strftime("%Y-%m-%d")
    fim     = agora.strftime("%Y-%m-%d")
    modo    = "última hora" if dias_back == 0 else f"{ini} → {fim}"

    cod_col = next(
        (c for c in ["station_id", "CD_ESTACAO"] if c in inventario.columns),
        inventario.columns[0],
    )
    codigos = inventario[cod_col].dropna().astype(str).tolist()
    if max_estacoes:
        codigos = codigos[:max_estacoes]

    logger.info(f"INMET: coletando {len(codigos)} estacoes RS ({modo})...")

    frames: list[pd.DataFrame] = []
    ok = erros = 0

    for i, cod in enumerate(codigos, 1):
        try:
            if dias_back == 0:
                df_raw = client.dados_estacao(cod)
            else:
                df_raw = client.dados_periodo(cod, ini, fim)
            if df_raw.empty:
                logger.debug(f"  [{i}/{len(codigos)}] {cod}: sem dados")
                continue
            df = _normalizar_df(df_raw, cod)
            frames.append(df)
            ok += 1
            logger.debug(f"  [{i}/{len(codigos)}] {cod}: {len(df)} leituras")
            time.sleep(delay_s)
        except RetryError as exc:
            logger.warning(f"  [{i}/{len(codigos)}] {cod}: falhou após retries — {exc}")
            erros += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  [{i}/{len(codigos)}] {cod}: {exc}")
            erros += 1

    logger.info(f"INMET: {ok} OK · {erros} erros · {sum(len(f) for f in frames)} leituras")

    if not frames:
        logger.warning("Nenhum dado INMET coletado.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(_OUT_PARQUET, index=False, engine="pyarrow")
    logger.success(f"inmet_rs.parquet salvo: {len(df_all)} leituras.")
    return df_all


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def collect_inmet(
    dias_back: int = 0,
    max_estacoes: int | None = None,
    force_inventario: bool = False,
    skip_hourly: bool = False,
) -> dict[str, Any]:
    """Pipeline de coleta INMET para o RS.

    1. Obtém inventário de 98 estações RS (cache 24h em Parquet).
    2. Opcionalmente coleta dados horários (requer IP brasileiro).
    3. Persiste em data/processed/inmet_{stations_rs,hourly}.parquet.
    4. Persiste via HybridWriter (Supabase + R2).

    Args:
        dias_back: 0 = apenas última hora; >0 = histórico dos últimos N dias.
        max_estacoes: Limite de estações para testes rápidos (None = todas 98).
        force_inventario: Força atualização do inventário mesmo com cache 24h.
        skip_hourly: Se True, coleta apenas o inventário (sem dados horários).
            Use em ambientes fora do Brasil onde /estacao/dados retorna 204.

    Returns:
        Dict com stations, rain_readings, estacoes_inv, leituras_raw,
        parquet, duration_s.

    Raises:
        RetryError: Se a API INMET falhar irreversivelmente após todos os retries.
    """
    t_start = datetime.now(tz=timezone.utc)
    logger.info("=" * 60)
    logger.info("INMET Collector — iniciando coleta RS")
    logger.info("=" * 60)

    client = INMETClient()

    try:
        inventario = coletar_inventario_rs(client, force=force_inventario)
        if skip_hourly:
            logger.info("  Modo inventário — pulando dados horários (skip_hourly=True)")
            leituras = pd.DataFrame()
        else:
            leituras = coletar_dados_rs(
                client, inventario,
                dias_back=dias_back,
                max_estacoes=max_estacoes,
            )
    except RetryError as exc:
        logger.error(f"INMET: falha irrecuperável — {exc}")
        raise

    if _HW_OK:
        from database.hybrid_writer import HybridWriter as _HW
        writer = _HW()
        res_st = writer.write_stations(inventario, path=_INV_PARQUET)
        res_rd = writer.write_rain_readings(leituras, path=_OUT_PARQUET) if not leituras.empty else None
        counts = {
            "stations":     res_st.pg_rows,
            "rain_readings": res_rd.pg_rows if res_rd else 0,
        }
    else:
        logger.warning("HybridWriter indisponível — contagens zeradas (sem fallback DuckDB).")
        counts = {"stations": 0, "rain_readings": 0}

    duration = (datetime.now(tz=timezone.utc) - t_start).total_seconds()
    logger.info("=" * 60)
    logger.success(
        f"INMET Collector — concluído em {duration:.1f}s | "
        f"{len(inventario)} estações RS | "
        f"{len(leituras)} leituras horárias"
    )
    logger.info("=" * 60)

    return {
        "stations":     counts["stations"],
        "rain_readings":counts["rain_readings"],
        "estacoes_inv": len(inventario),
        "leituras_raw": len(leituras),
        "parquet":      str(_OUT_PARQUET),
        "duration_s":   round(duration, 2),
    }


# ---------------------------------------------------------------------------
# Coleta histórica via ZIP público (sem geo-restrição)
# ---------------------------------------------------------------------------

def _download_zip_inmet(year: int) -> bytes:
    """
    Baixa o ZIP histórico INMET do ano especificado.

    URL pública, sem autenticação, sem geo-restrição.
    Arquivo de ~20-80 MB dependendo do ano.

    Args:
        year: Ano de referência (ex: 2026).

    Returns:
        Bytes do arquivo ZIP.

    Raises:
        RuntimeError: Se o download falhar após retries.
    """
    url = f"{_HIST_BASE}/{year}.zip"
    logger.info(f"INMET histórico: baixando {url} ...")

    _headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/zip, application/octet-stream, */*",
    }
    try:
        resp = niquests.get(url, timeout=120, stream=True, headers=_headers)
    except Exception as exc:
        raise RuntimeError(f"Falha ao baixar ZIP INMET {year}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"ZIP INMET {year}: HTTP {resp.status_code} → {url}"
        )

    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=1024 * 256):  # 256 KB chunks
        if chunk:
            chunks.append(chunk)
            total += len(chunk)

    data = b"".join(chunks)
    logger.info(f"  ZIP baixado: {len(data) / 1024 / 1024:.1f} MB")
    return data


def _parse_inmet_csv(
    csv_bytes: bytes,
    filename: str,
) -> tuple[dict, pd.DataFrame]:
    """
    Parseia um CSV INMET histórico retornando metadados da estação e dados horários.

    Formato esperado:
      Linhas 0-7 : metadados (key;value separados por ';')
      Linha  8   : cabeçalho de colunas
      Linha  9+  : dados horários

    Args:
        csv_bytes: Conteúdo bruto do arquivo CSV.
        filename: Nome do arquivo (para logs).

    Returns:
        Tupla (station_meta dict, DataFrame de leituras).
        DataFrame vazio se o arquivo não for RS ou estiver corrompido.
    """
    try:
        text = csv_bytes.decode("latin-1")
    except UnicodeDecodeError:
        text = csv_bytes.decode("utf-8", errors="replace")

    lines = text.splitlines()
    if len(lines) < 10:
        return {}, pd.DataFrame()

    # Lê metadados das 8 primeiras linhas
    meta: dict[str, str] = {}
    for line in lines[:8]:
        parts = line.split(";")
        if len(parts) >= 2:
            key = parts[0].strip().rstrip(":").upper()
            val = parts[1].strip()
            meta[key] = val

    # Filtra apenas RS
    uf = meta.get("UF", "").strip().upper()
    if uf != "RS":
        return {}, pd.DataFrame()

    station_meta = {
        "station_id":  meta.get("CODIGO (WMO)", "").strip(),
        "name":        meta.get("ESTACAO", "").strip().title(),
        "lat":         _safe_float(meta.get("LATITUDE", "")),
        "lon":         _safe_float(meta.get("LONGITUDE", "")),
        "elevation_m": _safe_float(meta.get("ALTITUDE", "")),
        "state":       "RS",
        "source":      "INMET",
        "active":      True,
    }

    if not station_meta["station_id"]:
        # Tenta extrair do nome do arquivo: INMET_S_RS_A826_...
        parts = filename.replace(".CSV", "").replace(".csv", "").split("_")
        if len(parts) >= 4:
            station_meta["station_id"] = parts[3]

    # Parseia dados (skiprows=8 → linha 8 é o header)
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=";",
            skiprows=8,
            encoding="latin-1",
            decimal=",",
            na_values=["", "-9999", "-9999.0", "null", "NULL"],
        )
    except Exception as exc:
        logger.debug(f"  {filename}: erro ao parsear CSV — {exc}")
        return station_meta, pd.DataFrame()

    if df.empty:
        return station_meta, pd.DataFrame()

    # Normaliza nomes de colunas — remove acentos e caracteres especiais
    col_map: dict[str, str] = {}
    for col in df.columns:
        col_clean = col.strip()
        col_upper = col_clean.upper()
        if "DATA" in col_upper and "HORA" not in col_upper:
            col_map[col] = "data_str"
        elif "HORA" in col_upper and "UTC" in col_upper:
            col_map[col] = "hora_str"
        elif "PRECIPITA" in col_upper and "TOTAL" in col_upper:
            col_map[col] = "rain_1h_mm"
        elif "PRESSAO" in col_upper or "PRESS" in col_upper and "NIVEL" in col_upper:
            if "rain_1h_mm" in col_map.values():  # primeira coluna de pressão
                col_map[col] = "pressure_hpa"
        elif "TEMPERATURA DO AR" in col_upper or "BULBO SECO" in col_upper:
            col_map[col] = "temperature"
        elif "UMIDADE RELATIVA DO AR" in col_upper or ("UMIDADE" in col_upper and "HORARIA" in col_upper.replace("Á", "A").replace("Â", "A")):
            col_map[col] = "humidity"
        elif "VENTO" in col_upper and "DIRE" in col_upper:
            col_map[col] = "wind_dir"
        elif "VENTO" in col_upper and "VELOCIDADE" in col_upper and "HORARIA" in col_upper.replace("Á", "A").replace("Â", "A"):
            col_map[col] = "wind_speed"

    df = df.rename(columns=col_map)

    # Garante colunas mínimas para timestamp
    if "data_str" not in df.columns or "hora_str" not in df.columns:
        # Tenta identificar por posição (Data = col 0, Hora = col 1)
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "data_str",
                                     df.columns[1]: "hora_str"})
        else:
            return station_meta, pd.DataFrame()

    # Constrói timestamp UTC
    def _parse_ts(row: pd.Series) -> pd.Timestamp | None:
        try:
            hora = str(row["hora_str"]).replace(" UTC", "").strip().zfill(4)
            ts_str = f"{row['data_str']} {hora[:2]}:{hora[2:]}"
            return pd.Timestamp(ts_str, tz="UTC")
        except (ValueError, TypeError):
            return None

    df["ts"] = df.apply(_parse_ts, axis=1)
    df = df.dropna(subset=["ts"])
    df["station_id"] = station_meta["station_id"]
    df["source"] = "INMET"

    # Mantém só colunas relevantes
    keep = ["station_id", "ts", "rain_1h_mm", "temperature",
            "humidity", "pressure_hpa", "wind_speed", "wind_dir", "source"]
    df = df[[c for c in keep if c in df.columns]].copy()

    return station_meta, df


def collect_historico_rs(
    year: int | None = None,
    days_back: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coleta dados históricos INMET RS via ZIP público — sem geo-restrição.

    Baixa o ZIP anual do INMET, extrai em memória, filtra estações RS,
    parseia os CSVs e retorna os últimos `days_back` dias de dados.

    Salva:
      - data/processed/inmet_historico_rs.parquet  (leituras)
      - data/processed/inmet_stations_rs.parquet   (metadados estações)
      - data/raw/inmet_stations_rs.parquet          (cópia raw)

    Args:
        year: Ano do ZIP (padrão: ano atual).
        days_back: Quantos dias para trás filtrar (padrão 30).

    Returns:
        Tupla (df_stations, df_leituras). Ambos podem ser vazios em caso de falha.

    Raises:
        RuntimeError: Se o download do ZIP falhar.
    """
    if year is None:
        year = datetime.now(tz=timezone.utc).year

    t0 = datetime.now(tz=timezone.utc)
    logger.info("=" * 60)
    logger.info(f"INMET Histórico {year} — coleta via ZIP público RS")
    logger.info(f"  Janela: últimos {days_back} dias")
    logger.info("=" * 60)

    # Download
    zip_bytes = _download_zip_inmet(year)

    # Abre ZIP em memória
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days_back))
    cutoff_naive = cutoff.replace(tzinfo=None)

    stations_meta: list[dict] = []
    all_frames: list[pd.DataFrame] = []
    rs_files = ok = skip = err = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        all_names = zf.namelist()
        logger.info(f"  ZIP: {len(all_names)} arquivos totais")

        for name in all_names:
            # Filtra apenas CSVs RS pelo nome do arquivo
            upper = name.upper()
            if not upper.endswith(".CSV"):
                continue

            # Formato: INMET_S_RS_A826_... ou INMET_CO_RS_...
            parts = upper.replace(".CSV", "").split("_")
            is_rs = (
                len(parts) >= 3 and parts[2] == "RS"
            ) or "_RS_" in upper

            if not is_rs:
                skip += 1
                continue

            rs_files += 1
            try:
                csv_bytes = zf.read(name)
                station_meta, df = _parse_inmet_csv(csv_bytes, name)

                if df.empty:
                    continue

                # Filtra últimos N dias
                if "ts" in df.columns:
                    ts_naive = pd.to_datetime(df["ts"]).dt.tz_localize(None)
                    df = df[ts_naive >= cutoff_naive]

                if not df.empty:
                    all_frames.append(df)
                    ok += 1

                if station_meta and station_meta.get("station_id"):
                    stations_meta.append(station_meta)

            except Exception as exc:
                logger.warning(f"  {name}: erro — {exc}")
                err += 1

    logger.info(
        f"  CSV processados: {rs_files} RS | {ok} com dados | "
        f"{skip} pulados | {err} erros"
    )

    # Consolida estações + situação oficial (Operante/Pane) do inventário
    df_stations = pd.DataFrame(stations_meta).drop_duplicates("station_id") \
        if stations_meta else pd.DataFrame()
    df_stations = _aplicar_situacao_oficial(df_stations)

    # Consolida leituras
    if not all_frames:
        logger.warning("  Nenhuma leitura extraída do ZIP histórico")
        df_leituras = pd.DataFrame()
    else:
        df_leituras = pd.concat(all_frames, ignore_index=True)
        df_leituras = df_leituras.drop_duplicates(subset=["station_id", "ts"])
        logger.success(
            f"  Total leituras RS (últimos {days_back}d): {len(df_leituras)} "
            f"| {df_leituras['station_id'].nunique() if not df_leituras.empty else 0} estações"
        )

    # Persistência híbrida (Supabase + R2) via HybridWriter
    if _HW_OK:
        writer = _HybridWriter()
        # Stations primeiro (rain_readings tem FK → stations)
        res_st = writer.write_stations(
            df_stations,
            path=_INV_PARQUET,
            extra_paths=[_RAW_DIR / "inmet_stations_rs.parquet"],
        )
        res_rd = writer.write_rain_readings(df_leituras, path=_HIST_PARQUET)
        counts = {
            "stations":     res_st.pg_rows,
            "rain_readings": res_rd.pg_rows,
        }
    else:
        # Fallback sem HybridWriter
        _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        if not df_leituras.empty:
            df_leituras.to_parquet(_HIST_PARQUET, index=False)
            logger.info(f"  Salvo: {_HIST_PARQUET}")
        if not df_stations.empty:
            df_stations.to_parquet(_INV_PARQUET, index=False)
            df_stations.to_parquet(_RAW_DIR / "inmet_stations_rs.parquet", index=False)
            logger.info(f"  Salvo: {_INV_PARQUET} ({len(df_stations)} estações RS)")
        logger.warning("HybridWriter indisponível — contagens zeradas (sem fallback DuckDB).")
        counts = {"stations": 0, "rain_readings": 0}

    duration = (datetime.now(tz=timezone.utc) - t0).total_seconds()
    logger.info("=" * 60)
    logger.success(
        f"INMET Histórico — concluído em {duration:.1f}s | "
        f"{counts['stations']} estações | {counts['rain_readings']} leituras "
        f"{'Supabase' if _HW_OK else 'sem persistência'}"
    )
    logger.info("=" * 60)

    return df_stations, df_leituras


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

    import argparse

    parser = argparse.ArgumentParser(
        description="Coletor INMET RS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de operação:
  --mode=historico  ZIP público anual — sem geo-restrição (GitHub Actions)
  --mode=inventory  Apenas inventário via API (funciona globalmente)
  --mode=full       Inventário + dados horários via API (requer IP BR)
  --mode=hourly     Só dados horários via API (requer IP BR)

Exemplos:
  python -m collectors.inmet_collector --mode=historico
  python -m collectors.inmet_collector --mode=historico --dias=7
  python -m collectors.inmet_collector --mode=inventory
  python -m collectors.inmet_collector --mode=full --max=10
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "inventory", "hourly", "historico"],
        default="historico",
        help="Modo de coleta (padrão: historico)",
    )
    parser.add_argument("--dias",  type=int, default=30,
                        help="Dias para trás — usado em --mode=historico (padrão 30)")
    parser.add_argument("--ano",   type=int, default=None,
                        help="Ano do ZIP histórico (padrão: ano atual)")
    parser.add_argument("--max",   type=int, default=None,
                        help="Limite de estacoes para teste com --mode=full")
    parser.add_argument("--force-inv", action="store_true",
                        help="Forca atualizacao do inventario")
    args = parser.parse_args()

    if args.mode == "historico":
        df_st, df_leit = collect_historico_rs(year=args.ano, days_back=args.dias)
        print("\n--- INMET Histórico RS ---")
        print(f"  estacoes_rs: {len(df_st)}")
        print(f"  leituras:    {len(df_leit)}")
        print(f"  parquet:     {_HIST_PARQUET}")
        sys.exit(0)

    if args.mode == "inventory":
        client  = INMETClient()
        inv     = coletar_inventario_rs(client, force=args.force_inv)
        if _HW_OK:
            from database.hybrid_writer import HybridWriter as _HW
            counts = {"stations": _HW().write_stations(inv, path=_INV_PARQUET).pg_rows, "rain_readings": 0}
        else:
            counts = {"stations": 0, "rain_readings": 0}
        print("\n--- Inventário INMET RS ---")
        print(f"  estacoes:        {len(inv)}")
        n_op = inv["active"].sum() if "active" in inv.columns else len(inv)
        print(f"  operantes:       {n_op}")
        print(f"  salvo em:        {_INV_PARQUET}")
        print(f"  stations gravadas: {counts['stations']}")
        sys.exit(0)

    result = collect_inmet(
        dias_back=args.dias,
        max_estacoes=args.max,
        force_inventario=args.force_inv,
        skip_hourly=(args.mode == "inventory"),
    )

    print("\n--- Resultado -----------------------------------")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("-------------------------------------------------")
