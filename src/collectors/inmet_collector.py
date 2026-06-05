"""
Agente 1 — Arquiteto de Dados
Coletor INMET — estações meteorológicas automáticas do RS.

Coleta via API pública apitempo.inmet.gov.br (sem token):
  - Inventário das ~500 estações automáticas do RS
  - Dados horários das últimas 24h por estação
  - Persiste em DuckDB (tabelas stations + rain_readings) e Parquet
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import niquests
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

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

from database.db_manager import ClimateDB  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_BASE_URL    = "https://apitempo.inmet.gov.br"
_PARQUET_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_OUT_PARQUET = _PARQUET_DIR / "inmet_rs.parquet"
_INV_PARQUET = _PARQUET_DIR / "inmet_inventario_rs.parquet"
_CACHE_TTL_S = 600   # 10 minutos

# Colunas da API → padrão interno
_COL_MAP: dict[str, str] = {
    # identificação
    "CD_ESTACAO":  "station_id",
    "DC_NOME":     "name",
    "SG_ESTADO":   "state",
    "CD_MUNICIPIO":"municipality",
    "VL_LATITUDE": "lat",
    "VL_LONGITUDE":"lon",
    "VL_ALTITUDE": "elevation_m",
    # observações
    "DT_MEDICAO":  "date",
    "HR_MEDICAO":  "hour_utc",
    "TEMP_INS":    "temperature",
    "TEMP_MAX":    "temp_max",
    "TEMP_MIN":    "temp_min",
    "UMID_INS":    "humidity",
    "UMID_MAX":    "humidity_max",
    "UMID_MIN":    "humidity_min",
    "PRES_INS":    "pressure_hpa",
    "VENTO_VEL":   "wind_speed",
    "VENTO_DIR":   "wind_dir",
    "CHUVA":       "rain_1h_mm",
    "VEN_VEL":     "wind_speed",    # alias em algumas versões da API
    "PRE_INS":     "pressure_hpa",  # alias
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
        """Lista todas as estações automáticas do RS — endpoint /estacoes/T.

        Filtra por SG_ESTADO == 'RS' na resposta.

        Returns:
            DataFrame com metadados de cada estação RS.

        Raises:
            niquests.exceptions.RequestException: Se a API falhar após retries.
        """
        logger.info("INMET: listando estacoes automaticas RS...")
        data = self._get("estacoes/T")
        df = pd.DataFrame(data)
        df = df[df.get("SG_ESTADO", pd.Series(dtype=str)) == "RS"].copy()
        df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
        for col in ["lat", "lon", "elevation_m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )
        logger.success(f"INMET: {len(df)} estacoes automaticas no RS.")
        return df

    def dados_estacao(self, station_id: str, date_str: str) -> pd.DataFrame:
        """Retorna dados horários de uma estação para uma data específica.

        Endpoint: /estacao/dados/{data}/{codWsi}
        Formato da data: aaaa-MM-dd

        Args:
            station_id: Código da estação (ex.: 'A803').
            date_str: Data no formato 'yyyy-MM-dd'.

        Returns:
            DataFrame com leituras horárias. Vazio se não houver dados.

        Raises:
            niquests.exceptions.RequestException: Após retries esgotados.
        """
        try:
            data = self._get(f"estacao/dados/{date_str}/{station_id}")
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

        Endpoint: /estacao/{data_ini}/{data_fim}/{codWsi}

        Args:
            station_id: Código da estação (ex.: 'A803').
            date_ini: Data início 'yyyy-MM-dd'.
            date_fim: Data fim 'yyyy-MM-dd'.

        Returns:
            DataFrame com leituras horárias do período. Vazio se sem dados.

        Raises:
            niquests.exceptions.RequestException: Após retries esgotados.
        """
        try:
            data = self._get(f"estacao/{date_ini}/{date_fim}/{station_id}")
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
    """Coleta dados horários de todas as estações RS para os últimos N dias.

    Itera o inventário, chama dados_periodo por estação, concatena tudo.
    Salva leituras normalizadas em data/raw/inmet_rs.parquet.

    Args:
        client: INMETClient inicializado.
        inventario: DataFrame do inventário RS (precisa de station_id).
        dias_back: Quantos dias atrás coletar (padrão 1 = ontem + hoje).
        delay_s: Pausa entre chamadas para não sobrecarregar a API (segundos).
        max_estacoes: Limita o número de estações para testes. None = todas.

    Returns:
        DataFrame consolidado com todas as leituras coletadas.

    Raises:
        OSError: Se não for possível criar o diretório de saída.
    """
    agora   = datetime.now(timezone.utc).replace(tzinfo=None)
    ini     = (agora - timedelta(days=dias_back)).strftime("%Y-%m-%d")
    fim     = agora.strftime("%Y-%m-%d")

    cod_col = next(
        (c for c in ["station_id", "CD_ESTACAO"] if c in inventario.columns),
        inventario.columns[0],
    )
    codigos = inventario[cod_col].dropna().astype(str).tolist()
    if max_estacoes:
        codigos = codigos[:max_estacoes]

    logger.info(
        f"INMET: coletando {len(codigos)} estacoes RS "
        f"({ini} → {fim})..."
    )

    frames: list[pd.DataFrame] = []
    ok = erros = 0

    for cod in codigos:
        try:
            df_raw = client.dados_periodo(cod, ini, fim)
            if df_raw.empty:
                continue
            df = _normalizar_df(df_raw, cod)
            frames.append(df)
            ok += 1
            time.sleep(delay_s)
        except RetryError as exc:
            logger.warning(f"  {cod}: falhou após retries — {exc}")
            erros += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"  {cod}: {exc}")
            erros += 1

    logger.info(f"INMET: {ok} estacoes com dados, {erros} erros.")

    if not frames:
        logger.warning("Nenhum dado INMET coletado.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(_OUT_PARQUET, index=False, engine="pyarrow")
    logger.success(f"inmet_rs.parquet salvo: {len(df_all)} leituras.")
    return df_all


# ---------------------------------------------------------------------------
# Persistência no DuckDB
# ---------------------------------------------------------------------------

def upsert_inmet_duckdb(
    db: ClimateDB,
    inventario: pd.DataFrame,
    leituras: pd.DataFrame,
) -> dict[str, int]:
    """Insere estações INMET no DuckDB e faz upsert das leituras.

    Insere metadados em `stations` (upsert por station_id) e leituras em
    `rain_readings` (INSERT OR IGNORE por station_id + ts).

    Args:
        db: Instância ClimateDB com conexão ativa.
        inventario: DataFrame do inventário (precisa de station_id, name, lat, lon).
        leituras: DataFrame de leituras normalizado (precisa de station_id, ts).

    Returns:
        Dict com chaves stations e rain_readings indicando linhas processadas.

    Raises:
        duckdb.Error: Se alguma operação DuckDB falhar.
    """
    # Stations — apenas colunas que a tabela espera
    inv_cols = [c for c in ["station_id", "name", "lat", "lon", "elevation_m",
                             "municipality", "state"] if c in inventario.columns]
    df_st = inventario[inv_cols].drop_duplicates("station_id").copy()
    df_st["source"] = "INMET"
    df_st["state"]  = df_st.get("state", pd.Series("RS", index=df_st.index)).fillna("RS")
    # upsert_stations exige colunas river e municipality mesmo que nulas
    if "river"        not in df_st.columns:
        df_st["river"]        = None
    if "municipality" not in df_st.columns:
        df_st["municipality"] = None
    if "active"       not in df_st.columns:
        df_st["active"]       = True
    n_st = db.upsert_stations(df_st)

    if leituras.empty:
        return {"stations": n_st, "rain_readings": 0}

    # Leituras — filtra colunas compatíveis com rain_readings
    reading_cols = [
        "station_id", "ts",
        "rain_1h_mm", "temperature", "humidity",
        "pressure_hpa", "wind_speed", "wind_dir", "source",
    ]
    df_rd = leituras[[c for c in reading_cols if c in leituras.columns]].copy()
    df_rd = df_rd.dropna(subset=["station_id", "ts"])
    n_rd = db.insert_rain_readings(df_rd)

    return {"stations": n_st, "rain_readings": n_rd}


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def collect_inmet(
    db: ClimateDB | None = None,
    dias_back: int = 1,
    max_estacoes: int | None = None,
    force_inventario: bool = False,
) -> dict[str, Any]:
    """Pipeline completo de coleta INMET para o RS.

    1. Obtém inventário de estações automáticas (com cache 24h).
    2. Coleta dados horários de todas as estações RS.
    3. Persiste em Parquet (data/raw/inmet_rs.parquet).
    4. Faz upsert no DuckDB.

    Args:
        db: Instância ClimateDB. Se None, abre conexão temporária.
        dias_back: Janela de coleta em dias (padrão 1).
        max_estacoes: Limite de estações para testes (None = todas).
        force_inventario: Força reatualização do inventário mesmo com cache.

    Returns:
        Dict com stations (int), rain_readings (int), parquet (str),
        duration_s (float).

    Raises:
        niquests.exceptions.RequestException: Se a API falhar irreversivelmente.
    """
    t_start = datetime.now(tz=timezone.utc)
    logger.info("=== INMETCollector — iniciando coleta RS ===")

    client = INMETClient()

    try:
        inventario = coletar_inventario_rs(client, force=force_inventario)
        leituras   = coletar_dados_rs(
            client, inventario,
            dias_back=dias_back,
            max_estacoes=max_estacoes,
        )
    except RetryError as exc:
        logger.error(f"INMET: falha irrecuperável — {exc}")
        raise

    _owns_db = db is None
    if _owns_db:
        db = ClimateDB()
    try:
        counts = upsert_inmet_duckdb(db, inventario, leituras)
    finally:
        if _owns_db:
            db.close()

    duration = (datetime.now(tz=timezone.utc) - t_start).total_seconds()
    logger.info(f"=== INMETCollector concluído em {duration:.1f}s ===")

    return {
        "stations":     counts["stations"],
        "rain_readings":counts["rain_readings"],
        "estacoes_inv": len(inventario),
        "leituras_raw": len(leituras),
        "parquet":      str(_OUT_PARQUET),
        "duration_s":   round(duration, 2),
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

    import argparse

    parser = argparse.ArgumentParser(description="Coletor INMET RS")
    parser.add_argument("--dias",  type=int, default=1,
                        help="Dias para trás (padrão 1)")
    parser.add_argument("--max",   type=int, default=None,
                        help="Limite de estacoes para teste (padrao: todas)")
    parser.add_argument("--force-inv", action="store_true",
                        help="Forca atualizacao do inventario")
    args = parser.parse_args()

    result = collect_inmet(
        dias_back=args.dias,
        max_estacoes=args.max,
        force_inventario=args.force_inv,
    )

    print("\n--- Resultado -----------------------------------")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("-------------------------------------------------")
