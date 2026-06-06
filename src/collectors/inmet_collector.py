"""
Agente 1 — Arquiteto de Dados
Coletor INMET — estações meteorológicas automáticas do RS.

Coleta via API pública apitempo.inmet.gov.br (sem token):
  - Inventário das ~500 estações automáticas do RS
  - Dados horários das últimas 24h por estação
  - Persiste em DuckDB (tabelas stations + rain_readings) e Parquet
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
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

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_BASE_URL    = "https://apitempo.inmet.gov.br"
_ROOT        = Path(__file__).resolve().parents[2]
_PARQUET_DIR = _ROOT / "data" / "processed"
_OUT_PARQUET = _PARQUET_DIR / "inmet_hourly.parquet"
_INV_PARQUET = _PARQUET_DIR / "inmet_stations_rs.parquet"
_CACHE_TTL_S = 600   # 10 minutos
_DB_PATH     = Path(os.getenv("DB_PATH", str(_ROOT / "data" / "climate.duckdb")))

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
# Persistência no DuckDB
# ---------------------------------------------------------------------------

def upsert_inmet_duckdb(
    inventario: pd.DataFrame,
    leituras: pd.DataFrame,
) -> dict[str, int]:
    """Insere estações INMET no DuckDB e faz upsert das leituras.

    Insere metadados em `stations` (INSERT OR REPLACE por station_id) e
    leituras em `rain_readings` (INSERT OR IGNORE por station_id + ts).
    Usa DuckDB diretamente — sem dependência de ClimateDB.

    Args:
        inventario: DataFrame do inventário RS (station_id, name, lat, lon, …).
        leituras: DataFrame de leituras normalizado (station_id, ts, …).

    Returns:
        Dict com chaves "stations" e "rain_readings" indicando linhas inseridas.

    Raises:
        duckdb.Error: Se alguma operação DuckDB falhar de modo irrecuperável.
    """
    try:
        conn = duckdb.connect(str(_DB_PATH))
    except duckdb.IOException as exc:
        logger.warning(f"  DuckDB bloqueado por outro processo — skip upsert: {exc}")
        return {"stations": 0, "rain_readings": 0}

    n_st = n_rd = 0

    try:
        # ── Stations ───────────────────────────────────────────────────────
        for _, row in inventario.iterrows():
            try:
                conn.execute("""
                    INSERT INTO stations
                        (station_id, name, source, lat, lon, elevation_m,
                         state, municipality, river, active)
                    VALUES (?, ?, 'INMET', ?, ?, ?, ?, ?, NULL, ?)
                    ON CONFLICT (station_id) DO UPDATE SET
                        name         = excluded.name,
                        lat          = excluded.lat,
                        lon          = excluded.lon,
                        elevation_m  = excluded.elevation_m,
                        municipality = excluded.municipality,
                        active       = excluded.active
                """, [
                    str(row.get("station_id", "") or ""),
                    str(row.get("name", "") or ""),
                    row.get("lat"),
                    row.get("lon"),
                    row.get("elevation_m"),
                    str(row.get("state", "RS") or "RS"),
                    row.get("municipality"),
                    bool(row.get("active", True)),
                ])
                n_st += 1
            except duckdb.Error as exc:
                logger.debug(f"  station {row.get('station_id')}: {exc}")

        conn.commit()
        logger.info(f"  DuckDB stations: {n_st} estações INMET inseridas/atualizadas")

        # ── Rain readings ──────────────────────────────────────────────────
        if not leituras.empty:
            df_rd = leituras.dropna(subset=["station_id", "ts"]).copy()
            df_rd["source"] = "INMET"

            for _, row in df_rd.iterrows():
                try:
                    sid = str(row["station_id"])
                    ts  = row["ts"]
                    key = f"INMET:{sid}:{ts}"
                    rid = int.from_bytes(
                        hashlib.sha256(key.encode()).digest()[:8], "big"
                    ) & 0x7FFFFFFFFFFFFFFF

                    conn.execute("""
                        INSERT OR IGNORE INTO rain_readings
                            (id, station_id, ts, rain_1h_mm,
                             temperature, humidity, pressure_hpa,
                             wind_speed, wind_dir, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'INMET')
                    """, [
                        rid, sid, ts,
                        row.get("rain_1h_mm"),
                        row.get("temperature"),
                        row.get("humidity"),
                        row.get("pressure_hpa"),
                        row.get("wind_speed"),
                        row.get("wind_dir"),
                    ])
                    n_rd += 1
                except duckdb.Error as exc:
                    logger.debug(f"  reading {row.get('station_id')} {row.get('ts')}: {exc}")

            conn.commit()
            logger.info(f"  DuckDB rain_readings: {n_rd} leituras INMET inseridas")

    finally:
        conn.close()

    return {"stations": n_st, "rain_readings": n_rd}


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
    4. Faz upsert no DuckDB (stations + rain_readings).

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
    logger.info(f"  DB_PATH: {_DB_PATH.resolve()}")
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

    counts = upsert_inmet_duckdb(inventario, leituras)

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
  --mode=full       Inventário + dados horários (requer IP brasileiro)
  --mode=inventory  Apenas inventário de estações (funciona globalmente)
  --mode=hourly     Apenas dados horários, assume inventário em cache

Exemplos:
  python -m collectors.inmet_collector --mode=inventory
  python -m collectors.inmet_collector --mode=full --dias=7
  python -m collectors.inmet_collector --mode=full --max=10
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "inventory", "hourly"],
        default="full",
        help="Modo de coleta (padrão: full)",
    )
    parser.add_argument("--dias",  type=int, default=0,
                        help="Dias para trás (0 = apenas última hora, padrão)")
    parser.add_argument("--max",   type=int, default=None,
                        help="Limite de estacoes para teste (padrao: todas)")
    parser.add_argument("--force-inv", action="store_true",
                        help="Forca atualizacao do inventario")
    args = parser.parse_args()

    if args.mode == "inventory":
        # Coleta apenas inventário — funciona de qualquer IP
        client  = INMETClient()
        inv     = coletar_inventario_rs(client, force=args.force_inv)
        counts  = upsert_inmet_duckdb(inv, pd.DataFrame())
        print(f"\n--- Inventário INMET RS ---")
        print(f"  estacoes: {len(inv)}")
        n_op = inv['active'].sum() if 'active' in inv.columns else len(inv)
        print(f"  operantes: {n_op}")
        print(f"  salvo em: {_INV_PARQUET}")
        print(f"  DuckDB stations: {counts['stations']}")
        sys.exit(0)

    # --mode=hourly: pula inventário, usa cache existente
    skip_inv = (args.mode == "hourly")

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
