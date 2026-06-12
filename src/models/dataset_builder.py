"""
Agente 2 — Cientista de Dados / ML
Unifica INMET (BigQuery) + ANA (SOAP) + R2 (recentes) no dataset do LSTM.

Fontes:
  A. INMET via BigQuery basedosdados — meteorologia diária agregada por
     bacia (a agregação roda DENTRO do BigQuery: mesmo custo de varredura,
     download ~1000x menor do que puxar as linhas horárias).
  B. ANA HidroWeb SOAP (ana_fetcher) — cotas diárias 2000-2026 (alvo).
  C. R2 Cloudflare (historico/live_river_levels) — telemetria recente
     posterior ao fim da série convencional da ANA.

Saída: DataFrame diário com features + alvo, salvo no R2 em
treino/{rio}_lstm_dataset_v1.parquet (Snappy).
"""

from __future__ import annotations

import io
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from loguru import logger

_SRC_DIR = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

from models.ana_fetcher import ESTACOES_RS, fetch_serie_historica  # noqa: E402
from models.data_loader import BIGQUERY_PROJECT_ID, log_consumption  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Municípios representativos da bacia hidrográfica de cada rio-alvo
# (códigos IBGE de 7 dígitos — estáveis; cf. grade NWP do config.yaml).
_BACIAS_IBGE: dict[str, dict[str, str]] = {
    "guaiba": {
        "4314902": "Porto Alegre",
        "4305108": "Caxias do Sul",
        "4316907": "Santa Maria",
        "4314100": "Passo Fundo",
    },
    "jacui": {
        "4316907": "Santa Maria",
        "4314100": "Passo Fundo",
        "4305355": "Cruz Alta",
    },
    "taquari": {
        "4311403": "Lajeado",
        "4305108": "Caxias do Sul",
        "4314050": "Passo Fundo",
    },
    "sinos": {
        "4318705": "São Leopoldo",
        "4313409": "Novo Hamburgo",
        "4314902": "Porto Alegre",
    },
    "camaqua": {
        "4303509": "Camaquã",
        "4300406": "Bagé",
    },
}

_R2_RIVER_PREFIX = "historico/live_river_levels"
_R2_TREINO_PREFIX = "treino"
_LIMITE_AVISO_MB = 500.0

_MAX_BYTES_BILLED = 10 * 1024**3


# ---------------------------------------------------------------------------
# Passo A — INMET agregado por bacia (agregação server-side no BigQuery)
# ---------------------------------------------------------------------------

def _load_inmet_bacia_diario(rio_alvo: str, start_year: int) -> pd.DataFrame:
    """Meteorologia diária da bacia, agregada dentro do BigQuery.

    Por estação/dia: precipitação somada, temperatura/umidade médias.
    Pela bacia/dia: média entre as estações dos municípios da bacia.

    Args:
        rio_alvo: Chave de _BACIAS_IBGE (guaiba, jacui, taquari, sinos, camaqua).
        start_year: Ano inicial da série.

    Returns:
        DataFrame diário com data, precip_bacia_mm, temp_media_c,
        umidade_media_pct.

    Raises:
        KeyError: Se rio_alvo não estiver mapeado em _BACIAS_IBGE.
        google.api_core.exceptions.GoogleAPIError: Falha na query.
    """
    municipios = _BACIAS_IBGE[rio_alvo]
    ids = ", ".join(f"'{m}'" for m in municipios)
    logger.info(
        f"INMET bacia {rio_alvo}: {', '.join(municipios.values())} "
        f"(desde {start_year})"
    )

    sql = f"""
        WITH por_estacao_dia AS (
            SELECT
                m.data,
                m.id_estacao,
                SUM(m.precipitacao_total)      AS precip_dia_mm,
                AVG(m.temperatura_bulbo_hora)  AS temp_dia_c,
                AVG(m.umidade_rel_hora)        AS umid_dia_pct
            FROM `basedosdados.br_inmet_bdmep.microdados` AS m
            JOIN `basedosdados.br_inmet_bdmep.estacao` AS e USING (id_estacao)
            WHERE e.id_municipio IN ({ids})
              AND m.ano >= {int(start_year)}
            GROUP BY m.data, m.id_estacao
        )
        SELECT
            data,
            AVG(precip_dia_mm) AS precip_bacia_mm,
            AVG(temp_dia_c)    AS temp_media_c,
            AVG(umid_dia_pct)  AS umidade_media_pct
        FROM por_estacao_dia
        GROUP BY data
        ORDER BY data
    """

    client = bigquery.Client(project=BIGQUERY_PROJECT_ID)

    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    log_consumption(dry.total_bytes_processed, 0)

    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        maximum_bytes_billed=_MAX_BYTES_BILLED, use_query_cache=True))
    df = job.to_dataframe()
    log_consumption(job.total_bytes_processed or 0, job.total_bytes_billed or 0)

    df["data"] = pd.to_datetime(df["data"]).dt.date
    logger.info(f"  INMET: {len(df):,} dias agregados da bacia.")
    return df


# ---------------------------------------------------------------------------
# Passo B — Cotas históricas ANA
# ---------------------------------------------------------------------------

def _load_cotas_ana(rio_alvo: str, start_year: int) -> pd.DataFrame:
    """Cota diária histórica do rio-alvo (estação primária do ana_fetcher).

    Args:
        rio_alvo: Chave de ESTACOES_RS.
        start_year: Ano inicial.

    Returns:
        DataFrame com data e cota_m.
    """
    cod = ESTACOES_RS[rio_alvo][0]
    logger.info(f"ANA SOAP: estação {cod} ({rio_alvo}), {start_year}-2026...")
    df = fetch_serie_historica(
        cod, f"01/01/{start_year}", "31/12/2026", tipo_dados=1
    )
    if df.empty:
        logger.warning(f"  ANA: estação {cod} sem dados no período.")
        return pd.DataFrame(columns=["data", "cota_m"])
    logger.info(f"  ANA: {len(df):,} dias de cota ({df['data'].min()} a {df['data'].max()}).")
    return df[["data", "cota_m"]]


# ---------------------------------------------------------------------------
# Passo C — Complemento recente via R2
# ---------------------------------------------------------------------------

def _r2_client() -> Optional["boto3.client"]:
    """Cria cliente boto3 para o R2 (None se variáveis ausentes)."""
    if not os.getenv("R2_ENDPOINT_URL"):
        return None
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    )


def _load_cotas_r2(rio_alvo: str, apos: date) -> pd.DataFrame:
    """Cotas telemétricas recentes do R2 para datas posteriores à série ANA.

    Lê os Parquets diários de historico/live_river_levels (1 leitura média
    por dia/rio) apenas para os dias após `apos`.

    Args:
        rio_alvo: Slug do rio (coluna rio_nome normalizada).
        apos: Última data presente na série histórica ANA.

    Returns:
        DataFrame com data e cota_m (vazio se nada novo no R2).
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        return pd.DataFrame(columns=["data", "cota_m"])

    import unicodedata

    def _slug(s: str) -> str:
        return (unicodedata.normalize("NFD", str(s))
                .encode("ascii", "ignore").decode().lower().strip())

    registros: list[dict] = []
    dia = apos + timedelta(days=1)
    hoje = datetime.now(timezone.utc).date()
    while dia <= hoje:
        prefix = (f"{_R2_RIVER_PREFIX}/ano={dia.year}"
                  f"/mes={dia.month:02d}/dia={dia.day:02d}/")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=200)
        objs = sorted(resp.get("Contents", []), key=lambda o: o["LastModified"])
        if objs:
            body = s3.get_object(Bucket=bucket, Key=objs[-1]["Key"])["Body"].read()
            dfp = pd.read_parquet(io.BytesIO(body))
            col_nivel = ("current_level_m" if "current_level_m" in dfp.columns
                         else "nivel_atual_m" if "nivel_atual_m" in dfp.columns
                         else None)
            if col_nivel and "rio_nome" in dfp.columns:
                sel = dfp[dfp["rio_nome"].map(_slug) == rio_alvo]
                if not sel.empty:
                    registros.append({
                        "data":   dia,
                        "cota_m": float(sel[col_nivel].mean()),
                    })
        dia += timedelta(days=1)

    df = pd.DataFrame(registros)
    logger.info(f"  R2: {len(df)} dias recentes complementados (após {apos}).")
    return df


# ---------------------------------------------------------------------------
# Passos D + E — Features e limpeza
# ---------------------------------------------------------------------------

def _features_e_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas e aplica interpolação/limpeza.

    Args:
        df: DataFrame diário com data, cota_m, precip_bacia_mm,
            temp_media_c, umidade_media_pct.

    Returns:
        DataFrame limpo com features de janela, lags, sazonalidade e delta.
    """
    df = df.sort_values("data").reset_index(drop=True)
    dt = pd.to_datetime(df["data"])

    # Interpolação linear apenas para gaps curtos (< 3 dias) — manutenção
    # de sensor; gaps longos (ex.: mai/2024) permanecem NaN e são removidos.
    antes_nan = df["cota_m"].isna().sum()
    df["cota_m"] = df["cota_m"].interpolate(method="linear", limit=2,
                                            limit_area="inside")
    interpolados = antes_nan - df["cota_m"].isna().sum()

    # Janelas deslizantes de precipitação
    for j in (3, 7, 15):
        df[f"precip_acum_{j}d"] = (
            df["precip_bacia_mm"].rolling(window=j, min_periods=j).sum()
        )

    # Lags da cota e taxa de variação
    for lag in (1, 2, 3):
        df[f"cota_lag_{lag}d"] = df["cota_m"].shift(lag)
    df["cota_delta_1d"] = df["cota_m"] - df["cota_lag_1d"]

    # Sazonalidade anual
    df["mes_seno"]    = np.sin(2 * np.pi * dt.dt.month / 12)
    df["mes_cosseno"] = np.cos(2 * np.pi * dt.dt.month / 12)

    criticas = ["cota_m", "precip_bacia_mm"]
    antes = len(df)
    df = df.dropna(subset=criticas).reset_index(drop=True)
    removidos = antes - len(df)
    logger.info(f"Limpeza: {removidos:,} dias removidos por NaN, "
                f"{interpolados:,} dias interpolados (gaps < 3 dias).")
    return df


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

def build_dataset(rio_alvo: str = "guaiba", start_year: int = 2000) -> pd.DataFrame:
    """Monta o dataset diário unificado (INMET + ANA + R2) para o LSTM.

    Args:
        rio_alvo: guaiba | jacui | taquari | sinos | camaqua.
        start_year: Ano inicial da série.

    Returns:
        DataFrame diário limpo com alvo (cota_m) e features.

    Raises:
        KeyError: rio_alvo desconhecido.
    """
    logger.info(f"=== build_dataset(rio={rio_alvo}, desde {start_year}) ===")

    inmet = _load_inmet_bacia_diario(rio_alvo, start_year)          # Passo A
    cotas = _load_cotas_ana(rio_alvo, start_year)                   # Passo B

    if not cotas.empty:                                              # Passo C
        recentes = _load_cotas_r2(rio_alvo, apos=cotas["data"].max())
        if not recentes.empty:
            cotas = pd.concat([cotas, recentes], ignore_index=True)

    df = inmet.merge(cotas, on="data", how="outer").sort_values("data")
    df = _features_e_limpeza(df)                                     # Passos D+E

    _relatorio_qualidade(df, rio_alvo)
    return df


def _relatorio_qualidade(df: pd.DataFrame, rio_alvo: str) -> None:
    """Loga o relatório de qualidade do dataset final.

    Args:
        df: Dataset final.
        rio_alvo: Identificador do rio (para o cabeçalho do log).
    """
    logger.info(f"--- Relatório de qualidade [{rio_alvo}] " + "-" * 30)
    logger.info(f"Total de dias na série: {len(df):,}")
    if df.empty:
        logger.warning("Dataset vazio — nada a reportar.")
        return
    logger.info(f"Período coberto: {df['data'].min()} → {df['data'].max()}")
    for col in df.columns:
        if col == "data":
            continue
        pct = df[col].notna().mean() * 100
        logger.info(f"  completude {col}: {pct:.1f}%")
    p10, p50, p90 = df["cota_m"].quantile([0.10, 0.50, 0.90])
    logger.info(
        f"Cotas — p10: {p10:.2f} m | mediana: {p50:.2f} m | "
        f"p90: {p90:.2f} m | máx histórico: {df['cota_m'].max():.2f} m "
        f"({df.loc[df['cota_m'].idxmax(), 'data']})"
    )


def save_to_r2(df: pd.DataFrame, rio_alvo: str = "guaiba") -> None:
    """Salva o dataset em Parquet/Snappy no R2 (treino/{rio}_lstm_dataset_v1).

    Args:
        df: Dataset final de build_dataset().
        rio_alvo: Slug do rio (compõe a chave do objeto).

    Raises:
        botocore.exceptions.BotoCoreError: Falha no upload.
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        logger.error("R2 não configurado — dataset não enviado.")
        return

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    tamanho_mb = buf.tell() / 1024**2
    buf.seek(0)

    key = f"{_R2_TREINO_PREFIX}/{rio_alvo}_lstm_dataset_v1.parquet"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.success(f"R2 upload: {key} ({tamanho_mb:.1f} MB)")
    if tamanho_mb > _LIMITE_AVISO_MB:
        logger.warning(
            f"WARNING: dataset com {tamanho_mb:.0f} MB — acima de "
            f"{_LIMITE_AVISO_MB:.0f} MB, próximo do conforto do R2 free."
        )


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

    dataset = build_dataset(rio_alvo="guaiba", start_year=2000)
    print(f"\nshape final: {dataset.shape}")
    print(dataset.tail().to_string())
    save_to_r2(dataset, rio_alvo="guaiba")
    print("\nDataset de treino construído com sucesso!")
