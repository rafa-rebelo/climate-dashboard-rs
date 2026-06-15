"""
Agente 2 — Cientista de Dados / ML
Carregador de dados históricos INMET via BigQuery (basedosdados) para o LSTM.

Pipeline de treino do modelo de previsão de cotas dos rios RS:
  BigQuery `basedosdados.br_inmet_bdmep.microdados` (séries desde 2000)
    → DataFrame pandas padronizado (precip_mm, temp_c, umidade_pct)
    → features do LSTM (src/models/lstm_river.py — Fase 2).

Controle de custo (free tier BigQuery = 1 TB/mês):
  1. DRY RUN antes de toda query — estima bytes sem custo;
  2. maximum_bytes_billed = 10 GB — aborta query que exceda o teto;
  3. log append em logs/bigquery_consumption.jsonl + alerta em 80% do limite.

Uso (PowerShell, venv ativo):
  python src/models/data_loader.py

Requisitos:
  pip install google-cloud-bigquery db-dtypes
  gcloud auth application-default login  (projeto clima-rs-lstm)
  .env: BIGQUERY_PROJECT_ID=clima-rs-lstm
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# google-cloud-bigquery é dep de TREINO (requirements-ml.txt), não instalada
# no CI enxuto. Import opcional para que dataset_builder/inference (que só
# reusam log_consumption/BIGQUERY_PROJECT_ID) carreguem sem bigquery.
try:
    from google.api_core import exceptions as gcp_exceptions
    from google.cloud import bigquery
    _BIGQUERY_OK = True
except ImportError:
    gcp_exceptions = None  # type: ignore[assignment]
    bigquery = None  # type: ignore[assignment]
    _BIGQUERY_OK = False

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "")

_ROOT_DIR = Path(__file__).resolve().parents[2]
_LOG_DIR  = _ROOT_DIR / "logs"
_LOG_FILE = _LOG_DIR / "bigquery_consumption.jsonl"

# Teto de cobrança por query (hard stop do BigQuery): 10 GB
_MAX_BYTES_BILLED = 10 * 1024**3

# Limite mensal de referência do free tier (1 TB) — alerta em 80% (800 GB)
_ALERTA_GB_MENSAL = 800.0


# ---------------------------------------------------------------------------
# Log de consumo
# ---------------------------------------------------------------------------

def log_consumption(bytes_processed: int, bytes_billed: int) -> None:
    """Registra o consumo de uma query BigQuery em logs/bigquery_consumption.jsonl.

    Converte bytes para GB, imprime o resumo, persiste em JSONL (append) e
    alerta quando o valor cobrado ultrapassa 80% do limite mensal gratuito.

    Args:
        bytes_processed: Bytes estimados/processados pela query
            (job.total_bytes_processed).
        bytes_billed: Bytes efetivamente cobrados
            (job.total_bytes_billed; 0 em dry run).

    Returns:
        None.

    Raises:
        OSError: Se não for possível criar a pasta logs/ ou escrever o arquivo.
    """
    gb_processados = (bytes_processed or 0) / 1024**3
    gb_cobrados    = (bytes_billed or 0) / 1024**3

    logger.info(f"Estimativa: {gb_processados:.2f} GB | Cobrado: {gb_cobrados:.2f} GB")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    registro = {
        "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gb_processados": round(gb_processados, 4),
        "gb_cobrados":    round(gb_cobrados, 4),
    }
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(registro, ensure_ascii=False) + "\n")

    if gb_cobrados > _ALERTA_GB_MENSAL:
        logger.warning("WARNING: CONSUMO ACIMA DE 80% DO LIMITE MENSAL")


# ---------------------------------------------------------------------------
# Carga INMET histórico (basedosdados)
# ---------------------------------------------------------------------------

def load_inmet_rs(start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    """Carrega microdados INMET do RS via BigQuery público basedosdados.

    Executa DRY RUN para estimar o volume antes da query real (que roda com
    teto de 10 GB cobrados e cache habilitado). Todo consumo é registrado em
    logs/bigquery_consumption.jsonl.

    Args:
        start_year: Ano inicial (inclusive) do filtro EXTRACT(YEAR FROM data).
        end_year: Ano final (inclusive).

    Returns:
        DataFrame com colunas data, id_estacao, precip_mm, temp_c,
        umidade_pct — ordenado por data e estação.

    Raises:
        SystemExit: Se BIGQUERY_PROJECT_ID não estiver configurado, se a API
            estiver desabilitada (Forbidden) ou em erro inesperado de execução.
    """
    if not _BIGQUERY_OK:
        logger.error(
            "google-cloud-bigquery não instalado — caminho de treino. "
            "Use: pip install -r requirements-ml.txt"
        )
        sys.exit(1)
    if not BIGQUERY_PROJECT_ID:
        logger.error(
            "BIGQUERY_PROJECT_ID ausente no .env — adicione "
            "BIGQUERY_PROJECT_ID=clima-rs-lstm"
        )
        sys.exit(1)

    # Schema real do basedosdados (verificado em 12/06/2026):
    # - microdados NÃO tem sigla_uf nem temperatura_media/umidade_relativa_media;
    #   as colunas horárias são temperatura_bulbo_hora e umidade_rel_hora.
    # - O filtro RS vem do JOIN com a tabela estacao: id_municipio (IBGE)
    #   começa com '43' para municípios gaúchos.
    # - Filtro por coluna `ano` (não EXTRACT) — reduz bytes varridos.
    # - `hora` incluída: a série precisa ser horária para o LSTM.
    sql = f"""
        SELECT m.data, m.hora, m.id_estacao,
               m.precipitacao_total,
               m.temperatura_bulbo_hora,
               m.umidade_rel_hora,
               m.radiacao_global
        FROM `basedosdados.br_inmet_bdmep.microdados` AS m
        JOIN `basedosdados.br_inmet_bdmep.estacao` AS e
          USING (id_estacao)
        WHERE STARTS_WITH(e.id_municipio, '43')
          AND m.ano BETWEEN {int(start_year)} AND {int(end_year)}
        ORDER BY m.data, m.hora, m.id_estacao
    """

    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT_ID)

        # ── 1. DRY RUN — estima bytes sem custo ──────────────────────────
        dry_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_job = client.query(sql, job_config=dry_config)
        log_consumption(dry_job.total_bytes_processed, 0)
        logger.info("Estimativa calculada. Prosseguindo com query real...")

        # ── 2. Query real — teto de 10 GB cobrados + cache ───────────────
        real_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=_MAX_BYTES_BILLED,
            use_query_cache=True,
        )
        job = client.query(sql, job_config=real_config)
        df: pd.DataFrame = job.to_dataframe()
        log_consumption(job.total_bytes_processed or 0, job.total_bytes_billed or 0)

    except gcp_exceptions.Forbidden as exc:
        logger.error(
            "Acesso negado ao BigQuery (403). Habilite a BigQuery API no "
            "projeto 'clima-rs-lstm': console.cloud.google.com → APIs & "
            "Services → Enable APIs → BigQuery API. Confirme também o "
            "billing/quota do projeto e a autenticação "
            f"(gcloud auth application-default login). Detalhe: {exc}"
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — contrato do CLI: reportar e encerrar
        logger.error(f"Erro inesperado na carga BigQuery: {exc}")
        sys.exit(1)

    # ── 3. Padronização de colunas para o pipeline LSTM ──────────────────
    df = df.rename(columns={
        "precipitacao_total":     "precip_mm",
        "temperatura_bulbo_hora": "temp_c",
        "umidade_rel_hora":       "umidade_pct",
        "radiacao_global":        "radiacao_kjm2",
    })

    logger.success(
        f"INMET RS {start_year}-{end_year}: {len(df):,} linhas | "
        f"{df['id_estacao'].nunique() if 'id_estacao' in df.columns else 0} estações"
    )
    return df


# ---------------------------------------------------------------------------
# Standalone — teste com 1 ano (volume mínimo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="DEBUG",
    )

    df = load_inmet_rs(start_year=2023, end_year=2023)
    print(f"\nshape: {df.shape}")
    print(df.head().to_string())
    print("\nTeste concluído com sucesso!")
