"""
Agente 1 — Arquiteto de Dados
Processador de acumulados de chuva — arquitetura híbrida v4 (sem DuckDB).

Fluxo de execução:
  1. Baixa do R2 o Parquet mais recente de live_rain_readings
     (série de ~30 dias gravada pelo inmet_collector a cada run).
  2. Calcula acumulados 1h/3h/6h/12h/24h/48h/72h/7d por estação com pandas,
     ancorados no MAX(ts) de cada estação (não em now() — robusto a lag).
  3. UPSERT dos campos precip_1h/precip_6h/precip_24h na tabela
     live_rain_readings do Supabase.
  4. Upload do DataFrame completo de acumulados para o R2
     (historico/rain_accumulated/…) e export local opcional para o
     dashboard legado.

O status dos rios NÃO é calculado aqui na v4 — o ana_collector já grava
status/percentual em live_river_levels via HybridWriter.write_river_status.
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

from database.hybrid_writer import (  # noqa: E402
    WriteResult,
    _pg_connect,
    _r2_client,
    _r2_upload,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_PROC_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed"
_OUT_ACCUM = _PROC_DIR / "accumulated_rain.parquet"

_R2_RAIN_PREFIX = "historico/live_rain_readings"

# Janelas de acumulação (nome da coluna → timedelta)
_WINDOWS: dict[str, timedelta] = {
    "rain_1h":  timedelta(hours=1),
    "rain_3h":  timedelta(hours=3),
    "rain_6h":  timedelta(hours=6),
    "rain_12h": timedelta(hours=12),
    "rain_24h": timedelta(hours=24),
    "rain_48h": timedelta(hours=48),
    "rain_72h": timedelta(hours=72),
    "rain_7d":  timedelta(days=7),
}


# ---------------------------------------------------------------------------
# Leitura da série de chuva no R2
# ---------------------------------------------------------------------------

def _latest_rain_key(s3: Any, bucket: str, days_back: int = 3) -> Optional[str]:
    """Encontra a chave do Parquet de chuva mais recente no R2.

    Varre as partições Hive de hoje para trás (até ``days_back`` dias)
    e retorna a chave com o LastModified mais novo — evita listar o
    bucket inteiro.

    Args:
        s3: Cliente boto3 configurado para o R2.
        bucket: Nome do bucket R2.
        days_back: Quantos dias retroceder na busca por partições.

    Returns:
        Chave S3 do Parquet mais recente, ou None se nada encontrado.
    """
    now = datetime.now(timezone.utc)
    candidatos: list[dict[str, Any]] = []
    for d in range(days_back + 1):
        dia = now - timedelta(days=d)
        prefix = (
            f"{_R2_RAIN_PREFIX}/ano={dia.year}"
            f"/mes={dia.month:02d}/dia={dia.day:02d}/"
        )
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        candidatos.extend(resp.get("Contents", []))
        if candidatos:
            break
    if not candidatos:
        return None
    return max(candidatos, key=lambda o: o["LastModified"])["Key"]


def load_rain_series_from_r2() -> pd.DataFrame:
    """Baixa do R2 a série de leituras de chuva mais recente.

    Returns:
        DataFrame com colunas station_id, ts (tz-aware UTC) e rain_1h_mm.
        Vazio se o R2 estiver indisponível ou sem Parquets recentes.
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        logger.warning("R2 indisponível — sem fonte de série de chuva.")
        return pd.DataFrame()

    key = _latest_rain_key(s3, bucket)
    if key is None:
        logger.warning("Nenhum Parquet de chuva encontrado no R2 (últimos 3 dias).")
        return pd.DataFrame()

    logger.info(f"Série de chuva: r2://{bucket}/{key}")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    df = pd.read_parquet(io.BytesIO(body))

    if df.empty or "station_id" not in df.columns:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["rain_1h_mm"] = pd.to_numeric(df.get("rain_1h_mm"), errors="coerce")
    df = df.dropna(subset=["station_id", "ts"])
    logger.info(
        f"  {len(df):,} leituras | {df['station_id'].nunique()} estações "
        f"| {df['ts'].min():%d/%m %H:%M} → {df['ts'].max():%d/%m %H:%M} UTC"
    )
    return df


# ---------------------------------------------------------------------------
# Cálculo dos acumulados
# ---------------------------------------------------------------------------

def compute_rain_accumulated(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula acumulados por estação ancorados no MAX(ts) de cada uma.

    Args:
        df: DataFrame com station_id, ts (UTC) e rain_1h_mm.

    Returns:
        DataFrame com 1 linha por estação: station_id, date (do MAX ts),
        rain_1h…rain_7d e updated_at. Vazio se a entrada for vazia.
    """
    if df.empty:
        logger.warning("Série de chuva vazia — nada a acumular.")
        return pd.DataFrame()

    chuva = df.dropna(subset=["rain_1h_mm"])
    linhas: list[dict[str, Any]] = []
    agora = datetime.now(timezone.utc)

    for sid, grupo in chuva.groupby("station_id"):
        max_ts = grupo["ts"].max()
        row: dict[str, Any] = {
            "station_id": str(sid),
            "date":       max_ts.date(),
            "updated_at": agora,
        }
        for col, janela in _WINDOWS.items():
            row[col] = float(
                grupo.loc[grupo["ts"] >= max_ts - janela, "rain_1h_mm"].sum()
            )
        linhas.append(row)

    out = pd.DataFrame(linhas)
    logger.success(f"rain_accumulated: {len(out)} estações calculadas.")
    return out


# ---------------------------------------------------------------------------
# Persistência — Supabase (UPSERT) + R2 + Parquet local
# ---------------------------------------------------------------------------

def upsert_accumulated_supabase(df_acc: pd.DataFrame) -> int:
    """Grava as 8 janelas (1h/3h/6h/12h/24h/48h/72h/7d) em live_rain_readings.

    UPDATE puro (não INSERT…ON CONFLICT): a linha da estação já existe
    com "timestamp" NOT NULL gravado pelo inmet_collector — acumulado só
    faz sentido para estação existente, e o caminho INSERT violaria a
    constraint de timestamp nulo. As colunas precip_3h/12h/48h/72h/7d são
    adicionadas pelo schema (ADD COLUMN IF NOT EXISTS).

    Args:
        df_acc: Saída de compute_rain_accumulated().

    Returns:
        Número de estações efetivamente atualizadas no Supabase.
    """
    if df_acc.empty:
        return 0

    conn = _pg_connect()
    if conn is None:
        logger.warning("Supabase indisponível — acumulados não gravados no PG.")
        return 0

    import psycopg2
    import psycopg2.extras

    rows = [
        (
            str(r["station_id"]),
            float(r["rain_1h"]), float(r["rain_3h"]), float(r["rain_6h"]),
            float(r["rain_12h"]), float(r["rain_24h"]), float(r["rain_48h"]),
            float(r["rain_72h"]), float(r["rain_7d"]),
        )
        for _, r in df_acc.iterrows()
    ]
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, """
                UPDATE live_rain_readings AS l
                SET precip_1h  = v.p1,
                    precip_3h  = v.p3,
                    precip_6h  = v.p6,
                    precip_12h = v.p12,
                    precip_24h = v.p24,
                    precip_48h = v.p48,
                    precip_72h = v.p72,
                    precip_7d  = v.p7d,
                    updated_at = NOW()
                FROM (VALUES %s) AS v(station_id, p1, p3, p6, p12, p24, p48, p72, p7d)
                WHERE l.station_id = v.station_id
            """, rows, page_size=200)
            atualizadas = cur.rowcount
        conn.commit()
        logger.success(
            f"  PG live_rain_readings: 8 janelas de {atualizadas} estações."
        )
        return int(atualizadas)
    except psycopg2.Error as exc:
        logger.warning(f"  PG acumulados: {exc}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def upload_accumulated_r2(df_acc: pd.DataFrame) -> Optional[str]:
    """Envia o DataFrame de acumulados completo (8 janelas) ao R2.

    Args:
        df_acc: Saída de compute_rain_accumulated().

    Returns:
        Chave S3 gravada, ou None se o upload não ocorreu.
    """
    if df_acc.empty:
        return None
    s3 = _r2_client()
    if s3 is None:
        return None
    result = WriteResult(table="rain_accumulated")
    _r2_upload(df_acc, "rain_accumulated", result, s3)
    return result.r2_key


def export_accumulated_local(
    df_acc: pd.DataFrame,
    output_path: Path = _OUT_ACCUM,
) -> int:
    """Exporta acumulados para Parquet local (dashboard legado).

    Best-effort: falha de escrita local não interrompe o pipeline.

    Args:
        df_acc: Saída de compute_rain_accumulated().
        output_path: Destino do arquivo Parquet.

    Returns:
        Número de linhas exportadas (0 se vazio ou em falha de I/O).
    """
    if df_acc.empty:
        return 0
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_acc.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"accumulated_rain.parquet exportado: {output_path} ({len(df_acc)} linhas)")
        return len(df_acc)
    except OSError as exc:
        logger.warning(f"Export local falhou (ignorado): {exc}")
        return 0


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def run_accumulator() -> dict[str, Any]:
    """Executa o pipeline completo de acumulados de chuva (v4 híbrido).

    Fluxo: R2 (série 30d) → pandas (8 janelas) → Supabase (1h/6h/24h)
    + R2 (acumulados completos) + Parquet local (dashboard legado).

    Returns:
        Dict com chaves: source_rows (int), stations (int), pg_rows (int),
        r2_key (str | None), local_rows (int), duration_s (float).
    """
    t_start = datetime.now(timezone.utc)
    logger.info("=== RainAccumulator v4 — R2 → Supabase ===")

    serie  = load_rain_series_from_r2()
    df_acc = compute_rain_accumulated(serie)

    pg_rows    = upsert_accumulated_supabase(df_acc)
    r2_key     = upload_accumulated_r2(df_acc)
    local_rows = export_accumulated_local(df_acc)

    duration = (datetime.now(timezone.utc) - t_start).total_seconds()
    logger.info(f"=== Pipeline concluído em {duration:.1f}s ===")

    return {
        "source_rows": int(len(serie)),
        "stations":    int(len(df_acc)),
        "pg_rows":     pg_rows,
        "r2_key":      r2_key,
        "local_rows":  local_rows,
        "duration_s":  round(duration, 2),
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

    result = run_accumulator()

    print("\n--- Resultado -----------------------------------")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("-------------------------------------------------")
