"""
Agente 1 — Arquiteto de Dados
Processador de acumulados de chuva e status de rios via SQL puro no DuckDB.

Fluxo de execução:
  1. Se rain_readings / river_levels estiverem vazios, ingere river_levels.parquet.
  2. Calcula acumulados 1h/3h/6h/12h/24h/48h/72h/7d com CTEs + FILTER aggregates.
  3. Calcula river_status (% da cota, tendência, classificação) em SQL puro.
  4. Persiste rain_accumulated e river_status no DuckDB.
  5. Exporta data/processed/accumulated_rain.parquet e river_status.parquet.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

from database.db_manager import ClimateDB  # noqa: E402

# ---------------------------------------------------------------------------
# Caminhos padrão
# ---------------------------------------------------------------------------

_CONFIG_PATH   = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
_PROC_DIR      = Path(__file__).resolve().parents[2] / "data" / "processed"
_RIVER_PARQUET = _RAW_DIR  / "river_levels.parquet"
_OUT_ACCUM     = _PROC_DIR / "accumulated_rain.parquet"
_OUT_STATUS    = _PROC_DIR / "river_status.parquet"

# ---------------------------------------------------------------------------
# SQL — Migração de schema (idempotente)
# ---------------------------------------------------------------------------

_SQL_ADD_RAIN_7D = """
    ALTER TABLE rain_accumulated
    ADD COLUMN IF NOT EXISTS rain_7d DOUBLE DEFAULT 0
"""

# ---------------------------------------------------------------------------
# SQL — Acumulados de chuva
#
# Usa MAX(ts) por estação como ancora temporal (não `now()`) para funcionar
# corretamente quando os dados são históricos (ANA DIAS_7 etc.).
# ---------------------------------------------------------------------------

_SQL_RAIN_ACCUM = """
INSERT OR REPLACE INTO rain_accumulated
    (station_id, date,
     rain_1h, rain_3h, rain_6h, rain_12h,
     rain_24h, rain_48h, rain_72h, rain_7d,
     updated_at)
WITH latest AS (
    SELECT station_id, MAX(ts) AS max_ts
    FROM   rain_readings
    GROUP  BY station_id
)
SELECT
    r.station_id,
    CAST(sl.max_ts AS DATE)                                                             AS date,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '1 hour'),   0) AS rain_1h,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '3 hours'),  0) AS rain_3h,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '6 hours'),  0) AS rain_6h,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '12 hours'), 0) AS rain_12h,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '24 hours'), 0) AS rain_24h,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '48 hours'), 0) AS rain_48h,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '72 hours'), 0) AS rain_72h,
    COALESCE(SUM(r.rain_1h_mm) FILTER (WHERE r.ts >= sl.max_ts - INTERVAL '7 days'),   0) AS rain_7d,
    now()                                                                                AS updated_at
FROM   rain_readings r
JOIN   latest        sl ON r.station_id = sl.station_id
WHERE  r.ts >= sl.max_ts - INTERVAL '7 days'
  AND  r.rain_1h_mm IS NOT NULL
GROUP  BY r.station_id, sl.max_ts
"""

# ---------------------------------------------------------------------------
# SQL — Status dos rios
#
# {thresholds_values} é substituído por uma cláusula VALUES gerada em Python
# com os limiares do config.yaml.
# ---------------------------------------------------------------------------

_SQL_RIVER_STATUS_TEMPLATE = """
WITH thresholds (river, cota_atencao_m, cota_alerta_m, cota_emergencia_m) AS (
    VALUES {thresholds_values}
),
latest_levels AS (
    SELECT
        river,
        station_id,
        ARGMAX(level_m,  ts) AS level_m,
        ARGMAX(flow_cms, ts) AS flow_cms,
        MAX(ts)              AS ts
    FROM   river_levels
    WHERE  level_m IS NOT NULL
    GROUP  BY river, station_id
),
levels_3h_ago AS (
    SELECT
        rl.river,
        rl.station_id,
        ARGMAX(rl.level_m, rl.ts) AS level_3h
    FROM   river_levels rl
    JOIN   latest_levels ll
           ON rl.river = ll.river AND rl.station_id = ll.station_id
    WHERE  rl.ts BETWEEN ll.ts - INTERVAL '4 hours'
                     AND ll.ts - INTERVAL '2 hours'
      AND  rl.level_m IS NOT NULL
    GROUP  BY rl.river, rl.station_id
)
SELECT
    l.river,
    l.station_id                                              AS segment,
    l.ts,
    ROUND(l.level_m, 3)                                      AS level_m,
    CASE
        WHEN l.level_m >= t.cota_emergencia_m THEN 'EMERGENCIA'
        WHEN l.level_m >= t.cota_alerta_m     THEN 'ALERTA'
        WHEN l.level_m >= t.cota_atencao_m    THEN 'ATENCAO'
        ELSE                                       'NORMAL'
    END                                                       AS status,
    CASE
        WHEN l.level_m > COALESCE(la.level_3h, l.level_m) + 0.05 THEN 'SUBINDO'
        WHEN l.level_m < COALESCE(la.level_3h, l.level_m) - 0.05 THEN 'DESCENDO'
        ELSE                                                            'ESTAVEL'
    END                                                       AS trend,
    ROUND(l.level_m / NULLIF(t.cota_alerta_m, 0) * 100, 1)  AS pct_cota_alerta,
    l.flow_cms
FROM   latest_levels  l
LEFT   JOIN thresholds   t  ON l.river = t.river
LEFT   JOIN levels_3h_ago la ON l.river = la.river AND l.station_id = la.station_id
WHERE  l.level_m IS NOT NULL
ORDER  BY l.river, l.station_id
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Carrega config.yaml com limiares dos rios e alertas.

    Returns:
        Dicionário com as seções do config.yaml.

    Raises:
        FileNotFoundError: Se config.yaml não for encontrado.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml não encontrado: {_CONFIG_PATH}")
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_threshold_values(thresholds: dict[str, dict[str, float]]) -> str:
    """Gera a cláusula VALUES para o CTE de limiares no SQL.

    Args:
        thresholds: Dict {nome_rio: {cota_atencao_m, cota_alerta_m, cota_emergencia_m}}.

    Returns:
        String com tuplas SQL prontas para substituição em {thresholds_values}.

    Example::

        "('Sinos', 3.5, 4.5, 5.5), ('Taquari', 3.5, 5.0, 7.0)"
    """
    rows = []
    for river, cfg in thresholds.items():
        atencao   = float(cfg.get("cota_atencao_m",    9999.0))
        alerta    = float(cfg.get("cota_alerta_m",     9999.0))
        emergencia = float(cfg.get("cota_emergencia_m", 9999.0))
        rows.append(f"('{river}', {atencao}, {alerta}, {emergencia})")
    return ", ".join(rows)


def _table_row_count(db: ClimateDB, table: str) -> int:
    """Retorna o número de linhas de uma tabela DuckDB.

    Args:
        db: Instância ClimateDB com conexão ativa.
        table: Nome da tabela.

    Returns:
        Contagem de linhas (int >= 0).
    """
    row = db._con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0] if row else 0


def _parquet_path_sql(path: Path) -> str:
    """Converte Path para string com barras normalizadas para uso em SQL DuckDB.

    Args:
        path: Caminho do arquivo Parquet.

    Returns:
        String com separadores '/' (compatível com DuckDB read_parquet).
    """
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Ingestão de Parquet → DuckDB
# ---------------------------------------------------------------------------

def ingest_parquet(
    db: ClimateDB,
    parquet_path: Path = _RIVER_PARQUET,
) -> dict[str, int]:
    """Ingere river_levels.parquet no DuckDB (stations → rain_readings → river_levels).

    Usa INSERT OR IGNORE para idempotência — execuções repetidas são seguras.
    Insere stations primeiro para satisfazer a FK de rain_readings.

    Args:
        db: Instância ClimateDB com conexão ativa.
        parquet_path: Caminho para river_levels.parquet.

    Returns:
        Dict com chaves stations, rain_readings, river_levels contendo
        o número de linhas efetivamente inseridas.

    Raises:
        FileNotFoundError: Se o arquivo Parquet não existir.
        duckdb.Error: Se alguma instrução SQL falhar.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {parquet_path}")

    pq = _parquet_path_sql(parquet_path)
    logger.info(f"Ingerindo {parquet_path.name} → DuckDB...")

    # 1 — Stations (stubs sem coordenadas — serão enriquecidos depois)
    db._con.execute(f"""
        INSERT OR IGNORE INTO stations
            (station_id, name, source, lat, lon, river)
        SELECT DISTINCT
            CAST(station_code AS VARCHAR),
            CAST(station_code AS VARCHAR),
            'ANA',
            0.0, 0.0,
            rio_nome
        FROM read_parquet('{pq}')
        WHERE station_code IS NOT NULL
    """)
    db._con.commit()
    n_stations = _table_row_count(db, "stations")

    # 2 — Rain readings (linhas com chuva_mm não nulo)
    before_rain = _table_row_count(db, "rain_readings")
    db._con.execute(f"""
        INSERT OR IGNORE INTO rain_readings
            (id, station_id, ts, rain_1h_mm, source)
        SELECT
            nextval('seq_rain_readings'),
            CAST(station_code AS VARCHAR),
            timestamp,
            chuva_mm,
            'ANA'
        FROM read_parquet('{pq}')
        WHERE chuva_mm  IS NOT NULL
          AND timestamp IS NOT NULL
          AND station_code IS NOT NULL
    """)
    db._con.commit()
    n_rain = _table_row_count(db, "rain_readings") - before_rain

    # 3 — River levels (linhas com nivel_m não nulo)
    before_levels = _table_row_count(db, "river_levels")
    db._con.execute(f"""
        INSERT OR IGNORE INTO river_levels
            (id, river, station_id, ts, level_m, flow_cms, source)
        SELECT
            nextval('seq_river_levels'),
            rio_nome,
            CAST(station_code AS VARCHAR),
            timestamp,
            nivel_m,
            vazao_m3s,
            'ANA'
        FROM read_parquet('{pq}')
        WHERE nivel_m   IS NOT NULL
          AND timestamp IS NOT NULL
          AND station_code IS NOT NULL
    """)
    db._con.commit()
    n_levels = _table_row_count(db, "river_levels") - before_levels

    logger.success(
        f"Ingestão concluída — stations: {n_stations}, "
        f"rain_readings: +{n_rain}, river_levels: +{n_levels}"
    )
    return {"stations": n_stations, "rain_readings": n_rain, "river_levels": n_levels}


# ---------------------------------------------------------------------------
# Acumulados de chuva
# ---------------------------------------------------------------------------

def compute_rain_accumulated(db: ClimateDB) -> int:
    """Calcula acumulados 1h/3h/6h/12h/24h/48h/72h/7d por estação via SQL puro.

    Usa CTEs com FILTER aggregates ancorados no MAX(ts) de cada estação.
    Resultados são escritos diretamente em rain_accumulated (INSERT OR REPLACE).

    Args:
        db: Instância ClimateDB com conexão ativa e rain_readings populada.

    Returns:
        Número de linhas gravadas em rain_accumulated.

    Raises:
        duckdb.Error: Se a query SQL falhar.
    """
    n_source = _table_row_count(db, "rain_readings")
    if n_source == 0:
        logger.warning("rain_readings está vazio — nada a acumular.")
        return 0

    # Garante que a coluna rain_7d existe (schema migration idempotente)
    db._con.execute(_SQL_ADD_RAIN_7D)
    db._con.commit()

    db._con.execute(_SQL_RAIN_ACCUM)
    db._con.commit()

    n_result = _table_row_count(db, "rain_accumulated")
    logger.success(
        f"rain_accumulated: {n_result} registros calculados "
        f"a partir de {n_source} leituras."
    )
    return n_result


# ---------------------------------------------------------------------------
# Status dos rios
# ---------------------------------------------------------------------------

def compute_river_status(
    db: ClimateDB,
    thresholds: dict[str, dict[str, float]] | None = None,
    municipalities: dict[str, list[str]] | None = None,
) -> int:
    """Calcula status operacional dos rios via SQL e persiste em river_status.

    Classifica cada estação em NORMAL/ATENCAO/ALERTA/EMERGENCIA conforme as
    cotas do config.yaml. Calcula tendência (SUBINDO/ESTAVEL/DESCENDO) com
    base na variação de nível nas últimas 3 horas.

    Args:
        db: Instância ClimateDB com conexão ativa e river_levels populada.
        thresholds: Dict {rio: {cota_atencao_m, cota_alerta_m, cota_emergencia_m}}.
                    Se None, carrega do config.yaml.
        municipalities: Dict {rio: [municipios_em_risco]}.
                        Se None, carrega do config.yaml.

    Returns:
        Número de linhas gravadas em river_status.

    Raises:
        FileNotFoundError: Se config.yaml não for encontrado e thresholds for None.
        duckdb.Error: Se a query SQL falhar.
    """
    n_source = _table_row_count(db, "river_levels")
    if n_source == 0:
        logger.warning("river_levels está vazio — nada a processar.")
        return 0

    if thresholds is None or municipalities is None:
        cfg = _load_config()
        thresholds    = thresholds    or cfg.get("rios", {})
        municipalities = municipalities or {
            rio: v.get("municipios_risco", [])
            for rio, v in cfg.get("rios", {}).items()
        }

    # Monta o VALUES clause com os limiares de todos os rios
    threshold_values = _build_threshold_values(thresholds)
    sql = _SQL_RIVER_STATUS_TEMPLATE.replace("{thresholds_values}", threshold_values)

    df: pd.DataFrame = db._con.execute(sql).df()

    if df.empty:
        logger.warning("Nenhum nível de rio encontrado no DuckDB.")
        return 0

    # Enriquece com municípios em risco conforme status
    df["municipalities_risk"] = df.apply(
        lambda row: municipalities.get(row["river"], [])
        if row["status"] in ("ATENCAO", "ALERTA", "EMERGENCIA")
        else [],
        axis=1,
    )
    df["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    # Persiste via INSERT OR REPLACE (PK: river + segment)
    db._con.register("_river_status_temp", df)
    db._con.execute("""
        INSERT OR REPLACE INTO river_status
            (river, segment, ts, level_m, status, trend,
             municipalities_risk, updated_at)
        SELECT
            river, segment, ts, level_m, status, trend,
            municipalities_risk,
            updated_at
        FROM _river_status_temp
    """)
    db._con.commit()
    db._con.unregister("_river_status_temp")

    n_result = _table_row_count(db, "river_status")

    # Log resumo por rio
    for _, row in df.iterrows():
        pct = row.get("pct_cota_alerta", 0.0) or 0.0
        logger.info(
            f"  [{row['status']:10s}] {row['river']:12s} "
            f"[{row['segment']}]: {row['level_m']:.2f}m "
            f"({pct:.1f}% cota) — {row['trend']}"
        )

    logger.success(f"river_status: {n_result} segmentos atualizados.")
    return n_result


# ---------------------------------------------------------------------------
# Exportação para Parquet
# ---------------------------------------------------------------------------

def export_accumulated_parquet(
    db: ClimateDB,
    output_path: Path = _OUT_ACCUM,
) -> int:
    """Exporta rain_accumulated do DuckDB para Parquet.

    Args:
        db: Instância ClimateDB com conexão ativa.
        output_path: Destino do arquivo Parquet.

    Returns:
        Número de linhas exportadas.

    Raises:
        duckdb.Error: Se a query falhar.
        OSError: Se não for possível criar o diretório de saída.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = db._con.execute("""
        SELECT ra.station_id, ra.date,
               ra.rain_1h, ra.rain_3h, ra.rain_6h, ra.rain_12h,
               ra.rain_24h, ra.rain_48h, ra.rain_72h, ra.rain_7d,
               ra.updated_at,
               s.lat, s.lon,
               s.name AS station_name,
               s.municipality,
               s.river
        FROM   rain_accumulated ra
        LEFT JOIN stations s ON s.station_id = ra.station_id
        ORDER  BY ra.station_id, ra.date
    """).df()
    # lat=0.0 são stubs ANA sem coordenada real — substituir por NaN
    df.loc[df["lat"] == 0.0, ["lat", "lon"]] = None
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"accumulated_rain.parquet exportado: {output_path} ({len(df)} linhas)")
    return len(df)


def export_river_status_parquet(
    db: ClimateDB,
    output_path: Path = _OUT_STATUS,
) -> int:
    """Exporta river_status do DuckDB para Parquet.

    Args:
        db: Instância ClimateDB com conexão ativa.
        output_path: Destino do arquivo Parquet.

    Returns:
        Número de linhas exportadas.

    Raises:
        duckdb.Error: Se a query falhar.
        OSError: Se não for possível criar o diretório de saída.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = db._con.execute(
        "SELECT * FROM river_status ORDER BY river, segment"
    ).df()
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"river_status.parquet exportado: {output_path} ({len(df)} linhas)")
    return len(df)


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def run_accumulator(
    db: ClimateDB | None = None,
    parquet_path: Path = _RIVER_PARQUET,
    force_ingest: bool = False,
) -> dict[str, Any]:
    """Executa o pipeline completo de acumulados e status dos rios.

    Fluxo:
    1. Verifica se DuckDB tem dados; se não, ingere do Parquet.
    2. Calcula acumulados de chuva via SQL puro.
    3. Calcula status dos rios via SQL puro.
    4. Exporta resultados para Parquet.

    Args:
        db: Instância ClimateDB. Se None, abre conexão temporária.
        parquet_path: Caminho do river_levels.parquet para ingestão.
        force_ingest: Se True, reingere mesmo com tabelas populadas.

    Returns:
        Dict com chaves: ingested (bool), rain_rows (int), status_rows (int),
        accum_parquet (str), status_parquet (str), duration_s (float).

    Raises:
        FileNotFoundError: Se parquet_path não existir e tabelas estiverem vazias.
    """
    t_start = datetime.now(timezone.utc)
    logger.info("=== RainAccumulator — iniciando pipeline ===")

    _owns_db = db is None
    if _owns_db:
        db = ClimateDB()

    try:
        ingested = False
        rain_empty   = _table_row_count(db, "rain_readings") == 0
        levels_empty = _table_row_count(db, "river_levels")  == 0

        if force_ingest or rain_empty or levels_empty:
            if parquet_path.exists():
                ingest_parquet(db, parquet_path)
                ingested = True
            else:
                logger.warning(
                    f"Parquet não encontrado ({parquet_path}) e tabelas vazias. "
                    "Acumulados não serão calculados."
                )

        rain_rows   = compute_rain_accumulated(db)
        status_rows = compute_river_status(db)

        accum_rows  = export_accumulated_parquet(db)
        status_exp  = export_river_status_parquet(db)

    finally:
        if _owns_db:
            db.close()

    duration = (datetime.now(timezone.utc) - t_start).total_seconds()
    logger.info(f"=== Pipeline concluído em {duration:.1f}s ===")

    return {
        "ingested":      ingested,
        "rain_rows":     rain_rows,
        "status_rows":   status_rows,
        "accum_rows":    accum_rows,
        "status_exported": status_exp,
        "accum_parquet": str(_OUT_ACCUM),
        "status_parquet": str(_OUT_STATUS),
        "duration_s":    round(duration, 2),
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

    parser = argparse.ArgumentParser(description="Processador de acumulados — Monitor RS")
    parser.add_argument(
        "--parquet", type=Path, default=_RIVER_PARQUET,
        help="Caminho do river_levels.parquet (padrão: data/raw/river_levels.parquet)",
    )
    parser.add_argument(
        "--force-ingest", action="store_true",
        help="Reingere Parquet mesmo com tabelas DuckDB populadas",
    )
    args = parser.parse_args()

    result = run_accumulator(
        parquet_path=args.parquet,
        force_ingest=args.force_ingest,
    )

    print("\n--- Resultado -----------------------------------")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("-------------------------------------------------")
