"""
Agente 1 — Arquiteto de Dados
Gerenciador central do banco DuckDB para o Monitor Hidrometeorológico RS.
11 tabelas cobrindo estações, chuva, previsões, alertas, rios, qualidade
de água, status operacional, previsões AI e resultados WRF-Hydro.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import duckdb
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Schema SQL — 11 tabelas
# ---------------------------------------------------------------------------

_DDL_STATEMENTS: list[str] = [
    # 1 — Estações de monitoramento (ANA HidroWeb + INMET)
    """
    CREATE TABLE IF NOT EXISTS stations (
        station_id   VARCHAR PRIMARY KEY,
        name         VARCHAR NOT NULL,
        source       VARCHAR NOT NULL,           -- ANA | INMET | REDEMET
        lat          DOUBLE NOT NULL,
        lon          DOUBLE NOT NULL,
        river        VARCHAR,
        municipality VARCHAR,
        state        VARCHAR DEFAULT 'RS',
        elevation_m  DOUBLE,
        active       BOOLEAN DEFAULT TRUE,
        created_at   TIMESTAMPTZ DEFAULT now()
    )
    """,

    # 2 — Leituras brutas de chuva e variáveis meteorológicas
    """
    CREATE TABLE IF NOT EXISTS rain_readings (
        id           BIGINT PRIMARY KEY,
        station_id   VARCHAR NOT NULL REFERENCES stations(station_id),
        ts           TIMESTAMPTZ NOT NULL,
        rain_1h_mm   DOUBLE,
        rain_3h_mm   DOUBLE,
        rain_6h_mm   DOUBLE,
        rain_12h_mm  DOUBLE,
        rain_24h_mm  DOUBLE,
        temperature  DOUBLE,
        humidity     DOUBLE,
        pressure_hpa DOUBLE,
        wind_speed   DOUBLE,
        wind_dir     DOUBLE,
        source       VARCHAR,
        created_at   TIMESTAMPTZ DEFAULT now(),
        UNIQUE (station_id, ts)
    )
    """,

    # 3 — Acumulados pré-calculados por estação/dia (evita GROUP BY custoso no dashboard)
    """
    CREATE TABLE IF NOT EXISTS rain_accumulated (
        station_id VARCHAR NOT NULL REFERENCES stations(station_id),
        date       DATE NOT NULL,
        rain_1h    DOUBLE DEFAULT 0,
        rain_3h    DOUBLE DEFAULT 0,
        rain_6h    DOUBLE DEFAULT 0,
        rain_12h   DOUBLE DEFAULT 0,
        rain_24h   DOUBLE DEFAULT 0,
        rain_48h   DOUBLE DEFAULT 0,
        rain_72h   DOUBLE DEFAULT 0,
        updated_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (station_id, date)
    )
    """,

    # 4 — Previsões NWP (Open-Meteo / NOAA GFS / ECMWF)
    """
    CREATE TABLE IF NOT EXISTS forecasts (
        id               BIGINT PRIMARY KEY,
        location_name    VARCHAR NOT NULL,
        lat              DOUBLE NOT NULL,
        lon              DOUBLE NOT NULL,
        forecast_ts      TIMESTAMPTZ NOT NULL,   -- quando a previsão foi gerada
        valid_ts         TIMESTAMPTZ NOT NULL,   -- para quando é válida
        rain_mm          DOUBLE,
        temperature      DOUBLE,
        wind_speed       DOUBLE,
        cape_j_kg        DOUBLE,
        lifted_index     DOUBLE,
        k_index          DOUBLE,
        model_source     VARCHAR NOT NULL,       -- openmeteo | noaa | ecmwf
        created_at       TIMESTAMPTZ DEFAULT now(),
        UNIQUE (location_name, model_source, valid_ts)
    )
    """,

    # 5 — Log de alertas emitidos (Telegram + webhook)
    """
    CREATE TABLE IF NOT EXISTS alerts_log (
        id              BIGINT PRIMARY KEY,
        ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
        alert_type      VARCHAR NOT NULL,   -- CHUVA_INTENSA | NIVEL_RIO | TEMPESTADE | ...
        severity        VARCHAR NOT NULL,   -- INFO | ATENCAO | ALERTA | EMERGENCIA
        location        VARCHAR,
        river           VARCHAR,
        message         TEXT NOT NULL,
        level_m         DOUBLE,
        threshold_m     DOUBLE,
        telegram_sent   BOOLEAN DEFAULT FALSE,
        resolved        BOOLEAN DEFAULT FALSE,
        resolved_at     TIMESTAMPTZ
    )
    """,

    # 6 — Métricas de performance dos modelos ML (rastreamento de drift)
    """
    CREATE TABLE IF NOT EXISTS model_metrics (
        id             BIGINT PRIMARY KEY,
        model_name     VARCHAR NOT NULL,   -- convlstm_v1 | lstm_sinos_v1 | ...
        river          VARCHAR,
        evaluated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        mae            DOUBLE,
        rmse           DOUBLE,
        csi            DOUBLE,
        ets            DOUBLE,
        horizon_h      INTEGER,
        n_samples      INTEGER,
        drift_detected BOOLEAN DEFAULT FALSE
    )
    """,

    # 7 — Níveis dos rios em tempo real (ANA HidroWeb)
    """
    CREATE TABLE IF NOT EXISTS river_levels (
        id         BIGINT PRIMARY KEY,
        river      VARCHAR NOT NULL,
        station_id VARCHAR REFERENCES stations(station_id),
        ts         TIMESTAMPTZ NOT NULL,
        level_m    DOUBLE NOT NULL,
        flow_cms   DOUBLE,
        source     VARCHAR DEFAULT 'ANA',
        created_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (river, station_id, ts)
    )
    """,

    # 8 — Qualidade da água (ANA HidroWeb — parâmetros físico-químicos)
    """
    CREATE TABLE IF NOT EXISTS water_quality (
        id               BIGINT PRIMARY KEY,
        station_id       VARCHAR REFERENCES stations(station_id),
        ts               TIMESTAMPTZ NOT NULL,
        ph               DOUBLE,
        turbidity_ntu    DOUBLE,
        do_mgl           DOUBLE,   -- oxigênio dissolvido mg/L
        temperature_c    DOUBLE,
        conductivity_uS  DOUBLE,
        source           VARCHAR DEFAULT 'ANA',
        created_at       TIMESTAMPTZ DEFAULT now(),
        UNIQUE (station_id, ts)
    )
    """,

    # 9 — Status operacional atual dos rios por segmento (calculado pelo processor)
    """
    CREATE TABLE IF NOT EXISTS river_status (
        river                VARCHAR NOT NULL,
        segment              VARCHAR NOT NULL,
        ts                   TIMESTAMPTZ NOT NULL DEFAULT now(),
        level_m              DOUBLE,
        status               VARCHAR NOT NULL,   -- NORMAL | ATENCAO | ALERTA | EMERGENCIA
        trend                VARCHAR,            -- SUBINDO | ESTAVEL | DESCENDO
        municipalities_risk  VARCHAR[],
        updated_at           TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (river, segment)
    )
    """,

    # 10 — Previsões LSTM por rio com intervalo de confiança (MC-Dropout)
    """
    CREATE TABLE IF NOT EXISTS river_ai_forecasts (
        id              BIGINT PRIMARY KEY,
        river           VARCHAR NOT NULL,
        generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        valid_at        TIMESTAMPTZ NOT NULL,
        horizon_h       INTEGER NOT NULL,   -- 6 | 12 | 24
        level_m_p50     DOUBLE NOT NULL,    -- mediana
        level_m_p05     DOUBLE,             -- percentil 5 (limite inferior)
        level_m_p95     DOUBLE,             -- percentil 95 (limite superior)
        uncertainty     DOUBLE,             -- desvio padrão MC-Dropout
        model_version   VARCHAR NOT NULL,
        UNIQUE (river, model_version, valid_at)
    )
    """,

    # 11 — Resultados WRF-Hydro (grade espacial por bacia)
    """
    CREATE TABLE IF NOT EXISTS wrf_hydro_results (
        id             BIGINT PRIMARY KEY,
        run_ts         TIMESTAMPTZ NOT NULL,
        valid_ts       TIMESTAMPTZ NOT NULL,
        basin          VARCHAR NOT NULL,
        flow_cms       DOUBLE,
        soil_moisture  DOUBLE,
        runoff_mm      DOUBLE,
        lat            DOUBLE NOT NULL,
        lon            DOUBLE NOT NULL,
        resolution_km  DOUBLE DEFAULT 4.0,
        created_at     TIMESTAMPTZ DEFAULT now()
    )
    """,
]

# Sequences para PKs BIGINT auto-incremento
_SEQUENCE_DDL: list[str] = [
    "CREATE SEQUENCE IF NOT EXISTS seq_rain_readings START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_forecasts START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_alerts_log START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_model_metrics START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_river_levels START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_water_quality START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_river_ai_forecasts START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_wrf_hydro_results START 1",
]


# ---------------------------------------------------------------------------
# ClimateDB
# ---------------------------------------------------------------------------

class ClimateDB:
    """Gerenciador central do DuckDB para o Monitor Hidrometeorológico RS.

    Mantém uma conexão persistente, cria o schema na primeira vez que é
    instanciado e expõe métodos tipados de leitura/escrita para todos os
    módulos do sistema.

    Usage::

        db = ClimateDB()
        with db.connection() as con:
            con.execute("SELECT count(*) FROM stations")

        # Ou usando o objeto diretamente como context manager:
        with ClimateDB() as db:
            db.upsert_stations(df)
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Inicializa a conexão DuckDB e garante que o schema existe.

        Args:
            db_path: Caminho para o arquivo .duckdb. Se None, usa DB_PATH
                do .env ou o padrão ``data/climate.duckdb``.

        Raises:
            OSError: Se o diretório pai do banco não puder ser criado.
            duckdb.Error: Se a conexão ou criação de tabelas falhar.
        """
        resolved = db_path or os.getenv("DB_PATH", "data/climate.duckdb")
        self._path = Path(resolved)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(str(self._path))
        self._init_schema()
        logger.info(f"ClimateDB conectado: {self._path}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ClimateDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @contextmanager
    def connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager para operações atômicas na mesma conexão.

        Yields:
            duckdb.DuckDBPyConnection: Conexão ativa.

        Example::

            with db.connection() as con:
                con.execute("INSERT INTO ...")
        """
        try:
            yield self._con
        except duckdb.Error as exc:
            logger.error(f"DuckDB erro: {exc}")
            raise

    def close(self) -> None:
        """Fecha a conexão com o banco."""
        try:
            self._con.close()
            logger.info("ClimateDB desconectado.")
        except duckdb.Error as exc:
            logger.warning(f"Erro ao fechar conexão: {exc}")

    # ------------------------------------------------------------------
    # Inicialização do schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Cria sequências e tabelas se ainda não existirem (idempotente).

        Raises:
            duckdb.Error: Se alguma instrução DDL falhar.
        """
        for ddl in _SEQUENCE_DDL + _DDL_STATEMENTS:
            self._con.execute(ddl)
        self._con.commit()
        logger.debug("Schema verificado/criado com sucesso.")

    # ------------------------------------------------------------------
    # Escrita — estações
    # ------------------------------------------------------------------

    def upsert_stations(self, df: pd.DataFrame) -> int:
        """Insere ou atualiza estações de monitoramento.

        Usa INSERT OR REPLACE baseado na PK ``station_id``.

        Args:
            df: DataFrame com colunas obrigatórias: station_id, name, source,
                lat, lon. Opcionais: river, municipality, state, elevation_m, active.

        Returns:
            Número de linhas afetadas.

        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes.
            duckdb.Error: Se a inserção falhar.
        """
        required = {"station_id", "name", "source", "lat", "lon"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        self._con.register("_stations_temp", df)
        self._con.execute("""
            INSERT OR REPLACE INTO stations
                (station_id, name, source, lat, lon, river, municipality,
                 state, elevation_m, active)
            SELECT
                station_id, name, source, lat, lon,
                river, municipality,
                COALESCE(state, 'RS'),
                elevation_m,
                COALESCE(active, TRUE)
            FROM _stations_temp
        """)
        self._con.commit()
        n = len(df)
        logger.info(f"upsert_stations: {n} estações processadas.")
        return n

    # ------------------------------------------------------------------
    # Escrita — leituras de chuva
    # ------------------------------------------------------------------

    def insert_rain_readings(self, df: pd.DataFrame) -> int:
        """Insere leituras brutas de chuva/meteorologia ignorando duplicatas.

        Duplicatas são detectadas pelo índice único (station_id, ts).

        Args:
            df: DataFrame com colunas: station_id, ts e ao menos uma coluna
                de medição (rain_1h_mm, temperature, etc.).

        Returns:
            Número de linhas efetivamente inseridas.

        Raises:
            ValueError: Se station_id ou ts estiverem ausentes.
            duckdb.Error: Se a inserção falhar.
        """
        required = {"station_id", "ts"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        self._con.register("_rain_temp", df)
        result = self._con.execute("""
            INSERT OR IGNORE INTO rain_readings
                (id, station_id, ts, rain_1h_mm, rain_3h_mm, rain_6h_mm,
                 rain_12h_mm, rain_24h_mm, temperature, humidity,
                 pressure_hpa, wind_speed, wind_dir, source)
            SELECT
                nextval('seq_rain_readings'),
                station_id, ts,
                rain_1h_mm, rain_3h_mm, rain_6h_mm, rain_12h_mm, rain_24h_mm,
                temperature, humidity, pressure_hpa, wind_speed, wind_dir,
                source
            FROM _rain_temp
        """)
        self._con.commit()
        inserted = result.rowcount if result.rowcount >= 0 else len(df)
        logger.info(f"insert_rain_readings: {inserted} linhas inseridas de {len(df)}.")
        return inserted

    # ------------------------------------------------------------------
    # Escrita — acumulados
    # ------------------------------------------------------------------

    def update_accumulated(self, df: pd.DataFrame) -> int:
        """Atualiza os acumulados pré-calculados de chuva por estação/dia.

        Args:
            df: DataFrame com colunas: station_id, date e colunas de acumulado
                (rain_1h … rain_72h).

        Returns:
            Número de linhas afetadas.

        Raises:
            ValueError: Se station_id ou date estiverem ausentes.
            duckdb.Error: Se a operação falhar.
        """
        required = {"station_id", "date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        self._con.register("_accum_temp", df)
        self._con.execute("""
            INSERT OR REPLACE INTO rain_accumulated
                (station_id, date, rain_1h, rain_3h, rain_6h,
                 rain_12h, rain_24h, rain_48h, rain_72h, updated_at)
            SELECT
                station_id, date,
                COALESCE(rain_1h,  0), COALESCE(rain_3h,  0),
                COALESCE(rain_6h,  0), COALESCE(rain_12h, 0),
                COALESCE(rain_24h, 0), COALESCE(rain_48h, 0),
                COALESCE(rain_72h, 0),
                now()
            FROM _accum_temp
        """)
        self._con.commit()
        n = len(df)
        logger.info(f"update_accumulated: {n} registros atualizados.")
        return n

    # ------------------------------------------------------------------
    # Escrita — previsões NWP
    # ------------------------------------------------------------------

    def upsert_forecasts(self, df: pd.DataFrame) -> int:
        """Insere ou atualiza previsões NWP.

        Chave única: (location_name, model_source, valid_ts).

        Args:
            df: DataFrame com colunas obrigatórias: location_name, lat, lon,
                forecast_ts, valid_ts, model_source.

        Returns:
            Número de linhas afetadas.

        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes.
            duckdb.Error: Se a operação falhar.
        """
        required = {"location_name", "lat", "lon", "forecast_ts", "valid_ts", "model_source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        self._con.register("_fc_temp", df)
        self._con.execute("""
            INSERT INTO forecasts
                (id, location_name, lat, lon, forecast_ts, valid_ts,
                 rain_mm, temperature, wind_speed, cape_j_kg,
                 lifted_index, k_index, model_source)
            SELECT
                nextval('seq_forecasts'),
                location_name, lat, lon, forecast_ts, valid_ts,
                rain_mm, temperature, wind_speed, cape_j_kg,
                lifted_index, k_index, model_source
            FROM _fc_temp
            ON CONFLICT (location_name, model_source, valid_ts) DO UPDATE SET
                forecast_ts  = EXCLUDED.forecast_ts,
                lat          = EXCLUDED.lat,
                lon          = EXCLUDED.lon,
                rain_mm      = EXCLUDED.rain_mm,
                temperature  = EXCLUDED.temperature,
                wind_speed   = EXCLUDED.wind_speed,
                cape_j_kg    = EXCLUDED.cape_j_kg,
                lifted_index = EXCLUDED.lifted_index,
                k_index      = EXCLUDED.k_index
        """)
        self._con.commit()
        n = len(df)
        logger.info(f"upsert_forecasts: {n} previsões processadas.")
        return n

    # ------------------------------------------------------------------
    # Escrita — alertas
    # ------------------------------------------------------------------

    def log_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        location: str | None = None,
        river: str | None = None,
        level_m: float | None = None,
        threshold_m: float | None = None,
        telegram_sent: bool = False,
    ) -> int:
        """Registra um alerta no log.

        Args:
            alert_type: Tipo do alerta (ex.: CHUVA_INTENSA, NIVEL_RIO).
            severity: Nível de severidade (INFO | ATENCAO | ALERTA | EMERGENCIA).
            message: Texto descritivo do alerta.
            location: Município ou localidade afetada.
            river: Nome do rio (se aplicável).
            level_m: Nível atual do rio em metros.
            threshold_m: Cota que foi superada.
            telegram_sent: True se já enviado para o canal Telegram.

        Returns:
            ID do alerta inserido.

        Raises:
            duckdb.Error: Se a inserção falhar.
        """
        result = self._con.execute("""
            INSERT INTO alerts_log
                (id, ts, alert_type, severity, location, river,
                 message, level_m, threshold_m, telegram_sent)
            VALUES (
                nextval('seq_alerts_log'), now(),
                ?, ?, ?, ?, ?, ?, ?, ?
            ) RETURNING id
        """, [alert_type, severity, location, river, message,
              level_m, threshold_m, telegram_sent]).fetchone()
        self._con.commit()
        alert_id: int = result[0]  # type: ignore[index]
        logger.info(f"log_alert: [{severity}] {alert_type} — id={alert_id}")
        return alert_id

    # ------------------------------------------------------------------
    # Leitura — mapa de chuva (dashboard Folium)
    # ------------------------------------------------------------------

    def get_rain_map_data(self, window_hours: int = 24) -> pd.DataFrame:
        """Retorna dados de chuva recente de todas as estações para o mapa.

        Busca o acumulado mais recente de cada estação dentro do período
        informado para renderizar o mapa de bolhas no Folium.

        Args:
            window_hours: Janela de tempo em horas (padrão 24h).

        Returns:
            DataFrame com colunas: station_id, name, lat, lon, river,
            municipality, rain_24h, rain_72h.

        Raises:
            duckdb.Error: Se a query falhar.
        """
        df: pd.DataFrame = self._con.execute(f"""
            SELECT
                s.station_id,
                s.name,
                s.lat,
                s.lon,
                s.river,
                s.municipality,
                a.rain_24h,
                a.rain_72h
            FROM stations s
            LEFT JOIN (
                SELECT station_id, rain_24h, rain_72h
                FROM rain_accumulated
                WHERE date >= current_date - INTERVAL '{window_hours // 24 + 1} days'
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY station_id ORDER BY date DESC
                ) = 1
            ) a ON a.station_id = s.station_id
            WHERE s.active = TRUE
            ORDER BY s.name
        """).df()
        logger.debug(f"get_rain_map_data: {len(df)} estações retornadas.")
        return df

    # ------------------------------------------------------------------
    # Leitura — série temporal de uma estação
    # ------------------------------------------------------------------

    def get_timeseries(
        self,
        station_id: str,
        start: datetime,
        end: datetime,
        variables: list[str] | None = None,
    ) -> pd.DataFrame:
        """Retorna série temporal de leituras para uma estação.

        Args:
            station_id: Identificador da estação.
            start: Início do período (UTC ou com tzinfo).
            end: Fim do período (UTC ou com tzinfo).
            variables: Lista de colunas a retornar. None = todas.

        Returns:
            DataFrame indexado por ``ts`` com as variáveis solicitadas.

        Raises:
            ValueError: Se station_id for vazio.
            duckdb.Error: Se a query falhar.
        """
        if not station_id:
            raise ValueError("station_id não pode ser vazio.")

        all_vars = [
            "rain_1h_mm", "rain_3h_mm", "rain_6h_mm",
            "rain_12h_mm", "rain_24h_mm",
            "temperature", "humidity", "pressure_hpa",
            "wind_speed", "wind_dir",
        ]
        cols = variables if variables else all_vars
        cols_sql = ", ".join(cols)

        df: pd.DataFrame = self._con.execute(f"""
            SELECT ts, {cols_sql}
            FROM rain_readings
            WHERE station_id = ?
              AND ts BETWEEN ? AND ?
            ORDER BY ts ASC
        """, [station_id, start, end]).df()
        df = df.set_index("ts")
        logger.debug(
            f"get_timeseries: estação={station_id} "
            f"{start:%Y-%m-%d} → {end:%Y-%m-%d} — {len(df)} pontos."
        )
        return df

    # ------------------------------------------------------------------
    # Leitura — alertas ativos
    # ------------------------------------------------------------------

    def get_active_alerts(self, severity: str | None = None) -> pd.DataFrame:
        """Retorna alertas não resolvidos, opcionalmente filtrados por severidade.

        Args:
            severity: Filtro de severidade (INFO | ATENCAO | ALERTA | EMERGENCIA).
                      None retorna todos os níveis.

        Returns:
            DataFrame com todos os campos de alerts_log, ordenado por ts desc.

        Raises:
            duckdb.Error: Se a query falhar.
        """
        params: list[Any] = []
        where = "WHERE resolved = FALSE"
        if severity:
            where += " AND severity = ?"
            params.append(severity)

        df: pd.DataFrame = self._con.execute(
            f"SELECT * FROM alerts_log {where} ORDER BY ts DESC",
            params,
        ).df()
        logger.debug(f"get_active_alerts: {len(df)} alertas ativos.")
        return df

    # ------------------------------------------------------------------
    # Leitura — status dos rios
    # ------------------------------------------------------------------

    def get_river_status(self, river: str | None = None) -> pd.DataFrame:
        """Retorna o status operacional atual dos rios monitorados.

        Args:
            river: Nome do rio para filtrar. None retorna todos.

        Returns:
            DataFrame com colunas: river, segment, level_m, status,
            trend, municipalities_risk, updated_at.

        Raises:
            duckdb.Error: Se a query falhar.
        """
        params: list[Any] = []
        where = ""
        if river:
            where = "WHERE river = ?"
            params.append(river)

        df: pd.DataFrame = self._con.execute(
            f"SELECT * FROM river_status {where} ORDER BY river, segment",
            params,
        ).df()
        logger.debug(f"get_river_status: {len(df)} segmentos retornados.")
        return df

    # ------------------------------------------------------------------
    # Leitura — qualidade da água
    # ------------------------------------------------------------------

    def get_water_quality(
        self,
        station_id: str | None = None,
        days: int = 7,
    ) -> pd.DataFrame:
        """Retorna dados de qualidade da água.

        Args:
            station_id: Filtro por estação. None retorna todas.
            days: Número de dias para trás a partir de agora.

        Returns:
            DataFrame com ph, turbidity_ntu, do_mgl, temperature_c,
            conductivity_uS, ts, station_id.

        Raises:
            duckdb.Error: Se a query falhar.
        """
        params: list[Any] = [days]
        where = "WHERE ts >= now() - INTERVAL ? || ' days'"
        if station_id:
            where += " AND station_id = ?"
            params.append(station_id)

        df: pd.DataFrame = self._con.execute(
            f"""
            SELECT station_id, ts, ph, turbidity_ntu,
                   do_mgl, temperature_c, conductivity_uS
            FROM water_quality
            {where}
            ORDER BY ts DESC
            """,
            params,
        ).df()
        logger.debug(f"get_water_quality: {len(df)} registros (últimos {days}d).")
        return df

    # ------------------------------------------------------------------
    # Leitura — estatísticas do banco
    # ------------------------------------------------------------------

    def get_db_stats(self) -> dict[str, Any]:
        """Retorna estatísticas de uso do banco (contagens + tamanho em disco).

        Returns:
            Dicionário com chaves: tables (dict nome→contagem),
            db_size_mb (float), oldest_reading (str), newest_reading (str).

        Raises:
            duckdb.Error: Se alguma query falhar.
        """
        tables = [
            "stations", "rain_readings", "rain_accumulated", "forecasts",
            "alerts_log", "model_metrics", "river_levels", "water_quality",
            "river_status", "river_ai_forecasts", "wrf_hydro_results",
        ]
        counts: dict[str, int] = {}
        for t in tables:
            row = self._con.execute(f"SELECT count(*) FROM {t}").fetchone()
            counts[t] = row[0] if row else 0

        size_mb = round(self._path.stat().st_size / 1_048_576, 2) if self._path.exists() else 0.0

        oldest = newest = "N/A"
        if counts["rain_readings"] > 0:
            row_o = self._con.execute(
                "SELECT min(ts)::VARCHAR, max(ts)::VARCHAR FROM rain_readings"
            ).fetchone()
            if row_o:
                oldest, newest = row_o[0], row_o[1]

        stats: dict[str, Any] = {
            "tables": counts,
            "db_size_mb": size_mb,
            "oldest_reading": oldest,
            "newest_reading": newest,
        }
        logger.info(f"DB stats: {size_mb} MB | leituras={counts['rain_readings']}")
        return stats
