-- =============================================================================
-- schema_supabase.sql
-- Monitor Hidrometeorológico RS — Supabase PostgreSQL
-- Executar no SQL Editor do Supabase (projeto: wyvyibdxkcizepgooexd)
-- =============================================================================

-- =============================================================================
-- SEÇÃO 1 — DESABILITAR RLS EM TODAS AS TABELAS DO PROJETO
-- Causa confirmada: RLS bloqueava leitura via role 'postgres' (Session Pooler)
-- enquanto o hybrid_writer reportava COUNT(*) = 17.724 na própria conexão.
-- Executar esta seção primeiro, depois confirmar no SQL Editor.
-- =============================================================================

-- Tabelas com prefixo live_ (schema existente no Supabase)
ALTER TABLE IF EXISTS live_gpm_precip      DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS live_rain_readings   DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS live_river_levels    DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS river_ai_forecasts   DISABLE ROW LEVEL SECURITY;

-- Tabelas sem prefixo (criadas pelo hybrid_writer via CREATE TABLE IF NOT EXISTS)
ALTER TABLE IF EXISTS stations             DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS rain_readings        DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS forecasts            DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS river_levels         DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS river_status         DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS water_quality        DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS gpm_precip           DISABLE ROW LEVEL SECURITY;

-- Verificação imediata: deve retornar rowsecurity = false para todas
SELECT
    schemaname,
    tablename,
    rowsecurity AS rls_ativa
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN (
      'live_gpm_precip', 'live_rain_readings', 'live_river_levels',
      'river_ai_forecasts', 'stations', 'rain_readings', 'forecasts',
      'river_levels', 'river_status', 'water_quality', 'gpm_precip'
  )
ORDER BY tablename;


-- =============================================================================
-- SEÇÃO 2 — CONFIRMAR DADOS APÓS DESABILITAR RLS
-- Executar após a Seção 1. Deve mostrar os 17.724 pontos GPM.
-- =============================================================================

SELECT 'gpm_precip'      AS tabela, COUNT(*) AS total FROM gpm_precip
UNION ALL
SELECT 'rain_readings'   AS tabela, COUNT(*) AS total FROM rain_readings
UNION ALL
SELECT 'stations'        AS tabela, COUNT(*) AS total FROM stations
UNION ALL
SELECT 'river_levels'    AS tabela, COUNT(*) AS total FROM river_levels
UNION ALL
SELECT 'forecasts'       AS tabela, COUNT(*) AS total FROM forecasts;


-- =============================================================================
-- SEÇÃO 3 — DDL COMPLETO (hybrid_writer tables)
-- Só executar se as tabelas NÃO existirem ainda.
-- O hybrid_writer já executa este DDL via _ensure_schema() na primeira conexão.
-- =============================================================================

CREATE TABLE IF NOT EXISTS stations (
    station_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    source       TEXT NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    river        TEXT,
    municipality TEXT,
    state        TEXT DEFAULT 'RS',
    elevation_m  DOUBLE PRECISION,
    active       BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rain_readings (
    station_id   TEXT NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    rain_1h_mm   DOUBLE PRECISION,
    temperature  DOUBLE PRECISION,
    humidity     DOUBLE PRECISION,
    pressure_hpa DOUBLE PRECISION,
    wind_speed   DOUBLE PRECISION,
    wind_dir     DOUBLE PRECISION,
    source       TEXT,
    PRIMARY KEY (station_id, ts)
);

CREATE TABLE IF NOT EXISTS forecasts (
    location_name TEXT             NOT NULL,
    lat           DOUBLE PRECISION,
    lon           DOUBLE PRECISION,
    forecast_ts   TIMESTAMPTZ,
    valid_ts      TIMESTAMPTZ      NOT NULL,
    rain_mm       DOUBLE PRECISION,
    temperature   DOUBLE PRECISION,
    wind_speed    DOUBLE PRECISION,
    cape_j_kg     DOUBLE PRECISION,
    lifted_index  DOUBLE PRECISION,
    k_index       DOUBLE PRECISION,
    model_source  TEXT             NOT NULL,
    PRIMARY KEY (location_name, model_source, valid_ts)
);

CREATE TABLE IF NOT EXISTS river_levels (
    river      TEXT             NOT NULL,
    station_id TEXT             NOT NULL,
    ts         TIMESTAMPTZ      NOT NULL,
    level_m    DOUBLE PRECISION NOT NULL,
    flow_cms   DOUBLE PRECISION,
    source     TEXT DEFAULT 'ANA',
    PRIMARY KEY (river, station_id, ts)
);

CREATE TABLE IF NOT EXISTS river_status (
    river               TEXT        NOT NULL,
    segment             TEXT        NOT NULL,
    ts                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level_m             DOUBLE PRECISION,
    status              TEXT        NOT NULL,
    trend               TEXT,
    municipalities_risk TEXT[],
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (river, segment)
);

CREATE TABLE IF NOT EXISTS water_quality (
    station_id      TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    ph              DOUBLE PRECISION,
    turbidity_ntu   DOUBLE PRECISION,
    do_mgl          DOUBLE PRECISION,
    temperature_c   DOUBLE PRECISION,
    conductivity_us DOUBLE PRECISION,
    source          TEXT DEFAULT 'ANA',
    PRIMARY KEY (station_id, ts)
);

CREATE TABLE IF NOT EXISTS gpm_precip (
    lat       DOUBLE PRECISION NOT NULL,
    lon       DOUBLE PRECISION NOT NULL,
    precip_mm REAL,
    "timestamp" TIMESTAMPTZ    NOT NULL,
    source    TEXT DEFAULT 'GPM_IMERG_EARLY',
    PRIMARY KEY (lat, lon, "timestamp")
);

-- RLS desabilitado para todas as tabelas recém-criadas
ALTER TABLE IF EXISTS stations       DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS rain_readings  DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS forecasts      DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS river_levels   DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS river_status   DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS water_quality  DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS gpm_precip     DISABLE ROW LEVEL SECURITY;


-- =============================================================================
-- SEÇÃO 4 — GRANTS (caso o dashboard use role anon/authenticated)
-- Executar se o Streamlit Cloud ou a API FastAPI usam a chave anon do Supabase.
-- =============================================================================

GRANT SELECT ON stations, rain_readings, forecasts, river_levels,
               river_status, water_quality, gpm_precip
    TO anon, authenticated;

GRANT INSERT, UPDATE ON stations, rain_readings, forecasts, river_levels,
                        river_status, water_quality, gpm_precip
    TO service_role;
