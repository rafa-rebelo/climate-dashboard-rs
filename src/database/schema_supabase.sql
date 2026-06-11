-- ============================================================
-- Schema Supabase — Sistema Autônomo de Análise Climática RS
-- Camada quente (estado atual) — Fase 1, arquitetura híbrida
--
-- Tabelas live_* / stations / river_ai_forecasts foram criadas
-- originalmente via SQL Editor; este arquivo versiona o schema
-- completo e é idempotente (IF NOT EXISTS) — pode ser reexecutado.
-- RLS desabilitado: escrita via psycopg2 (GitHub Actions / API).
-- ============================================================

-- ── stations — inventário INMET/ANA (PK: station_id) ─────────
CREATE TABLE IF NOT EXISTS stations (
    station_id  TEXT PRIMARY KEY,
    nome        TEXT,
    municipio   TEXT,
    estado      TEXT,
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    fonte       TEXT,
    ativa       BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE stations DISABLE ROW LEVEL SECURITY;

-- ── live_rain_readings — snapshot 1 linha/estação ────────────
CREATE TABLE IF NOT EXISTS live_rain_readings (
    station_id  TEXT PRIMARY KEY,
    precip_1h   DOUBLE PRECISION,
    precip_6h   DOUBLE PRECISION,
    precip_24h  DOUBLE PRECISION,
    temperatura DOUBLE PRECISION,
    umidade     DOUBLE PRECISION,
    "timestamp" TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE live_rain_readings DISABLE ROW LEVEL SECURITY;

-- ── live_river_levels — snapshot 1 linha/rio ─────────────────
CREATE TABLE IF NOT EXISTS live_river_levels (
    rio_id            TEXT PRIMARY KEY,
    rio_nome          TEXT NOT NULL,
    nivel_atual_m     DOUBLE PRECISION,
    vazao_m3s         DOUBLE PRECISION,
    cota_atencao_m    DOUBLE PRECISION NOT NULL,
    cota_alerta_m     DOUBLE PRECISION NOT NULL,
    cota_emergencia_m DOUBLE PRECISION NOT NULL,
    status            TEXT,
    percentual_cota   DOUBLE PRECISION,
    "timestamp"       TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE live_river_levels DISABLE ROW LEVEL SECURITY;

-- ── live_gpm_precip — grade satélite (PK: lat_lon_key) ───────
CREATE TABLE IF NOT EXISTS live_gpm_precip (
    lat_lon_key TEXT PRIMARY KEY,
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    precip_mm   DOUBLE PRECISION,
    source      TEXT,
    "timestamp" TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gpm_latlon
    ON live_gpm_precip (latitude, longitude);
ALTER TABLE live_gpm_precip DISABLE ROW LEVEL SECURITY;

-- ── forecasts — previsões NWP Open-Meteo (10 pontos RS) ──────
-- PK composta: 1 linha por ponto × modelo × horário previsto.
-- UPSERT do hybrid_writer.write_forecasts().
CREATE TABLE IF NOT EXISTS forecasts (
    location_name TEXT NOT NULL,
    lat           DOUBLE PRECISION,
    lon           DOUBLE PRECISION,
    forecast_ts   TIMESTAMPTZ,
    valid_ts      TIMESTAMPTZ NOT NULL,
    rain_mm       DOUBLE PRECISION,
    temperature   DOUBLE PRECISION,
    wind_speed    DOUBLE PRECISION,
    cape_j_kg     DOUBLE PRECISION,
    lifted_index  DOUBLE PRECISION,
    k_index       DOUBLE PRECISION,
    model_source  TEXT NOT NULL DEFAULT 'openmeteo',
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (location_name, model_source, valid_ts)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_valid_ts ON forecasts (valid_ts);
ALTER TABLE forecasts DISABLE ROW LEVEL SECURITY;

-- ── water_quality — físico-química ANA (PK: station_id, ts) ──
CREATE TABLE IF NOT EXISTS water_quality (
    station_id      TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    ph              DOUBLE PRECISION,
    turbidity_ntu   DOUBLE PRECISION,
    do_mgl          DOUBLE PRECISION,
    temperature_c   DOUBLE PRECISION,
    conductivity_us DOUBLE PRECISION,
    source          TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (station_id, ts)
);
ALTER TABLE water_quality DISABLE ROW LEVEL SECURITY;

-- ── alerts_log — histórico de alertas enviados ───────────────
-- Também é a base do throttle anti-spam (janela de 1h por chave):
-- o runner do GitHub Actions é efêmero, então a de-duplicação
-- precisa ser persistente, não em memória.
CREATE TABLE IF NOT EXISTS alerts_log (
    id            BIGSERIAL PRIMARY KEY,
    alert_key     TEXT NOT NULL,
    alert_type    TEXT,
    severity      TEXT,
    rio_id        TEXT,
    message       TEXT,
    level_m       DOUBLE PRECISION,
    threshold_m   DOUBLE PRECISION,
    telegram_sent BOOLEAN DEFAULT FALSE,
    sent_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alerts_key_time
    ON alerts_log (alert_key, sent_at DESC);
ALTER TABLE alerts_log DISABLE ROW LEVEL SECURITY;

-- ── river_ai_forecasts — previsões LSTM (Fase 2, vazia) ──────
CREATE TABLE IF NOT EXISTS river_ai_forecasts (
    rio_id          TEXT NOT NULL,
    forecast_ts     TIMESTAMPTZ NOT NULL,
    valid_ts        TIMESTAMPTZ NOT NULL,
    nivel_previsto_m DOUBLE PRECISION,
    ic_inferior_m   DOUBLE PRECISION,
    ic_superior_m   DOUBLE PRECISION,
    horizonte_h     INTEGER,
    modelo_versao   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (rio_id, valid_ts, horizonte_h)
);
ALTER TABLE river_ai_forecasts DISABLE ROW LEVEL SECURITY;
