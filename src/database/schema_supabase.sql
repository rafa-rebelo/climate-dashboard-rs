-- ============================================================
-- Schema Supabase — Sistema Autônomo de Análise Climática RS
-- Camada quente (estado atual) — Fase 1, arquitetura híbrida
--
-- Tabelas live_* / stations / river_ai_forecasts foram criadas
-- originalmente via SQL Editor; este arquivo versiona o schema
-- completo e é idempotente (IF NOT EXISTS) — pode ser reexecutado.
-- RLS desabilitado: escrita via psycopg2 (GitHub Actions / API).
-- ============================================================

-- ── stations — inventário INMET/ANA/CEMADEN (PK: station_id) ─
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
-- Fontes aceitas — o check original (INMET/ANA) rejeitava o coletor CEMADEN;
-- recriado em 03/07/2026 incluindo CEMADEN/REDEMET e depois DCRS (Defesa
-- Civil RS). Idempotente.
ALTER TABLE stations DROP CONSTRAINT IF EXISTS stations_fonte_check;
ALTER TABLE stations ADD CONSTRAINT stations_fonte_check
    CHECK (fonte IN ('INMET', 'ANA', 'CEMADEN', 'REDEMET', 'DCRS'));

-- ── live_dcrs_stations — Rede Hidrometeorológica Defesa Civil RS ──
-- Snapshot 1 linha/estação DCRS (GraphQL tags_data, ~130 estações, 26
-- bacias). rio_nivel na unidade BRUTA da rede (mista cm/m por estação —
-- heurística de exibição no dashboard). Cotas de alarme vêm da própria
-- API quando definidas (hoje nulas no client público).
CREATE TABLE IF NOT EXISTS live_dcrs_stations (
    codigo            TEXT PRIMARY KEY,
    nome              TEXT,
    local             TEXT,
    bacia             TEXT,
    latitude          DOUBLE PRECISION,
    longitude         DOUBLE PRECISION,
    rio_nome          TEXT,
    rio_nivel         DOUBLE PRECISION,
    rio_vazao         DOUBLE PRECISION,
    rio_tendencia     DOUBLE PRECISION,
    cota_atencao      DOUBLE PRECISION,
    cota_alerta       DOUBLE PRECISION,
    cota_emergencia   DOUBLE PRECISION,
    inundacao_status  DOUBLE PRECISION,
    chuva_1h          DOUBLE PRECISION,
    chuva_3h          DOUBLE PRECISION,
    chuva_6h          DOUBLE PRECISION,
    chuva_12h         DOUBLE PRECISION,
    chuva_24h         DOUBLE PRECISION,
    chuva_48h         DOUBLE PRECISION,
    chuva_72h         DOUBLE PRECISION,
    chuva_168h        DOUBLE PRECISION,
    temperatura       DOUBLE PRECISION,
    umidade           DOUBLE PRECISION,
    vento_vel         DOUBLE PRECISION,
    vento_dir         DOUBLE PRECISION,
    pressao           DOUBLE PRECISION,
    senstermica       DOUBLE PRECISION,
    radiacao          DOUBLE PRECISION,
    "timestamp"       TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE live_dcrs_stations DISABLE ROW LEVEL SECURITY;

-- ── live_rain_readings — snapshot 1 linha/estação ────────────
CREATE TABLE IF NOT EXISTS live_rain_readings (
    station_id  TEXT PRIMARY KEY,
    precip_1h   DOUBLE PRECISION,
    precip_3h   DOUBLE PRECISION,
    precip_6h   DOUBLE PRECISION,
    precip_12h  DOUBLE PRECISION,
    precip_24h  DOUBLE PRECISION,
    precip_48h  DOUBLE PRECISION,
    precip_72h  DOUBLE PRECISION,
    precip_7d   DOUBLE PRECISION,
    temperatura DOUBLE PRECISION,
    umidade     DOUBLE PRECISION,
    "timestamp" TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
-- Janelas de acumulado adicionadas após a criação original (idempotente):
-- as 8 já eram calculadas pelo rain_accumulator e gravadas no R2; agora também
-- na camada quente para a API expor sem ler Parquet a cada request.
ALTER TABLE live_rain_readings ADD COLUMN IF NOT EXISTS precip_3h  DOUBLE PRECISION;
ALTER TABLE live_rain_readings ADD COLUMN IF NOT EXISTS precip_12h DOUBLE PRECISION;
ALTER TABLE live_rain_readings ADD COLUMN IF NOT EXISTS precip_48h DOUBLE PRECISION;
ALTER TABLE live_rain_readings ADD COLUMN IF NOT EXISTS precip_72h DOUBLE PRECISION;
ALTER TABLE live_rain_readings ADD COLUMN IF NOT EXISTS precip_7d  DOUBLE PRECISION;
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

-- ── river_ai_forecasts — previsões LSTM (snapshot vivo) ──────
-- Semântica de snapshot: 1 linha por rio × horizonte, sempre a previsão
-- mais recente (igual às tabelas live_*). PK (rio_id, horizonte_h) para
-- o UPSERT da inferência. horizonte_h em HORAS: 24/48/72/144 = 1/2/3/6 dias.
CREATE TABLE IF NOT EXISTS river_ai_forecasts (
    rio_id           TEXT NOT NULL,
    horizonte_h      INTEGER NOT NULL,
    valid_ts         TIMESTAMPTZ,
    nivel_previsto_m DOUBLE PRECISION,
    ic_inferior_m    DOUBLE PRECISION,
    ic_superior_m    DOUBLE PRECISION,
    modelo_versao    TEXT,
    status           TEXT DEFAULT 'ok',
    gerado_em        TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (rio_id, horizonte_h)
);
ALTER TABLE river_ai_forecasts DISABLE ROW LEVEL SECURITY;
