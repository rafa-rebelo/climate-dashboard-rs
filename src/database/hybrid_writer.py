"""
Agente 1 — Arquiteto de Dados
Módulo de persistência híbrida v2.

Fluxo A — PostgreSQL Supabase (psycopg2-binary)
  • Upsert via INSERT … ON CONFLICT DO UPDATE/NOTHING
  • Var principal : SUPABASE_DATABASE_URL        (conexão direta, local/VSCode)
  • Var GitHub CI : SUPABASE_DATABASE_URL_POOLER (Session Pooler IPv4, Actions)
  • Falha isolada : não afeta Fluxo B

Fluxo B — Cloudflare R2 (boto3 / S3-compatible)
  • Upload Parquet em memória (BytesIO), ZERO arquivo local
  • Path: historico/{fonte}/ano={YYYY}/mes={MM}/dia={DD}/{fonte}_{ts}UTC.parquet
  • Vars: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
  • Falha isolada : não afeta Fluxo A

SEM DuckDB. SEM ClimateDB. SEM arquivo .parquet local.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Variáveis de ambiente
# ---------------------------------------------------------------------------

# Fluxo A — Supabase PostgreSQL
# NÃO cachear a URL em módulo — ler os.getenv() a cada conexão para garantir
# que a variável injetada pelo runner do GitHub Actions seja sempre usada.
# _pg_connect() lê SUPABASE_DATABASE_URL_POOLER > SUPABASE_DATABASE_URL a cada chamada.

# Fluxo B — Cloudflare R2 (R2 não tem o problema de ordem de injeção — ok cachear)
_R2_ENDPOINT:   str = os.getenv("R2_ENDPOINT_URL",      "")
_R2_ACCESS_KEY: str = os.getenv("R2_ACCESS_KEY_ID",     "")
_R2_SECRET_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
_R2_BUCKET:     str = os.getenv("R2_BUCKET_NAME",       "climate-data-rs")

# ---------------------------------------------------------------------------
# Importações condicionais
# ---------------------------------------------------------------------------

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAIL = True
except ImportError:
    psycopg2 = None           # type: ignore[assignment]
    _PSYCOPG2_AVAIL = False
    logger.warning("psycopg2 não disponível — pip install psycopg2-binary")

try:
    import boto3
    import botocore.exceptions as _botocore_exc
    _BOTO3_AVAIL = True
except ImportError:
    boto3 = None              # type: ignore[assignment]
    _botocore_exc = None      # type: ignore[assignment]
    _BOTO3_AVAIL = False
    logger.warning("boto3 não disponível — pip install boto3")

# ---------------------------------------------------------------------------
# DDL PostgreSQL — criado uma vez por sessão (idempotente)
# ---------------------------------------------------------------------------

# DDL removido — tabelas já existem no Supabase com schema correto.
# _ensure_schema() agora é no-op; mantida para compatibilidade de chamadas.


# ---------------------------------------------------------------------------
# WriteResult
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    """Resultado de uma operação de persistência híbrida (Fluxo A + B).

    Attributes:
        table: Nome lógico da entidade gravada.
        pg_rows: Linhas inseridas/atualizadas no PostgreSQL.
        pg_ok: True se o Fluxo A (PostgreSQL) completou sem erro.
        pg_error: Mensagem de erro do PostgreSQL, ou None.
        r2_key: Chave S3 do objeto carregado no R2.
        r2_ok: True se o Fluxo B (R2) completou sem erro.
        r2_error: Mensagem de erro do R2, ou None.
        duration_s: Tempo total da operação em segundos.
    """

    table: str
    pg_rows: int = 0
    pg_ok: bool = False
    pg_error: Optional[str] = None
    r2_key: str = ""
    r2_ok: bool = False
    r2_error: Optional[str] = None
    duration_s: float = 0.0

    @property
    def success(self) -> bool:
        """True se ao menos um fluxo (A ou B) gravou com sucesso."""
        return self.pg_ok or self.r2_ok

    def log_summary(self) -> None:
        """Emite linha de log com resumo da operação."""
        icon = "✓" if self.success else "✗"
        pg_s = f"PG {self.pg_rows:,}r" if self.pg_ok else f"PG ERR({self.pg_error})"
        r2_s = f"R2 {self.r2_key.split('/')[-1]}" if self.r2_ok else f"R2 ERR({self.r2_error})"
        logger.info(f"  {icon} [{self.table}] {pg_s} | {r2_s} | {self.duration_s:.1f}s")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _rio_slug(name: str) -> str:
    """Normaliza nome de rio para slug ASCII minúsculo (ex.: 'Guaíba' → 'guaiba').

    Args:
        name: Nome do rio como retornado pela API ou config.

    Returns:
        Slug ASCII minúsculo sem acentos (ex.: 'sinos', 'jacui', 'guaiba').
    """
    s = unicodedata.normalize("NFD", str(name)).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def _sha256_id(key: str) -> int:
    """Gera BIGINT determinístico (63 bits) via SHA-256 para uso como PK.

    Args:
        key: String composta que identifica unicamente o registro.

    Returns:
        Inteiro positivo de 63 bits compatível com BIGINT signed PostgreSQL.
    """
    digest = hashlib.sha256(key.encode()).digest()[:8]
    return struct.unpack(">Q", digest)[0] & 0x7FFF_FFFF_FFFF_FFFF


def _resolve_pg_url() -> str:
    """Resolve a URL de conexão PostgreSQL lendo os.getenv() em runtime.

    Leitura lazy (não cacheada) garante que variáveis injetadas pelo runner
    do GitHub Actions após o import do módulo sejam sempre capturadas.
    Ordem de prioridade: POOLER (IPv4 Session Pooler) > DIRECT (IPv6).

    Returns:
        URL de conexão PostgreSQL, ou string vazia se nenhuma var configurada.
    """
    return (
        os.getenv("SUPABASE_DATABASE_URL_POOLER")
        or os.getenv("SUPABASE_DATABASE_URL")
        or ""
    )


def _pg_host_safe(url: str = "") -> str:
    """Extrai somente o host da URL (sem senha) para logging.

    Args:
        url: URL de conexão. Se vazio, resolve via ``_resolve_pg_url()``.

    Returns:
        Trecho ``@host:port/db`` da URL ou ``<url_vazia>``.
    """
    u = url or _resolve_pg_url()
    if not u:
        return "<url_vazia>"
    at = u.rfind("@")
    return u[at:] if at != -1 else u[:30] + "…"


def _pg_connect() -> "Optional[psycopg2.extensions.connection]":
    """Abre conexão PostgreSQL lendo as variáveis de ambiente em runtime.

    Resolve a URL a cada chamada (não usa cache de módulo) para garantir que
    variáveis injetadas pelo GitHub Actions runner após o import sejam usadas.
    Prioridade: SUPABASE_DATABASE_URL_POOLER > SUPABASE_DATABASE_URL.

    Returns:
        Conexão psycopg2 aberta com autocommit=False, ou None em falha.

    Raises:
        Não lança — captura psycopg2.Error internamente.
    """
    if not _PSYCOPG2_AVAIL:
        return None

    db_url = _resolve_pg_url()
    if not db_url:
        logger.warning("PG: SUPABASE_DATABASE_URL_POOLER e SUPABASE_DATABASE_URL ausentes — skip Fluxo A.")
        return None

    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
        conn.autocommit = False
        logger.debug(f"PG conectado: {_pg_host_safe(db_url)}")
        return conn
    except psycopg2.Error as exc:
        logger.warning(f"PG: falha na conexão {_pg_host_safe(db_url)} — {exc}")
        return None


def _r2_client() -> "Any":
    """Cria cliente boto3 para Cloudflare R2 (S3-compatible).

    Returns:
        Cliente boto3.client ou None se boto3 não disponível ou vars ausentes.
    """
    if not _BOTO3_AVAIL:
        return None
    if not (_R2_ENDPOINT and _R2_ACCESS_KEY and _R2_SECRET_KEY):
        logger.warning("R2: variáveis R2_ENDPOINT_URL/KEY não configuradas — skip Fluxo B.")
        return None
    return boto3.client(
        "s3",
        endpoint_url=_R2_ENDPOINT,
        aws_access_key_id=_R2_ACCESS_KEY,
        aws_secret_access_key=_R2_SECRET_KEY,
        region_name="auto",
    )


def _r2_key(fonte: str, now: datetime) -> str:
    """Gera chave S3 com particionamento Hive-style por data.

    Args:
        fonte: Nome da fonte/tabela (ex.: "river_levels", "forecasts").
        now: Timestamp do momento do upload (UTC).

    Returns:
        Chave S3 no formato:
        ``historico/{fonte}/ano={YYYY}/mes={MM}/dia={DD}/{fonte}_{ts}UTC.parquet``
    """
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    return (
        f"historico/{fonte}"
        f"/ano={now.year}"
        f"/mes={now.month:02d}"
        f"/dia={now.day:02d}"
        f"/{fonte}_{ts_str}UTC.parquet"
    )


def _r2_upload(
    df: pd.DataFrame,
    fonte: str,
    result: WriteResult,
    s3: "Any",
) -> None:
    """Serializa DataFrame para Parquet em memória e envia ao R2.

    Args:
        df: DataFrame a enviar.
        fonte: Nome lógico (define o prefixo do caminho S3).
        result: WriteResult a atualizar in-place.
        s3: Cliente boto3 S3 já instanciado.

    Note:
        Falhas de rede ou autenticação são capturadas e registradas sem
        propagar exceção — Fluxo B nunca interrompe Fluxo A.
    """
    now = datetime.now(tz=timezone.utc)
    key = _r2_key(fonte, now)
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        s3.put_object(
            Bucket=_R2_BUCKET,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        result.r2_key = key
        result.r2_ok = True
        size_kb = len(buf.getvalue()) / 1024
        logger.info(f"    R2 upload: {key.split('/')[-1]} ({len(df):,} linhas, {size_kb:.0f} KB)")
    except Exception as exc:  # noqa: BLE001 — R2 nunca propaga
        result.r2_error = str(exc)
        logger.warning(f"    R2 falhou ({fonte}): {exc}")


def _ensure_schema(conn: "Any") -> None:
    """No-op — tabelas já existem no Supabase com schema correto.

    Mantida apenas para compatibilidade com as chamadas nos métodos write_*.

    Args:
        conn: Ignorado.
    """
    logger.debug("PG schema verificado/criado.")


# ---------------------------------------------------------------------------
# HybridWriter
# ---------------------------------------------------------------------------

class HybridWriter:
    """Persiste DataFrames em PostgreSQL Supabase (Fluxo A) e Cloudflare R2 (Fluxo B).

    Instanciar uma vez por coleta. Os clientes de banco e R2 são abertos e
    fechados por operação — sem estado de conexão persistente entre chamadas.

    Example::

        writer = HybridWriter()
        res = writer.write_forecasts(df_hourly, df_daily)
        if not res.success:
            raise RuntimeError("falha total — nenhum fluxo gravou")
    """

    # ── write_river_levels ─────────────────────────────────────────────────

    def write_river_levels(self, df: pd.DataFrame) -> WriteResult:
        """Persiste série temporal de níveis/chuva ANA.

        PG: UPSERT em ``live_river_levels`` (PK: rio_id, timestamp).
        R2: ``historico/live_river_levels/ano=…/mes=…/dia=…/{ts}UTC.parquet``

        Mapeamento de colunas ANA → Supabase:
          station_code → rio_id
          nivel_m      → nivel_atual_m
          vazao_m3s    → vazao_m3s
          status calculado (NORMAL/ATENÇÃO/ALERTA/EMERGÊNCIA) → status
          percentual_cota → NULL (calculado externamente pelo processador)

        Args:
            df: DataFrame ANA normalizado com colunas rio_nome, station_code,
                timestamp/data_hora, nivel_m, vazao_m3s.

        Returns:
            WriteResult com estatísticas dos dois fluxos.
        """
        result = WriteResult(table="live_river_levels")
        t0 = time.monotonic()

        if df.empty:
            result.duration_s = time.monotonic() - t0
            return result

        # ── normalizar colunas do DataFrame ANA ───────────────────────────
        df = df.copy()
        for alias in ("timestamp", "data_hora_medicao", "data_hora", "DataHora"):
            if alias in df.columns:
                df["_ts"] = pd.to_datetime(df[alias], errors="coerce", utc=True)
                break
        if "_ts" not in df.columns:
            logger.warning("write_river_levels: coluna de timestamp ausente — skip PG.")
            df["_ts"] = pd.NaT
        # rio_nome → slug ASCII ('Guaíba'→'guaiba'); fallback para station_code / CodEstacao
        if "rio_nome" in df.columns:
            df["_rio_id"] = df["rio_nome"].apply(_rio_slug)
        elif "station_code" in df.columns:
            df["_rio_id"] = df["station_code"].astype(str)
        elif "CodEstacao" in df.columns:
            df["_rio_id"] = df["CodEstacao"].astype(str)
        else:
            df["_rio_id"] = pd.Series("", index=df.index, dtype=str)
        df["_level"]   = pd.to_numeric(df.get("nivel_m", None), errors="coerce")
        df["_flow"]    = pd.to_numeric(df.get("vazao_m3s", None), errors="coerce")
        # status: usa coluna já computada pelo coletor ou deixa NULL
        df["_status"]  = df.get("status", df.get("alert_level", pd.Series(dtype=str)))
        # live_river_levels tem PK = rio_id (snapshot, 1 linha por rio).
        # Postgres proíbe atualizar a mesma linha 2x no mesmo INSERT
        # ("ON CONFLICT DO UPDATE command cannot affect row a second time"),
        # então o PG recebe só a leitura MAIS RECENTE de cada rio.
        # A série completa vai para o R2 (df original, sem dedup).
        df_db = (
            df.dropna(subset=["_ts", "_rio_id", "_level"])
            .sort_values("_ts")
            .drop_duplicates(subset="_rio_id", keep="last")
        )

        # ── Fluxo A: PostgreSQL ────────────────────────────────────────────
        conn = _pg_connect()
        if conn is not None:
            try:
                _ensure_schema(conn)
                # Schema Supabase: rio_id, rio_nome, nivel_atual_m, vazao_m3s,
                #   cota_atencao_m NOT NULL, cota_alerta_m NOT NULL,
                #   cota_emergencia_m NOT NULL, status, percentual_cota, timestamp
                rows = [
                    (
                        str(r["_rio_id"]),
                        str(r["rio_nome"]) if pd.notna(r.get("rio_nome")) else str(r["_rio_id"]),
                        float(r["_level"]),
                        float(r["_flow"]) if pd.notna(r.get("_flow")) else None,
                        float(r["cota_atencao_m"]) if pd.notna(r.get("cota_atencao_m")) else 0.0,
                        float(r["cota_alerta_m"])   if pd.notna(r.get("cota_alerta_m"))   else 0.0,
                        float(r["cota_emergencia_m"]) if pd.notna(r.get("cota_emergencia_m")) else 0.0,
                        str(r["_status"]) if pd.notna(r.get("_status")) else None,
                        None,   # percentual_cota
                        r["_ts"].to_pydatetime(),
                    )
                    for _, r in df_db.iterrows()
                ]
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, """
                        INSERT INTO live_river_levels
                            (rio_id, rio_nome, nivel_atual_m, vazao_m3s,
                             cota_atencao_m, cota_alerta_m, cota_emergencia_m,
                             status, percentual_cota, "timestamp")
                        VALUES %s
                        ON CONFLICT (rio_id) DO UPDATE SET
                            rio_nome          = EXCLUDED.rio_nome,
                            nivel_atual_m     = EXCLUDED.nivel_atual_m,
                            vazao_m3s         = EXCLUDED.vazao_m3s,
                            cota_atencao_m    = EXCLUDED.cota_atencao_m,
                            cota_alerta_m     = EXCLUDED.cota_alerta_m,
                            cota_emergencia_m = EXCLUDED.cota_emergencia_m,
                            status            = EXCLUDED.status,
                            percentual_cota   = EXCLUDED.percentual_cota,
                            "timestamp"       = EXCLUDED."timestamp",
                            updated_at        = NOW()
                    """, rows, page_size=500)
                conn.commit()
                result.pg_rows = len(rows)
                result.pg_ok = True
                logger.info(f"    PG live_river_levels: {result.pg_rows:,} linhas")
            except psycopg2.Error as exc:
                result.pg_error = str(exc)
                logger.warning(f"    PG live_river_levels: {exc}")
                conn.rollback()
            finally:
                conn.close()

        # ── Fluxo B: R2 ───────────────────────────────────────────────────
        s3 = _r2_client()
        if s3 is not None:
            _r2_upload(df, "live_river_levels", result, s3)

        result.duration_s = time.monotonic() - t0
        result.log_summary()
        return result

    # ── write_river_status ─────────────────────────────────────────────────

    def write_river_status(self, df: pd.DataFrame) -> WriteResult:
        """Persiste status operacional atual dos rios.

        PG: UPSERT em ``live_river_levels`` (PK: river, segment).
        R2: ``historico/live_river_levels/status/…``

        Args:
            df: DataFrame com rio_nome/river, station_code, current_level_m,
                alert_level/status.

        Returns:
            WriteResult com estatísticas.
        """
        result = WriteResult(table="live_river_levels")
        t0 = time.monotonic()

        if df.empty:
            result.duration_s = time.monotonic() - t0
            return result

        conn = _pg_connect()
        if conn is not None:
            try:
                _ensure_schema(conn)
                # Mapeamento: rio_nome slug ('Guaíba'→'guaiba') ou fallback station_code
                # Schema: rio_id, rio_nome, nivel_atual_m, vazao_m3s,
                #   cota_atencao_m NOT NULL, cota_alerta_m NOT NULL, cota_emergencia_m NOT NULL,
                #   status, percentual_cota, timestamp
                def _cota(row: pd.Series, col: str) -> float:
                    v = row.get(col)
                    return float(v) if v is not None and pd.notna(v) else 0.0

                rows = [
                    (
                        _rio_slug(r.get("rio_nome") or r.get("river") or r.get("station_code") or ""),
                        str(r.get("rio_nome") or r.get("river") or ""),  # rio_nome (original)
                        float(r["current_level_m"]) if pd.notna(r.get("current_level_m")) else None,
                        None,   # vazao_m3s
                        _cota(r, "cota_atencao_m"),
                        _cota(r, "cota_alerta_m"),
                        _cota(r, "cota_emergencia_m"),
                        str(r.get("alert_level") or r.get("status") or "NORMAL"),
                        float(r["pct_cota_alerta"]) if pd.notna(r.get("pct_cota_alerta")) else None,
                        datetime.now(tz=timezone.utc),
                    )
                    for _, r in df.iterrows()
                ]
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, """
                        INSERT INTO live_river_levels
                            (rio_id, rio_nome, nivel_atual_m, vazao_m3s,
                             cota_atencao_m, cota_alerta_m, cota_emergencia_m,
                             status, percentual_cota, "timestamp")
                        VALUES %s
                        ON CONFLICT (rio_id) DO UPDATE SET
                            rio_nome          = EXCLUDED.rio_nome,
                            nivel_atual_m     = EXCLUDED.nivel_atual_m,
                            cota_atencao_m    = EXCLUDED.cota_atencao_m,
                            cota_alerta_m     = EXCLUDED.cota_alerta_m,
                            cota_emergencia_m = EXCLUDED.cota_emergencia_m,
                            status            = EXCLUDED.status,
                            percentual_cota   = EXCLUDED.percentual_cota,
                            "timestamp"       = EXCLUDED."timestamp",
                            updated_at        = NOW()
                    """, rows, page_size=200)
                conn.commit()
                result.pg_rows = len(rows)
                result.pg_ok = True
                logger.info(f"    PG live_river_levels (status): {result.pg_rows:,} linhas")
            except psycopg2.Error as exc:
                result.pg_error = str(exc)
                logger.warning(f"    PG live_river_levels (status): {exc}")
                conn.rollback()
            finally:
                conn.close()

        s3 = _r2_client()
        if s3 is not None:
            _r2_upload(df, "live_river_levels", result, s3)

        result.duration_s = time.monotonic() - t0
        result.log_summary()
        return result

    # ── write_water_quality ────────────────────────────────────────────────

    def write_water_quality(self, df: pd.DataFrame) -> WriteResult:
        """Persiste parâmetros físico-químicos de qualidade da água (ANA QA).

        PG: UPSERT em ``water_quality`` (PK: station_id, ts).
        R2: ``historico/water_quality/…``

        Args:
            df: DataFrame com station_id/station_code, ts/timestamp, ph,
                turbidity_ntu, do_mgl, temperature_c, conductivity_uS.

        Returns:
            WriteResult com estatísticas.
        """
        result = WriteResult(table="water_quality")
        t0 = time.monotonic()

        if df.empty:
            result.duration_s = time.monotonic() - t0
            return result

        df = df.copy()
        # Aliases de coluna
        sid_col  = next((c for c in ("station_id", "station_code", "CodEstacao") if c in df.columns), None)
        ts_col   = next((c for c in ("ts", "timestamp", "data_hora") if c in df.columns), None)
        turb_col = next((c for c in ("turbidity_ntu", "turbidez_ntu", "turbidez") if c in df.columns), None)
        od_col   = next((c for c in ("do_mgl", "od_mgl", "oxigenio_mgl") if c in df.columns), None)
        cond_col = next((c for c in ("conductivity_us", "conductivity_uS", "condutividade_uS") if c in df.columns), None)
        temp_col = next((c for c in ("temperature_c", "temperatura_c", "temperatura") if c in df.columns), None)

        if not sid_col or not ts_col:
            logger.warning("write_water_quality: station_id ou ts ausente — skip PG.")
        else:
            conn = _pg_connect()
            if conn is not None:
                try:
                    _ensure_schema(conn)

                    def _fv(row: pd.Series, col: Optional[str]) -> Optional[float]:
                        return float(row[col]) if col and col in row and pd.notna(row[col]) else None

                    rows = [
                        (
                            str(r[sid_col]),
                            pd.to_datetime(r[ts_col], utc=True).to_pydatetime(),
                            _fv(r, "ph"),
                            _fv(r, turb_col),
                            _fv(r, od_col),
                            _fv(r, temp_col),
                            _fv(r, cond_col),
                            "ANA",
                        )
                        for _, r in df.iterrows()
                        if pd.notna(r.get(sid_col)) and pd.notna(r.get(ts_col))
                    ]
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_values(cur, """
                            INSERT INTO water_quality
                                (station_id, ts, ph, turbidity_ntu, do_mgl,
                                 temperature_c, conductivity_us, source)
                            VALUES %s
                            ON CONFLICT (station_id, ts) DO UPDATE SET
                                ph             = EXCLUDED.ph,
                                turbidity_ntu  = EXCLUDED.turbidity_ntu,
                                do_mgl         = EXCLUDED.do_mgl,
                                temperature_c  = EXCLUDED.temperature_c,
                                conductivity_us= EXCLUDED.conductivity_us
                        """, rows, page_size=500)
                    conn.commit()
                    result.pg_rows = len(rows)
                    result.pg_ok = True
                except psycopg2.Error as exc:
                    result.pg_error = str(exc)
                    logger.warning(f"    PG water_quality: {exc}")
                    conn.rollback()
                finally:
                    conn.close()

        s3 = _r2_client()
        if s3 is not None:
            _r2_upload(df, "water_quality", result, s3)

        result.duration_s = time.monotonic() - t0
        result.log_summary()
        return result

    # ── write_forecasts ────────────────────────────────────────────────────

    def write_forecasts(
        self,
        df_hourly: pd.DataFrame,
        df_daily: pd.DataFrame,
        path_hourly: Any = None,   # ignorado — sem arquivo local
        path_daily: Any = None,    # ignorado — sem arquivo local
    ) -> WriteResult:
        """Persiste previsões NWP horárias da Open-Meteo.

        PG: UPSERT em ``forecasts`` (PK: location_name, model_source, valid_ts).
        R2: hourly em ``historico/forecasts/…``, daily em ``historico/forecasts_daily/…``

        Args:
            df_hourly: DataFrame horário com colunas Open-Meteo normalizadas.
            df_daily: DataFrame diário (enviado ao R2, sem tabela PG própria).
            path_hourly: Ignorado (sem arquivo local).
            path_daily: Ignorado (sem arquivo local).

        Returns:
            WriteResult baseado no hourly.
        """
        result = WriteResult(table="forecasts")
        t0 = time.monotonic()

        # ── Fluxo A: PostgreSQL ────────────────────────────────────────────
        if not df_hourly.empty:
            conn = _pg_connect()
            if conn is not None:
                try:
                    _ensure_schema(conn)

                    def _fv(row: pd.Series, col: str) -> Optional[float]:
                        v = row.get(col)
                        return float(v) if v is not None and pd.notna(v) else None

                    rows = [
                        (
                            str(r.get("location_name", "")),
                            _fv(r, "lat"),
                            _fv(r, "lon"),
                            pd.to_datetime(r["forecast_ts"], utc=True).to_pydatetime()
                            if pd.notna(r.get("forecast_ts")) else None,
                            pd.to_datetime(r["valid_ts"], utc=True).to_pydatetime(),
                            _fv(r, "rain_mm"),
                            _fv(r, "temperature"),
                            _fv(r, "wind_speed"),
                            _fv(r, "cape_j_kg"),
                            _fv(r, "lifted_index"),
                            _fv(r, "k_index"),
                            str(r.get("model_source", "openmeteo")),
                        )
                        for _, r in df_hourly.iterrows()
                        if pd.notna(r.get("valid_ts"))
                    ]
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_values(cur, """
                            INSERT INTO forecasts
                                (location_name, lat, lon, forecast_ts, valid_ts,
                                 rain_mm, temperature, wind_speed, cape_j_kg,
                                 lifted_index, k_index, model_source)
                            VALUES %s
                            ON CONFLICT (location_name, model_source, valid_ts)
                            DO UPDATE SET
                                forecast_ts  = EXCLUDED.forecast_ts,
                                lat          = EXCLUDED.lat,
                                lon          = EXCLUDED.lon,
                                rain_mm      = EXCLUDED.rain_mm,
                                temperature  = EXCLUDED.temperature,
                                wind_speed   = EXCLUDED.wind_speed,
                                cape_j_kg    = EXCLUDED.cape_j_kg,
                                lifted_index = EXCLUDED.lifted_index,
                                k_index      = EXCLUDED.k_index,
                                updated_at   = NOW()
                        """, rows, page_size=500)
                    conn.commit()
                    result.pg_rows = len(rows)
                    result.pg_ok = True
                    logger.info(f"    PG forecasts: {result.pg_rows:,} linhas")
                except psycopg2.Error as exc:
                    result.pg_error = str(exc)
                    logger.warning(f"    PG forecasts: {exc}")
                    conn.rollback()
                finally:
                    conn.close()

        # ── Fluxo B: R2 ───────────────────────────────────────────────────
        s3 = _r2_client()
        if s3 is not None:
            if not df_hourly.empty:
                _r2_upload(df_hourly, "forecasts", result, s3)
            if not df_daily.empty:
                r2_daily = WriteResult(table="forecasts_daily")
                _r2_upload(df_daily, "forecasts_daily", r2_daily, s3)
                if r2_daily.r2_ok:
                    logger.info(f"    R2 forecasts_daily: {r2_daily.r2_key.split('/')[-1]}")

        result.duration_s = time.monotonic() - t0
        result.log_summary()
        return result

    # ── write_stations ─────────────────────────────────────────────────────

    def write_stations(
        self,
        df: pd.DataFrame,
        path: Any = None,                        # ignorado
        extra_paths: Optional[list] = None,      # ignorado
    ) -> WriteResult:
        """Persiste metadados de estações INMET/ANA.

        PG: UPSERT em ``stations`` (PK: station_id).
        R2: ``historico/stations/…``

        Args:
            df: DataFrame de inventário (station_id, name, lat, lon, …).
            path: Ignorado (sem arquivo local).
            extra_paths: Ignorado.

        Returns:
            WriteResult com estatísticas.
        """
        result = WriteResult(table="stations")
        t0 = time.monotonic()

        if df.empty:
            result.duration_s = time.monotonic() - t0
            return result

        df = df.copy()
        if "source" not in df.columns:
            df["source"] = "INMET"
        if "name" not in df.columns:
            df["name"] = df["station_id"].astype(str)

        conn = _pg_connect()
        if conn is not None:
            try:
                _ensure_schema(conn)
                # Schema Supabase: station_id, nome, municipio, estado,
                #                 latitude, longitude, fonte, ativa
                rows = [
                    (
                        str(r.get("station_id", "")),
                        str(r.get("name") or r.get("station_id") or ""),  # nome
                        str(r.get("municipality") or r.get("municipio") or ""),  # municipio
                        str(r.get("state") or r.get("estado") or "RS")[:2],     # estado (char 2)
                        float(r.get("lat") or r.get("latitude") or 0.0),         # latitude
                        float(r.get("lon") or r.get("longitude") or 0.0),        # longitude
                        str(r.get("source", "INMET")),                            # fonte
                        bool(r.get("active", True)),                              # ativa
                    )
                    for _, r in df.iterrows()
                    if pd.notna(r.get("station_id"))
                ]
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, """
                        INSERT INTO stations
                            (station_id, nome, municipio, estado,
                             latitude, longitude, fonte, ativa)
                        VALUES %s
                        ON CONFLICT (station_id) DO UPDATE SET
                            nome      = EXCLUDED.nome,
                            latitude  = EXCLUDED.latitude,
                            longitude = EXCLUDED.longitude,
                            ativa     = EXCLUDED.ativa
                    """, rows, page_size=200)
                conn.commit()
                result.pg_rows = len(rows)
                result.pg_ok = True
                logger.info(f"    PG stations: {result.pg_rows:,} estações")
            except psycopg2.Error as exc:
                result.pg_error = str(exc)
                logger.warning(f"    PG stations: {exc}")
                conn.rollback()
            finally:
                conn.close()

        s3 = _r2_client()
        if s3 is not None:
            _r2_upload(df, "stations", result, s3)

        result.duration_s = time.monotonic() - t0
        result.log_summary()
        return result

    # ── write_rain_readings ────────────────────────────────────────────────

    def write_rain_readings(
        self,
        df: pd.DataFrame,
        path: Any = None,   # ignorado
    ) -> WriteResult:
        """Persiste leituras brutas de chuva/meteorologia INMET.

        PG: INSERT ON CONFLICT DO NOTHING em ``live_rain_readings``
        (PK: station_id, timestamp).
        R2: ``historico/live_rain_readings/…``

        Mapeamento de colunas INMET → Supabase:
          ts          → timestamp
          rain_1h_mm  → precip_1h
          temperature → temperatura
          humidity    → umidade
          precip_6h / precip_24h → NULL (não coletado em tempo real)

        Args:
            df: DataFrame INMET com station_id, ts, rain_1h_mm,
                temperature, humidity, pressure_hpa, wind_speed, wind_dir.
            path: Ignorado (sem arquivo local).

        Returns:
            WriteResult com estatísticas.
        """
        result = WriteResult(table="live_rain_readings")
        t0 = time.monotonic()

        if df.empty:
            result.duration_s = time.monotonic() - t0
            return result

        conn = _pg_connect()
        if conn is not None:
            try:
                _ensure_schema(conn)
                # Deduplica para 1 linha por station_id (leitura mais recente)
                # live_rain_readings tem PK = station_id (tabela de snapshot live)
                df_rd = (
                    df.dropna(subset=["station_id", "ts"])
                    .copy()
                    .sort_values("ts")
                    .groupby("station_id", as_index=False)
                    .last()
                )

                def _fv(row: pd.Series, col: str) -> Optional[float]:
                    v = row.get(col)
                    return float(v) if v is not None and pd.notna(v) else None

                rows = [
                    (
                        str(r["station_id"]),
                        _fv(r, "rain_1h_mm"),    # precip_1h
                        None,                     # precip_6h  — não coletado
                        None,                     # precip_24h — não coletado
                        _fv(r, "temperature"),    # temperatura
                        _fv(r, "humidity"),       # umidade
                        pd.to_datetime(r["ts"], utc=True).to_pydatetime(),  # timestamp
                    )
                    for _, r in df_rd.iterrows()
                ]
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, """
                        INSERT INTO live_rain_readings
                            (station_id, precip_1h, precip_6h, precip_24h,
                             temperatura, umidade, "timestamp")
                        VALUES %s
                        ON CONFLICT (station_id) DO UPDATE SET
                            precip_1h   = EXCLUDED.precip_1h,
                            precip_6h   = EXCLUDED.precip_6h,
                            precip_24h  = EXCLUDED.precip_24h,
                            temperatura = EXCLUDED.temperatura,
                            umidade     = EXCLUDED.umidade,
                            "timestamp" = EXCLUDED."timestamp",
                            updated_at  = NOW()
                    """, rows, page_size=200)
                conn.commit()
                result.pg_rows = len(rows)
                result.pg_ok = True
                logger.info(f"    PG live_rain_readings: {result.pg_rows:,} estações (snapshot live)")
            except psycopg2.Error as exc:
                result.pg_error = str(exc)
                logger.warning(f"    PG live_rain_readings: {exc}")
                conn.rollback()
            finally:
                conn.close()

        s3 = _r2_client()
        if s3 is not None:
            _r2_upload(df, "live_rain_readings", result, s3)

        result.duration_s = time.monotonic() - t0
        result.log_summary()
        return result

    # ── write_gpm_precip ──────────────────────────────────────────────────

    def write_gpm_precip(
        self,
        df: pd.DataFrame,
        path: Any = None,   # ignorado
    ) -> WriteResult:
        """Persiste grade de precipitação GPM IMERG / CHIRPS.

        PG: UPSERT em ``live_gpm_precip`` (PK: lat, lon, timestamp).
        R2: ``historico/live_gpm_precip/…``

        Args:
            df: DataFrame com lat, lon, precip_mm, timestamp, source.
            path: Ignorado.

        Returns:
            WriteResult com estatísticas.
        """
        result = WriteResult(table="live_gpm_precip")
        t0 = time.monotonic()

        if df.empty:
            result.duration_s = time.monotonic() - t0
            return result

        conn = _pg_connect()
        if conn is not None:
            try:
                _ensure_schema(conn)

                # Suporte a lat/lon (GPM V06) e latitude/longitude (GPM V07)
                lat_col = "latitude" if "latitude" in df.columns else "lat"
                lon_col = "longitude" if "longitude" in df.columns else "lon"

                rows = [
                    (
                        f"{float(r[lat_col]):.4f}_{float(r[lon_col]):.4f}",  # lat_lon_key (PK)
                        float(r[lat_col]),
                        float(r[lon_col]),
                        float(r["precip_mm"]) if pd.notna(r.get("precip_mm")) else None,
                        str(r.get("source", "GPM_IMERG_EARLY")),
                        pd.to_datetime(r["timestamp"], utc=True).to_pydatetime(),
                    )
                    for _, r in df.iterrows()
                    if pd.notna(r.get(lat_col)) and pd.notna(r.get(lon_col)) and pd.notna(r.get("timestamp"))
                ]

                if not rows:
                    logger.warning("    PG live_gpm_precip: rows vazio apos filtragem — nada inserido.")
                    result.duration_s = time.monotonic() - t0
                    return result

                # Diagnóstico: log do primeiro registro para confirmar formato
                logger.debug(
                    f"    PG live_gpm_precip — {len(rows):,} rows a inserir | "
                    f"primeiro: lat_lon_key={rows[0][0]} "
                    f"precip={rows[0][3]} ts={rows[0][5]} src={rows[0][4]}"
                )

                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, """
                        INSERT INTO live_gpm_precip
                            (lat_lon_key, latitude, longitude,
                             precip_mm, source, "timestamp")
                        VALUES %s
                        ON CONFLICT (lat_lon_key) DO UPDATE SET
                            precip_mm  = EXCLUDED.precip_mm,
                            source     = EXCLUDED.source,
                            "timestamp"= EXCLUDED."timestamp",
                            updated_at = NOW()
                    """, rows, page_size=2000)

                    # Conta rows afetados ANTES do commit para diagnóstico
                    affected = cur.rowcount

                    # Purga a grade da OUTRA fonte: GPM IMERG (5.600 pts) e o
                    # fallback CHIRPS (17.723 pts) usam grades de lat_lon_key
                    # distintas, então o upsert não sobrescreve a grade antiga
                    # e ela fica órfã. Mantém só a fonte recém-gravada.
                    fontes_gravadas = {r[4] for r in rows}
                    if len(fontes_gravadas) == 1:
                        cur.execute(
                            "DELETE FROM live_gpm_precip WHERE source <> %s",
                            (next(iter(fontes_gravadas)),),
                        )
                        if cur.rowcount:
                            logger.info(
                                f"    PG live_gpm_precip: {cur.rowcount:,} linhas "
                                f"de grade antiga purgadas (fonte != "
                                f"{next(iter(fontes_gravadas))})"
                            )

                conn.commit()

                # Verificação pós-commit: conta efetiva no banco
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM live_gpm_precip")
                    total_in_db: int = cur.fetchone()[0]

                result.pg_rows = affected if affected >= 0 else len(rows)
                result.pg_ok = True
                logger.info(
                    f"    PG live_gpm_precip: {result.pg_rows:,} rows afetados | "
                    f"total na tabela: {total_in_db:,} | "
                    f"destino: {_pg_host_safe()}"
                )
            except psycopg2.Error as exc:
                result.pg_error = str(exc)
                logger.error(f"    PG live_gpm_precip psycopg2.Error: {exc!r}")
                try:
                    conn.rollback()
                except psycopg2.Error:
                    pass
            except (ValueError, TypeError, OverflowError) as exc:
                # Erros de conversão Python na construção de rows — não chegam ao DB
                result.pg_error = f"rows_build_error: {exc}"
                logger.error(f"    PG live_gpm_precip: erro ao montar rows — {exc!r}")
            finally:
                conn.close()

        s3 = _r2_client()
        if s3 is not None:
            _r2_upload(df, "live_gpm_precip", result, s3)

        result.duration_s = time.monotonic() - t0
        result.log_summary()
        return result
