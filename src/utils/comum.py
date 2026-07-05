"""
Agente 1 — Arquiteto de Dados
Helpers compartilhados do projeto (fonte ÚNICA — não duplicar).

Consolidação da auditoria de 05/07/2026: _pg_connect existia em 3 módulos
(com timeouts JÁ divergentes: 10 s vs 20 s), _r2_client em 2, slug de rio
em 4 e _safe_float em 2. Este módulo é o canônico; os antigos pontos de
definição re-exportam daqui para não quebrar imports existentes.

Também é a fonte de verdade da lista de rios com modelo LSTM (RIOS_LSTM),
antes cravada em inference/api/train separadamente.
"""

from __future__ import annotations

import math
import os
import unicodedata
from typing import Any, Optional

from loguru import logger

# Rios com modelo LSTM publicável (slug = rio_id em river_ai_forecasts).
# A inferência descobre os .pt no R2 dinamicamente; esta lista é o fallback
# conhecido e a referência para API/treino.
RIOS_LSTM: list[str] = [
    "guaiba", "jacui", "taquari", "sinos", "camaqua",
    "cai", "ibicui", "ijui", "gravatai", "pardo",
]


def resolve_pg_url() -> str:
    """URL de conexão Postgres lida em runtime (nunca cachear em módulo).

    Prioridade: SUPABASE_DATABASE_URL_POOLER (Session Pooler IPv4, GitHub
    Actions) > SUPABASE_DATABASE_URL (direta). Leitura lazy garante que
    variáveis injetadas pelo runner após o import sejam capturadas.

    Returns:
        URL de conexão, ou string vazia se nenhuma variável configurada.
    """
    return (
        os.getenv("SUPABASE_DATABASE_URL_POOLER")
        or os.getenv("SUPABASE_DATABASE_URL")
        or ""
    )


def pg_connect(
    connect_timeout: int = 15,
    statement_timeout_ms: int | None = None,
) -> Optional[Any]:
    """Abre conexão psycopg2 com o Supabase (None em falha/config ausente).

    Args:
        connect_timeout: Timeout de conexão em segundos (padrão unificado 15 —
            a auditoria achou 10 s e 20 s divergentes nas cópias antigas).
        statement_timeout_ms: Se definido, aplica ``statement_timeout`` na
            sessão — protege o chamador de query presa (essencial na API).

    Returns:
        Conexão psycopg2 (autocommit=False), ou None se indisponível.
    """
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 não disponível — pip install psycopg2-binary")
        return None

    url = resolve_pg_url()
    if not url:
        logger.warning("PG: SUPABASE_DATABASE_URL_POOLER/SUPABASE_DATABASE_URL ausentes.")
        return None
    try:
        opts = (f"-c statement_timeout={int(statement_timeout_ms)}"
                if statement_timeout_ms else None)
        conn = psycopg2.connect(url, connect_timeout=connect_timeout,
                                options=opts)
        conn.autocommit = False
        return conn
    except psycopg2.Error as exc:
        at = url.rfind("@")
        logger.warning(f"PG: falha na conexão {url[at:] if at != -1 else '<url>'} — {exc}")
        return None


def r2_client(connect_timeout: int = 10, read_timeout: int = 60) -> Optional[Any]:
    """Cliente boto3 para o Cloudflare R2 (None se env/boto3 ausentes).

    Lê as variáveis em runtime (R2_ENDPOINT_URL/R2_ACCESS_KEY_ID/
    R2_SECRET_ACCESS_KEY) — mesma garantia de injeção tardia do runner.

    Args:
        connect_timeout: Timeout de conexão (s).
        read_timeout: Timeout de leitura (s).

    Returns:
        boto3 S3 client configurado, ou None.
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        logger.warning("boto3 não disponível — pip install boto3")
        return None

    endpoint = os.getenv("R2_ENDPOINT_URL", "")
    key = os.getenv("R2_ACCESS_KEY_ID", "")
    secret = os.getenv("R2_SECRET_ACCESS_KEY", "")
    if not (endpoint and key and secret):
        logger.warning("R2: variáveis R2_ENDPOINT_URL/KEY não configuradas.")
        return None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="auto",
        config=BotoConfig(connect_timeout=connect_timeout, read_timeout=read_timeout),
    )


def rio_slug(name: Any) -> str:
    """Slug ASCII minúsculo de nome de rio ('Guaíba' → 'guaiba').

    Args:
        name: Nome como vem da API/config (aceita qualquer tipo).

    Returns:
        Slug sem acentos, minúsculo, sem espaços nas pontas.
    """
    s = unicodedata.normalize("NFD", str(name)).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def safe_float(val: Any) -> float | None:
    """Converte para float; None para None/NaN/inconversível.

    Args:
        val: Valor de origem (DB, CSV, JSON).

    Returns:
        float, ou None.
    """
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
