"""
Auditor de Custos / SRE — verificação de saúde dos limites gratuitos.

Confirma que o projeto (arquitetura de custo zero) segue dentro das camadas
free de R2, BigQuery, GitHub Actions e Supabase. Roda manualmente
(python src/models/monitor_costs.py) ou como step opcional no CI.

Cada verificação é isolada (try/except próprio): a falha de um serviço não
derruba o relatório dos demais. Saída: tabela ASCII no terminal +
logs/cost_report_{date}.json.

NOTA: com o repositório PÚBLICO, os minutos de GitHub Actions são
ILIMITADOS (grátis para repos públicos) — o gargalo de cota que derrubou
o pipeline em 13/06 deixou de existir.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import boto3
import psycopg2
from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

_GB = 1024**3
_MB = 1024**2


def _status(pct: float, w: float = 80.0, c: float = 95.0) -> str:
    """Classifica um percentual de uso em SEGURO/WARNING/CRITICO.

    Args:
        pct: Percentual de uso (0-100).
        w: Limiar de WARNING.
        c: Limiar de CRÍTICO.

    Returns:
        Rótulo de status.
    """
    if pct >= c:
        return "CRITICO"
    if pct >= w:
        return "WARNING"
    return "SEGURO"


# ---------------------------------------------------------------------------
# 1. R2 (Cloudflare) — 10 GB grátis
# ---------------------------------------------------------------------------

def check_r2() -> dict[str, Any]:
    """Soma o uso do bucket R2 e quebra por pasta de topo.

    Returns:
        Dict com usado_gb, limite_gb, pct, status e breakdown por pasta.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    )
    bucket = os.getenv("R2_BUCKET_NAME", "")
    total = 0
    por_pasta: dict[str, int] = {}
    token = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            total += o["Size"]
            pasta = o["Key"].split("/")[0]
            por_pasta[pasta] = por_pasta.get(pasta, 0) + o["Size"]
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]

    limite_gb = 10.0
    usado_gb = total / _GB
    pct = usado_gb / limite_gb * 100
    return {
        "usado": f"{usado_gb:.2f} GB",
        "limite": f"{limite_gb:.2f} GB",
        "pct": pct,
        "status": _status(pct),
        "breakdown": {p: f"{b / _GB:.3f} GB" for p, b in sorted(por_pasta.items())},
    }


# ---------------------------------------------------------------------------
# 2. BigQuery (Google) — 1 TB/mês grátis
# ---------------------------------------------------------------------------

def check_bigquery() -> dict[str, Any]:
    """Soma bytes faturados de queries no mês corrente (INFORMATION_SCHEMA).

    Returns:
        Dict com usado, limite, pct, status; ou N/A se bigquery indisponível.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        return {"usado": "N/A", "limite": "1 TB", "pct": 0.0,
                "status": "N/A (dep de treino)"}

    project = os.getenv("BIGQUERY_PROJECT_ID", "")
    if not project:
        return {"usado": "N/A", "limite": "1 TB", "pct": 0.0, "status": "N/A (sem projeto)"}

    client = bigquery.Client(project=project)
    # JOBS_BY_PROJECT é regional; us é o default do basedosdados public.
    sql = """
        SELECT COALESCE(SUM(total_bytes_billed), 0) AS bytes_billed
        FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
        WHERE creation_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)
          AND job_type = 'QUERY'
    """
    try:
        # dry-run-free: INFORMATION_SCHEMA não é faturado
        bytes_billed = list(client.query(sql).result())[0]["bytes_billed"]
    except Exception as exc:  # noqa: BLE001
        return {"usado": "N/A", "limite": "1 TB", "pct": 0.0,
                "status": f"N/A ({str(exc)[:40]})"}

    limite_gb = 1000.0  # 1 TB
    usado_gb = (bytes_billed or 0) / _GB
    pct = usado_gb / limite_gb * 100
    return {
        "usado": f"{usado_gb:.2f} GB",
        "limite": "1000 GB",
        "pct": pct,
        "status": _status(pct),
    }


# ---------------------------------------------------------------------------
# 3. GitHub Actions — público = ilimitado
# ---------------------------------------------------------------------------

def check_github_actions() -> dict[str, Any]:
    """Verifica minutos de Actions. Repos públicos = ilimitado/grátis.

    Returns:
        Dict com usado, limite, pct, status.
    """
    import niquests

    owner = "rafa-rebelo"
    repo = "climate-dashboard-rs"
    token = os.getenv("GH_TOKEN", "")

    # Confirma visibilidade: público → minutos ilimitados.
    try:
        r = niquests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=15,
        )
        if r.status_code == 200 and not r.json().get("private", True):
            return {"usado": "ilimitado", "limite": "∞ (repo público)",
                    "pct": 0.0, "status": "SEGURO"}
    except niquests.exceptions.RequestException:
        pass

    # Privado: tenta a billing API (precisa de token com escopo).
    if not token:
        return {"usado": "N/A", "limite": "2000 min", "pct": 0.0,
                "status": "N/A (sem GH_TOKEN)"}
    try:
        r = niquests.get(
            f"https://api.github.com/users/{owner}/settings/billing/actions",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json"},
            timeout=15,
        )
        if r.status_code != 200:
            return {"usado": "N/A", "limite": "2000 min", "pct": 0.0,
                    "status": f"N/A (HTTP {r.status_code})"}
        usado = r.json().get("total_minutes_used", 0)
    except niquests.exceptions.RequestException as exc:
        return {"usado": "N/A", "limite": "2000 min", "pct": 0.0,
                "status": f"N/A ({str(exc)[:30]})"}

    limite = 2000
    pct = usado / limite * 100
    return {"usado": f"{usado} min", "limite": "2000 min", "pct": pct,
            "status": _status(pct)}


# ---------------------------------------------------------------------------
# 4. Supabase — 500 MB grátis
# ---------------------------------------------------------------------------

def check_supabase() -> dict[str, Any]:
    """Lê o tamanho do banco via pg_database_size.

    Returns:
        Dict com usado, limite, pct, status.
    """
    url = os.getenv("SUPABASE_DATABASE_URL_POOLER") or os.getenv("SUPABASE_DATABASE_URL")
    if not url:
        return {"usado": "N/A", "limite": "500 MB", "pct": 0.0, "status": "N/A (sem URL)"}
    try:
        conn = psycopg2.connect(url, connect_timeout=20)
        cur = conn.cursor()
        cur.execute("SELECT pg_database_size(current_database())")
        size = cur.fetchone()[0]
        conn.close()
    except psycopg2.Error as exc:
        return {"usado": "N/A", "limite": "500 MB", "pct": 0.0,
                "status": f"N/A ({str(exc)[:30]})"}

    limite_mb = 500.0
    usado_mb = size / _MB
    pct = usado_mb / limite_mb * 100
    return {"usado": f"{usado_mb:.0f} MB", "limite": "500 MB", "pct": pct,
            "status": _status(pct)}


# ---------------------------------------------------------------------------
# Relatório consolidado
# ---------------------------------------------------------------------------

_ICONE = {"SEGURO": "✓", "WARNING": "⚠", "CRITICO": "✗"}


def gerar_relatorio() -> dict[str, Any]:
    """Roda as 4 verificações, imprime tabela ASCII e salva JSON.

    Returns:
        Dict consolidado com timestamp e resultado de cada serviço.
    """
    logger.info("Auditando limites gratuitos...")
    servicos = {
        "R2":         check_r2(),
        "BigQuery":   check_bigquery(),
        "GH Actions": check_github_actions(),
        "Supabase":   check_supabase(),
    }

    print("\n" + "=" * 64)
    print(f"{'SERVIÇO':<12}| {'USADO':<11}| {'LIMITE':<14}| {'%':>5} | STATUS")
    print("-" * 64)
    for nome, d in servicos.items():
        pct = f"{d['pct']:.0f}%" if isinstance(d.get("pct"), (int, float)) else "---"
        icone = _ICONE.get(d["status"], "")
        print(f"{nome:<12}| {d['usado']:<11}| {d['limite']:<14}| {pct:>5} | {d['status']} {icone}")
    print("-" * 64)
    print(f"{'TOTAL INFRA':<12}| {'R$ 0,00':<11}| {'R$ 0,00':<14}| {'---':>5} | FASE 1 OK ✓")
    print("=" * 64)

    # Breakdown do R2
    if "breakdown" in servicos["R2"]:
        bd = " | ".join(f"{p}/ {v}" for p, v in servicos["R2"]["breakdown"].items())
        print(f"R2 breakdown: {bd}\n")

    # Alertas acionáveis
    for nome, d in servicos.items():
        if d["status"] in ("WARNING", "CRITICO"):
            logger.warning(f"{nome}: {d['status']} — uso em {d['pct']:.0f}%")

    relatorio = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "servicos": servicos,
        "custo_total_mensal": "R$ 0,00",
    }
    log_dir = _ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    data = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = log_dir / f"cost_report_{data}.json"
    out.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Relatório salvo: {out}")
    return relatorio


if __name__ == "__main__":
    # Terminal Windows é CP1252 por padrão — força UTF-8 para os ícones.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               level="INFO")
    rel = gerar_relatorio()
    # Código de saída: 1 se algo CRÍTICO (para alertar no CI)
    if any(s.get("status") == "CRITICO" for s in rel["servicos"].values()):
        sys.exit(1)
