"""Monitor de estabilização — amostra o lag das tabelas live_* do Supabase.

Uso: python scripts/stability_monitor.py [n_amostras] [intervalo_s]
Acrescenta linhas em data/stability_log.csv (gitignored via data/).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

_LOG = _ROOT / "data" / "stability_log.csv"
_TABLES = ("live_gpm_precip", "live_rain_readings", "live_river_levels")
_FIELDS = ["sample_utc"] + [f"{t}_lag_min" for t in _TABLES] + ["erro"]


def sample() -> dict[str, object]:
    """Coleta uma amostra de lag (minutos desde o último updated_at).

    Returns:
        Dict com sample_utc, lag por tabela e campo erro em falhas.
    """
    now = datetime.now(timezone.utc)
    row: dict[str, object] = {"sample_utc": now.isoformat(timespec="seconds")}
    try:
        url = os.getenv("SUPABASE_DATABASE_URL_POOLER") or os.getenv("SUPABASE_DATABASE_URL")
        conn = psycopg2.connect(url, connect_timeout=20)
        cur = conn.cursor()
        for t in _TABLES:
            cur.execute(f"SELECT MAX(updated_at) FROM {t}")
            mx = cur.fetchone()[0]
            row[f"{t}_lag_min"] = (
                round((now - mx).total_seconds() / 60, 1) if mx else None
            )
        conn.close()
    except psycopg2.Error as exc:
        row["erro"] = str(exc)[:100]
    return row


def main() -> None:
    """Roda n amostras com intervalo fixo, acrescentando ao CSV."""
    n        = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 900

    _LOG.parent.mkdir(parents=True, exist_ok=True)
    novo = not _LOG.exists()
    for i in range(n):
        row = sample()
        with _LOG.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_FIELDS)
            if novo:
                w.writeheader()
                novo = False
            w.writerow(row)
        print(f"[{i + 1}/{n}] {row}", flush=True)
        if i < n - 1:
            time.sleep(interval)


if __name__ == "__main__":
    main()
