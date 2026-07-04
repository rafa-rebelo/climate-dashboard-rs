"""
Agente 1 — Arquiteto de Dados
Coletor DCRS — Rede Hidrometeorológica da Defesa Civil RS (GraphQL, ~5 min).

Fonte: https://redehidrometeorologica.defesacivil.rs.gov.br/graphql (pública,
sem autenticação). ~130 estações DCRS-xxxxx em 26 bacias hidrográficas do RS,
com nível de rio, chuva acumulada (1h…168h), temperatura, umidade, vento,
pressão, sensação térmica e radiação — cadência de minutos.

Por que ela importa: é a ÚNICA fonte com nível de rio POR BACIA em tempo real
para rios fora da telemetria ANA (Taquari-Antas tem 20 estações; Sinos 10;
Camaquã 10; Baixo Jacuí 9…). A query `historic` é bloqueada para acesso
anônimo — o histórico é construído por NÓS, acumulando snapshots no R2
(mesma estratégia do CEMADEN), o que futuramente alimenta o treino (Agente 2).

Unidade do rio_nivel: BRUTA da rede — mista por estação (lagoas ~0,3 m;
serra ~860, aparenta cm). Persistimos o valor cru; a heurística de exibição
fica no dashboard, sinalizada.

Persistência híbrida:
  A. Supabase `live_dcrs_stations` (PK codigo, snapshot 1 linha/estação)
     + `stations` (fonte=DCRS, prefixo DCRS- já vem no código).
  B. R2 `historico/live_dcrs/...` (Parquet por run — série p/ histórico).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import niquests
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from database.hybrid_writer import (  # noqa: E402
    HybridWriter,
    WriteResult,
    _pg_connect,
    _r2_client,
    _r2_upload,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

_GRAPHQL_URL = os.getenv(
    "DCRS_GRAPHQL_URL",
    "https://redehidrometeorologica.defesacivil.rs.gov.br/graphql",
)
_CLIENT = os.getenv("DCRS_CLIENT", "casa-militar-defesa-civil-rs")

_session = niquests.Session()
_session.headers.update({
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
})

# Query tags_data (tempo real, todas as estações do client) — introspecção
# 03/07/2026. `Valores{value}` embrulha cada métrica; alarmes de inundação
# trazem as cotas oficiais quando definidas (hoje nulas no client público,
# mas mantidas para o dia em que forem preenchidas).
_QUERY = """
{
  tags_data(clients: ["%s"]) {
    qualle_meteorologia {
      codigo
      timestamp
      name { general local }
      position { bacia latitude longitude }
      data {
        rio {
          rio_nome { value }
          rio_nivel { value }
          rio_vazao { value }
          rio_nivel_tendencia { value }
          rio_alarmes {
            inundacao {
              atencao { value } alerta { value }
              emergencia { value } status { value }
            }
          }
        }
        chuva {
          acumulado {
            h001 { value } h003 { value } h006 { value } h012 { value }
            h024 { value } h048 { value } h072 { value } h168 { value }
          }
        }
        temperatura { atual { value } }
        umidade { atual { value } }
        vento { velocidade_media { value } direcao { value } }
        pressaoatmos { atual { value } }
        senstermica { atual { value } }
        radiacaosolar { atual { value } }
      }
    }
  }
}
""" % _CLIENT


def _v(obj: Any, *caminho: str) -> Optional[float]:
    """Extrai ``caminho -> {value}`` de um dict aninhado, tolerante a None.

    Args:
        obj: Dict raiz (ex.: data.rio).
        caminho: Sequência de chaves até o nó que contém {"value": ...}.

    Returns:
        float do value, ou None se qualquer nível faltar/for nulo.
    """
    atual = obj
    for k in caminho:
        if not isinstance(atual, dict) or atual.get(k) is None:
            return None
        atual = atual[k]
    val = atual.get("value") if isinstance(atual, dict) else None
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _vs(obj: Any, *caminho: str) -> Optional[str]:
    """Como ``_v`` mas para value string (ex.: rio_nome)."""
    atual = obj
    for k in caminho:
        if not isinstance(atual, dict) or atual.get(k) is None:
            return None
        atual = atual[k]
    val = atual.get("value") if isinstance(atual, dict) else None
    return str(val).strip() if val not in (None, "") else None


# ---------------------------------------------------------------------------
# 1. Coleta GraphQL
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(niquests.exceptions.RequestException),
    reraise=True,
)
def fetch_tags_data() -> list[dict]:
    """POST GraphQL tags_data e devolve a lista de estações.

    Returns:
        Lista de dicts Estacao (codigo, timestamp, name, position, data).

    Raises:
        niquests.exceptions.RequestException: Erro de rede após retries.
        ValueError: Resposta com errors do GraphQL.
    """
    resp = _session.post(_GRAPHQL_URL, json={"query": _QUERY}, timeout=90)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise ValueError(f"GraphQL errors: {str(payload['errors'])[:200]}")
    est = ((payload.get("data") or {}).get("tags_data") or {}).get(
        "qualle_meteorologia") or []
    logger.info(f"DCRS: {len(est)} estações no tags_data.")
    return est


def normalize(estacoes: list[dict]) -> pd.DataFrame:
    """Achata as estações GraphQL no DataFrame de live_dcrs_stations.

    Args:
        estacoes: Saída de fetch_tags_data().

    Returns:
        DataFrame com 1 linha por estação (colunas do schema
        live_dcrs_stations). Vazio se nada utilizável.
    """
    linhas: list[dict[str, Any]] = []
    for e in estacoes:
        if not isinstance(e, dict) or not e.get("codigo"):
            continue
        pos = e.get("position") or {}
        nome = e.get("name") or {}
        data = e.get("data") or {}
        rio = data.get("rio") or {}
        inu = ((rio.get("rio_alarmes") or {}).get("inundacao")) or {}
        acc = ((data.get("chuva") or {}).get("acumulado")) or {}
        lat, lon = pos.get("latitude"), pos.get("longitude")
        if lat is None or lon is None:
            continue
        linhas.append({
            "codigo":           str(e["codigo"]),
            "nome":             str(nome.get("general") or e["codigo"]).strip(),
            "local":            str(nome.get("local") or "").strip(),
            "bacia":            str(pos.get("bacia") or "").strip(),
            "latitude":         float(lat),
            "longitude":        float(lon),
            "rio_nome":         _vs(rio, "rio_nome"),
            "rio_nivel":        _v(rio, "rio_nivel"),
            "rio_vazao":        _v(rio, "rio_vazao"),
            "rio_tendencia":    _v(rio, "rio_nivel_tendencia"),
            "cota_atencao":     _v(inu, "atencao"),
            "cota_alerta":      _v(inu, "alerta"),
            "cota_emergencia":  _v(inu, "emergencia"),
            "inundacao_status": _v(inu, "status"),
            "chuva_1h":         _v(acc, "h001"),
            "chuva_3h":         _v(acc, "h003"),
            "chuva_6h":         _v(acc, "h006"),
            "chuva_12h":        _v(acc, "h012"),
            "chuva_24h":        _v(acc, "h024"),
            "chuva_48h":        _v(acc, "h048"),
            "chuva_72h":        _v(acc, "h072"),
            "chuva_168h":       _v(acc, "h168"),
            "temperatura":      _v(data, "temperatura", "atual"),
            "umidade":          _v(data, "umidade", "atual"),
            "vento_vel":        _v(data, "vento", "velocidade_media"),
            "vento_dir":        _v(data, "vento", "direcao"),
            "pressao":          _v(data, "pressaoatmos", "atual"),
            "senstermica":      _v(data, "senstermica", "atual"),
            "radiacao":         _v(data, "radiacaosolar", "atual"),
            "timestamp":        e.get("timestamp"),
        })

    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.drop_duplicates(subset="codigo", keep="last")
    logger.info(
        f"DCRS: {len(df)} estações normalizadas | "
        f"{df['bacia'].nunique()} bacias | "
        f"{df['rio_nivel'].notna().sum()} com nível de rio"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Persistência
# ---------------------------------------------------------------------------

_COLS = [
    "codigo", "nome", "local", "bacia", "latitude", "longitude",
    "rio_nome", "rio_nivel", "rio_vazao", "rio_tendencia",
    "cota_atencao", "cota_alerta", "cota_emergencia", "inundacao_status",
    "chuva_1h", "chuva_3h", "chuva_6h", "chuva_12h", "chuva_24h",
    "chuva_48h", "chuva_72h", "chuva_168h",
    "temperatura", "umidade", "vento_vel", "vento_dir",
    "pressao", "senstermica", "radiacao", "timestamp",
]


def upsert_supabase(df: pd.DataFrame) -> int:
    """UPSERT do snapshot em live_dcrs_stations (PK codigo).

    Args:
        df: Saída de normalize().

    Returns:
        Linhas gravadas (0 em falha de conexão/query).
    """
    if df.empty:
        return 0
    conn = _pg_connect()
    if conn is None:
        logger.warning("Supabase indisponível — snapshot DCRS não gravado no PG.")
        return 0

    import psycopg2
    import psycopg2.extras

    def _cell(v: Any) -> Any:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
        return v

    rows = [tuple(_cell(r[c]) for c in _COLS) for _, r in df.iterrows()]
    updates = ",\n".join(
        f"{c} = EXCLUDED.{c}" for c in _COLS[1:-1]
    ) + ',\n"timestamp" = EXCLUDED."timestamp",\nupdated_at = NOW()'
    cols_sql = ", ".join(f'"{c}"' if c == "timestamp" else c for c in _COLS)
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, f"""
                INSERT INTO live_dcrs_stations ({cols_sql})
                VALUES %s
                ON CONFLICT (codigo) DO UPDATE SET
                {updates}
            """, rows, page_size=200)
        conn.commit()
        logger.success(f"  PG live_dcrs_stations: {len(rows)} estações.")
        return len(rows)
    except psycopg2.Error as exc:
        logger.warning(f"  PG live_dcrs_stations: {exc}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def _build_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Inventário p/ a tabela stations (fonte=DCRS, ids DCRS-xxxxx)."""
    return pd.DataFrame({
        "station_id":   df["codigo"],
        "name":         df["nome"],
        "municipality": df["nome"],   # nome da rede já é o município/local
        "state":        "RS",
        "lat":          df["latitude"],
        "lon":          df["longitude"],
        "source":       "DCRS",
        "active":       True,
    })


# ---------------------------------------------------------------------------
# 3. Orquestração
# ---------------------------------------------------------------------------

def run() -> dict[str, Any]:
    """Executa a coleta DCRS e persiste (Supabase + R2 + stations).

    Returns:
        Dict com estações, bacias, pg_rows e r2_key.
    """
    estacoes = fetch_tags_data()
    df = normalize(estacoes)
    if df.empty:
        return {"estacoes": 0, "bacias": 0, "pg_rows": 0, "r2_key": None}

    pg_rows = upsert_supabase(df)

    r2_key: Optional[str] = None
    s3 = _r2_client()
    if s3 is not None:
        result = WriteResult(table="live_dcrs")
        _r2_upload(df, "live_dcrs", result, s3)
        r2_key = result.r2_key or None

    HybridWriter().write_stations(_build_stations(df))

    return {
        "estacoes": int(len(df)),
        "bacias":   int(df["bacia"].nunique()),
        "pg_rows":  pg_rows,
        "r2_key":   r2_key,
    }


def main() -> int:
    """Entry point: falha vira log (step roda com continue-on-error).

    Returns:
        0 sempre — a rede DCRS é complementar, nunca derruba o pipeline.
    """
    try:
        r = run()
        logger.success(
            f"DCRS OK — {r['estacoes']} estações / {r['bacias']} bacias "
            f"(PG: {r['pg_rows']})."
        )
        return 0
    except niquests.exceptions.RequestException as exc:
        logger.error(f"DCRS — falha de rede após retries: {exc}")
        return 0
    except (ValueError, KeyError) as exc:
        logger.error(f"DCRS — resposta inesperada: {exc}")
        return 0


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO",
    )
    sys.exit(main())
