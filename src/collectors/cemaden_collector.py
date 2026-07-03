"""
Agente 1 — Arquiteto de Dados
Coletor CEMADEN — pluviômetros automáticos (PCD) em tempo real (~10 min).

Motivação: a chuva por estação do INMET no pipeline vem do ZIP histórico
público — geo-livre, mas o INMET o republica com semanas de atraso (medido:
~19 dias). A API horária do INMET, que teria o dado atual, é geo-bloqueada
(HTTP 204) no GitHub Actions. O CEMADEN tem pluviômetros automáticos com
cadência ~10 min, cobertura densa nos municípios de risco do RS (Vale do
Taquari, Vale dos Sinos, Serra) e — confirmado pelo usuário no mapa interativo
(camadas Pluviometros_Cemaden / Hidrologicas_Cemaden) — os dados são ABERTOS,
sem login. Este coletor traz chuva observada realmente atual. Custo: R$ 0,00.

Acesso (duas rotas):
  A. OFICIAL (preferida): API PED (sws.cemaden.gov.br/PED/rest) com token do
     SGAA (sgaa.cemaden.gov.br) — cadastro gratuito; credenciais nos secrets
     CEMADEN_EMAIL/CEMADEN_PASSWORD. Endpoint: /pcds/pcds-dados-recentes?uf=RS.
  B. Fallback: GET aberto na URL JSON do mapa interativo (CEMADEN_DADOS_URL,
     capturável no DevTools → Network) — rota interna que pode mudar sem aviso.

Persistência (reuso do HybridWriter): write_stations (fonte=CEMADEN) +
write_rain_readings (UPSERT live_rain_readings + R2 historico/...). Os ids
recebem prefixo CEM_ para coexistir com os do INMET sem colidir na PK.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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

from database.hybrid_writer import HybridWriter  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração — duas rotas de acesso
# ---------------------------------------------------------------------------
# Rota A (OFICIAL, preferida): API PED com token do SGAA.
#   1. POST {_SGAA_TOKEN_URL} {"email","password"} → {"token","timeToExp"(ms)}
#   2. GET  {_PED_BASE}/pcds/pcds-dados-recentes?token=...&uf=RS
#   Cadastro gratuito no CEMADEN; credenciais em CEMADEN_EMAIL/CEMADEN_PASSWORD.
#   Swagger: sgaa.cemaden.gov.br/SGAA/api/ui + sws.cemaden.gov.br/PED/api/ui.
# Rota B (fallback): URL JSON aberta capturada no DevTools do mapa interativo
#   (CEMADEN_DADOS_URL) — rota interna que pode mudar sem aviso.

_DADOS_URL = os.getenv("CEMADEN_DADOS_URL", "").strip()
_UF = os.getenv("CEMADEN_UF", "RS")

_SGAA_TOKEN_URL = os.getenv(
    "CEMADEN_TOKEN_URL",
    "https://sgaa.cemaden.gov.br/SGAA/rest/controle-token/tokens",
)
_PED_BASE = os.getenv("CEMADEN_PED_BASE", "https://sws.cemaden.gov.br/PED/rest").rstrip("/")
_EMAIL = os.getenv("CEMADEN_EMAIL", "").strip()
_SENHA = os.getenv("CEMADEN_PASSWORD", "").strip()

# Cache do token em memória (o SGAA devolve timeToExp em ms; renovamos com folga).
_TOKEN_CACHE: dict[str, Any] = {"token": None, "exp": None}

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

_session = niquests.Session()
_session.headers.update({"Accept": "application/json, */*", "User-Agent": _UA})

# Variantes de nome de campo aceitas (o CEMADEN mistura camelCase/UPPER/snake).
_F_COD = ("codEstacao", "COD_ESTACAO", "codestacao", "codibge", "codigo", "cod")
_F_NOME = ("nomeEstacao", "NOME_ESTACAO", "nomeestacao", "nome")
_F_SENSOR = ("sensor", "sensordescricao", "descricao_sensor", "tiposensor")
_F_MUN = ("municipio", "MUNICIPIO", "cidade")
_F_UF = ("uf", "UF", "sigla_uf")
_F_LAT = ("latitude", "LATITUDE", "lat")
_F_LON = ("longitude", "LONGITUDE", "lon")
_F_DH = ("datahora", "DATA_HORA", "dataHora", "datahoraUltimovalor", "ultimovalor_data")
# 1h: acumulado da última hora; se ausente, usa a medida pontual.
_F_1H = ("acc1hr", "acc1h", "acumulado1h", "chuva1h", "valor1h")
_F_VAL = ("valorMedida", "MEDIDA", "valor", "chuva", "acumulado", "ultimovalor")


def _pick(d: dict, chaves: tuple[str, ...]) -> Any:
    """Retorna o 1º valor presente entre as variantes de chave, ou None."""
    for k in chaves:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _num(v: Any) -> float | None:
    """Converte para float aceitando vírgula decimal; None se inválido."""
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. Busca aberta (sem autenticação)
# ---------------------------------------------------------------------------

def _unwrap(data: Any) -> list[dict]:
    """Desembrulha o JSON do CEMADEN em lista de dicts.

    Aceita lista direta, envelope {"items"/"dados"/...: [...]} e GeoJSON
    (features com properties/geometry).

    Args:
        data: JSON já decodificado da resposta.

    Returns:
        Lista de dicts (uma por estação/leitura); vazia se irreconhecível.
    """
    if isinstance(data, dict):
        for k in ("items", "dados", "data", "result", "features", "estacoes"):
            if isinstance(data.get(k), list):
                bruto = data[k]
                # GeoJSON: cada feature traz {properties: {...}, geometry: {...}}
                if bruto and isinstance(bruto[0], dict) and "properties" in bruto[0]:
                    return [_flatten_feature(f) for f in bruto]
                return bruto
        return [data]
    return data if isinstance(data, list) else []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(niquests.exceptions.RequestException),
    reraise=True,
)
def _get_token() -> str:
    """Obtém (e cacheia) o token do SGAA via e-mail/senha cadastrados.

    O SGAA devolve o mesmo token enquanto ele estiver ativo (timeToExp em
    milissegundos, ~7 dias); renovamos com folga de 1 h.

    Returns:
        Token JWT para o parâmetro ``token`` da API PED.

    Raises:
        ValueError: Credenciais ausentes/recusadas ou resposta sem token.
        niquests.exceptions.RequestException: Erro de rede após retries.
    """
    agora = datetime.now(timezone.utc)
    if _TOKEN_CACHE["token"] and _TOKEN_CACHE["exp"] and agora < _TOKEN_CACHE["exp"]:
        return _TOKEN_CACHE["token"]
    if not _EMAIL or not _SENHA:
        raise ValueError("CEMADEN_EMAIL/CEMADEN_PASSWORD ausentes no ambiente.")

    resp = _session.post(
        _SGAA_TOKEN_URL, params={"format": "json"},
        json={"email": _EMAIL, "password": _SENHA}, timeout=60,
    )
    if resp.status_code == 400:
        # SGAA responde 400 com [{"Alerta": "..."}] p/ credencial inválida.
        raise ValueError(f"SGAA recusou as credenciais: {resp.text[:120]}")
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        raise ValueError(f"SGAA sem token na resposta: {str(data)[:120]}")
    ttl_ms = float(data.get("timeToExp") or 3_600_000)
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["exp"] = agora + timedelta(milliseconds=ttl_ms) - timedelta(hours=1)
    logger.success(f"Token SGAA obtido (expira em ~{ttl_ms / 3_600_000:.0f} h).")
    return token


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(niquests.exceptions.RequestException),
    reraise=True,
)
def _get_dados_ped() -> list[dict]:
    """GET oficial PED: leituras recentes de todas as PCDs da UF.

    Endpoint ``/pcds/pcds-dados-recentes`` (Swagger PED). O token vai como
    query param, conforme contrato da API.

    Returns:
        Lista de dicts de leituras recentes.

    Raises:
        ValueError: Token recusado (401/403) — credencial a revisar.
        niquests.exceptions.RequestException: Erro de rede após retries.
    """
    token = _get_token()
    resp = _session.get(
        f"{_PED_BASE}/pcds/pcds-dados-recentes",
        params={"token": token, "uf": _UF, "formato": "json"},
        timeout=120,
    )
    if resp.status_code in (401, 403):
        _TOKEN_CACHE["token"] = None   # força renovação no próximo run
        raise ValueError(f"PED recusou o token (HTTP {resp.status_code}).")
    resp.raise_for_status()
    return _unwrap(resp.json())


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(niquests.exceptions.RequestException),
    reraise=True,
)
def _get_dados() -> list[dict]:
    """GET aberto das PCDs/leituras do RS no CEMADEN (rota B, fallback).

    Returns:
        Lista de dicts (uma por estação/leitura). Desembrulha respostas
        no formato {"items"/"dados"/"data"/"features": [...]}.

    Raises:
        niquests.exceptions.RequestException: Erro de rede após retries.
    """
    resp = _session.get(_DADOS_URL, params={"uf": _UF}, timeout=60)
    resp.raise_for_status()
    return _unwrap(resp.json())


def _flatten_feature(feat: dict) -> dict:
    """Achata uma feature GeoJSON (properties + geometry) em um dict plano."""
    props = dict(feat.get("properties") or {})
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if isinstance(coords, list) and len(coords) >= 2:
        props.setdefault("longitude", coords[0])
        props.setdefault("latitude", coords[1])
    return props


def fetch_dados_rs() -> pd.DataFrame:
    """Coleta e normaliza as leituras de pluviômetros CEMADEN do RS.

    Rota A (PED oficial com token SGAA) quando CEMADEN_EMAIL/PASSWORD estão
    definidos; senão rota B (URL aberta CEMADEN_DADOS_URL).

    Returns:
        DataFrame normalizado: cod, nome, municipio, uf, lat, lon,
        datahora (UTC), chuva_mm. Vazio se nada retornar.
    """
    if _EMAIL and _SENHA:
        logger.info("CEMADEN: rota oficial PED (token SGAA).")
        registros = _get_dados_ped()
    else:
        logger.info("CEMADEN: rota aberta (CEMADEN_DADOS_URL).")
        registros = _get_dados()
    if not registros:
        logger.warning("CEMADEN: nenhum registro retornado.")
        return pd.DataFrame()

    linhas: list[dict[str, Any]] = []
    for d in registros:
        if not isinstance(d, dict):
            continue
        cod = _pick(d, _F_COD)
        lat = _num(_pick(d, _F_LAT))
        lon = _num(_pick(d, _F_LON))
        if cod is None or lat is None or lon is None:
            continue
        # 1h dedicado se houver; senão a medida pontual mais recente.
        chuva = _num(_pick(d, _F_1H))
        if chuva is None:
            chuva = _num(_pick(d, _F_VAL))
        linhas.append({
            "cod":       str(cod),
            "nome":      str(_pick(d, _F_NOME) or cod),
            "municipio": str(_pick(d, _F_MUN) or ""),
            "uf":        str(_pick(d, _F_UF) or _UF),
            "lat":       lat,
            "lon":       lon,
            "datahora":  _pick(d, _F_DH),
            "chuva_mm":  chuva,
            "sensor":    str(_pick(d, _F_SENSOR) or ""),
        })

    df = pd.DataFrame(linhas)
    if df.empty:
        return df

    # A PED devolve TODOS os sensores da PCD (chuva, nível, temp...). Se o
    # feed identifica o sensor, mantém só chuva; sem identificação, mantém
    # tudo (feeds simples do mapa já são só pluviômetros).
    if df["sensor"].str.strip().any():
        chuva_mask = df["sensor"].str.lower().str.contains(
            "chuva|pluv|precip", regex=True, na=False
        )
        if chuva_mask.any():
            df = df[chuva_mask]
    df["datahora"] = pd.to_datetime(df["datahora"], utc=True, errors="coerce")
    # Sem datahora utilizável → usa o instante da coleta (snapshot).
    df["datahora"] = df["datahora"].fillna(pd.Timestamp.utcnow())
    # Filtra por UF (defensivo, caso o endpoint não filtre).
    df = df[df["uf"].str.upper() == _UF.upper()] if "uf" in df else df
    if df.empty:
        return df
    logger.info(
        f"CEMADEN: {len(df):,} leituras | {df['cod'].nunique()} estações {_UF} "
        f"| {df['datahora'].min():%d/%m %H:%M} → {df['datahora'].max():%d/%m %H:%M} UTC"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Transformação para os schemas do HybridWriter
# ---------------------------------------------------------------------------

def _build_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Monta o inventário de estações (1 linha por estação) p/ write_stations."""
    inv = df.sort_values("datahora").groupby("cod", as_index=False).last()
    return pd.DataFrame({
        "station_id":   "CEM_" + inv["cod"].astype(str),
        "name":         inv["nome"],
        "municipality": inv["municipio"],
        "state":        inv["uf"],
        "lat":          inv["lat"],
        "lon":          inv["lon"],
        "source":       "CEMADEN",
        "active":       True,
    })


def _build_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Monta as leituras (chuva 1h por estação) p/ write_rain_readings.

    Usa o acumulado de 1h quando o endpoint o fornece; caso o feed traga só
    incrementos de 10 min, soma os da última hora. ``ts`` = leitura mais recente.
    """
    linhas: list[dict[str, Any]] = []
    for cod, g in df.dropna(subset=["chuva_mm"]).groupby("cod"):
        g = g.sort_values("datahora")
        ult = g["datahora"].max()
        if len(g) > 1:
            rain_1h = float(g.loc[g["datahora"] >= ult - timedelta(hours=1), "chuva_mm"].sum())
        else:
            rain_1h = float(g["chuva_mm"].iloc[-1])
        linhas.append({
            "station_id":  f"CEM_{cod}",
            "ts":          ult,
            "rain_1h_mm":  rain_1h,
            "temperature": None,   # pluviômetro CEMADEN não mede temp/umidade
            "humidity":    None,
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 3. Orquestração
# ---------------------------------------------------------------------------

def run() -> dict[str, Any]:
    """Executa a coleta CEMADEN e persiste via HybridWriter.

    Returns:
        Dict com estações, leituras e linhas gravadas no Supabase.
    """
    df = fetch_dados_rs()
    if df.empty:
        return {"estacoes": 0, "leituras": 0, "pg_stations": 0, "pg_readings": 0}

    writer = HybridWriter()
    est_df = _build_stations(df)
    rd_df = _build_readings(df)

    r_st = writer.write_stations(est_df)
    # R2 em prefixo PRÓPRIO (historico/live_rain_cemaden) — o CEMADEN grava só
    # um snapshot de 1 leitura/estação; se caísse no prefixo do INMET, o
    # rain_accumulator (que pega o Parquet mais recente de live_rain_readings)
    # passaria a ler esse snapshot e perderia a série de 30 dias do INMET,
    # zerando os acumulados 6h/24h/72h/7d das estações INMET. O Supabase
    # (live_rain_readings) continua recebendo o snapshot para a API expor.
    r_rd = writer.write_rain_readings(rd_df, r2_fonte="live_rain_cemaden")

    return {
        "estacoes":    len(est_df),
        "leituras":    len(rd_df),
        "pg_stations": r_st.pg_rows,
        "pg_readings": r_rd.pg_rows,
    }


def main() -> int:
    """Entry point com guard: sem credencial/URL configurada, não quebra o pipeline.

    Returns:
        0 sempre que a coleta não for erro fatal (o step roda com
        continue-on-error); apenas loga e sai.
    """
    if not (_DADOS_URL or (_EMAIL and _SENHA)):
        logger.warning(
            "CEMADEN sem acesso configurado — coletor ignorado. Preferido: "
            "cadastre-se no CEMADEN (PED) e defina CEMADEN_EMAIL/CEMADEN_PASSWORD "
            "nos Secrets (token SGAA automático). Alternativa: CEMADEN_DADOS_URL "
            "com a URL JSON aberta do mapa interativo."
        )
        return 0

    try:
        r = run()
        logger.success(
            f"CEMADEN OK — {r['estacoes']} estações, {r['leituras']} leituras "
            f"(PG: {r['pg_readings']})."
        )
        return 0
    except niquests.exceptions.RequestException as exc:
        logger.error(f"CEMADEN — falha de rede após retries: {exc}")
        return 0
    except (ValueError, KeyError) as exc:
        logger.error(f"CEMADEN — credencial/resposta inválida: {exc}")
        return 0


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO",
    )
    sys.exit(main())
