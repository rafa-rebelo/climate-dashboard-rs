"""
Agente 1 — Engenheiro de Dados / APIs hidrológicas
Cliente da API REST nova da ANA (hidrowebservice, OAuth) — DIAGNÓSTICO.

Complementa o ana_fetcher.py (SOAP telemetriaws1) com a API REST mais rica
(telemetria 15 min, status de qualidade por leitura, flag "Operando" no
inventário). Por ora é SÓ diagnóstico — reconcilia os códigos hardcoded
do Taquari/Jacuí contra o inventário de estações realmente ativas;
NÃO sobrescreve ana_fetcher.py.

Base URL: usa CF_WORKER_URL (proxy Cloudflare) se presente — a ANA bloqueia
IPs do GitHub Actions, e o Worker é a rota que funciona em todo lugar.
Localmente, sem o Worker, cai no acesso direto www.ana.gov.br.

Credenciais: ANA_IDENTIFICADOR + ANA_SENHA no .env.
Custo: R$ 0,00 — API pública com login gratuito.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import niquests
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

# Mesma resolução de base do ana_collector: Worker se houver, senão direto.
_CF_PROXY = os.getenv("CF_WORKER_URL", "").rstrip("/")
_BASE = f"{_CF_PROXY}/hidrowebservice" if _CF_PROXY else "https://www.ana.gov.br/hidrowebservice"
_EST = f"{_BASE}/EstacoesTelemetricas"

_IDENT = os.getenv("ANA_IDENTIFICADOR", "")
_SENHA = os.getenv("ANA_SENHA", "")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Cache do token em memória (válido ~60 min; renovamos com folga em 50).
_TOKEN_CACHE: dict[str, Any] = {"token": None, "exp": None}

_session = niquests.Session()
_session.headers.update({"Accept": "*/*", "User-Agent": _UA})


# ---------------------------------------------------------------------------
# 1. Autenticação OAuth (token cacheado)
# ---------------------------------------------------------------------------

def get_token(force: bool = False) -> str:
    """Obtém (e cacheia) o token OAuth da ANA via OAUth/v1.

    O token vale ~60 min; renovamos com folga aos 50. Cache em memória
    evita autenticações repetidas (a ANA bloqueia IPs por excesso).

    Args:
        force: Se True, ignora o cache e renova imediatamente.

    Returns:
        Token JWT (string) para o header Authorization: Bearer.

    Raises:
        ValueError: Se credenciais ausentes ou token não vier na resposta.
        niquests.exceptions.HTTPError: Se a autenticação falhar (após retry).
    """
    agora = datetime.now(timezone.utc)
    if (not force and _TOKEN_CACHE["token"] and _TOKEN_CACHE["exp"]
            and agora < _TOKEN_CACHE["exp"]):
        return _TOKEN_CACHE["token"]

    if not _IDENT or not _SENHA:
        raise ValueError("ANA_IDENTIFICADOR e ANA_SENHA ausentes no .env.")

    logger.info("Autenticando na ANA (hidrowebservice OAuth)...")
    resp = _request_com_retry(
        "GET", f"{_EST}/OAUth/v1",
        headers={"Identificador": _IDENT, "Senha": _SENHA},
        autenticar_em_401=False,
    )
    data = resp.json()
    items = data.get("items") or data
    token = items.get("tokenautenticacao") or items.get("token")
    if not token:
        raise ValueError(f"Token não encontrado na resposta: {str(data)[:200]}")

    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["exp"] = agora + timedelta(minutes=50)
    _session.headers["Authorization"] = f"Bearer {token}"
    logger.success("Token ANA obtido (válido ~50 min, cacheado).")
    return token


# ---------------------------------------------------------------------------
# Requisição com retry (417/429/5xx intermitentes + 401 → renova token)
# ---------------------------------------------------------------------------

def _request_com_retry(
    metodo: str,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    autenticar_em_401: bool = True,
    max_retries: int = 2,
) -> "niquests.Response":
    """GET/POST com backoff de 30s para erros transitórios e 1 renovação de token.

    A ANA monitora IPs: o backoff é longo (30s) e o nº de tentativas baixo,
    e NUNCA reautenticamos em loop apertado.

    Args:
        metodo: "GET" ou "POST".
        url: URL completa.
        params: Query params.
        headers: Headers extras (ex.: Identificador/Senha na auth).
        autenticar_em_401: Se True, renova token e tenta 1x ao receber 401.
        max_retries: Tentativas extras para 417/429/5xx.

    Returns:
        Response com status 2xx.

    Raises:
        niquests.exceptions.HTTPError: Se esgotar as tentativas.
    """
    transitorios = {417, 429, 500, 502, 503, 504}
    tentativa = 0
    while True:
        resp = _session.request(metodo, url, params=params, headers=headers, timeout=60)

        if resp.status_code == 401 and autenticar_em_401:
            logger.warning("401 — renovando token e tentando 1x...")
            get_token(force=True)
            resp = _session.request(metodo, url, params=params, headers=headers, timeout=60)

        if resp.status_code in transitorios and tentativa < max_retries:
            tentativa += 1
            logger.warning(f"HTTP {resp.status_code} transitório — backoff 30s "
                           f"(tentativa {tentativa}/{max_retries})...")
            time.sleep(30)
            continue

        resp.raise_for_status()
        return resp


# ---------------------------------------------------------------------------
# 2. Inventário de estações ativas
# ---------------------------------------------------------------------------

def _to_df(data: Any) -> pd.DataFrame:
    """Extrai o DataFrame do envelope da ANA (items/list/dict).

    Args:
        data: JSON da resposta.

    Returns:
        DataFrame com os registros.
    """
    if isinstance(data, list):
        return pd.DataFrame(data)
    for k in ("items", "data", "dados", "result", "estacoes"):
        if isinstance(data.get(k), list):
            return pd.DataFrame(data[k])
    return pd.DataFrame([data]) if data else pd.DataFrame()


def fetch_inventario_estacoes() -> pd.DataFrame:
    """Inventário de estações telemétricas do RS, SÓ as ativas (Operando=1).

    Returns:
        DataFrame com codigoestacao, Estacao_Nome, Municipio_Nome,
        Bacia_Nome, Latitude, Longitude, Tipo_Estacao (telemétrica/escala)
        e Data_Periodo_Telemetrica_Inicio — apenas estações operando.
    """
    get_token()
    # "Unidade Federativa" espera a SIGLA (RS), não o nome completo
    # (confirmado em ana_collector.inventario_estacoes).
    resp = _request_com_retry("GET", f"{_EST}/HidroInventarioEstacoes/v1",
                              params={"Unidade Federativa": "RS"})
    df = _to_df(resp.json())
    if df.empty:
        logger.warning("Inventário vazio.")
        return df

    # Filtro 1: Operando == 1 (ativas de fato)
    if "Operando" in df.columns:
        df = df[df["Operando"].astype(str) == "1"]

    # Filtro 2: RS (defensivo — a API pode não respeitar o param de UF)
    uf_col = next((c for c in df.columns
                   if c.lower() in ("uf", "uf_estacao", "estado_sigla")), None)
    if uf_col:
        df = df[df[uf_col].astype(str).str.upper().str.contains("RS|RIO GRANDE", na=False)]

    # Seleção de colunas (tolerante a variações de nome no schema)
    def _col(*nomes: str) -> Optional[str]:
        for n in nomes:
            if n in df.columns:
                return n
        return None

    cols = {
        "codigoestacao":  _col("codigoestacao", "Codigo", "Codigo_Estacao"),
        "Estacao_Nome":   _col("Estacao_Nome", "Nome"),
        "Municipio_Nome": _col("Municipio_Nome", "nmMunicipio"),
        "Bacia_Nome":     _col("Bacia_Nome"),
        "Rio_Nome":       _col("Rio_Nome"),
        "Latitude":       _col("Latitude", "lat"),
        "Longitude":      _col("Longitude", "lon"),
        "Tipo_Telem":     _col("Tipo_Estacao_Telemetrica"),
        "Telem_Inicio":   _col("Data_Periodo_Telemetrica_Inicio"),
    }
    presentes = {k: v for k, v in cols.items() if v}
    out = df[list(presentes.values())].rename(columns={v: k for k, v in presentes.items()})
    logger.info(f"Inventário RS ativo (Operando=1): {len(out)} estações.")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Série telemétrica adotada (com status de qualidade)
# ---------------------------------------------------------------------------

def fetch_serie_telemetrica(cod_estacao: int, dias_busca: int = 30) -> pd.DataFrame:
    """Série telemétrica adotada de uma estação, filtrando qualidade=OK.

    Args:
        cod_estacao: Código ANA da estação.
        dias_busca: Janela em dias (enum RangeIntervaloDeBusca=DIAS_{n}).

    Returns:
        DataFrame com data_hora_medicao, cota_m (cm/100), vazao_m3s,
        chuva_mm, status_qualidade — só leituras com status 0 (OK).
    """
    get_token()
    resp = _request_com_retry(
        "GET", f"{_EST}/HidroinfoanaSerieTelemetricaAdotada/v1",
        params={
            "Código da Estação":        cod_estacao,
            "Tipo Filtro Data":         "DATA_LEITURA",
            "Range Intervalo de busca": f"DIAS_{dias_busca}",
        },
    )
    df = _to_df(resp.json())
    if df.empty:
        return df

    def _col(*nomes: str) -> Optional[str]:
        for n in nomes:
            if n in df.columns:
                return n
        return None

    c_dt   = _col("Data_Hora_Medicao", "DataHora", "data_hora_medicao")
    c_cota = _col("Cota_Adotada", "Cota_Adotada_cm", "cota")
    c_vaz  = _col("Vazao_Adotada", "vazao")
    c_chu  = _col("Chuva_Adotada", "chuva")
    c_st   = _col("Cota_Adotada_Status", "Status_Qualidade", "status")

    out = pd.DataFrame()
    out["data_hora_medicao"] = pd.to_datetime(df[c_dt], errors="coerce") if c_dt else pd.NaT
    out["cota_m"]    = pd.to_numeric(df[c_cota], errors="coerce") / 100.0 if c_cota else float("nan")
    out["vazao_m3s"] = pd.to_numeric(df[c_vaz], errors="coerce") if c_vaz else float("nan")
    out["chuva_mm"]  = pd.to_numeric(df[c_chu], errors="coerce") if c_chu else float("nan")
    out["status_qualidade"] = pd.to_numeric(df[c_st], errors="coerce") if c_st else 0

    antes = len(out)
    out = out[out["status_qualidade"].fillna(0) == 0]
    logger.debug(f"  {cod_estacao}: {len(out)}/{antes} leituras OK (status=0).")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Reconciliação Taquari/Jacuí
# ---------------------------------------------------------------------------

def reconciliar_estacoes_taquari_jacui() -> dict[str, Any]:
    """Compara os códigos hardcoded (ana_fetcher.ESTACOES_RS) com o inventário
    ativo do Taquari/Jacuí da API REST nova.

    Returns:
        Dict por rio com: usados (códigos atuais), inativos (usados mas não
        operando), e disponiveis (estações ativas da bacia não usadas).
    """
    from models.ana_fetcher import ESTACOES_RS

    inv = fetch_inventario_estacoes()
    # Na ANA o rio fica em Rio_Nome; Bacia_Nome é a macro-bacia
    # ("Atlântico, Trecho Sudeste"), por isso filtramos por Rio_Nome.
    if inv.empty or "Rio_Nome" not in inv.columns:
        logger.error("Inventário sem coluna Rio_Nome — não dá para reconciliar.")
        return {}

    inv["_rio"] = inv["Rio_Nome"].astype(str).str.upper()
    inv["_cod"] = inv["codigoestacao"].astype(str)
    tem_telem = "Tipo_Telem" in inv.columns
    ativos = set(inv["_cod"])

    alvos = {"taquari": "TAQUARI", "jacui": "JACU"}
    relatorio: dict[str, Any] = {}

    for rio, termo in alvos.items():
        usados = [str(c) for c in ESTACOES_RS.get(rio, [])]
        inativos = [c for c in usados if c not in ativos]
        nrio = inv[inv["_rio"].str.contains(termo, na=False)]
        # Telemétricas (Tipo_Telem=1) são as melhores candidatas (15 min).
        candidatas = nrio[~nrio["_cod"].isin(usados)]
        if tem_telem:
            candidatas = candidatas[candidatas["Tipo_Telem"].astype(str) == "1"]

        cols = [c for c in ("codigoestacao", "Estacao_Nome", "Municipio_Nome")
                if c in candidatas.columns]
        relatorio[rio] = {
            "usados": usados,
            "usados_inativos": inativos,
            "ativos_no_rio": len(nrio),
            "telemetricas_candidatas": candidatas[cols].head(15).to_dict("records"),
        }
    return relatorio


# ---------------------------------------------------------------------------
# Standalone — só diagnóstico (NÃO sobrescreve ana_fetcher.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               level="INFO")

    rel = reconciliar_estacoes_taquari_jacui()
    print("\n" + "=" * 60)
    print("RELATÓRIO DE RECONCILIAÇÃO — API REST nova vs ana_fetcher.py")
    print("=" * 60)
    for rio, d in rel.items():
        print(f"\n### {rio.upper()} ###")
        print(f"  Códigos usados hoje: {d['usados']}")
        if d["usados_inativos"]:
            print(f"  ⚠ NÃO aparecem como ativos (Operando=1): {d['usados_inativos']}")
        else:
            print("  ✓ Todos os códigos usados estão ativos.")
        print(f"  Estações ativas no rio: {d['ativos_no_rio']}")
        if d["telemetricas_candidatas"]:
            print("  Telemétricas ativas NÃO usadas (candidatas a melhorar a bacia):")
            for e in d["telemetricas_candidatas"]:
                cod = e.get("codigoestacao", "?")
                nome = e.get("Estacao_Nome", "")
                mun = e.get("Municipio_Nome", "")
                print(f"    {cod} | {nome} ({mun})")
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO APENAS — ana_fetcher.py NÃO foi modificado.")
    print("=" * 60)
