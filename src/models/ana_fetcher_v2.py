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
# 5. Série de rio (REST recente) + fallback SOAP (histórico profundo)
# ---------------------------------------------------------------------------

# Alias semântico pedido na spec.
fetch_inventario_ativo = fetch_inventario_estacoes


def fetch_serie_rio(
    cod_estacao: int,
    data_inicio: str,
    data_fim: str,
) -> pd.DataFrame:
    """Série diária do rio via REST, agregando as leituras 15-min OK por dia.

    IMPORTANTE: a API REST telemétrica só serve os ~últimos 30 dias. Pedir
    janelas antigas retorna vazio. Esta função pagina em janelas de 30 dias
    do mais recente para o mais antigo e PARA assim que uma janela volta
    vazia (limite real da REST) — proteção contra centenas de chamadas e
    bloqueio de IP. Para histórico profundo, use fetch_rio_com_fallback.

    Args:
        cod_estacao: Código ANA da estação.
        data_inicio: Data inicial "dd/MM/yyyy".
        data_fim: Data final "dd/MM/yyyy".

    Returns:
        DataFrame com data (diária), cota_m, vazao_m3s, chuva_mm —
        só dias com leituras de qualidade OK (status=0).
    """
    ini = datetime.strptime(data_inicio, "%d/%m/%Y").date()
    fim = datetime.strptime(data_fim, "%d/%m/%Y").date()
    # A REST telemétrica serve só a janela mais recente (~30 dias relativos a
    # hoje) — uma única chamada cobre tudo o que existe; janelas antigas
    # retornam vazio e insistir arrisca bloqueio de IP.
    df = fetch_serie_telemetrica(cod_estacao, dias_busca=30)
    frames: list[pd.DataFrame] = [df] if not df.empty else []

    if not frames:
        return pd.DataFrame(columns=["data", "cota_m", "vazao_m3s", "chuva_mm"])

    bruto = pd.concat(frames, ignore_index=True)
    bruto["data"] = pd.to_datetime(bruto["data_hora_medicao"]).dt.date
    diario = (
        bruto.groupby("data")
        .agg(cota_m=("cota_m", "mean"),
             vazao_m3s=("vazao_m3s", "mean"),
             chuva_mm=("chuva_mm", "sum"))
        .reset_index()
    )
    diario = diario[(diario["data"] >= ini) & (diario["data"] <= fim)]
    return diario.reset_index(drop=True)


def reconciliar_estacoes(rio_alvo: str) -> dict[str, Any]:
    """Verifica/corrige o código de estação do rio contra o inventário ativo.

    Args:
        rio_alvo: 'taquari' ou 'jacui'.

    Returns:
        Dict com codigo_atual, ativo (bool), e codigo_recomendado (mantém o
        atual se ativo; senão sugere a 1ª telemétrica ativa do rio).
    """
    from models.ana_fetcher import ESTACOES_RS

    atual = str(ESTACOES_RS.get(rio_alvo, [None])[0])
    inv = fetch_inventario_ativo()
    if inv.empty or "codigoestacao" not in inv.columns:
        return {"codigo_atual": atual, "ativo": None, "codigo_recomendado": atual}

    ativos = set(inv["codigoestacao"].astype(str))
    ativo = atual in ativos
    recomendado = atual
    if not ativo and "Rio_Nome" in inv.columns and "Tipo_Telem" in inv.columns:
        termo = "TAQUARI" if rio_alvo == "taquari" else "JACU"
        cand = inv[
            inv["Rio_Nome"].astype(str).str.upper().str.contains(termo, na=False)
            & (inv["Tipo_Telem"].astype(str) == "1")
        ]
        if not cand.empty:
            recomendado = str(cand.iloc[0]["codigoestacao"])
            logger.warning(f"{rio_alvo}: código {atual} inativo → recomendado {recomendado}.")
    if ativo:
        logger.info(f"{rio_alvo}: código {atual} ativo no inventário REST — mantido.")
    return {"codigo_atual": atual, "ativo": ativo, "codigo_recomendado": recomendado}


def fetch_rio_com_fallback(
    rio_alvo: str,
    start_year: int,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Cotas diárias do rio: SOAP (histórico profundo) + REST (recente OK).

    A REST telemétrica só tem ~30 dias, então NÃO serve como fonte do
    histórico 2000-2026 — o SOAP (HidroSerieHistorica) é a espinha dorsal.
    A REST entra como CAMADA DE QUALIDADE nos dias recentes (leituras
    15-min com status filtrado), sobrescrevendo o SOAP onde houver
    sobreposição. Loga quantos dias cada fonte contribuiu.

    Args:
        rio_alvo: 'taquari' ou 'jacui'.
        start_year: Ano inicial.
        end_year: Ano final.

    Returns:
        DataFrame com data e cota_m (união SOAP+REST, REST tem prioridade
        nos dias recentes).
    """
    from models import ana_fetcher as soap

    rec = reconciliar_estacoes(rio_alvo)
    cod = int(rec["codigo_recomendado"])

    # 1. Espinha dorsal: SOAP histórico
    df_soap = soap.fetch_serie_historica(
        cod, f"01/01/{start_year}", f"31/12/{end_year}", tipo_dados=1
    )
    n_soap = len(df_soap)
    base = df_soap[["data", "cota_m"]].copy() if not df_soap.empty else \
        pd.DataFrame(columns=["data", "cota_m"])

    # 2. Camada de qualidade: REST recente
    n_rest = 0
    try:
        df_rest = fetch_serie_rio(cod, "01/01/2026", "31/12/2026")
        if not df_rest.empty:
            n_rest = len(df_rest)
            rest = df_rest[["data", "cota_m"]].dropna(subset=["cota_m"])
            base = (
                pd.concat([base, rest], ignore_index=True)
                .sort_values("data")
                .drop_duplicates(subset="data", keep="last")  # REST sobrescreve
                .reset_index(drop=True)
            )
    except Exception as exc:  # noqa: BLE001 — REST nunca derruba a fonte SOAP
        logger.warning(f"{rio_alvo}: REST indisponível ({str(exc)[:50]}) — só SOAP.")

    fonte = "SOAP+REST" if n_rest else "SOAP (REST sem dados)"
    logger.info(
        f"{rio_alvo} [{cod}]: {len(base):,} dias | fonte: {fonte} "
        f"(SOAP {n_soap:,} dias + REST {n_rest} dias recentes)"
    )
    return base[["data", "cota_m"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standalone — só diagnóstico (NÃO sobrescreve ana_fetcher.py)
