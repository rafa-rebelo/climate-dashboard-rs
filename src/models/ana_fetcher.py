"""
Agente 1 — Arquiteto de Dados
Série HISTÓRICA de cotas/vazões via API SOAP do HidroWeb (telemetriaws1).

Módulo do pipeline de TREINO do LSTM — separado do ana_collector.py
(tempo real via Cloudflare Worker). A API SOAP histórica usa outro host
(telemetriaws1.ana.gov.br) e dispensa o Worker e autenticação.

CUSTO: R$ 0,00 — API pública da ANA, sem autenticação, sem chave e sem
limite de uso documentado. Nenhuma dependência paga.

Estrutura da resposta HidroSerieHistorica: 1 elemento <SerieHistorica>
por MÊS, com colunas diárias Cota01..Cota31 (cm) ou Vazao01..Vazao31
(m³/s) — o parser "derrete" os dias em linhas diárias.

Códigos confirmados via HidroInventario SOAP em 12/06/2026.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from calendar import monthrange
from datetime import date, datetime
from typing import Any, Optional

import niquests
import pandas as pd
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_BASE = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"
_TIMEOUT_S = 60

# Estações de régua (escala=1) confirmadas no HidroInventario em 12/06/2026.
# Nota: os códigos do tempo real (config) nem sempre servem para histórico —
# 87010000 é TRIUNFO/Rio Jacuí (não Guaíba) e 86500000 é Rio Carreiro.
ESTACOES_RS: dict[str, list[int]] = {
    # PONTA DOS COATIS — Porto Alegre, Rio Guaíba: a régua com a série
    # convencional mais completa (335 dias em 2024). O CAIS MAUÁ C6
    # (87450004) é só telemétrico (ErrorTable no histórico) e a Ilha da
    # Pintada (87450005) tem 2024 quase vazio.
    # ATENÇÃO: lacuna em mai/2024 — as réguas foram submersas na enchente
    # (pico ~5,3 m ausente da série convencional; preencher via telemetria
    # REST se necessário para o treino).
    "guaiba":  [87500020],
    # RIO PARDO — Rio Jacuí, série centenária (principal do médio Jacuí)
    "jacui":   [85900000],
    # MUÇUM — Rio Taquari (epicentro das cheias de 2023/2024)
    "taquari": [86510000],
    # SÃO LEOPOLDO + CAMPO BOM — Rio dos Sinos
    "sinos":   [87382000, 87380000],
    # PASSO DO MENDONÇA (Cristal) — Rio Camaquã
    "camaqua": [87905000],
    # ── Onda DCRS (censo 04/07/2026 — sondagem SOAP 2005/2015/2024/2026) ──
    # BARCA DO CAÍ — Rio Caí, série completa 2005→2026 (10 estações DCRS na bacia)
    "cai":      [87170000],
    # MANOEL VIANA — Rio Ibicuí, série completa + telemetria ANA ativa
    "ibicui":   [76560000],
    # SANTO ÂNGELO — Rio Ijuí, série completa + telemetria ANA ativa
    "ijui":     [75230000],
    # ALBATROZ (Canoas) — Rio Gravataí. RESSALVA: série 2005→abr/2024 e PAROU
    # (estação possivelmente perdida na enchente de mai/2024) — validação
    # recente e costura ficam 100% na régua DCRS (zero diferente).
    "gravatai": [87406000],
    # CANDELÁRIA MONTANTE — Rio Pardo. RESSALVA: série 2005→jun/2024 e parou
    # (mesmo caso do Gravataí).
    "pardo":    [85735000],
}

# tipo_dados da API: 1=Cotas, 2=Chuvas, 3=Vazões
_PREFIXO_COLUNA = {1: "Cota", 2: "Chuva", 3: "Vazao"}


# ---------------------------------------------------------------------------
# Parser XML
# ---------------------------------------------------------------------------

def _parse_serie(xml_bytes: bytes, tipo_dados: int) -> pd.DataFrame:
    """Converte o XML mensal da HidroSerieHistorica em DataFrame diário.

    Cada <SerieHistorica> representa um mês; os valores diários vêm em
    colunas {Prefixo}01..{Prefixo}31. Dias inexistentes do mês são
    descartados via calendar.monthrange.

    Args:
        xml_bytes: Corpo bruto da resposta SOAP.
        tipo_dados: 1=Cotas (cm→m), 2=Chuvas (mm), 3=Vazões (m³/s).

    Returns:
        DataFrame com colunas data, cota_m, vazao_m3s, chuva_mm e
        consistido (1=bruto, 2=consistido). Colunas não aplicáveis = NaN.

    Raises:
        ET.ParseError: Se o XML for inválido.
    """
    root = ET.fromstring(xml_bytes)
    prefixo = _PREFIXO_COLUNA[tipo_dados]
    registros: list[dict[str, Any]] = []

    for serie in root.iter("SerieHistorica"):
        dt_el = serie.find("DataHora")
        nc_el = serie.find("NivelConsistencia")
        if dt_el is None or not dt_el.text:
            continue
        primeiro_dia = datetime.strptime(dt_el.text.strip()[:10], "%Y-%m-%d").date()
        consistido = int(nc_el.text) if nc_el is not None and nc_el.text else 1
        _, dias_no_mes = monthrange(primeiro_dia.year, primeiro_dia.month)

        for dia in range(1, dias_no_mes + 1):
            el = serie.find(f"{prefixo}{dia:02d}")
            if el is None or el.text is None or not el.text.strip():
                continue
            try:
                valor = float(el.text.strip().replace(",", "."))
            except ValueError:
                continue
            registros.append({
                "data":       date(primeiro_dia.year, primeiro_dia.month, dia),
                "valor":      valor,
                "consistido": consistido,
            })

    if not registros:
        return pd.DataFrame(columns=["data", "cota_m", "vazao_m3s", "chuva_mm", "consistido"])

    df = pd.DataFrame(registros)
    # Meses podem vir duplicados (bruto + consistido) — mantém o de maior
    # nível de consistência por data.
    df = (
        df.sort_values(["data", "consistido"])
        .drop_duplicates(subset="data", keep="last")
        .reset_index(drop=True)
    )

    df["cota_m"]    = df["valor"] / 100.0 if tipo_dados == 1 else float("nan")
    df["vazao_m3s"] = df["valor"] if tipo_dados == 3 else float("nan")
    df["chuva_mm"]  = df["valor"] if tipo_dados == 2 else float("nan")
    return df[["data", "cota_m", "vazao_m3s", "chuva_mm", "consistido"]]


# ---------------------------------------------------------------------------
# Fetch com retry
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(niquests.exceptions.RequestException),
    reraise=True,
)
def _get(params: dict[str, Any]) -> bytes:
    """GET na API SOAP com 3 tentativas e 5s entre elas (API gov instável).

    Args:
        params: Query params da HidroSerieHistorica.

    Returns:
        Corpo bruto (bytes) da resposta.

    Raises:
        niquests.exceptions.RequestException: Após 3 tentativas sem sucesso.
    """
    resp = niquests.get(_BASE, params=params, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    return resp.content


def fetch_serie_historica(
    cod_estacao: int,
    data_inicio: str,
    data_fim: str,
    tipo_dados: int = 1,
    nivel_consistencia: int = 2,
) -> pd.DataFrame:
    """Busca a série histórica diária de uma estação na API SOAP da ANA.

    Tenta primeiro o nível de consistência pedido (2=consistido); se a
    resposta vier vazia, faz fallback automático para 1=bruto.

    Args:
        cod_estacao: Código ANA de 8 dígitos da estação.
        data_inicio: Data inicial "dd/MM/yyyy".
        data_fim: Data final "dd/MM/yyyy".
        tipo_dados: 1=Cotas, 2=Chuvas, 3=Vazões.
        nivel_consistencia: 2=Consistido (preferência) ou 1=Bruto.

    Returns:
        DataFrame com data, cota_m, vazao_m3s, chuva_mm, consistido —
        vazio se a estação não tiver dados no período.

    Raises:
        niquests.exceptions.RequestException: Falha de rede após retries.
        xml.etree.ElementTree.ParseError: Resposta não-XML da API.

    Note:
        Quando nivel_consistencia=2, a busca é feita com TODAS as
        consistências e o consistido é preferido POR DATA (o parser
        deduplica mantendo o maior nível). Pedir nc=2 direto à API
        omitiria os meses recentes ainda não consistidos — ex.: a cheia
        de mai/2024 só existe como bruto.
    """
    nc_api: Any = "" if nivel_consistencia == 2 else nivel_consistencia
    xml = _get({
        "codEstacao":        cod_estacao,
        "dataInicio":        data_inicio,
        "dataFim":           data_fim,
        "tipoDados":         tipo_dados,
        "nivelConsistencia": nc_api,
    })
    return _parse_serie(xml, tipo_dados)


# ---------------------------------------------------------------------------
# Orquestrador — todos os rios
# ---------------------------------------------------------------------------

def fetch_todos_rios(start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    """Baixa cotas E vazões históricas de todas as estações de ESTACOES_RS.

    Para cada estação: cotas (tipo 1) + vazões (tipo 3), unidas por data.
    Resultado concatenado com colunas rio_id e cod_estacao.

    Args:
        start_year: Ano inicial (inclusive).
        end_year: Ano final (inclusive).

    Returns:
        DataFrame com data, rio_id, cod_estacao, cota_m, vazao_m3s,
        consistido — ordenado por rio_id, cod_estacao, data.
    """
    ini = f"01/01/{start_year}"
    fim = f"31/12/{end_year}"
    frames: list[pd.DataFrame] = []

    for rio_id, codigos in ESTACOES_RS.items():
        for cod in codigos:
            logger.info(f"Buscando {rio_id.capitalize()} [{cod}] ({start_year}–{end_year})...")
            try:
                cotas  = fetch_serie_historica(cod, ini, fim, tipo_dados=1)
                vazoes = fetch_serie_historica(cod, ini, fim, tipo_dados=3)
            except niquests.exceptions.RequestException as exc:
                logger.error(f"  {rio_id} [{cod}]: falha de rede — {exc}")
                continue
            except ET.ParseError as exc:
                logger.error(f"  {rio_id} [{cod}]: XML inválido — {exc}")
                continue

            base = cotas[["data", "cota_m", "consistido"]]
            if not vazoes.empty:
                base = base.merge(
                    vazoes[["data", "vazao_m3s"]], on="data", how="outer"
                )
            else:
                base = base.assign(vazao_m3s=float("nan"))

            base["rio_id"]      = rio_id
            base["cod_estacao"] = str(cod)
            frames.append(base)
            logger.info(
                f"  {rio_id.capitalize()} [{cod}]: {len(base):,} registros recebidos "
                f"(cotas: {cotas['cota_m'].notna().sum():,} | "
                f"vazões: {len(vazoes):,})"
            )

    if not frames:
        logger.warning("Nenhuma estação retornou dados.")
        return pd.DataFrame(
            columns=["data", "rio_id", "cod_estacao", "cota_m", "vazao_m3s", "consistido"]
        )

    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["rio_id", "cod_estacao", "data"])
        .reset_index(drop=True)
    )
    logger.success(
        f"Série histórica completa: {len(df):,} linhas | "
        f"{df['rio_id'].nunique()} rios | {df['cod_estacao'].nunique()} estações"
    )
    return df[["data", "rio_id", "cod_estacao", "cota_m", "vazao_m3s", "consistido"]]


# ---------------------------------------------------------------------------
# Standalone — teste rápido (Guaíba 2023–2024, inclui a cheia de mai/2024)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="DEBUG",
    )

    df = fetch_serie_historica(87500020, "01/01/2023", "31/12/2024", tipo_dados=1)
    print(f"\nGuaíba (Ponta dos Coatis) 2023-2024 — shape: {df.shape}")
    print(df.head().to_string())
    if not df.empty:
        pico = df.loc[df["cota_m"].idxmax()]
        print(f"\nPico do período: {pico['cota_m']:.2f} m em {pico['data']} "
              f"(régua submersa na cheia de mai/2024 — pico real ~5,3 m "
              f"fica fora da série convencional)")
    print("\nTeste concluído com sucesso!")
