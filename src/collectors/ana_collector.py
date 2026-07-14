"""
Agente 1 — Arquiteto de Dados
Coletor ANA HidroWeb v1.0.3984.2 — COMPLETO
API: https://www.ana.gov.br/hidrowebservice

23 endpoints mapeados do Swagger oficial.
Endpoints utilizados para o Sistema Climático RS:

  AUTENTICAÇÃO (2):
    OAUth/v1                          → token SSO
    OAUthPermissoes/v1                → permissões do token

  INVENTÁRIO / CATÁLOGO (8):
    HidroInventarioEstacoes/v1        → inventário completo de estações
    HidroUF/v1                        → lista por UF
    HidroSubBacia/v1                  → sub-bacias
    HidroBacia/v1                     → bacias hidrográficas
    HidroRio/v1                       → corpos hídricos
    HidroMunicipio/v1                 → municípios
    HidroEntidade/v1                  → entidades responsáveis
    HidrosatInventarioEstacoes/v1     → estações virtuais satélite

  SÉRIES TELEMETRICAS (2):  <- PRINCIPAL — tempo real
    HidroinfoanaSerieTelemetricaAdotada/v1   → chuva+nível+vazão (até 30d)
    HidroinfoanaSerieTelemetricaDetalhada/v1 → dados brutos (até 30d)

  SÉRIES CONVENCIONAIS (6): <- histórico/qualidade
    HidroSerieChuva/v1            → chuva convencional (até 366d)
    HidroSerieCotas/v1            → cotas convencionais (até 366d)
    HidroSerieVazao/v1            → vazão convencional (até 366d)
    HidroSerieQA/v1               → qualidade da água (até 366d)
    HidroSerieSedimentos/v1       → sedimentos (até 366d)
    HidroSerieResumoDescarga/v1   → descarga líquida (até 366d)

  SÉRIES HIDRÁULICAS (3):
    HidroSerieCurvaDescarga/v1    → curvas de descarga
    HidroSeriePerfilTransversal/v1→ perfil transversal
    HidroSerieGranulometria/v1    → granulometria

  SATÉLITE (1):
    HidrosatSerieDados/v1         → estimativa satélite (até 366d)

NOTA DE AMBIENTE:
  Este módulo usa `niquests` (fork moderno de requests, compatível com
  urllib3-future instalado pelo openmeteo-requests). Não use `requests`
  diretamente neste ambiente — urllib3.exceptions não está disponível.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Garante que src/ está no sys.path quando executado como script standalone
_SRC_DIR = Path(__file__).resolve().parent.parent  # src/collectors/.. = src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import niquests
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

# Importação lazy para evitar circular import e manter compatibilidade como script standalone
try:
    from database.hybrid_writer import HybridWriter as _HybridWriter
    _HW_OK = True
except ImportError:  # execução fora do src/ sem PYTHONPATH correto
    _HW_OK = False

# CF_WORKER_URL → Cloudflare Worker proxy (resolve bloqueio de IP da ANA em GH Actions).
# Sem a variável: usa ANA diretamente (funciona localmente, bloqueado no runner).
_CF_PROXY = os.getenv("CF_WORKER_URL", "").rstrip("/")
BASE_URL = f"{_CF_PROXY}/hidrowebservice" if _CF_PROXY else "https://www.ana.gov.br/hidrowebservice"

# ── Estações dos rios críticos do RS ─────────────────────────────────────────
# 03/07/2026 — COTAS POR ESTAÇÃO (não mais por rio). Cada régua tem zero
# próprio: Muçum inunda aos 18 m, Encantado aos 14 m — uma cota única por rio
# gerava FALSO EMERGENCIA (7,3 m em Muçum era classificado 146% da "cota 5 m").
# Fonte das cotas do Taquari: SACE-SGB / Defesa Civil (Alerta Hidrológico da
# Bacia do Rio Taquari). Demais rios: cotas herdadas do config antigo até o
# Agente 5 fornecer os valores oficiais por ponto de controle.
# Códigos mortos removidos (fora do inventário ativo, telemetria vazia):
# 86724000, 86696000, 86600000, 87030000 (Lagoa dos Patos — SEM estação
# telemétrica de nível com dado na ANA; monitoramento ao vivo inviável),
# 86900000 Porto Gomes (Operando=1 mas 0 leituras em 30 dias).
RIOS_RS: dict[str, dict] = {
    "Sinos": {
        # 11/06/2026: códigos antigos (87386000/87374000/87392000/87358000)
        # estavam sem telemetria ou desativados — 0 leituras em 30 dias.
        "estacoes": {
            87382000: {"nome": "São Leopoldo",     "cota_atencao_m": 3.5, "cota_max_hist_m": 8.64,
                       "cota_alerta_m": 4.5, "cota_emergencia_m": 5.5},
            87380000: {"nome": "Campo Bom",        "cota_atencao_m": 3.5,
                       "cota_alerta_m": 4.5, "cota_emergencia_m": 5.5},
            87376000: {"nome": "Foz do Paranhana", "cota_atencao_m": 3.5,
                       "cota_alerta_m": 4.5, "cota_emergencia_m": 5.5},
        },
        "municipios": ["São Leopoldo", "Novo Hamburgo", "Canoas"],
    },
    "Taquari": {
        # Cotas oficiais SGB/Defesa Civil (atenção/alerta/inundação):
        # Muçum 15/17/18 · Encantado 11/13/14. 86720000 Encantado validado
        # em 02/07/2026 (2.780 leituras/30d, cota ao vivo).
        "estacoes": {
            86510000: {"nome": "Muçum",     "cota_atencao_m": 15.0, "cota_max_hist_m": 26.11,
                       "cota_alerta_m": 17.0, "cota_emergencia_m": 18.0},
            86720000: {"nome": "Encantado", "cota_atencao_m": 11.0,
                       "cota_alerta_m": 13.0, "cota_emergencia_m": 14.0},
        },
        "municipios": ["Lajeado", "Estrela", "Encantado", "Muçum"],
    },
    "Jacuí": {
        # 05/07/2026: 86500000 (Passo Carreiro — RIO CARREIRO, rio errado)
        # morreu em 04/07 e foi REMOVIDO. Substituído por 85900000 RIO PARDO
        # (cidade) — leito do médio Jacuí e a MESMA régua usada no treino do
        # LSTM (série centenária). Telemetria esparsa (~3 leituras/dia, lag
        # de ~2 dias) porém do rio certo. Cotas PROVISÓRIAS por percentil da
        # série de treino (P90/P95/P99) — substituir por oficiais (Agente 5).
        "estacoes": {
            85900000: {"nome": "Rio Pardo (cidade)", "cota_atencao_m": 7.6, "cota_max_hist_m": 20.21,
                       "cota_alerta_m": 9.2, "cota_emergencia_m": 11.7},
        },
        "municipios": ["Rio Pardo", "Santa Cruz do Sul", "Cachoeira do Sul"],
    },
    "Guaíba": {
        # PROXY DOCUMENTADO (Agente 5, reavaliado 05/07/2026): 87010000
        # Triunfo e 87020000 São Jerônimo são estações do BAIXO JACUÍ na
        # zona de remanso do delta — proxy de montante do Guaíba. As réguas
        # próprias do Guaíba seguem inviáveis na telemetria ANA (Ponta dos
        # Coatis 87500020 sem cota; Cais Mauá 87450004 instável/504).
        # Cotas 2,5/3,0/3,6 = referência oficial Guaíba/POA.
        "estacoes": {
            87010000: {"nome": "Triunfo (baixo Jacuí)",      "cota_atencao_m": 2.5,
                       "cota_alerta_m": 3.0, "cota_emergencia_m": 3.6},
            87020000: {"nome": "São Jerônimo (baixo Jacuí)", "cota_atencao_m": 2.5,
                       "cota_alerta_m": 3.0, "cota_emergencia_m": 3.6},
        },
        "municipios": ["Porto Alegre", "Eldorado do Sul"],
    },
    "Camaquã": {
        # 15/06: 87540000/87530000 sem telemetria (0 leituras, congelado
        # desde 12/06). Substituído por 87905000 Passo do Mendonça (Cristal),
        # telemétrica ativa — mesmo padrão do fix do Sinos.
        "estacoes": {
            87905000: {"nome": "Passo do Mendonça", "cota_atencao_m": 3.0, "cota_max_hist_m": 7.97,
                       "cota_alerta_m": 4.0, "cota_emergencia_m": 5.0},
        },
        "municipios": ["Camaquã", "Cristal"],
    },
    # ── Onda DCRS (05/07/2026) — mesmas réguas ANA usadas no TREINO do LSTM,
    # telemetria ativa verificada (2.700+ leituras/30d cada). Cotas
    # PROVISÓRIAS por percentil da série histórica de treino (P90/P95/P99,
    # ~20 anos) — SUBSTITUIR pelas cotas oficiais quando o Agente 5 as
    # homologar (SGB/Defesa Civil municipal). Gravataí fica FORA daqui:
    # régua ANA (Albatroz) morta desde abr/2024 — observado só via DCRS.
    "Caí": {
        "estacoes": {
            87170000: {"nome": "Barca do Caí", "cota_atencao_m": 5.3, "cota_max_hist_m": 17.51,
                       "cota_alerta_m": 7.2, "cota_emergencia_m": 11.1},
        },
        "municipios": ["São Sebastião do Caí", "Montenegro", "Feliz"],
    },
    "Ibicuí": {
        "estacoes": {
            76560000: {"nome": "Manoel Viana", "cota_atencao_m": 6.6, "cota_max_hist_m": 14.84,
                       "cota_alerta_m": 7.8, "cota_emergencia_m": 10.3},
        },
        "municipios": ["Manoel Viana", "Alegrete", "São Vicente do Sul"],
    },
    "Ijuí": {
        "estacoes": {
            75230000: {"nome": "Santo Ângelo", "cota_atencao_m": 2.7, "cota_max_hist_m": 7.58,
                       "cota_alerta_m": 3.4, "cota_emergencia_m": 4.6},
        },
        "municipios": ["Santo Ângelo", "Entre-Ijuís", "Ijuí"],
    },
}


# ── Utilitários ───────────────────────────────────────────────────────────────

def _fmt(dt: datetime) -> str:
    """Formata datetime para o padrão ANA: 'DD/MM/AAAA HH:MM:SS'.

    Args:
        dt: Objeto datetime a formatar.

    Returns:
        String no formato esperado pela API ANA HidroWeb.
    """
    return dt.strftime("%d/%m/%Y %H:%M:%S")


def _to_df(data: dict | list) -> pd.DataFrame:
    """Converte resposta JSON da ANA em DataFrame de forma segura.

    Tenta chaves comuns de envelope antes de tratar o objeto raiz.

    Args:
        data: Resposta JSON da API (dict ou list).

    Returns:
        DataFrame com os registros retornados. Vazio se data for vazio.
    """
    if isinstance(data, list):
        return pd.DataFrame(data)
    for key in ("items", "data", "dados", "result", "estacoes"):
        if key in data and isinstance(data[key], list):
            return pd.DataFrame(data[key])
    return pd.DataFrame([data]) if data else pd.DataFrame()


def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas da API ANA para padrão interno.

    A ANA retorna nomes variados por endpoint e versão. Este mapeamento
    centraliza a normalização para evitar duplicação nos métodos.

    Args:
        df: DataFrame com colunas no formato original da API.

    Returns:
        DataFrame com colunas renomeadas para o padrão interno do sistema.
    """
    mapa = {
        # timestamp — snake_case e camelCase da API real
        "DataHora":          "timestamp", "dataHora":          "timestamp",
        "Data_Hora_Medicao": "timestamp", "Data_Hora_Leitura": "timestamp",
        "Data":              "timestamp", "data":              "timestamp",
        # código da estação — variantes da API real
        "CodEstacao":    "station_code", "codEstacao":    "station_code",
        "codigoestacao": "station_code", "CodigoEstacao": "station_code",
        "Codigo_Estacao":"station_code", "codigo_estacao":"station_code",
        "Codigo":        "station_code", "codigo":        "station_code",
        # chuva — com e sem underscore
        "Chuva":          "chuva_mm", "chuva":          "chuva_mm",
        "Chuva_Adotada":  "chuva_mm", "ChuvaAdotada":   "chuva_mm",
        "Precipitacao":   "chuva_mm", "precipitacao":   "chuva_mm",
        # nível / cota
        "Nivel":          "nivel_m", "nivel":          "nivel_m",
        "Cota":           "nivel_m", "cota":           "nivel_m",
        "Cota_Adotada":   "nivel_m", "CotaAdotada":    "nivel_m",
        "NivelAdotado":   "nivel_m",
        # vazão
        "Vazao":          "vazao_m3s", "vazao":         "vazao_m3s",
        "Vazao_Adotada":  "vazao_m3s", "VazaoAdotada":  "vazao_m3s",
        # qualidade da água
        "pH":                   "ph",                 "ph":               "ph",
        "Turbidez":             "turbidez_ntu",       "turbidez":         "turbidez_ntu",
        "OxigenioDissolvidoMg": "od_mgl",             "oxigenioDissolvidoMg": "od_mgl",
        "Condutividade":        "condutividade_us_cm","condutividade":    "condutividade_us_cm",
        "TemperaturaAgua":      "temp_agua_c",        "temperaturaAgua":  "temp_agua_c",
        # sedimentos / descarga
        "ConcentracaoSedimento": "sedimento_mg_l",
        "DescargaLiquida":       "descarga_m3s",
    }
    return df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})


def classificar_nivel(nivel_m: float, cfg: dict) -> str:
    """Classifica o status do rio conforme cotas configuradas.

    Args:
        nivel_m: Nível atual em metros.
        cfg: Dicionário com cota_atencao_m, cota_alerta_m, cota_emergencia_m.

    Returns:
        "EMERGENCIA" | "ALERTA" | "ATENCAO" | "NORMAL"
    """
    if nivel_m >= cfg["cota_emergencia_m"]:
        return "EMERGENCIA"
    if nivel_m >= cfg["cota_alerta_m"]:
        return "ALERTA"
    if nivel_m >= cfg["cota_atencao_m"]:
        return "ATENCAO"
    return "NORMAL"


# ── Cliente HTTP ──────────────────────────────────────────────────────────────

class ANAClient:
    """Cliente REST para ANA HidroWeb v1.0.3984.2.

    Cobre todos os 23 endpoints do Swagger oficial.
    Autenticação SSO via OAUth/v1 com renovação automática.
    Retry com backoff exponencial via tenacity em todo GET.

    Usa ``niquests.Session`` (fork compatível com urllib3-future)
    em vez de ``requests.Session`` — ver nota no módulo.

    Args:
        identificador: CPF/CNPJ cadastrado no portal ANA (env: ANA_IDENTIFICADOR).
        senha: Senha do portal ANA (env: ANA_SENHA).
        token: Token JWT pré-obtido (env: ANA_TOKEN). Dispensa user/pass.
    """

    def __init__(
        self,
        identificador: str = "",
        senha:         str = "",
        token:         str = "",
    ) -> None:
        self.identificador = identificador
        self.senha         = senha
        self._token:     Optional[str]      = token or None
        self._token_exp: Optional[datetime] = (
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=50) if token else None
        )
        self.session = niquests.Session()
        # Content-Type e Accept específicos por endpoint — não no nível da sessão
        # (ANA retorna XML em vários endpoints; Accept: application/json causa 406)
        self.session.headers.update({
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
            logger.info("Token ANA carregado diretamente.")

    # ── Autenticação ─────────────────────────────────────────────

    @staticmethod
    def _auth_erro_transitorio(exc: BaseException) -> bool:
        """Indica se o erro de autenticação é transitório e vale retry.

        A ANA retorna 417 de forma intermitente para IPs de saída da
        Cloudflare (variam a cada request) — com as mesmas credenciais
        que funcionam em outras tentativas. 429/5xx também são passageiros.

        Args:
            exc: Exceção capturada durante a autenticação.

        Returns:
            True se a exceção indicar falha transitória (vale retentar).
        """
        if isinstance(exc, niquests.exceptions.HTTPError) and exc.response is not None:
            return exc.response.status_code in (417, 429, 500, 502, 503, 504)
        return isinstance(exc, niquests.exceptions.RequestException)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_auth_erro_transitorio.__func__),
        reraise=True,
    )
    def autenticar(self) -> str:
        """Obtém token SSO via GET /EstacoesTelemetricas/OAUth/v1.

        Retry exponencial (5 tentativas, 2s → 30s) para 417/429/5xx —
        a ANA rejeita intermitentemente requisições de IPs Cloudflare.

        Returns:
            Token JWT válido por ~60 minutos.

        Raises:
            ValueError: Se o token não estiver presente na resposta.
            niquests.HTTPError: Se a requisição HTTP falhar após retries.
        """
        logger.info("Autenticando na ANA HidroWeb...")
        resp = self.session.get(
            f"{BASE_URL}/EstacoesTelemetricas/OAUth/v1",
            headers={"Identificador": self.identificador, "Senha": self.senha},
            timeout=30,
        )
        resp.raise_for_status()
        data  = resp.json()
        # Envelope real: {"status":"OK","items":{"tokenautenticacao":"eyJ..."}}
        items = data.get("items") or data
        token = items.get("tokenautenticacao") or items.get("token")
        if not token:
            raise ValueError(f"Token não encontrado na resposta: {data}")

        self._token     = token
        self._token_exp = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=50)
        self.session.headers["Authorization"] = f"Bearer {token}"
        logger.success("Token ANA obtido — válido por 50min.")
        return token

    def verificar_permissoes(self) -> dict:
        """Verifica permissões do token via OAUthPermissoes/v1.

        Returns:
            Dict com as permissões concedidas ao usuário autenticado.

        Raises:
            niquests.HTTPError: Se a requisição falhar.
        """
        self._garantir_token()
        resp = self.session.get(
            f"{BASE_URL}/EstacoesTelemetricas/OAUthPermissoes/v1",
            timeout=15,
        )
        resp.raise_for_status()
        perms = resp.json()
        logger.info(f"Permissoes ANA: {perms}")
        return perms

    def _garantir_token(self) -> None:
        """Renova token se ausente ou próximo do vencimento (< 5min restantes)."""
        if (
            not self._token
            or not self._token_exp
            or datetime.now(timezone.utc).replace(tzinfo=None) >= self._token_exp
        ):
            self.autenticar()

    # ── GET genérico com retry ────────────────────────────────────

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(niquests.exceptions.RequestException),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict) -> dict | list:
        """Executa GET autenticado com retry exponencial (2s → 30s, 5 tentativas).

        Args:
            endpoint: Caminho após BASE_URL, sem barra inicial.
            params: Query parameters da requisição.

        Returns:
            Resposta JSON da API (dict ou list).

        Raises:
            niquests.exceptions.RequestException: Após 5 tentativas sem sucesso.
        """
        self._garantir_token()
        resp = self.session.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            timeout=60,
        )
        if resp.status_code == 401:
            logger.warning("401 recebido — renovando token e retentando...")
            self.autenticar()
            resp = self.session.get(
                f"{BASE_URL}/{endpoint}", params=params, timeout=60
            )
        resp.raise_for_status()
        return resp.json()

    # ══════════════════════════════════════════════════════════════
    # GRUPO 1 — INVENTÁRIO / CATÁLOGO
    # ══════════════════════════════════════════════════════════════

    def inventario_estacoes(
        self,
        cod_estacao: int | None = None,
        cod_bacia:   int | None = None,
        uf:          str = "RS",
    ) -> pd.DataFrame:
        """Inventário completo de estações — HidroInventarioEstacoes/v1.

        Parâmetros conforme Swagger v1.0.3984.2: nomes em português,
        código como inteiro, UF como enum completo.

        Args:
            cod_estacao: Código numérico da estação ANA (opcional).
            cod_bacia: Código numérico da bacia (opcional).
            uf: Sigla do estado — padrão "RS". Enum oficial da API.

        Returns:
            DataFrame com metadados completos das estações.

        Raises:
            niquests.exceptions.RequestException: Se todas as tentativas falharem.
        """
        logger.info(f"Inventario de estacoes — UF: {uf}")
        params: dict = {}
        if cod_estacao is not None:
            params["Código da Estação"] = cod_estacao
        if cod_bacia is not None:
            params["Código da Bacia"] = cod_bacia
        if uf:
            params["Unidade Federativa"] = uf
        data = self._get("EstacoesTelemetricas/HidroInventarioEstacoes/v1", params)
        df   = _to_df(data)
        logger.success(f"{len(df)} estacoes no inventario.")
        return df

    def listar_uf(self) -> pd.DataFrame:
        """Lista unidades federativas disponíveis — HidroUF/v1.

        Returns:
            DataFrame com códigos e nomes das UFs.
        """
        data = self._get("EstacoesTelemetricas/HidroUF/v1", {})
        return _to_df(data)

    def listar_sub_bacias(self) -> pd.DataFrame:
        """Lista sub-bacias hidrográficas — HidroSubBacia/v1.

        Returns:
            DataFrame com código e nome das sub-bacias.
        """
        data = self._get("EstacoesTelemetricas/HidroSubBacia/v1", {})
        return _to_df(data)

    def listar_bacias(self) -> pd.DataFrame:
        """Lista bacias hidrográficas principais — HidroBacia/v1.

        Returns:
            DataFrame com as bacias cadastradas.
        """
        data = self._get("EstacoesTelemetricas/HidroBacia/v1", {})
        return _to_df(data)

    def listar_rios(self) -> pd.DataFrame:
        """Lista corpos hídricos cadastrados — HidroRio/v1.

        Returns:
            DataFrame com código e nome dos rios.
        """
        data = self._get("EstacoesTelemetricas/HidroRio/v1", {})
        return _to_df(data)

    def listar_municipios(self) -> pd.DataFrame:
        """Lista municípios na base HIDRO — HidroMunicipio/v1.

        Returns:
            DataFrame com código IBGE e nome dos municípios.
        """
        data = self._get("EstacoesTelemetricas/HidroMunicipio/v1", {})
        return _to_df(data)

    def listar_entidades(self) -> pd.DataFrame:
        """Lista entidades responsáveis por estações — HidroEntidade/v1.

        Returns:
            DataFrame com código e nome das entidades operadoras.
        """
        data = self._get("EstacoesTelemetricas/HidroEntidade/v1", {})
        return _to_df(data)

    def inventario_hidrosat(self) -> pd.DataFrame:
        """Inventário de estações virtuais de satélite — HidrosatInventarioEstacoes/v1.

        Returns:
            DataFrame com estações virtuais do HidroSat.
        """
        logger.info("Inventario HidroSat (estacoes satelite)...")
        data = self._get("EstacoesTelemetricas/HidrosatInventarioEstacoes/v1", {})
        df   = _to_df(data)
        logger.success(f"{len(df)} estacoes HidroSat.")
        return df

    # ══════════════════════════════════════════════════════════════
    # GRUPO 2 — SÉRIES TELEMETRICAS (tempo real — limite 30 dias)
    # ══════════════════════════════════════════════════════════════

    def serie_adotada(
        self,
        cod: int,
        range_intervalo: str = "DIAS_30",
        data_busca: str | None = None,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série adotada telemétrica — chuva + nível + vazão.

        HidroinfoanaSerieTelemetricaAdotada/v1.
        Contract real (Swagger v1.0.3984.2): sem DataInicio/DataFim,
        usa Range Intervalo de busca + Tipo Filtro Data.

        Args:
            cod: Código numérico da estação ANA.
            range_intervalo: Janela de coleta. Enum:
                MINUTO_5/10/15/30, HORA_1..HORA_24, DIAS_2/7/14/21/30.
                Padrão: "DIAS_30".
            data_busca: Data de referência "yyyy-MM-dd" (opcional).
                        None = usa a data mais recente disponível.
            tipo_filtro: "DATA_LEITURA" (padrão) ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com colunas timestamp, chuva_mm, nivel_m, vazao_m3s,
            station_code.

        Raises:
            niquests.exceptions.RequestException: Se a coleta falhar após retries.
        """
        params: dict = {
            "Código da Estação":    cod,
            "Tipo Filtro Data":     tipo_filtro,
            "Range Intervalo de busca": range_intervalo,
        }
        if data_busca:
            params["Data de Busca (yyyy-MM-dd)"] = data_busca
        data = self._get(
            "EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1",
            params,
        )
        df = _normalizar_colunas(_to_df(data))
        df["station_code"] = str(cod)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], dayfirst=False, errors="coerce"
            )
        for col in ["chuva_mm", "nivel_m", "vazao_m3s"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # ANA telemetria retorna Cota_Adotada em centímetros — converte para metros
        if "nivel_m" in df.columns:
            df["nivel_m"] = df["nivel_m"] / 100.0
        return df

    def serie_detalhada(
        self,
        cod: int,
        range_intervalo: str = "DIAS_30",
        data_busca: str | None = None,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série detalhada telemétrica — dados adotados + brutos.

        HidroinfoanaSerieTelemetricaDetalhada/v1.
        Mesma contract de serie_adotada — sem DataInicio/DataFim.

        Args:
            cod: Código numérico da estação ANA.
            range_intervalo: Janela de coleta (enum igual a serie_adotada).
            data_busca: Data de referência "yyyy-MM-dd" (opcional).
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com leituras brutas e adotadas da estação.
        """
        params: dict = {
            "Código da Estação":        cod,
            "Tipo Filtro Data":         tipo_filtro,
            "Range Intervalo de busca": range_intervalo,
        }
        if data_busca:
            params["Data de Busca (yyyy-MM-dd)"] = data_busca
        data = self._get(
            "EstacoesTelemetricas/HidroinfoanaSerieTelemetricaDetalhada/v1",
            params,
        )
        df = _normalizar_colunas(_to_df(data))
        df["station_code"] = str(cod)
        return df

    # ══════════════════════════════════════════════════════════════
    # GRUPO 3 — SÉRIES CONVENCIONAIS (histórico — limite 366 dias)
    # ══════════════════════════════════════════════════════════════

    def _params_conv(
        self,
        cod: int,
        data_ini: str,
        data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> dict:
        """Monta params padrão para endpoints convencionais (Swagger v1.0.3984.2).

        Args:
            cod: Código numérico da estação.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            Dict pronto para passar a _get().
        """
        return {
            "Código da Estação":        cod,
            "Tipo Filtro Data":         tipo_filtro,
            "Data Inicial (yyyy-MM-dd)": data_ini,
            "Data Final (yyyy-MM-dd)":  data_fim,
        }

    def serie_chuva(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série de chuva convencional — HidroSerieChuva/v1. Limite: 366 dias.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" (padrão) ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com timestamp e chuva_mm.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSerieChuva/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _normalizar_colunas(_to_df(data))

    def serie_cotas(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série de cotas convencionais — HidroSerieCotas/v1. Limite: 366 dias.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com timestamp e nivel_m.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSerieCotas/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _normalizar_colunas(_to_df(data))

    def serie_vazao(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série de vazão convencional — HidroSerieVazao/v1. Limite: 366 dias.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com timestamp e vazao_m3s.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSerieVazao/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _normalizar_colunas(_to_df(data))

    def serie_qualidade_agua(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série de qualidade da água — HidroSerieQA/v1. Limite: 366 dias.

        Retorna pH, turbidez, oxigênio dissolvido, condutividade e temperatura.
        Inspirado nas estações HCMR Hydro Stations (projeto grego).

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com parâmetros físico-químicos da água.
        """
        logger.info(f"Qualidade da agua — estacao {cod}")
        data = self._get(
            "EstacoesTelemetricas/HidroSerieQA/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        df = _normalizar_colunas(_to_df(data))
        df["station_code"] = str(cod)
        return df

    def serie_sedimentos(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série de sedimentos — HidroSerieSedimentos/v1. Limite: 366 dias.

        Útil para análise de erosão e assoreamento de reservatórios.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com concentração de sedimentos em mg/L.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSerieSedimentos/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _normalizar_colunas(_to_df(data))

    def serie_descarga(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Resumo de descarga líquida — HidroSerieResumoDescarga/v1. Limite: 366 dias.

        Fornece vazão medida in situ, mais precisa que estimativas telemetricas.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com descarga_m3s por data de medição.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSerieResumoDescarga/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _normalizar_colunas(_to_df(data))

    # ══════════════════════════════════════════════════════════════
    # GRUPO 4 — SÉRIES HIDRÁULICAS
    # ══════════════════════════════════════════════════════════════

    def curva_descarga(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Curvas de descarga líquida — HidroSerieCurvaDescarga/v1.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com pontos da curva cota-descarga.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSerieCurvaDescarga/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _to_df(data)

    def perfil_transversal(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Perfil transversal do leito — HidroSeriePerfilTransversal/v1.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com batimetria transversal da seção.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSeriePerfilTransversal/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _to_df(data)

    def granulometria(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Granulometria do leito — HidroSerieGranulometria/v1.

        Args:
            cod: Código numérico da estação ANA.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com distribuição granulométrica do sedimento.
        """
        data = self._get(
            "EstacoesTelemetricas/HidroSerieGranulometria/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _to_df(data)

    # ══════════════════════════════════════════════════════════════
    # GRUPO 5 — SATÉLITE HidroSat
    # ══════════════════════════════════════════════════════════════

    def serie_hidrosat(
        self, cod: int, data_ini: str, data_fim: str,
        tipo_filtro: str = "DATA_LEITURA",
    ) -> pd.DataFrame:
        """Série HidroSat — estimativa de chuva por satélite — HidrosatSerieDados/v1.

        Limite: 366 dias. Útil para bacias sem estações telemetricas.

        Args:
            cod: Código numérico da estação virtual HidroSat.
            data_ini: Data início "yyyy-MM-dd".
            data_fim: Data fim   "yyyy-MM-dd".
            tipo_filtro: "DATA_LEITURA" ou "DATA_ULTIMA_ATUALIZACAO".

        Returns:
            DataFrame com estimativas de precipitação por satélite.
        """
        logger.info(f"HidroSat — estacao virtual {cod}")
        data = self._get(
            "EstacoesTelemetricas/HidrosatSerieDados/v1",
            self._params_conv(cod, data_ini, data_fim, tipo_filtro),
        )
        return _normalizar_colunas(_to_df(data))


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE COLETA DE ALTO NÍVEL
# ══════════════════════════════════════════════════════════════════════════════

def _horas_para_range(horas: int) -> str:
    """Converte horas para o enum Range Intervalo de busca da API ANA.

    Args:
        horas: Janela de coleta em horas.

    Returns:
        Valor do enum mais próximo (ex.: HORA_6, DIAS_7, DIAS_30).
    """
    if horas <= 24:
        return f"HORA_{min(horas, 24)}"
    dias = min(horas // 24, 30)
    for limite in (2, 7, 14, 21, 30):
        if dias <= limite:
            return f"DIAS_{limite}"
    return "DIAS_30"


def coletar_rios_rs(
    client: ANAClient,
    horas_back: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coleta nível, vazão e chuva dos rios críticos do RS via telemetria.

    Janela padrão de 7 dias (DIAS_7): réguas convencionais-telemetrizadas
    como Rio Pardo/Jacuí 85900000 reportam a cada 1-3 dias — com 72h elas
    caíam fora da janela e o snapshot congelava (bug do Jacuí, 04-05/07).
    O nível usa sempre a ÚLTIMA leitura válida da janela.

    Itera sobre todos os rios em RIOS_RS e suas estações, calcula o status
    operacional (NORMAL/ATENCAO/ALERTA/EMERGENCIA) e persiste em Parquet.

    Args:
        client: ANAClient autenticado.
        horas_back: Janela de coleta em horas. Máximo: 720h (30 dias).

    Returns:
        Tupla (df_serie, df_status):
            - df_serie: Série temporal completa de todos os rios.
            - df_status: Status atual (último valor) por estação.

    Raises:
        OSError: Se não for possível criar os diretórios de saída.
    """
    agora         = datetime.now(timezone.utc).replace(tzinfo=None)
    range_intervalo = _horas_para_range(horas_back)

    all_series:  list[pd.DataFrame] = []
    status_list: list[dict]         = []

    logger.info("Coletando rios criticos RS...")

    for rio, cfg in RIOS_RS.items():
        logger.info(f"  {rio} — {len(cfg['estacoes'])} estacoes")
        for cod, est in cfg["estacoes"].items():
            try:
                df = client.serie_adotada(cod, range_intervalo=range_intervalo)
                if df.empty:
                    logger.warning(f"    Sem dados: {cod} ({est['nome']})")
                    continue

                # Cotas POR ESTAÇÃO (cada régua tem zero próprio — SGB).
                df["rio_nome"]          = rio
                df["estacao_nome"]      = est["nome"]
                df["cota_atencao_m"]    = est["cota_atencao_m"]
                df["cota_alerta_m"]     = est["cota_alerta_m"]
                df["cota_emergencia_m"] = est["cota_emergencia_m"]
                df["cota_max_hist_m"]   = est.get("cota_max_hist_m")

                if "nivel_m" in df.columns:
                    serie_nivel = df["nivel_m"].dropna()
                    if not serie_nivel.empty:
                        nivel  = float(serie_nivel.iloc[-1])
                        status = classificar_nivel(nivel, est)
                        pct    = nivel / est["cota_alerta_m"] * 100
                        vazao: Optional[float] = None
                        if "vazao_m3s" in df.columns:
                            vz = df["vazao_m3s"].dropna()
                            if not vz.empty:
                                vazao = float(vz.iloc[-1])
                        df["status"] = status
                        status_list.append({
                            "rio_nome":          rio,
                            "station_code":      cod,
                            "estacao_nome":      est["nome"],
                            "current_level_m":   round(nivel, 3),
                            "cota_atencao_m":    est["cota_atencao_m"],
                            "cota_alerta_m":     est["cota_alerta_m"],
                            "cota_emergencia_m": est["cota_emergencia_m"],
                            "cota_max_hist_m":   est.get("cota_max_hist_m"),
                            "pct_cota_alerta":   round(pct, 1),
                            "flow_m3s":          vazao,
                            "alert_level":       status,
                            "updated_at":        agora.isoformat(),
                        })
                        _log_nivel(f"{rio}/{est['nome']}", cod, nivel, pct, status)

                all_series.append(df)
                time.sleep(0.5)

            except Exception as exc:
                logger.error(f"    {cod} ({rio}): {exc}")

    df_serie  = pd.concat(all_series,  ignore_index=True) if all_series  else pd.DataFrame()
    df_status = pd.DataFrame(status_list)

    if _HW_OK:
        writer = _HybridWriter()
        if not df_serie.empty:
            res = writer.write_river_levels(df_serie)
            logger.success(
                f"river_levels — {res.pg_rows:,} registros PG "
                f"| R2: {'ok' if res.r2_ok else 'skip'}"
            )
        if not df_status.empty:
            res_st = writer.write_river_status(df_status)
            logger.success(
                f"river_status — {res_st.pg_rows:,} registros PG "
                f"| R2: {'ok' if res_st.r2_ok else 'skip'}"
            )
            if not res_st.pg_ok and res_st.pg_error:
                logger.error(f"river_status PG falhou: {res_st.pg_error}")
            _imprimir_resumo(df_status)
    else:
        # Fallback direto (sem HybridWriter) — mantém compatibilidade
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        if not df_serie.empty:
            df_serie.to_parquet("data/raw/river_levels.parquet", index=False)
            logger.success(f"river_levels.parquet — {len(df_serie)} registros.")
        if not df_status.empty:
            df_status.to_parquet("data/processed/river_status.parquet", index=False)
            _imprimir_resumo(df_status)

    return df_serie, df_status


def coletar_qualidade_agua_rs(
    client: ANAClient,
    dias_back: int = 30,
) -> pd.DataFrame:
    """Coleta qualidade da água via HidroSerieQA/v1 para estações dos rios RS.

    Parâmetros coletados: pH, turbidez, oxigênio dissolvido, condutividade,
    temperatura da água. Metodologia inspirada nas estações HCMR Hydro Stations.

    Args:
        client: ANAClient autenticado.
        dias_back: Janela de coleta em dias. Máximo: 366 dias.

    Returns:
        DataFrame consolidado com todos os parâmetros de qualidade por estação.
        DataFrame vazio se nenhuma estação retornar dados.
    """
    agora    = datetime.now(timezone.utc).replace(tzinfo=None)
    data_ini = (agora - timedelta(days=min(dias_back, 366))).strftime("%Y-%m-%d")
    data_fim = agora.strftime("%Y-%m-%d")

    estacoes_qa = [
        cod
        for cfg in RIOS_RS.values()
        for cod in cfg["estacoes"]
    ]

    logger.info(f"Qualidade da agua — {len(estacoes_qa)} estacoes RS")
    all_qa: list[pd.DataFrame] = []

    for cod in estacoes_qa:
        try:
            df = client.serie_qualidade_agua(cod, data_ini, data_fim)
            if not df.empty:
                all_qa.append(df)
            time.sleep(0.5)
        except Exception as exc:
            logger.debug(f"QA {cod}: {exc}")

    if not all_qa:
        logger.warning("Nenhum dado de qualidade da agua disponivel.")
        return pd.DataFrame()

    df_qa = pd.concat(all_qa, ignore_index=True)
    if _HW_OK:
        res = _HybridWriter().write_water_quality(df_qa)
        logger.success(
            f"water_quality — {res.pg_rows:,} registros PG "
            f"| R2: {'ok' if res.r2_ok else 'skip'}"
        )
    else:
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df_qa.to_parquet("data/raw/water_quality.parquet", index=False)
        logger.success(f"water_quality.parquet — {len(df_qa)} registros.")
    return df_qa


# ── Helpers de log ────────────────────────────────────────────────────────────

def _log_nivel(
    rio: str, cod: str, nivel: float, pct: float, status: str
) -> None:
    """Loga o nível do rio com indicador visual de status."""
    label = {"EMERGENCIA": "[EMERGENCIA]", "ALERTA": "[ALERTA]",
             "ATENCAO": "[ATENCAO]", "NORMAL": "[NORMAL]"}.get(status, "")
    msg = f"    {label} {rio} [{cod}]: {nivel:.2f}m ({pct:.0f}% cota) — {status}"
    if status in ("EMERGENCIA", "ALERTA"):
        logger.warning(msg)
    else:
        logger.info(msg)


def _imprimir_resumo(df: pd.DataFrame) -> None:
    """Loga um resumo tabular do status de todos os rios monitorados."""
    logger.info("=" * 55)
    logger.info("RESUMO — RIOS DO RS")
    logger.info("=" * 55)
    for _, r in df.iterrows():
        label = {"EMERGENCIA": "[EMERGENCIA]", "ALERTA": "[ALERTA]",
                 "ATENCAO": "[ATENCAO]", "NORMAL": "[NORMAL]"}.get(
            r.get("alert_level", ""), ""
        )
        logger.info(
            f"  {label} {r['rio_nome']:<15} "
            f"{r['current_level_m']:.2f}m / {r['cota_alerta_m']}m "
            f"({r['pct_cota_alerta']:.0f}%) — {r['alert_level']}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ANA_IDENTIFICADOR = os.getenv("ANA_IDENTIFICADOR", "")
    ANA_SENHA         = os.getenv("ANA_SENHA",         "")
    ANA_TOKEN         = os.getenv("ANA_TOKEN",         "")

    if not ANA_IDENTIFICADOR and not ANA_TOKEN:
        logger.error("Configure ANA_IDENTIFICADOR+ANA_SENHA ou ANA_TOKEN no .env")
        sys.exit(1)

    logger.info("Coletor ANA HidroWeb — Sistema Climatico RS")
    logger.info(f"API: {BASE_URL}")
    logger.info("Endpoints disponiveis: 23 (Swagger v1.0.3984.2)")

    client = ANAClient(
        identificador=ANA_IDENTIFICADOR,
        senha=ANA_SENHA,
        token=ANA_TOKEN,
    )

    if not ANA_TOKEN:
        client.autenticar()
        try:
            client.verificar_permissoes()
        except Exception as exc:
            logger.warning(f"verificar_permissoes indisponivel: {exc}")

    # Janela 168h (7 dias): réguas convencionais-telemetrizadas (ex.: Rio
    # Pardo/Jacuí) reportam a cada 1-3 dias — 72h as deixava fora da janela.
    # Chuva ANA removida em 07/2026: redundante (rain_accumulator cobre via
    # INMET/CEMADEN) e não persistia no Supabase/R2.
    df_serie, df_status = coletar_rios_rs(client,           horas_back=168)
    df_qa               = coletar_qualidade_agua_rs(client, dias_back=30)

    logger.info("Coleta ANA finalizada.")
    logger.info(f"  Registros rios:  {len(df_serie)}")
    logger.info(f"  Rios c/ status:  {len(df_status)}")
    logger.info(f"  Registros QA:    {len(df_qa)}")
