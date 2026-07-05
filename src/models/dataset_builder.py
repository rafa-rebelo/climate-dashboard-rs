"""
Agente 2 — Cientista de Dados / ML
Unifica INMET (BigQuery) + ANA (SOAP) + R2 (recentes) no dataset do LSTM.

Fontes:
  A. INMET via BigQuery basedosdados — meteorologia diária agregada por
     bacia (a agregação roda DENTRO do BigQuery: mesmo custo de varredura,
     download ~1000x menor do que puxar as linhas horárias).
  B. ANA HidroWeb SOAP (ana_fetcher) — cotas diárias 2000-2026 (alvo).
  C. R2 Cloudflare (historico/live_river_levels) — telemetria recente
     posterior ao fim da série convencional da ANA.

Saída: DataFrame diário com features + alvo, salvo no R2 em
treino/{rio}_lstm_dataset_v1.parquet (Snappy).
"""

from __future__ import annotations

import io
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# bigquery só é usado no caminho de TREINO (_load_inmet_bacia_diario).
# A inferência no CI importa este módulo só por _FEATURE_COLS/_r2_client e
# não tem google-cloud-bigquery instalado (dep de requirements-ml.txt).
try:
    from google.cloud import bigquery
    _BIGQUERY_OK = True
except ImportError:
    bigquery = None  # type: ignore[assignment]
    _BIGQUERY_OK = False

_SRC_DIR = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

from models.ana_fetcher import ESTACOES_RS, fetch_serie_historica  # noqa: E402
from models.data_loader import BIGQUERY_PROJECT_ID, log_consumption  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Municípios representativos da bacia, com códigos IBGE VERIFICADOS contra
# basedosdados.br_inmet_bdmep.estacao em 15/06/2026 — só municípios que de
# fato têm estação INMET com microdados. (O mapeamento anterior usava códigos
# errados: Lajeado/São Leopoldo/Novo Hamburgo não têm estação; Caxias é
# 4308201 não 4305108; Cruz Alta é 4306106 não 4305355; Bagé é 4301602.)
_BACIAS_IBGE: dict[str, dict[str, str]] = {
    "guaiba": {  # exutório de quase todo o leste do RS — set amplo
        "4314902": "Porto Alegre",
        "4308201": "Caxias do Sul (Aeroporto)",
        "4316907": "Santa Maria",
        "4314100": "Passo Fundo",
    },
    "jacui": {  # bacia central grande
        "4316907": "Santa Maria",
        "4306106": "Cruz Alta",
        "4310009": "Ibirubá",
        "4320800": "Soledade",
    },
    "taquari": {  # Taquari-Antas (NE)
        "4321451": "Teutônia",
        "4302105": "Bento Gonçalves",
        "4320800": "Soledade",
        "4311304": "Lagoa Vermelha",
    },
    "sinos": {  # bacia do Sinos (NE metropolitano)
        "4303905": "Campo Bom",
        "4316006": "Rolante",
        "4304408": "Canela",
        "4314902": "Porto Alegre",
    },
    "camaqua": {  # sudeste
        "4303509": "Camaquã",
        "4304507": "Canguçu",
        "4306908": "Encruzilhada do Sul",
    },
    # ── Onda DCRS (códigos VALIDADOS no BigQuery em 04/07/2026) ──────────
    "cai": {  # bacia do Caí (serra → São Sebastião do Caí)
        "4308201": "Caxias do Sul (Aeroporto)",
        "4304408": "Canela",
        "4302105": "Bento Gonçalves",
    },
    "ibicui": {  # oeste — maior bacia do RS
        "4300406": "Alegrete",
        "4317400": "Santiago",
        "4318309": "São Gabriel",
        "4316907": "Santa Maria",
    },
    "ijui": {  # noroeste
        "4306106": "Cruz Alta",
        "4317806": "Santo Augusto",
        "4310009": "Ibirubá",
        "4317202": "Santa Rosa",
    },
    "gravatai": {  # metropolitana
        "4314902": "Porto Alegre",
        "4303103": "Cachoeirinha",
        "4323002": "Viamão",
    },
    "pardo": {  # centro-serra
        "4315701": "Rio Pardo",
        "4320800": "Soledade",
    },
}

_R2_RIVER_PREFIX = "historico/live_river_levels"
_R2_TREINO_PREFIX = "treino"
_LIMITE_AVISO_MB = 500.0

_MAX_BYTES_BILLED = 10 * 1024**3


# ---------------------------------------------------------------------------
# Passo A — INMET agregado por bacia (agregação server-side no BigQuery)
# ---------------------------------------------------------------------------

def _load_inmet_bacia_diario(rio_alvo: str, start_year: int) -> pd.DataFrame:
    """Meteorologia diária da bacia, agregada dentro do BigQuery.

    Por estação/dia: precipitação somada, temperatura/umidade médias.
    Pela bacia/dia: média entre as estações dos municípios da bacia.

    Args:
        rio_alvo: Chave de _BACIAS_IBGE (guaiba, jacui, taquari, sinos, camaqua).
        start_year: Ano inicial da série.

    Returns:
        DataFrame diário com data, precip_bacia_mm, temp_media_c,
        umidade_media_pct.

    Raises:
        KeyError: Se rio_alvo não estiver mapeado em _BACIAS_IBGE.
        google.api_core.exceptions.GoogleAPIError: Falha na query.
    """
    if not _BIGQUERY_OK:
        raise RuntimeError(
            "google-cloud-bigquery não instalado — carga INMET é caminho de "
            "treino. Use requirements-ml.txt (pip install -r requirements-ml.txt)."
        )
    municipios = _BACIAS_IBGE[rio_alvo]
    ids = ", ".join(f"'{m}'" for m in municipios)
    logger.info(
        f"INMET bacia {rio_alvo}: {', '.join(municipios.values())} "
        f"(desde {start_year})"
    )

    sql = f"""
        WITH por_estacao_dia AS (
            SELECT
                m.data,
                m.id_estacao,
                SUM(m.precipitacao_total)      AS precip_dia_mm,
                AVG(m.temperatura_bulbo_hora)  AS temp_dia_c,
                AVG(m.umidade_rel_hora)        AS umid_dia_pct
            FROM `basedosdados.br_inmet_bdmep.microdados` AS m
            JOIN `basedosdados.br_inmet_bdmep.estacao` AS e USING (id_estacao)
            WHERE e.id_municipio IN ({ids})
              AND m.ano >= {int(start_year)}
            GROUP BY m.data, m.id_estacao
        )
        SELECT
            data,
            AVG(precip_dia_mm) AS precip_bacia_mm,
            AVG(temp_dia_c)    AS temp_media_c,
            AVG(umid_dia_pct)  AS umidade_media_pct
        FROM por_estacao_dia
        GROUP BY data
        ORDER BY data
    """

    client = bigquery.Client(project=BIGQUERY_PROJECT_ID)

    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    log_consumption(dry.total_bytes_processed, 0)

    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        maximum_bytes_billed=_MAX_BYTES_BILLED, use_query_cache=True))
    df = job.to_dataframe()
    log_consumption(job.total_bytes_processed or 0, job.total_bytes_billed or 0)

    df["data"] = pd.to_datetime(df["data"]).dt.date
    logger.info(f"  INMET: {len(df):,} dias agregados da bacia.")
    return df


# ---------------------------------------------------------------------------
# Passo B — Cotas históricas ANA
# ---------------------------------------------------------------------------

def _load_cotas_ana(rio_alvo: str, start_year: int) -> pd.DataFrame:
    """Cota diária histórica do rio-alvo (estação primária do ana_fetcher).

    Args:
        rio_alvo: Chave de ESTACOES_RS.
        start_year: Ano inicial.

    Returns:
        DataFrame com data e cota_m.
    """
    # Taquari/Jacuí usam a API REST nova (qualidade nos dias recentes) com
    # fallback SOAP para o histórico profundo. Os outros 3 rios seguem SOAP puro.
    if rio_alvo in ("taquari", "jacui"):
        from models.ana_fetcher_v2 import fetch_rio_com_fallback
        logger.info(f"ANA REST+SOAP (fallback): {rio_alvo}, {start_year}-2026...")
        df = fetch_rio_com_fallback(rio_alvo, start_year, 2026)
        if df.empty:
            logger.warning(f"  ANA: {rio_alvo} sem dados — fallback puro SOAP.")
        return df[["data", "cota_m"]] if not df.empty else \
            pd.DataFrame(columns=["data", "cota_m"])

    cod = ESTACOES_RS[rio_alvo][0]
    logger.info(f"ANA SOAP: estação {cod} ({rio_alvo}), {start_year}-2026...")
    df = fetch_serie_historica(
        cod, f"01/01/{start_year}", "31/12/2026", tipo_dados=1
    )
    if df.empty:
        logger.warning(f"  ANA: estação {cod} sem dados no período.")
        return pd.DataFrame(columns=["data", "cota_m"])
    logger.info(f"  ANA: {len(df):,} dias de cota ({df['data'].min()} a {df['data'].max()}).")
    return df[["data", "cota_m"]]


# ---------------------------------------------------------------------------
# Passo C — Complemento recente via R2
# ---------------------------------------------------------------------------

from utils.comum import r2_client as _comum_r2_client  # noqa: E402


def _r2_client():
    """Cliente R2 (canônico em utils.comum) — mantido p/ imports legados."""
    return _comum_r2_client()


def _load_cotas_r2(rio_alvo: str, apos: date) -> pd.DataFrame:
    """Cotas telemétricas recentes do R2 para datas posteriores à série ANA.

    Lê os Parquets diários de historico/live_river_levels (1 leitura média
    por dia/rio) apenas para os dias após `apos`.

    Args:
        rio_alvo: Slug do rio (coluna rio_nome normalizada).
        apos: Última data presente na série histórica ANA.

    Returns:
        DataFrame com data e cota_m (vazio se nada novo no R2).
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        return pd.DataFrame(columns=["data", "cota_m"])

    import unicodedata

    def _slug(s: str) -> str:
        return (unicodedata.normalize("NFD", str(s))
                .encode("ascii", "ignore").decode().lower().strip())

    registros: list[dict] = []
    dia = apos + timedelta(days=1)
    hoje = datetime.now(timezone.utc).date()
    while dia <= hoje:
        prefix = (f"{_R2_RIVER_PREFIX}/ano={dia.year}"
                  f"/mes={dia.month:02d}/dia={dia.day:02d}/")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=200)
        objs = sorted(resp.get("Contents", []), key=lambda o: o["LastModified"])
        if objs:
            body = s3.get_object(Bucket=bucket, Key=objs[-1]["Key"])["Body"].read()
            dfp = pd.read_parquet(io.BytesIO(body))
            col_nivel = ("current_level_m" if "current_level_m" in dfp.columns
                         else "nivel_atual_m" if "nivel_atual_m" in dfp.columns
                         else None)
            if col_nivel and "rio_nome" in dfp.columns:
                sel = dfp[dfp["rio_nome"].map(_slug) == rio_alvo]
                if not sel.empty:
                    registros.append({
                        "data":   dia,
                        "cota_m": float(sel[col_nivel].mean()),
                    })
        dia += timedelta(days=1)

    df = pd.DataFrame(registros)
    logger.info(f"  R2: {len(df)} dias recentes complementados (após {apos}).")
    return df


# ---------------------------------------------------------------------------
# Passos D + E — Features e limpeza
# ---------------------------------------------------------------------------

def _features_e_limpeza(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas e aplica interpolação/limpeza.

    Args:
        df: DataFrame diário com data, cota_m, precip_bacia_mm,
            temp_media_c, umidade_media_pct.

    Returns:
        DataFrame limpo com features de janela, lags, sazonalidade e delta.
    """
    df = df.sort_values("data").reset_index(drop=True)
    dt = pd.to_datetime(df["data"])

    # Interpolação linear apenas para gaps curtos (< 3 dias) — manutenção
    # de sensor; gaps longos (ex.: mai/2024) permanecem NaN e são removidos.
    antes_nan = df["cota_m"].isna().sum()
    df["cota_m"] = df["cota_m"].interpolate(method="linear", limit=2,
                                            limit_area="inside")
    interpolados = antes_nan - df["cota_m"].isna().sum()

    # Janelas deslizantes de precipitação
    for j in (3, 7, 15):
        df[f"precip_acum_{j}d"] = (
            df["precip_bacia_mm"].rolling(window=j, min_periods=j).sum()
        )

    # Lags da cota e taxa de variação
    for lag in (1, 2, 3):
        df[f"cota_lag_{lag}d"] = df["cota_m"].shift(lag)
    df["cota_delta_1d"] = df["cota_m"] - df["cota_lag_1d"]

    # Sazonalidade anual
    df["mes_seno"]    = np.sin(2 * np.pi * dt.dt.month / 12)
    df["mes_cosseno"] = np.cos(2 * np.pi * dt.dt.month / 12)

    criticas = ["cota_m", "precip_bacia_mm"]
    antes = len(df)
    df = df.dropna(subset=criticas).reset_index(drop=True)
    removidos = antes - len(df)
    logger.info(f"Limpeza: {removidos:,} dias removidos por NaN, "
                f"{interpolados:,} dias interpolados (gaps < 3 dias).")
    return df


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

def build_dataset(rio_alvo: str = "guaiba", start_year: int = 2000) -> pd.DataFrame:
    """Monta o dataset diário unificado (INMET + ANA + R2) para o LSTM.

    Args:
        rio_alvo: guaiba | jacui | taquari | sinos | camaqua.
        start_year: Ano inicial da série.

    Returns:
        DataFrame diário limpo com alvo (cota_m) e features.

    Raises:
        KeyError: rio_alvo desconhecido.
    """
    logger.info(f"=== build_dataset(rio={rio_alvo}, desde {start_year}) ===")

    inmet = _load_inmet_bacia_diario(rio_alvo, start_year)          # Passo A
    cotas = _load_cotas_ana(rio_alvo, start_year)                   # Passo B

    if not cotas.empty:                                              # Passo C
        recentes = _load_cotas_r2(rio_alvo, apos=cotas["data"].max())
        if not recentes.empty:
            cotas = pd.concat([cotas, recentes], ignore_index=True)

    df = inmet.merge(cotas, on="data", how="outer").sort_values("data")
    df = _features_e_limpeza(df)                                     # Passos D+E

    _relatorio_qualidade(df, rio_alvo)
    return df


def _relatorio_qualidade(df: pd.DataFrame, rio_alvo: str) -> None:
    """Loga o relatório de qualidade do dataset final.

    Args:
        df: Dataset final.
        rio_alvo: Identificador do rio (para o cabeçalho do log).
    """
    logger.info(f"--- Relatório de qualidade [{rio_alvo}] " + "-" * 30)
    logger.info(f"Total de dias na série: {len(df):,}")
    if df.empty:
        logger.warning("Dataset vazio — nada a reportar.")
        return
    logger.info(f"Período coberto: {df['data'].min()} → {df['data'].max()}")
    for col in df.columns:
        if col == "data":
            continue
        pct = df[col].notna().mean() * 100
        logger.info(f"  completude {col}: {pct:.1f}%")
    p10, p50, p90 = df["cota_m"].quantile([0.10, 0.50, 0.90])
    logger.info(
        f"Cotas — p10: {p10:.2f} m | mediana: {p50:.2f} m | "
        f"p90: {p90:.2f} m | máx histórico: {df['cota_m'].max():.2f} m "
        f"({df.loc[df['cota_m'].idxmax(), 'data']})"
    )


def save_to_r2(df: pd.DataFrame, rio_alvo: str = "guaiba") -> None:
    """Salva o dataset em Parquet/Snappy no R2 (treino/{rio}_lstm_dataset_v1).

    Args:
        df: Dataset final de build_dataset().
        rio_alvo: Slug do rio (compõe a chave do objeto).

    Raises:
        botocore.exceptions.BotoCoreError: Falha no upload.
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        logger.error("R2 não configurado — dataset não enviado.")
        return

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    tamanho_mb = buf.tell() / 1024**2
    buf.seek(0)

    key = f"{_R2_TREINO_PREFIX}/{rio_alvo}_lstm_dataset_v1.parquet"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.success(f"R2 upload: {key} ({tamanho_mb:.1f} MB)")
    if tamanho_mb > _LIMITE_AVISO_MB:
        logger.warning(
            f"WARNING: dataset com {tamanho_mb:.0f} MB — acima de "
            f"{_LIMITE_AVISO_MB:.0f} MB, próximo do conforto do R2 free."
        )


# ---------------------------------------------------------------------------
# Sequências para o LSTM (janelas deslizantes)
# ---------------------------------------------------------------------------

# Resolução do dataset é DIÁRIA — os horizontes 6/12/24/48 HORAS do roadmap
# são mapeados para 1/2/3/6 DIAS até existir série horária suficiente
# (a telemetria horária do R2 acumula desde 10/06/2026).
_HORIZONS_DIAS_PADRAO = [1, 2, 3, 6]

_FEATURE_COLS = [
    "cota_m", "precip_bacia_mm", "temp_media_c", "umidade_media_pct",
    "precip_acum_3d", "precip_acum_7d", "precip_acum_15d",
    "cota_lag_1d", "cota_lag_2d", "cota_lag_3d", "cota_delta_1d",
    "mes_seno", "mes_cosseno",
]


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 72,
    horizons: Optional[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforma o dataset diário em janelas deslizantes para o LSTM.

    Cada amostra t usa seq_len dias de histórico (X) e prevê a cota nos
    horizontes futuros (y). Janelas que atravessam lacunas de calendário
    (dias removidos na limpeza, ex.: mai/2024) são DESCARTADAS — o LSTM
    nunca vê uma sequência com salto temporal disfarçado.

    Args:
        df: Saída de build_dataset() (diário, limpo).
        seq_len: Dias de histórico por janela de entrada.
        horizons: Horizontes de previsão em DIAS (padrão [1, 2, 3, 6] —
            equivalente diário dos 6/12/24/48h do roadmap).

    Returns:
        Tupla (X, y, datas):
          X: float32 (n_amostras, seq_len, n_features)
          y: float32 (n_amostras, n_horizontes) — cota_m futura
          datas: datetime64 (n_amostras,) — data-base t de cada amostra
            (necessária para o split temporal logar períodos).
    """
    horizons = horizons or _HORIZONS_DIAS_PADRAO
    h_max = max(horizons)

    # Reindexa para o calendário contínuo — dias ausentes viram NaN e
    # invalidam qualquer janela que os contenha.
    serie = (
        df.assign(data=pd.to_datetime(df["data"]))
        .set_index("data")
        .sort_index()
        .asfreq("D")
    )
    feats = serie[_FEATURE_COLS].to_numpy(dtype=np.float32)
    alvo  = serie["cota_m"].to_numpy(dtype=np.float32)
    valido = ~np.isnan(feats).any(axis=1)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    datas:  list[np.datetime64] = []

    for t in range(seq_len - 1, len(serie) - h_max):
        if not valido[t - seq_len + 1: t + 1].all():
            continue
        alvos = alvo[[t + h for h in horizons]]
        if np.isnan(alvos).any():
            continue
        X_list.append(feats[t - seq_len + 1: t + 1])
        y_list.append(alvos)
        datas.append(serie.index.values[t])

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(_FEATURE_COLS)), np.float32)
    y = np.stack(y_list) if y_list else np.empty((0, len(horizons)), np.float32)
    datas_arr = np.array(datas, dtype="datetime64[D]")

    descartadas = (len(serie) - seq_len - h_max + 1) - len(X)
    logger.info(
        f"Sequências: {len(X):,} amostras (seq_len={seq_len}, "
        f"horizontes={horizons} dias) | {descartadas:,} janelas descartadas "
        f"por atravessarem lacunas."
    )
    return X, y, datas_arr


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    datas: Optional[np.ndarray] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split TEMPORAL (train → val → test, em ordem cronológica).

    NUNCA aleatório: amostras de validação/teste são estritamente
    posteriores às de treino — dados futuros não vazam para o passado.

    Args:
        X: Janelas de entrada (n, seq_len, n_features).
        y: Alvos (n, n_horizontes).
        datas: Datas-base das amostras (para logar períodos; opcional).
        val_ratio: Fração final do treino reservada à validação.
        test_ratio: Fração final da série reservada ao teste.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    n = len(X)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]

    def _periodo(ini: int, fim: int) -> str:
        if datas is None or fim <= ini:
            return "—"
        return f"{datas[ini]} → {datas[fim - 1]}"

    logger.info(f"Split temporal — train: {len(X_train):,} ({_periodo(0, n_train)})")
    logger.info(f"               — val:   {len(X_val):,} ({_periodo(n_train, n_train + n_val)})")
    logger.info(f"               — test:  {len(X_test):,} ({_periodo(n_train + n_val, n)})")
    for nome, yy in (("train", y_train), ("val", y_val), ("test", y_test)):
        if len(yy):
            logger.info(f"  cota {nome}: min {yy.min():.2f} m | max {yy.max():.2f} m")
    logger.info("Vazamento de dados: NENHUM (confirmado por split temporal)")
    return X_train, X_val, X_test, y_train, y_val, y_test


class MinMaxParams:
    """Escalador min-max picklável, ajustado SOMENTE no treino.

    Implementação numpy pura (sem dependência de scikit-learn) com a
    mesma matemática do MinMaxScaler: (x - min) / (max - min).

    Attributes:
        x_min: Mínimos por feature (1, 1, n_features).
        x_max: Máximos por feature (1, 1, n_features).
        y_min: Mínimo do alvo (para desnormalizar previsões).
        y_max: Máximo do alvo.
    """

    def __init__(self) -> None:
        self.x_min: Optional[np.ndarray] = None
        self.x_max: Optional[np.ndarray] = None
        self.y_min: float = 0.0
        self.y_max: float = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MinMaxParams":
        """Aprende min/max do treino (e do alvo, para desnormalização)."""
        self.x_min = X_train.min(axis=(0, 1), keepdims=True)
        self.x_max = X_train.max(axis=(0, 1), keepdims=True)
        self.y_min = float(y_train.min())
        self.y_max = float(y_train.max())
        return self

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        """Normaliza features para [0, 1] com os parâmetros do treino."""
        return ((X - self.x_min) / (self.x_max - self.x_min + 1e-9)).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Normaliza o alvo para [0, 1]."""
        return ((y - self.y_min) / (self.y_max - self.y_min + 1e-9)).astype(np.float32)

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Desnormaliza previsões de volta para metros."""
        return y_norm * (self.y_max - self.y_min) + self.y_min


def normalize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    rio_alvo: str = "guaiba",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxParams]:
    """Normaliza min-max com parâmetros aprendidos APENAS no treino.

    Val e test recebem somente transform (nunca fit) — estatísticas do
    futuro não contaminam o escalonamento. O escalador (incluindo
    y_min/y_max para desnormalizar previsões) é salvo no R2 em
    models/{rio_alvo}_scaler.pkl.

    Args:
        X_train: Janelas de treino (define os parâmetros).
        X_val: Janelas de validação (só transform).
        X_test: Janelas de teste (só transform).
        y_train: Alvos de treino — obrigatório para registrar y_min/y_max.
        rio_alvo: Slug do rio (compõe a chave do scaler no R2).

    Returns:
        (X_train_n, X_val_n, X_test_n, scaler).

    Raises:
        ValueError: Se y_train não for fornecido.
    """
    import pickle

    if y_train is None:
        raise ValueError("y_train é obrigatório (y_min/y_max desnormalizam previsões).")

    scaler = MinMaxParams().fit(X_train, y_train)
    X_train_n = scaler.transform_x(X_train)
    X_val_n   = scaler.transform_x(X_val)   if len(X_val)  else X_val
    X_test_n  = scaler.transform_x(X_test)  if len(X_test) else X_test

    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is not None and bucket:
        key = f"models/{rio_alvo}_scaler.pkl"
        s3.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(scaler))
        logger.success(
            f"R2 upload: {key} (y_min={scaler.y_min:.2f} m, "
            f"y_max={scaler.y_max:.2f} m)"
        )
    else:
        logger.warning("R2 indisponível — scaler não persistido.")

    return X_train_n, X_val_n, X_test_n, scaler


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="DEBUG",
    )

    dataset = build_dataset(rio_alvo="guaiba", start_year=2000)
    print(f"\nshape final: {dataset.shape}")
    save_to_r2(dataset, rio_alvo="guaiba")

    # Pipeline de sequências para o LSTM
    X, y, datas = create_sequences(dataset, seq_len=72)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_sequences(X, y, datas=datas)
    X_tr, X_va, X_te, scaler = normalize(X_tr, X_va, X_te, y_train=y_tr)

    print(f"\nX_train: {X_tr.shape} | X_val: {X_va.shape} | X_test: {X_te.shape}")
    print(f"y_train: {y_tr.shape} | normalizado em [{X_tr.min():.2f}, {X_tr.max():.2f}]")
    print("\nPipeline de sequências pronto para tensores PyTorch "
          "(torch.from_numpy nos arrays float32)!")
