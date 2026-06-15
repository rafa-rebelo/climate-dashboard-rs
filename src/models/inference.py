"""
Agente 3 — Engenheiro DevOps Sênior
Inferência leve do RiverLSTM no GitHub Actions (< 30s, a cada 10 min).

Diferente do train.py (pesado, roda local/Colab), este script é otimizado
para o runner: baixa o .pt do R2 (~0,8 MB), monta a janela de 72 dias a
partir do dataset engenheirado no R2, roda o modelo e faz UPSERT em
river_ai_forecasts. Tolerante a falha: se o modelo não existir, grava
status='pendente' em vez de quebrar o pipeline.

Janela de features: vem do dataset de treino no R2 (treino/{rio}_..parquet),
que é a série diária já com lags/acumulados — reconstruí-la via BigQuery+ANA
a cada 10 min estouraria custo e os 30s. A última cota viva do Supabase é
costurada na ponta da janela para refletir o estado atual.

Horizontes: 24/48/72/144 h (= 1/2/3/6 dias do modelo).
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from models.dataset_builder import _FEATURE_COLS, _r2_client  # noqa: E402

load_dotenv()

_SEQ_LEN = 72
_HORIZONS_DIAS = [1, 2, 3, 6]
_HORIZONS_H = [d * 24 for d in _HORIZONS_DIAS]  # [24, 48, 72, 144]
_MODELO_VERSAO = "v1"

# Cache em memória durante a execução (evita re-download do .pt / scaler)
_MODEL_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Conexão Supabase
# ---------------------------------------------------------------------------

def _pg_connect() -> Optional["psycopg2.extensions.connection"]:
    """Abre conexão com o Supabase (pooler IPv4 no Actions).

    Returns:
        Conexão psycopg2 ou None se as variáveis estiverem ausentes/falharem.
    """
    url = os.getenv("SUPABASE_DATABASE_URL_POOLER") or os.getenv("SUPABASE_DATABASE_URL")
    if not url:
        return None
    try:
        return psycopg2.connect(url, connect_timeout=20)
    except psycopg2.OperationalError as exc:
        logger.error(f"Supabase indisponível: {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. Pesos do R2 (com cache em memória)
# ---------------------------------------------------------------------------

def load_weights_from_r2(rio: str = "guaiba"):
    """Baixa o modelo treinado do R2 para /tmp e o reconstrói em eval().

    Args:
        rio: Slug do rio.

    Returns:
        RiverLSTM carregado, ou None se o .pt não existir no R2.
    """
    if f"model_{rio}" in _MODEL_CACHE:
        return _MODEL_CACHE[f"model_{rio}"]

    import tempfile

    import torch

    from models.lstm_river import RiverLSTM

    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    key = f"models/{rio}_lstm_{_MODELO_VERSAO}.pt"
    if s3 is None or not bucket:
        logger.warning("R2 não configurado — sem pesos.")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            s3.download_fileobj(bucket, key, tmp)
            caminho = tmp.name
    except Exception as exc:  # noqa: BLE001 — chave ausente não deve quebrar
        logger.warning(f"Modelo {key} não encontrado no R2 ({exc}).")
        return None

    payload = torch.load(caminho, map_location="cpu", weights_only=False)
    os.unlink(caminho)
    hparams = payload.get("hparams") or payload.get("metadata", {}).get("hparams", {})
    model = RiverLSTM(**hparams)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _MODEL_CACHE[f"model_{rio}"] = model
    logger.info(f"Modelo carregado do R2: {key}")
    return model


def _load_scaler(rio: str):
    """Carrega o MinMaxParams (scaler) do R2 models/{rio}_scaler.pkl.

    Args:
        rio: Slug do rio.

    Returns:
        Instância do scaler, ou None se ausente.
    """
    if f"scaler_{rio}" in _MODEL_CACHE:
        return _MODEL_CACHE[f"scaler_{rio}"]
    import pickle

    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        return None
    try:
        body = s3.get_object(Bucket=bucket, Key=f"models/{rio}_scaler.pkl")["Body"].read()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Scaler de {rio} ausente no R2 ({exc}).")
        return None
    scaler = pickle.loads(body)
    _MODEL_CACHE[f"scaler_{rio}"] = scaler
    return scaler


# ---------------------------------------------------------------------------
# 2. Janela de features recentes
# ---------------------------------------------------------------------------

def get_recent_features(rio: str = "guaiba", seq_len: int = _SEQ_LEN) -> Optional[np.ndarray]:
    """Monta a janela de seq_len dias normalizada para o LSTM.

    Lê o dataset engenheirado do R2 (treino/{rio}_lstm_dataset_v1.parquet),
    pega os últimos seq_len dias e costura a cota viva do Supabase na última
    linha (para refletir o nível atual). Aplica o scaler do R2.

    Args:
        rio: Slug do rio.
        seq_len: Tamanho da janela (dias).

    Returns:
        Array (1, seq_len, n_features) normalizado, ou None se faltar
        dataset/scaler ou histórico insuficiente.
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    scaler = _load_scaler(rio)
    if s3 is None or not bucket or scaler is None:
        return None

    try:
        body = s3.get_object(
            Bucket=bucket, Key=f"treino/{rio}_lstm_dataset_v1.parquet"
        )["Body"].read()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Dataset de {rio} ausente no R2 ({exc}).")
        return None

    df = pd.read_parquet(io.BytesIO(body)).sort_values("data")
    if len(df) < seq_len:
        logger.warning(f"Histórico insuficiente ({len(df)} < {seq_len}).")
        return None

    # Costura a cota mais recente do Supabase na ponta (estado atual).
    conn = _pg_connect()
    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT nivel_atual_m FROM live_river_levels WHERE rio_id = %s",
                    (rio,),
                )
                row = cur.fetchone()
            if row and row[0] is not None:
                df = df.copy()
                df.iloc[-1, df.columns.get_loc("cota_m")] = float(row[0])
        except psycopg2.Error:
            pass
        finally:
            conn.close()

    janela = df[_FEATURE_COLS].tail(seq_len).to_numpy(dtype=np.float32)
    if np.isnan(janela).any():
        janela = np.nan_to_num(janela, nan=0.0)
    return scaler.transform_x(janela[np.newaxis, :, :])


# ---------------------------------------------------------------------------
# 3. Inferência
# ---------------------------------------------------------------------------

def run_inference(rio: str = "guaiba") -> dict:
    """Executa a previsão de cota para os 4 horizontes, com IC 95%.

    Args:
        rio: Slug do rio.

    Returns:
        Dict com status e, se ok, lista de previsões por horizonte
        (horizonte_h, nivel_previsto_m, ic_lower, ic_upper).
    """
    import torch

    model = load_weights_from_r2(rio)
    if model is None:
        return {"status": "pendente", "motivo": "modelo ausente no R2"}

    X = get_recent_features(rio)
    if X is None:
        return {"status": "pendente", "motivo": "features indisponíveis"}

    scaler = _load_scaler(rio)
    ic = model.predict_with_interval(torch.from_numpy(X).float(), n_samples=30, scaler=scaler)
    base = datetime.now(timezone.utc)

    previsoes = []
    for i, (dias, h) in enumerate(zip(_HORIZONS_DIAS, _HORIZONS_H)):
        previsoes.append({
            "horizonte_h":      h,
            "valid_ts":         base + timedelta(days=dias),
            "nivel_previsto_m": round(float(ic["media"][0, i]), 3),
            "ic_lower":         round(float(ic["ic_inferior"][0, i]), 3),
            "ic_upper":         round(float(ic["ic_superior"][0, i]), 3),
        })
    return {"status": "ok", "previsoes": previsoes}


# ---------------------------------------------------------------------------
# 4. Persistência
# ---------------------------------------------------------------------------

def save_forecast(resultado: dict, rio: str = "guaiba") -> bool:
    """UPSERT das previsões em river_ai_forecasts (snapshot por rio×horizonte).

    Args:
        resultado: Saída de run_inference().
        rio: Slug do rio.

    Returns:
        True se gravou; False em falha de conexão.

    Raises:
        psycopg2.Error: Propagada se a query falhar (guard no main trata).
    """
    conn = _pg_connect()
    if conn is None:
        return False
    try:
        if resultado["status"] == "pendente":
            # Marca pendência sem quebrar (1 linha sentinela por rio).
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO river_ai_forecasts
                        (rio_id, horizonte_h, modelo_versao, status, gerado_em)
                    VALUES (%s, 0, %s, 'pendente', NOW())
                    ON CONFLICT (rio_id, horizonte_h) DO UPDATE SET
                        status = 'pendente', gerado_em = NOW()
                """, (rio, _MODELO_VERSAO))
            conn.commit()
            logger.warning(f"{rio}: status=pendente ({resultado.get('motivo')})")
            return True

        rows = [
            (rio, p["horizonte_h"], p["valid_ts"], p["nivel_previsto_m"],
             p["ic_lower"], p["ic_upper"], _MODELO_VERSAO)
            for p in resultado["previsoes"]
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO river_ai_forecasts
                    (rio_id, horizonte_h, valid_ts, nivel_previsto_m,
                     ic_inferior_m, ic_superior_m, modelo_versao)
                VALUES %s
                ON CONFLICT (rio_id, horizonte_h) DO UPDATE SET
                    valid_ts         = EXCLUDED.valid_ts,
                    nivel_previsto_m = EXCLUDED.nivel_previsto_m,
                    ic_inferior_m    = EXCLUDED.ic_inferior_m,
                    ic_superior_m    = EXCLUDED.ic_superior_m,
                    modelo_versao    = EXCLUDED.modelo_versao,
                    status           = 'ok',
                    gerado_em        = NOW()
            """, rows)
        conn.commit()
        logger.success(f"{rio}: {len(rows)} previsões gravadas em river_ai_forecasts.")
        return True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main com guards
# ---------------------------------------------------------------------------

def main() -> int:
    """Pipeline de inferência para todos os rios com modelo treinado.

    Returns:
        0 em sucesso (mesmo com pendências); 1 se o Supabase falhar.
    """
    t0 = time.monotonic()
    # Rios com modelo treinado — só guaiba por ora; expandir conforme treino.
    rios = ["guaiba"]

    falha_db = False
    for rio in rios:
        try:
            resultado = run_inference(rio)
            ok = save_forecast(resultado, rio)
            if not ok:
                falha_db = True
        except psycopg2.Error as exc:
            logger.error(f"{rio}: erro Supabase — {exc}")
            falha_db = True
        except Exception as exc:  # noqa: BLE001 — inferência nunca derruba o pipeline
            logger.error(f"{rio}: erro inesperado — {exc}")

    dur = time.monotonic() - t0
    logger.info(f"Inferência concluída em {dur:.1f}s.")
    if falha_db:
        logger.error("Falha de persistência no Supabase — saindo com código 1.")
        return 1
    return 0


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               level="INFO")
    sys.exit(main())
