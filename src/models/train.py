# RODAR LOCALMENTE — não adicionar ao CI/CD
"""
Agente 2 — Engenheiro de Machine Learning Sênior
Treino do RiverLSTM (previsão de cota de rio RS com IC 95%).

EXECUÇÃO: localmente no VSCode (CPU, ~2-3h) ou Google Colab com GPU T4
(~20 min). NUNCA no GitHub Actions — o workflow de coleta roda a cada
10 min e treino é pesado/esporádico. Instruções de Colab no fim do arquivo.

Pipeline: dataset (R2) → sequências/split temporal/normalização →
treino (Huber + Adam + ReduceLROnPlateau + early stopping) → avaliação
(MAE/RMSE por horizonte + cobertura do IC 95%) → artefatos no R2.

Horizontes: o dataset é DIÁRIO, então os 4 horizontes são [1, 2, 3, 6]
dias. O critério de homologação do roadmap ("MAE < 0,20 m para 24h")
mapeia para o horizonte de 1 dia (índice 0).

Uso:
  python src/models/train.py --rio guaiba --epochs 100
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from models.dataset_builder import (  # noqa: E402
    _HORIZONS_DIAS_PADRAO,
    _r2_client,
    build_dataset,
    create_sequences,
    normalize,
    split_sequences,
)
from models.lstm_river import RiverLSTM, save_model  # noqa: E402

load_dotenv()

_HORIZON_LABELS = [f"t+{h}d" for h in _HORIZONS_DIAS_PADRAO]  # ['t+1d','t+2d','t+3d','t+6d']
_HOMOLOG_HORIZONTE_IDX = 0       # 1 dia ≈ "24h" do roadmap
_HOMOLOG_MAE_LIMITE_M = 0.20     # critério de aprovação


# ---------------------------------------------------------------------------
# Carga do dataset (R2 ou Drive/Colab ou reconstrução)
# ---------------------------------------------------------------------------

def _load_dataframe(rio: str, colab: bool) -> pd.DataFrame:
    """Carrega o dataset diário do rio (Parquet do R2, Drive ou reconstrói).

    Args:
        rio: Slug do rio (guaiba, jacui, taquari, sinos, camaqua).
        colab: Se True, tenta ler do Google Drive montado antes do R2.

    Returns:
        DataFrame diário com features + cota_m (saída de build_dataset).
    """
    nome = f"{rio}_lstm_dataset_v1.parquet"

    if colab:
        local = Path(f"/content/drive/MyDrive/climate_rs/treino/{nome}")
        if local.exists():
            logger.info(f"Dataset do Drive: {local}")
            return pd.read_parquet(local)

    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is not None and bucket:
        try:
            body = s3.get_object(Bucket=bucket, Key=f"treino/{nome}")["Body"].read()
            logger.info(f"Dataset do R2: treino/{nome}")
            return pd.read_parquet(io.BytesIO(body))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"R2 sem dataset pronto ({exc}) — reconstruindo via build_dataset.")

    logger.info("Reconstruindo dataset (BigQuery + ANA + R2)...")
    return build_dataset(rio_alvo=rio, start_year=2000)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def _avaliar(
    model: RiverLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Any,
    device: str,
) -> dict[str, Any]:
    """Calcula MAE/RMSE por horizonte e cobertura do IC 95% no teste.

    Args:
        model: Modelo treinado.
        X_test: Janelas de teste normalizadas.
        y_test: Cotas reais de teste (em metros, não normalizadas).
        scaler: MinMaxParams (desnormaliza previsões).
        device: 'cpu' ou 'cuda'.

    Returns:
        Dict com mae_m, rmse_m e cobertura_ic95 por horizonte + homologação.
    """
    Xt = torch.from_numpy(X_test).float().to(device)
    ic = model.predict_with_interval(Xt, n_samples=30, scaler=scaler)  # metros
    media, lo, hi = ic["media"], ic["ic_inferior"], ic["ic_superior"]

    res: dict[str, Any] = {"por_horizonte": {}}
    for i, label in enumerate(_HORIZON_LABELS):
        yt = y_test[:, i]
        err = media[:, i] - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        cobertura = float(np.mean((yt >= lo[:, i]) & (yt <= hi[:, i])) * 100)
        res["por_horizonte"][label] = {
            "mae_m": round(mae, 4),
            "rmse_m": round(rmse, 4),
            "cobertura_ic95_pct": round(cobertura, 1),
        }

    mae_homolog = res["por_horizonte"][_HORIZON_LABELS[_HOMOLOG_HORIZONTE_IDX]]["mae_m"]
    aprovado = mae_homolog < _HOMOLOG_MAE_LIMITE_M
    res["homologacao"] = {
        "horizonte": _HORIZON_LABELS[_HOMOLOG_HORIZONTE_IDX],
        "mae_m": mae_homolog,
        "limite_m": _HOMOLOG_MAE_LIMITE_M,
        "status": "APROVADO" if aprovado else "REPROVADO",
    }
    return res


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar(args: argparse.Namespace) -> dict[str, Any]:
    """Executa o pipeline completo de treino e avaliação.

    Args:
        args: Argumentos de linha de comando (rio, epochs, batch_size, lr,
            device, colab).

    Returns:
        Dict de métricas finais (também salvo em logs/).
    """
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"=== Treino RiverLSTM [{args.rio}] | device={device} ===")

    # 1-2. Dataset → sequências → split temporal → normalização
    df = _load_dataframe(args.rio, args.colab)
    X, y, datas = create_sequences(df, seq_len=72)
    if len(X) < 200:
        raise RuntimeError(f"Sequências insuficientes ({len(X)}) — dataset muito curto.")
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_sequences(X, y, datas=datas)
    X_tr, X_va, X_te, scaler = normalize(X_tr, X_va, X_te, y_train=y_tr, rio_alvo=args.rio)

    # Alvos normalizados para a loss; metros preservados para avaliação.
    y_tr_n = scaler.transform_y(y_tr)
    y_va_n = scaler.transform_y(y_va)

    def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    # shuffle=False: preserva a ordem temporal das janelas
    train_loader = _loader(X_tr, y_tr_n, shuffle=False)
    val_loader   = _loader(X_va, y_va_n, shuffle=False)

    # 3. Modelo + otimização
    model = RiverLSTM(input_size=X_tr.shape[2], output_size=y_tr.shape[1]).to(device)
    # Huber: robusto a outliers de cheia (menos sensível que MSE a picos extremos)
    criterio = nn.HuberLoss(delta=1.0)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, factor=0.5)

    melhor_val = float("inf")
    melhor_state: Optional[dict] = None
    paciencia, limite_paciencia = 0, 15
    hist: list[dict[str, float]] = []
    t0 = time.monotonic()

    for epoca in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterio(model(xb), yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterio(model(xb), yb).item() * len(xb)
        va_loss /= len(val_loader.dataset)
        sched.step(va_loss)
        hist.append({"epoca": epoca, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < melhor_val - 1e-6:
            melhor_val = va_loss
            melhor_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            paciencia = 0
        else:
            paciencia += 1

        if epoca % 10 == 0 or epoca == 1:
            logger.info(
                f"época {epoca:3d}/{args.epochs} | train {tr_loss:.5f} | "
                f"val {va_loss:.5f} | melhor {melhor_val:.5f} | "
                f"{time.monotonic() - t0:.0f}s"
            )
        if paciencia >= limite_paciencia:
            logger.info(f"Early stopping na época {epoca} (val sem melhora há {limite_paciencia}).")
            break

    if melhor_state is not None:
        model.load_state_dict(melhor_state)

    # 4. Avaliação no teste
    metricas = _avaliar(model, X_te, y_te, scaler, device)
    logger.info("--- Avaliação (teste) ---")
    for label, m in metricas["por_horizonte"].items():
        logger.info(f"  {label}: MAE {m['mae_m']:.3f} m | RMSE {m['rmse_m']:.3f} m | "
                    f"IC95 cobre {m['cobertura_ic95_pct']:.0f}%")
    h = metricas["homologacao"]
    nivel = logger.success if h["status"] == "APROVADO" else logger.error
    nivel(f"HOMOLOGAÇÃO ({h['horizonte']}): MAE {h['mae_m']:.3f} m "
          f"(limite {h['limite_m']} m) → {h['status']}")

    # 5. Persistência (R2 via save_model + scaler já salvo pelo normalize)
    metricas["treino"] = {
        "rio": args.rio, "device": device, "epocas_rodadas": len(hist),
        "melhor_val_loss": round(melhor_val, 6),
        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
        "duracao_s": round(time.monotonic() - t0, 1),
    }
    try:
        save_model(model, rio_alvo=args.rio, version="v1",
                   metrics={"mae_24h_m": h["mae_m"], "status": h["status"]})
    except RuntimeError as exc:
        logger.warning(f"Upload do modelo ao R2 falhou ({exc}) — salvando só local.")

    # Cópia local dos pesos
    local_pt = _ROOT / "models_saved" / f"{args.rio}_lstm_v1.pt"
    local_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hparams": model.hparams}, local_pt)
    logger.info(f"Pesos locais: {local_pt}")

    # Log JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = _ROOT / "logs" / f"training_{args.rio}_{ts}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metricas["historico"] = hist
    log_path.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Métricas: {log_path}")

    # Curva de treino (opcional — só se matplotlib disponível)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [h["epoca"] for h in hist]
        plt.figure(figsize=(9, 5))
        plt.plot(ep, [h["train_loss"] for h in hist], label="train")
        plt.plot(ep, [h["val_loss"] for h in hist], label="val")
        plt.xlabel("época"); plt.ylabel("Huber loss"); plt.legend()
        plt.title(f"Treino RiverLSTM — {args.rio}")
        png = _ROOT / "logs" / f"training_{args.rio}_{ts}.png"
        plt.savefig(png, dpi=110, bbox_inches="tight")
        logger.info(f"Curva: {png}")
    except ImportError:
        pass

    return metricas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino RiverLSTM — Monitor RS")
    p.add_argument("--rio", default="guaiba",
                   choices=["guaiba", "jacui", "taquari", "sinos", "camaqua"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--colab", action="store_true", help="Adapta paths ao Google Drive")
    return p.parse_args()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               level="INFO")
    treinar(_parse_args())


# ===========================================================================
# Para rodar no Google Colab com GPU:
#   1. Runtime → Change runtime type → T4 GPU
#   2. !pip install boto3 google-cloud-bigquery pyarrow torch python-dotenv loguru
#   3. from google.colab import drive; drive.mount('/content/drive')
#      (coloque o dataset em /content/drive/MyDrive/climate_rs/treino/ e o .env
#       com as credenciais R2 na raiz, ou exporte as R2_* como variáveis)
#   4. !python train.py --rio guaiba --colab --epochs 150
# ===========================================================================
