"""
Agente 2 — Cientista de Dados / ML
RiverLSTM — previsão de cotas de rios RS com intervalo de confiança 95%.

Arquitetura: LSTM(13→128, 2 camadas, dropout 0.2) → Dropout → FC(128→64)
→ ReLU → FC(64→4 horizontes [1/2/3/6 dias]).

Incerteza via MC Dropout (Gal & Ghahramani, 2016): n forward passes com
dropout ativo em inferência → média + IC 95% por percentis. Regra do
projeto: TODA previsão de nível de rio sai com intervalo de confiança.

Persistência: state_dict + metadata no R2 (models/{rio}_lstm_{versao}.pt).

Desafio documentado do dataset: o treino viu cota máx 2,24 m e o teste
contém 3,67 m (cheia mai/2024) — avaliar extremos separadamente.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger
from torch import nn

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

from models.dataset_builder import _FEATURE_COLS, _r2_client  # noqa: E402

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


class RiverLSTM(nn.Module):
    """LSTM multi-horizonte para previsão de cota de rio.

    Args:
        input_size: Número de features por passo de tempo (13).
        hidden_size: Dimensão do estado oculto do LSTM.
        num_layers: Camadas LSTM empilhadas.
        dropout: Dropout entre camadas LSTM e antes da cabeça FC
            (o PyTorch não aplica na última camada LSTM, conforme spec).
        output_size: Número de horizontes de previsão.
    """

    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 4,
    ) -> None:
        super().__init__()
        self.hparams: dict[str, Any] = {
            "input_size":  input_size,
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
            "dropout":     dropout,
            "output_size": output_size,
        }
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passo forward.

        Args:
            x: Tensor (batch, seq_len, input_size).

        Returns:
            Tensor (batch, output_size) — cota normalizada por horizonte.
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # último passo da sequência

    # ── Incerteza via MC Dropout ─────────────────────────────────────────

    def _ativar_mc_dropout(self) -> None:
        """eval() geral, mas dropout (FC e interno do LSTM) em modo train.

        O dropout interno do nn.LSTM só atua com o módulo em train();
        como o LSTM não tem camadas sensíveis a modo além do dropout
        (sem BatchNorm), reativá-lo é seguro para MC Dropout.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.LSTM)):
                m.train()

    @torch.no_grad()
    def predict_with_interval(
        self,
        X: torch.Tensor,
        n_samples: int = 10,
        scaler: Optional[Any] = None,
    ) -> dict[str, np.ndarray]:
        """Previsão com intervalo de confiança 95% via MC Dropout.

        Args:
            X: Tensor (batch, seq_len, input_size) normalizado.
            n_samples: Forward passes estocásticos.
            scaler: MinMaxParams do dataset_builder — se fornecido, as
                saídas são desnormalizadas para metros via inverse_y().

        Returns:
            Dict com arrays (batch, output_size):
              media, ic_inferior (p2.5), ic_superior (p97.5).
        """
        self._ativar_mc_dropout()
        amostras = torch.stack([self(X) for _ in range(n_samples)])  # (n, b, h)
        self.eval()

        arr = amostras.cpu().numpy()
        media = arr.mean(axis=0)
        p025  = np.percentile(arr, 2.5, axis=0)
        p975  = np.percentile(arr, 97.5, axis=0)

        if scaler is not None:
            media = scaler.inverse_y(media)
            p025  = scaler.inverse_y(p025)
            p975  = scaler.inverse_y(p975)

        return {"media": media, "ic_inferior": p025, "ic_superior": p975}


# ---------------------------------------------------------------------------
# Persistência no R2
# ---------------------------------------------------------------------------

def save_model(
    model: RiverLSTM,
    rio_alvo: str,
    version: str,
    metrics: dict[str, float],
    extra: dict | None = None,
) -> None:
    """Salva state_dict + metadata do modelo no R2.

    Args:
        model: Instância treinada de RiverLSTM.
        rio_alvo: Slug do rio (guaiba, taquari, ...).
        version: Identificador da versão (ex.: 'v1').
        metrics: Métricas de avaliação (ex.: {'mae_val': 0.08}).
        extra: Metadados adicionais mesclados no topo do metadata (ex.:
            periodo_treino da série usada, ressalva de fonte/régua).

    Raises:
        RuntimeError: Se o R2 não estiver configurado.
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        raise RuntimeError("R2 não configurado — modelo não pode ser salvo.")

    metadata = {
        "rio_alvo":   rio_alvo,
        "version":    version,
        "salvo_em":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics":    metrics,
        "hparams":    model.hparams,
        "features":   _FEATURE_COLS,
        "horizontes_dias": [1, 2, 3, 6],
        **(extra or {}),
    }
    payload = {"state_dict": model.state_dict(), "metadata": metadata}
    buf = io.BytesIO()
    torch.save(payload, buf)
    key = f"models/{rio_alvo}_lstm_{version}.pt"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.success(f"R2 upload: {key} ({buf.tell() / 1024**2:.1f} MB) | metrics={metrics}")

    # Sidecar JSON leve — permite à API (sem torch) ler os metadados de
    # transparência via GET /api/v3/forecasts/model-info.
    import json

    meta_key = f"models/{rio_alvo}_meta.json"
    s3.put_object(
        Bucket=bucket, Key=meta_key,
        Body=json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"R2 upload: {meta_key} (sidecar de metadados)")


def load_model(rio_alvo: str, version: str, device: str = "cpu") -> RiverLSTM:
    """Carrega modelo do R2 e retorna em modo eval().

    Args:
        rio_alvo: Slug do rio.
        version: Versão salva (ex.: 'v1').
        device: 'cpu' ou 'cuda'.

    Returns:
        RiverLSTM com pesos carregados, em eval().

    Raises:
        RuntimeError: Se o R2 não estiver configurado.
        botocore.exceptions.ClientError: Se a chave não existir.
    """
    s3 = _r2_client()
    bucket = os.getenv("R2_BUCKET_NAME", "")
    if s3 is None or not bucket:
        raise RuntimeError("R2 não configurado — modelo não pode ser carregado.")

    key = f"models/{rio_alvo}_lstm_{version}.pt"
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        s3.download_fileobj(bucket, key, tmp)
        caminho = tmp.name

    payload = torch.load(caminho, map_location=device, weights_only=False)
    os.unlink(caminho)

    model = RiverLSTM(**payload["metadata"]["hparams"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    logger.info(
        f"Modelo carregado: {key} (salvo em "
        f"{payload['metadata']['salvo_em']}, metrics={payload['metadata']['metrics']})"
    )
    return model


# ---------------------------------------------------------------------------
# Standalone — validação da arquitetura
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="DEBUG",
    )

    model = RiverLSTM(input_size=13)

    dummy = torch.randn(32, 72, 13)
    saida = model(dummy)
    assert saida.shape == (32, 4), f"shape inesperado: {saida.shape}"
    print(f"Forward OK: entrada {tuple(dummy.shape)} -> saida {tuple(saida.shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros treinaveis: {n_params:,}")

    # IC 95% com 1 amostra (scaler sintético só para a desnormalização)
    from models.dataset_builder import MinMaxParams

    scaler = MinMaxParams()
    scaler.y_min, scaler.y_max = 0.02, 2.24  # valores reais do treino guaiba

    uma = torch.randn(1, 72, 13)
    ic = model.predict_with_interval(uma, n_samples=10, scaler=scaler)
    for h, nome in enumerate(["t+1d", "t+2d", "t+3d", "t+6d"]):
        print(f"  {nome}: {ic['media'][0, h]:.3f} m "
              f"[{ic['ic_inferior'][0, h]:.3f}, {ic['ic_superior'][0, h]:.3f}]")
    largura = (ic["ic_superior"] - ic["ic_inferior"]).mean()
    assert largura > 0, "IC degenerado — dropout não ativo na inferência"
    print(f"Largura media do IC: {largura:.3f} m (dropout MC ativo OK)")

    print("\nArquitetura validada com sucesso!")
