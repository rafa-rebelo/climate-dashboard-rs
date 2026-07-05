"""Testes do pipeline de sequências do LSTM (sem rede/BigQuery)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.dataset_builder import (
    _FEATURE_COLS,
    MinMaxParams,
    create_sequences,
)


def _dataset_diario(n_dias: int, buraco: tuple[int, int] | None = None) -> pd.DataFrame:
    """Dataset sintético diário com todas as _FEATURE_COLS preenchidas."""
    datas = pd.date_range("2020-01-01", periods=n_dias, freq="D").date
    df = pd.DataFrame({"data": datas})
    rng = np.random.default_rng(42)
    for col in _FEATURE_COLS:
        df[col] = rng.uniform(0, 5, n_dias)
    if buraco:
        ini, fim = buraco
        df = df.drop(df.index[ini:fim]).reset_index(drop=True)
    return df


def test_create_sequences_shapes() -> None:
    df = _dataset_diario(120)
    X, y, datas = create_sequences(df, seq_len=30, horizons=[1, 2])
    # amostras = n - seq_len - h_max + 1 = 120 - 30 - 2 + 1 = 89
    assert X.shape == (89, 30, len(_FEATURE_COLS))
    assert y.shape == (89, 2)
    assert len(datas) == 89


def test_create_sequences_descarta_janelas_com_lacuna() -> None:
    # buraco de 10 dias no meio: janelas que o atravessam são descartadas
    df = _dataset_diario(120, buraco=(50, 60))
    X_furado, _, _ = create_sequences(df, seq_len=30, horizons=[1])
    X_cheio, _, _ = create_sequences(_dataset_diario(120), seq_len=30, horizons=[1])
    assert len(X_furado) < len(X_cheio)
    # nenhuma janela contém NaN (lacunas nunca entram disfarçadas)
    assert not np.isnan(X_furado).any()


def test_minmax_roundtrip_do_alvo() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, (50, 12, len(_FEATURE_COLS))).astype(np.float32)
    y = rng.uniform(1.0, 8.0, (50, 4)).astype(np.float32)
    sc = MinMaxParams().fit(X, y)
    Xn = sc.transform_x(X)
    assert Xn.min() >= 0.0 and Xn.max() <= 1.0 + 1e-6
    # desnormalizar o alvo devolve os metros originais
    y_volta = sc.inverse_y(sc.transform_y(y))
    assert np.allclose(y_volta, y, atol=1e-4)
