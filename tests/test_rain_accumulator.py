"""Testes do cálculo de acumulados de chuva (janelas ancoradas no MAX ts)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from processors.rain_accumulator import compute_rain_accumulated


def _serie(sid: str, horas_e_mm: list[tuple[int, float]]) -> pd.DataFrame:
    """Série sintética: (horas atrás do t0, mm) por estação."""
    t0 = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)
    return pd.DataFrame({
        "station_id": sid,
        "ts": [t0 - timedelta(hours=h) for h, _ in horas_e_mm],
        "rain_1h_mm": [mm for _, mm in horas_e_mm],
    })


def test_janelas_ancoradas_no_max_ts_da_estacao() -> None:
    # 1 mm agora, 2 mm há 2h, 4 mm há 5h, 8 mm há 30h (fora de 24h; dentro de 48h)
    df = _serie("A", [(0, 1.0), (2, 2.0), (5, 4.0), (30, 8.0)])
    out = compute_rain_accumulated(df).set_index("station_id")
    r = out.loc["A"]
    assert r["rain_1h"] == 1.0                 # só a leitura do próprio t0
    assert r["rain_3h"] == 3.0                 # 1 + 2
    assert r["rain_6h"] == 7.0                 # 1 + 2 + 4
    assert r["rain_24h"] == 7.0                # 30h fica fora
    assert r["rain_48h"] == 15.0               # inclui a de 30h
    assert r["rain_7d"] == 15.0


def test_estacoes_independentes_e_nan_ignorado() -> None:
    a = _serie("A", [(0, 5.0)])
    b = _serie("B", [(0, 2.0), (1, float("nan"))])
    out = compute_rain_accumulated(pd.concat([a, b], ignore_index=True))
    out = out.set_index("station_id")
    assert out.loc["A", "rain_24h"] == 5.0
    assert out.loc["B", "rain_24h"] == 2.0     # NaN não soma nem quebra


def test_entrada_vazia_retorna_vazio() -> None:
    assert compute_rain_accumulated(pd.DataFrame()).empty
