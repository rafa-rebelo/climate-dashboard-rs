"""Testes dos helpers canônicos (utils.comum)."""

from __future__ import annotations

import math

from utils.comum import RIOS_LSTM, rio_slug, safe_float


def test_rio_slug_remove_acentos_e_caixa() -> None:
    assert rio_slug("Guaíba") == "guaiba"
    assert rio_slug("  Jacuí ") == "jacui"
    assert rio_slug("Caí") == "cai"
    assert rio_slug("TAQUARI") == "taquari"


def test_rio_slug_aceita_nao_string() -> None:
    assert rio_slug(None) == "none"
    assert rio_slug(123) == "123"


def test_safe_float_conversoes() -> None:
    assert safe_float("3.14") == 3.14
    assert safe_float(2) == 2.0
    assert safe_float(None) is None
    assert safe_float("abc") is None
    assert safe_float(float("nan")) is None


def test_rios_lstm_slugs_validos() -> None:
    assert len(RIOS_LSTM) == 10
    # todo rio_id deve ser o próprio slug (contrato com river_ai_forecasts)
    for rio in RIOS_LSTM:
        assert rio == rio_slug(rio)
