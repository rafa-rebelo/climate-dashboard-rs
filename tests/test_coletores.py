"""Testes dos parsers puros dos coletores CEMADEN e DCRS (sem rede)."""

from __future__ import annotations

from collectors.cemaden_collector import _num, _pick, _unwrap
from collectors.dcrs_collector import _v, _vs, normalize


# ── CEMADEN ────────────────────────────────────────────────────────────────

def test_unwrap_lista_envelope_geojson() -> None:
    assert _unwrap([{"a": 1}]) == [{"a": 1}]
    assert _unwrap({"items": [{"a": 1}]}) == [{"a": 1}]
    geo = {"features": [{"properties": {"cod": "1"},
                         "geometry": {"coordinates": [-51.0, -29.0]}}]}
    u = _unwrap(geo)
    assert u[0]["latitude"] == -29.0 and u[0]["longitude"] == -51.0
    assert _unwrap("lixo") == []


def test_num_aceita_virgula_decimal() -> None:
    assert _num("3,5") == 3.5
    assert _num("2.5") == 2.5
    assert _num(None) is None
    assert _num("x") is None


def test_pick_variantes_de_chave() -> None:
    d = {"codestacao": "123", "vazio": ""}
    assert _pick(d, ("codEstacao", "codestacao")) == "123"
    assert _pick(d, ("vazio", "nada")) is None


# ── DCRS (GraphQL Valores{value}) ──────────────────────────────────────────

def test_v_extrai_value_aninhado_tolerante() -> None:
    rio = {"rio_nivel": {"value": 3.5}, "quebrado": None}
    assert _v(rio, "rio_nivel") == 3.5
    assert _v(rio, "quebrado") is None
    assert _v(rio, "inexistente") is None
    assert _v({}, "a", "b") is None


def test_vs_extrai_string() -> None:
    assert _vs({"rio_nome": {"value": " Caí "}}, "rio_nome") == "Caí"
    assert _vs({"rio_nome": {"value": None}}, "rio_nome") is None


def test_normalize_dcrs_estacao_minima() -> None:
    est = [{
        "codigo": "DCRS-00001", "timestamp": "2026-07-05T12:00:00.000Z",
        "name": {"general": "Teste", "local": ""},
        "position": {"bacia": "RS - Rio Caí", "latitude": -29.5, "longitude": -51.2},
        "data": {"rio": {"rio_nivel": {"value": 2.0}},
                 "chuva": {"acumulado": {"h024": {"value": 4.5}}}},
    }, {
        # sem lat/lon → descartada
        "codigo": "DCRS-00002", "name": {}, "position": {}, "data": {},
    }]
    df = normalize(est)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["codigo"] == "DCRS-00001"
    assert r["rio_nivel"] == 2.0
    assert r["chuva_24h"] == 4.5
    assert str(r["timestamp"].tzinfo) is not None   # tz-aware UTC
