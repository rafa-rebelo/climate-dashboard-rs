"""
Agente 1 — Arquiteto de Dados
Rastreador de progresso do pipeline hidrometeorológico RS.

Lê métricas reais do DuckDB + sistema de arquivos e gera dois artefatos:
  - data/processed/progress.json        — relatório operacional detalhado
  - JSONCRACK_FLUXOGRAMA.json           — fluxograma do pipeline para JSONCrack

Execute ao final de cada ciclo do pipeline ou de forma autônoma:
    python -m utils.progress_tracker
"""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).resolve().parents[2]
_DB_PATH   = Path(os.getenv("DB_PATH", str(_ROOT / "data" / "monitor_rs.duckdb")))
_OUT_PROG  = _ROOT / "data" / "processed" / "progress.json"
_OUT_FLOW  = _ROOT / "JSONCRACK_FLUXOGRAMA.json"

# Artefatos que o pipeline deveria produzir
_PARQUETS = {
    "accumulated_rain": _ROOT / "data" / "processed" / "accumulated_rain.parquet",
    "river_status":     _ROOT / "data" / "processed" / "river_status.parquet",
}
_MODELS = {
    "convlstm_v1":      _ROOT / "models_saved" / "convlstm_v1.pt",
    "lstm_sinos_v1":    _ROOT / "models_saved" / "lstm_sinos_v1.pt",
    "lstm_taquari_v1":  _ROOT / "models_saved" / "lstm_taquari_v1.pt",
    "lstm_jacui_v1":    _ROOT / "models_saved" / "lstm_jacui_v1.pt",
    "lstm_guaiba_v1":   _ROOT / "models_saved" / "lstm_guaiba_v1.pt",
}
_COLLECTORS = ["ana_collector", "inmet_collector", "noaa_collector"]
_API_PORT   = int(os.getenv("API_PORT", "8765"))

# Cotas dos rios para calcular % de risco
_COTAS: dict[str, dict[str, float]] = {
    "Sinos":   {"atencao": 3.5,  "alerta": 4.5,  "emergencia": 5.5},
    "Taquari": {"atencao": 3.5,  "alerta": 5.0,  "emergencia": 7.0},
    "Jacuí":   {"atencao": 5.0,  "alerta": 7.0,  "emergencia": 9.0},
    "Guaíba":  {"atencao": 2.5,  "alerta": 3.0,  "emergencia": 3.6},
    "Camaquã": {"atencao": 3.0,  "alerta": 4.0,  "emergencia": 5.0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Retorna timestamp ISO-8601 UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_ts(ts: Any) -> str:
    """Converte timestamp DuckDB → string legível ou 'nunca'."""
    if ts is None:
        return "nunca"
    try:
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d %H:%M UTC")
        return str(ts)[:19]
    except Exception:
        return str(ts)


def _file_info(path: Path) -> dict[str, Any]:
    """
    Retorna status, tamanho e data de modificação de um arquivo.

    Args:
        path: Caminho do arquivo a inspecionar.

    Returns:
        Dict com keys: exists, size_kb, modified_at.
    """
    if path.exists():
        stat = path.stat()
        return {
            "exists":      True,
            "size_kb":     round(stat.st_size / 1024, 1),
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
        }
    return {"exists": False, "size_kb": 0, "modified_at": None}


def _api_running(port: int) -> bool:
    """
    Verifica se a API FastAPI está respondendo na porta especificada.

    Args:
        port: Porta TCP a testar.

    Returns:
        True se a porta estiver aberta, False caso contrário.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _status_icon(ok: bool) -> str:
    """Retorna ✅ ou ❌ baseado em booleano."""
    return "✅ ATIVO" if ok else "❌ INATIVO"


def _river_risk(level_m: float | None, river: str) -> str:
    """
    Calcula categoria de risco para um nível de rio.

    Args:
        level_m: Nível em metros, ou None se não disponível.
        river: Nome do rio para lookup das cotas.

    Returns:
        String de categoria: NORMAL | ATENCAO | ALERTA | EMERGENCIA | DESCONHECIDO.
    """
    if level_m is None:
        return "DESCONHECIDO"
    cfg = _COTAS.get(river, {})
    if not cfg:
        return "DESCONHECIDO"
    if level_m >= cfg["emergencia"]:
        return "EMERGENCIA"
    if level_m >= cfg["alerta"]:
        return "ALERTA"
    if level_m >= cfg["atencao"]:
        return "ATENCAO"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Coleta de métricas do DuckDB
# ---------------------------------------------------------------------------

class _DBMetrics:
    """
    Coleta métricas de todas as 11 tabelas do DuckDB em uma única passagem.

    Attributes:
        available: True se a conexão foi estabelecida com sucesso.
        db_size_mb: Tamanho do arquivo .duckdb em MB.
        tables: Dict com contagens e timestamps por tabela.
        rivers: Dict com nível atual, status e tendência por rio.
        rain_summary: Resumo de acumulados de chuva.
        forecasts_summary: Resumo de previsões disponíveis.
        alerts_summary: Resumo de alertas ativos/histórico.
        model_metrics: Últimas métricas de performance dos modelos ML.
    """

    def __init__(self) -> None:
        self.available    = False
        self.db_size_mb   = 0.0
        self.tables: dict[str, Any]          = {}
        self.rivers: dict[str, Any]          = {}
        self.rain_summary: dict[str, Any]    = {}
        self.forecasts_summary: dict[str, Any] = {}
        self.alerts_summary: dict[str, Any]  = {}
        self.model_metrics: dict[str, Any]   = {}

        if not _DB_PATH.exists():
            logger.warning(f"Banco não encontrado: {_DB_PATH}")
            return

        self.db_size_mb = round(_DB_PATH.stat().st_size / 1024 / 1024, 2)

        try:
            con = duckdb.connect(str(_DB_PATH), read_only=True)
        except duckdb.IOException:
            con = duckdb.connect(str(_DB_PATH))

        try:
            self._collect_tables(con)
            self._collect_rivers(con)
            self._collect_rain(con)
            self._collect_forecasts(con)
            self._collect_alerts(con)
            self._collect_model_metrics(con)
            self.available = True
        except duckdb.Error as exc:
            logger.error(f"Erro ao coletar métricas: {exc}")
        finally:
            con.close()

    def _q(self, con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
        """Executa query e retorna lista de tuplas. Retorna [] em caso de erro."""
        try:
            return con.execute(sql).fetchall()
        except duckdb.Error:
            return []

    def _collect_tables(self, con: duckdb.DuckDBPyConnection) -> None:
        """Conta registros e obtém última inserção para cada tabela.

        Os timestamps são CAST para VARCHAR para evitar dependência de pytz
        ao devolver TIMESTAMPTZ ao Python.
        """
        table_queries = {
            "stations":         "SELECT COUNT(*), CAST(MAX(created_at) AS VARCHAR) FROM stations",
            "rain_readings":    "SELECT COUNT(*), CAST(MAX(ts) AS VARCHAR) FROM rain_readings",
            "rain_accumulated": "SELECT COUNT(*), CAST(MAX(updated_at) AS VARCHAR) FROM rain_accumulated",
            "forecasts":        "SELECT COUNT(*), CAST(MAX(forecast_ts) AS VARCHAR) FROM forecasts",
            "alerts_log":       "SELECT COUNT(*), CAST(MAX(ts) AS VARCHAR) FROM alerts_log",
            "river_levels":     "SELECT COUNT(*), CAST(MAX(ts) AS VARCHAR) FROM river_levels",
            "river_status":     "SELECT COUNT(*), CAST(MAX(updated_at) AS VARCHAR) FROM river_status",
            "water_quality":    "SELECT COUNT(*), CAST(MAX(ts) AS VARCHAR) FROM water_quality",
            "model_metrics":    "SELECT COUNT(*), CAST(MAX(evaluated_at) AS VARCHAR) FROM model_metrics",
            "river_ai_forecasts": "SELECT COUNT(*), CAST(MAX(generated_at) AS VARCHAR) FROM river_ai_forecasts",
            "wrf_hydro_results":  "SELECT COUNT(*), CAST(MAX(run_ts) AS VARCHAR) FROM wrf_hydro_results",
        }
        for table, sql in table_queries.items():
            rows = self._q(con, sql)
            if rows:
                count, last_ts = rows[0]
                self.tables[table] = {
                    "registros":    int(count or 0),
                    "ultima_entrada": _fmt_ts(last_ts),
                }
            else:
                self.tables[table] = {"registros": 0, "ultima_entrada": "nunca"}

    def _collect_rivers(self, con: duckdb.DuckDBPyConnection) -> None:
        """Obtém nível atual, status e tendência de cada rio monitorado."""
        rows = self._q(con, """
            SELECT river, ARGMAX(level_m, ts) AS level_m,
                   ARGMAX(flow_cms, ts) AS flow_cms,
                   CAST(MAX(ts) AS VARCHAR) AS last_ts
            FROM river_levels
            GROUP BY river
            ORDER BY river
        """)
        for river, level, flow, last_ts in rows:
            self.rivers[river] = {
                "nivel_m":   round(float(level), 3) if level else None,
                "vazao_cms": round(float(flow),  3) if flow  else None,
                "ultima_leitura": _fmt_ts(last_ts),
                "status":    _river_risk(float(level) if level else None, river),
            }
            # Tenta pegar status da tabela river_status (mais rico)
            rs_rows = self._q(con, f"""
                SELECT status, trend FROM river_status
                WHERE river = '{river}'
                LIMIT 1
            """)
            if rs_rows:
                self.rivers[river]["status"]    = rs_rows[0][0]
                self.rivers[river]["tendencia"] = rs_rows[0][1]

    def _collect_rain(self, con: duckdb.DuckDBPyConnection) -> None:
        """Obtém estatísticas de chuva acumulada e fontes ativas."""
        # Estações com dados
        rows = self._q(con, """
            SELECT COUNT(DISTINCT station_id),
                   CAST(MAX(date) AS VARCHAR),
                   MAX(rain_24h),
                   AVG(rain_24h)
            FROM rain_accumulated
        """)
        if rows and rows[0][0]:
            n, last_date, max_24h, avg_24h = rows[0]
            self.rain_summary = {
                "estacoes_com_dados": int(n),
                "data_referencia":    _fmt_ts(last_date),
                "max_rain_24h_mm":    round(float(max_24h or 0), 1),
                "media_rain_24h_mm":  round(float(avg_24h or 0), 1),
            }
        else:
            self.rain_summary = {"estacoes_com_dados": 0}

        # Fontes de dados
        src_rows = self._q(con, """
            SELECT source, COUNT(*) FROM stations
            WHERE active = TRUE GROUP BY source ORDER BY source
        """)
        self.rain_summary["fontes_ativas"] = {
            src: int(cnt) for src, cnt in src_rows
        }

    def _collect_forecasts(self, con: duckdb.DuckDBPyConnection) -> None:
        """Obtém resumo das previsões NWP disponíveis."""
        rows = self._q(con, """
            SELECT COUNT(DISTINCT location_name),
                   COUNT(DISTINCT model_source),
                   CAST(MIN(valid_ts) AS VARCHAR),
                   CAST(MAX(valid_ts) AS VARCHAR),
                   COUNT(*)
            FROM forecasts
            WHERE valid_ts >= now()
        """)
        if rows and rows[0][0]:
            locs, models, valid_from, valid_to, total = rows[0]
            self.forecasts_summary = {
                "localidades":    int(locs  or 0),
                "modelos":        int(models or 0),
                "total_horarios": int(total  or 0),
                "valido_de":      _fmt_ts(valid_from),
                "valido_ate":     _fmt_ts(valid_to),
            }
        else:
            self.forecasts_summary = {"localidades": 0, "total_horarios": 0}

        models_rows = self._q(con, """
            SELECT model_source, COUNT(*) FROM forecasts
            WHERE valid_ts >= now()
            GROUP BY model_source
        """)
        self.forecasts_summary["por_modelo"] = {
            m: int(c) for m, c in models_rows
        }

    def _collect_alerts(self, con: duckdb.DuckDBPyConnection) -> None:
        """Obtém contagem de alertas ativos e distribuição por severidade."""
        rows = self._q(con, """
            SELECT
                COUNT(*) FILTER (WHERE resolved = FALSE) AS ativos,
                COUNT(*) FILTER (WHERE resolved = TRUE)  AS resolvidos,
                COUNT(*) AS total,
                CAST(MAX(ts) AS VARCHAR) AS ultimo_alerta
            FROM alerts_log
        """)
        if rows:
            ativos, resolvidos, total, ultimo = rows[0]
            self.alerts_summary = {
                "ativos":      int(ativos    or 0),
                "resolvidos":  int(resolvidos or 0),
                "total":       int(total     or 0),
                "ultimo_em":   _fmt_ts(ultimo),
            }
        sev_rows = self._q(con, """
            SELECT severity, COUNT(*) FROM alerts_log
            WHERE resolved = FALSE
            GROUP BY severity ORDER BY severity
        """)
        self.alerts_summary["por_severidade"] = {
            s: int(c) for s, c in sev_rows
        }

    def _collect_model_metrics(self, con: duckdb.DuckDBPyConnection) -> None:
        """Obtém as últimas métricas de performance de cada modelo."""
        rows = self._q(con, """
            SELECT model_name, river, CAST(evaluated_at AS VARCHAR),
                   mae, rmse, csi, drift_detected
            FROM model_metrics
            WHERE (model_name, evaluated_at) IN (
                SELECT model_name, MAX(evaluated_at)
                FROM model_metrics GROUP BY model_name
            )
            ORDER BY model_name
        """)
        for name, river, ts, mae, rmse, csi, drift in rows:
            self.model_metrics[name] = {
                "rio":           river,
                "avaliado_em":   _fmt_ts(ts),
                "mae":           round(float(mae),  4) if mae  else None,
                "rmse":          round(float(rmse), 4) if rmse else None,
                "csi":           round(float(csi),  4) if csi  else None,
                "drift_detectado": bool(drift),
            }


# ---------------------------------------------------------------------------
# Construção dos artefatos
# ---------------------------------------------------------------------------

def _build_progress(m: _DBMetrics) -> dict[str, Any]:
    """
    Monta o relatório operacional completo do pipeline.

    Args:
        m: Instância de _DBMetrics com todas as métricas coletadas.

    Returns:
        Dict com estrutura completa do relatório para gravação em progress.json.
    """
    # Status geral: OPERACIONAL se banco disponível + >0 leituras recentes
    river_ok  = bool(m.rivers)
    rain_ok   = m.rain_summary.get("estacoes_com_dados", 0) > 0
    fc_ok     = m.forecasts_summary.get("total_horarios", 0) > 0
    api_ok    = _api_running(_API_PORT)
    overall   = "OPERACIONAL" if (m.available and (river_ok or rain_ok)) else \
                "DEGRADADO"   if m.available else "OFFLINE"

    parquets  = {k: _file_info(v) for k, v in _PARQUETS.items()}
    models    = {k: _file_info(v) for k, v in _MODELS.items()}
    n_models  = sum(1 for v in models.values() if v["exists"])

    return {
        "meta": {
            "gerado_em":     _now_iso(),
            "versao":        "2.0",
            "projeto":       "Monitor Hidrometeorológico RS",
            "status_geral":  overall,
            "banco_db":      str(_DB_PATH),
            "banco_size_mb": m.db_size_mb,
            "banco_online":  m.available,
            "api_online":    api_ok,
        },
        "banco_duckdb": {
            "11_tabelas": m.tables,
        },
        "coleta": {
            "ANA_HidroWeb": {
                "status":   _status_icon(m.tables.get("river_levels", {}).get("registros", 0) > 0),
                "registros_river_levels": m.tables.get("river_levels", {}).get("registros", 0),
                "registros_rain_readings": m.tables.get("rain_readings", {}).get("registros", 0),
                "ultima_leitura": m.tables.get("river_levels", {}).get("ultima_entrada", "nunca"),
            },
            "INMET": {
                "status":    _status_icon(
                    m.rain_summary.get("fontes_ativas", {}).get("INMET", 0) > 0
                ),
                "estacoes_inventariadas": m.rain_summary.get(
                    "fontes_ativas", {}
                ).get("INMET", 0),
                "nota": "dados horários requerem IP brasileiro",
            },
            "Open_Meteo_NWP": {
                "status":    _status_icon(fc_ok),
                "previsoes_disponiveis": m.forecasts_summary.get("total_horarios", 0),
                "localidades": m.forecasts_summary.get("localidades", 0),
                "valido_ate":  m.forecasts_summary.get("valido_ate", "nunca"),
            },
        },
        "processamento": {
            "rain_accumulator": {
                "status":    _status_icon(parquets["accumulated_rain"]["exists"]),
                "parquet":   parquets["accumulated_rain"],
                "estacoes_processadas": m.rain_summary.get("estacoes_com_dados", 0),
                "max_rain_24h_mm": m.rain_summary.get("max_rain_24h_mm", 0),
            },
            "river_status_calculator": {
                "status":  _status_icon(parquets["river_status"]["exists"]),
                "parquet": parquets["river_status"],
                "segmentos_calculados": m.tables.get("river_status", {}).get("registros", 0),
            },
        },
        "rios_monitorados": m.rivers,
        "modelos_ml": {
            "modelos_treinados": f"{n_models}/{len(_MODELS)}",
            "arquivos": models,
            "metricas_performance": m.model_metrics,
        },
        "previsoes_nwp": m.forecasts_summary,
        "alertas": m.alerts_summary,
        "api_fastapi": {
            "status":   _status_icon(api_ok),
            "porta":    _API_PORT,
            "endpoints": 8,
        },
        "dashboard_streamlit": {
            "status": "✅ DISPONÍVEL",
            "porta":  8501,
            "paginas": ["Visão Geral", "Rios", "Chuva",
                        "Previsões", "Alertas", "Sistema"],
        },
    }


def _build_jsoncrack(progress: dict[str, Any]) -> dict[str, Any]:
    """
    Converte o relatório de progresso no formato hierárquico para JSONCrack.

    O JSONCrack visualiza qualquer JSON como grafo de nós — cada chave vira
    um nó e o valor vira o rótulo. A estrutura abaixo representa o fluxo
    completo do pipeline em 7 etapas lineares + banco central.

    Args:
        progress: Dict retornado por _build_progress().

    Returns:
        Dict estruturado para visualização no JSONCrack (jsoncrack.com).
    """
    meta    = progress["meta"]
    coleta  = progress["coleta"]
    proc    = progress["processamento"]
    rios    = progress["rios_monitorados"]
    models  = progress["modelos_ml"]
    fc      = progress["previsoes_nwp"]
    alerts  = progress["alertas"]
    api     = progress["api_fastapi"]
    dash    = progress["dashboard_streamlit"]
    db      = progress["banco_duckdb"]["11_tabelas"]

    # Formata cada rio como string resumida para o nó do fluxograma
    def _rio_str(name: str, info: dict) -> str:
        nivel = info.get("nivel_m")
        status = info.get("status", "?")
        trend  = {"SUBINDO": "▲", "DESCENDO": "▼", "ESTAVEL": "→"}.get(
            info.get("tendencia", ""), "")
        if nivel is not None:
            return f"{nivel:.2f}m {trend} [{status}]"
        return f"[{status}]"

    rios_nodes = {
        nome: _rio_str(nome, info)
        for nome, info in rios.items()
    } if rios else {"nota": "sem dados — execute o coletor ANA"}

    # Contagens do banco como resumo compacto
    def _tbl(name: str) -> str:
        t = db.get(name, {})
        n = t.get("registros", 0)
        ts = t.get("ultima_entrada", "nunca")
        return f"{n:,} reg | {ts}"

    return {
        f"Monitor_RS_v{meta['versao']}": {
            "status_geral":  meta["status_geral"],
            "gerado_em":     meta["gerado_em"],
            "banco_size_mb": meta["banco_size_mb"],

            "ETAPA_1__Coleta": {
                "ANA_HidroWeb": {
                    "status":        coleta["ANA_HidroWeb"]["status"],
                    "river_levels":  _tbl("river_levels"),
                    "rain_readings": _tbl("rain_readings"),
                    "ultima_leitura": coleta["ANA_HidroWeb"]["ultima_leitura"],
                },
                "INMET_500_estacoes": {
                    "status":     coleta["INMET"]["status"],
                    "inventario": f"{coleta['INMET']['estacoes_inventariadas']} estações",
                    "nota":       coleta["INMET"]["nota"],
                },
                "Open_Meteo_NWP": {
                    "status":     coleta["Open_Meteo_NWP"]["status"],
                    "previsoes":  f"{fc.get('total_horarios', 0):,} horários",
                    "localidades": f"{fc.get('localidades', 0)} pontos",
                    "valido_ate": fc.get("valido_ate", "nunca"),
                },
            },

            "ETAPA_2__Banco_DuckDB_11_tabelas": {
                "stations":            _tbl("stations"),
                "rain_readings":       _tbl("rain_readings"),
                "rain_accumulated":    _tbl("rain_accumulated"),
                "river_levels":        _tbl("river_levels"),
                "river_status":        _tbl("river_status"),
                "forecasts":           _tbl("forecasts"),
                "alerts_log":          _tbl("alerts_log"),
                "water_quality":       _tbl("water_quality"),
                "model_metrics":       _tbl("model_metrics"),
                "river_ai_forecasts":  _tbl("river_ai_forecasts"),
                "wrf_hydro_results":   _tbl("wrf_hydro_results"),
            },

            "ETAPA_3__Processamento": {
                "rain_accumulator": {
                    "status":    proc["rain_accumulator"]["status"],
                    "parquet":   "accumulated_rain.parquet",
                    "tamanho":   f"{proc['rain_accumulator']['parquet'].get('size_kb', 0)} KB",
                    "estacoes":  proc["rain_accumulator"]["estacoes_processadas"],
                    "max_24h_mm": proc["rain_accumulator"]["max_rain_24h_mm"],
                },
                "river_status_calc": {
                    "status":    proc["river_status_calculator"]["status"],
                    "parquet":   "river_status.parquet",
                    "tamanho":   f"{proc['river_status_calculator']['parquet'].get('size_kb', 0)} KB",
                    "segmentos": proc["river_status_calculator"]["segmentos_calculados"],
                },
            },

            "ETAPA_4__Rios_Monitorados": rios_nodes,

            "ETAPA_5__Modelos_ML": {
                "treinados":         models["modelos_treinados"],
                "ConvLSTM_nowcast":  "❌ PENDENTE" if not models["arquivos"]["convlstm_v1"]["exists"] else "✅ PRONTO",
                "LSTM_Sinos":        "❌ PENDENTE" if not models["arquivos"]["lstm_sinos_v1"]["exists"] else "✅ PRONTO",
                "LSTM_Taquari":      "❌ PENDENTE" if not models["arquivos"]["lstm_taquari_v1"]["exists"] else "✅ PRONTO",
                "LSTM_Jacuí":        "❌ PENDENTE" if not models["arquivos"]["lstm_jacui_v1"]["exists"] else "✅ PRONTO",
                "LSTM_Guaíba":       "❌ PENDENTE" if not models["arquivos"]["lstm_guaiba_v1"]["exists"] else "✅ PRONTO",
                **(
                    {"metricas": {
                        n: f"MAE={v['mae']} RMSE={v['rmse']} CSI={v['csi']}"
                        for n, v in models["metricas_performance"].items()
                    }}
                    if models["metricas_performance"] else {}
                ),
            },

            "ETAPA_6__API_FastAPI": {
                "status":    api["status"],
                "porta":     api["porta"],
                "endpoints": api["endpoints"],
                "rotas": [
                    "POST /v1/token",
                    "GET  /v1/rivers/status",
                    "GET  /v1/rivers/{river}/levels",
                    "GET  /v1/rain/accumulated",
                    "GET  /v1/forecasts",
                    "GET  /v1/alerts",
                    "PUT  /v1/alerts/{id}/resolve",
                    "GET  /v1/stats",
                ],
            },

            "ETAPA_7__Alertas_e_Dashboard": {
                "Telegram_Bot": {
                    "alertas_ativos":   alerts.get("ativos", 0),
                    "alertas_total":    alerts.get("total", 0),
                    "ultimo_alerta":    alerts.get("ultimo_em", "nunca"),
                    "por_severidade":   alerts.get("por_severidade", {}),
                },
                "Dashboard_Streamlit": {
                    "status":  dash["status"],
                    "porta":   dash["porta"],
                    "paginas": dash["paginas"],
                },
            },
        }
    }


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def run(db_path: str | None = None,
        out_progress: Path | None = None,
        out_flowchart: Path | None = None) -> dict[str, Any]:
    """
    Executa o rastreador de progresso completo.

    Coleta métricas do DuckDB, gera os dois artefatos JSON e retorna
    o dicionário de progresso para uso programático.

    Args:
        db_path: Caminho alternativo para o .duckdb (None = usa DB_PATH).
        out_progress:  Destino alternativo para progress.json.
        out_flowchart: Destino alternativo para JSONCRACK_FLUXOGRAMA.json.

    Returns:
        Dict com o relatório de progresso gerado.

    Raises:
        OSError: Se não for possível criar os diretórios de saída.
    """
    if db_path:
        global _DB_PATH
        _DB_PATH = Path(db_path)

    prog_path = out_progress  or _OUT_PROG
    flow_path = out_flowchart or _OUT_FLOW

    prog_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Coletando métricas do DuckDB…")
    metrics  = _DBMetrics()
    progress = _build_progress(metrics)
    flowchart = _build_jsoncrack(progress)

    # Salva progress.json
    prog_path.write_text(
        json.dumps(progress,  ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success(f"progress.json  → {prog_path}")

    # Salva JSONCRACK_FLUXOGRAMA.json
    flow_path.write_text(
        json.dumps(flowchart, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success(f"JSONCRACK_FLUXOGRAMA.json → {flow_path}")

    # Log resumido no console
    meta = progress["meta"]
    logger.info(
        f"Status: {meta['status_geral']} | "
        f"DB: {meta['banco_size_mb']} MB | "
        f"API: {'online' if meta['api_online'] else 'offline'}"
    )

    rios = progress["rios_monitorados"]
    if rios:
        for nome, info in rios.items():
            nivel = info.get("nivel_m") or 0.0
            status = info.get("status", "?")
            logger.info(f"  Rio {nome}: {nivel:.2f}m [{status}]")

    return progress


if __name__ == "__main__":
    # Permite passar caminho alternativo do banco como argumento
    db_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(db_path=db_arg)
