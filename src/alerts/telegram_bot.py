"""
Agente 3 — Engenheiro de Software
Bot Telegram para alertas hidrometeorológicos do RS.

Funcionalidades:
  - Alertas de nível de rio quando cruza cota de atenção/alerta/emergência
  - Alertas de chuva intensa (1h/24h acima dos limiares do config.yaml)
  - Throttling anti-spam: 1 alerta/rio/hora (evita flood em enchentes)
  - Mensagens ricas em Markdown com emoji de severidade
  - Comando /status para consulta on-demand via polling
  - Persiste alertas enviados em DuckDB (tabela alerts_log)
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import niquests
import pandas as pd
import yaml
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR / "src"))

from database.db_manager import ClimateDB  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_CONFIG_PATH  = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_STATUS_PATH  = Path(__file__).resolve().parents[2] / "data" / "processed" / "river_status.parquet"
_ACCUM_PATH   = Path(__file__).resolve().parents[2] / "data" / "processed" / "accumulated_rain.parquet"

_TELEGRAM_API = "https://api.telegram.org/bot{token}"

# Emoji por severidade
_EMOJI: dict[str, str] = {
    "EMERGENCIA": "\U0001f6a8",   # 🚨
    "ALERTA":     "\U000026a0",   # ⚠️
    "ATENCAO":    "\U0001f7e1",   # 🟡
    "NORMAL":     "\U0001f7e2",   # 🟢
    "INFO":       "\U0001f4cb",   # 📋
}

# Throttle: intervalo mínimo em segundos entre alertas do mesmo rio
_THROTTLE_S = 3600   # 1 hora


# ---------------------------------------------------------------------------
# TelegramClient
# ---------------------------------------------------------------------------

class TelegramClient:
    """Cliente HTTP para a Bot API do Telegram.

    Usa niquests para compatibilidade com urllib3-future e tenacity para retry.
    """

    def __init__(self, token: str, chat_id: str) -> None:
        """Inicializa o cliente com token e chat destino.

        Args:
            token: Token do bot (BotFather).
            chat_id: ID do canal ou grupo destino (ex.: '-100xxxxxxxx').
        """
        self.token   = token
        self.chat_id = chat_id
        self._base   = _TELEGRAM_API.format(token=token)
        self.session = niquests.Session()

    @retry(
        retry=retry_if_exception_type((niquests.exceptions.RequestException,)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        """Envia mensagem de texto para o chat configurado.

        Args:
            text: Texto da mensagem (suporta Markdown ou HTML).
            parse_mode: "Markdown" ou "HTML".
            disable_notification: Se True, envia silenciosamente.

        Returns:
            Resposta JSON da API do Telegram.

        Raises:
            niquests.exceptions.RequestException: Após esgotar tentativas.
        """
        payload = {
            "chat_id":              self.chat_id,
            "text":                 text,
            "parse_mode":           parse_mode,
            "disable_notification": disable_notification,
        }
        resp = self.session.post(
            f"{self._base}/sendMessage",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        if not result.get("ok"):
            logger.warning(f"Telegram API nao-ok: {result}")
        return result

    def get_updates(self, offset: int = 0, timeout: int = 5) -> list[dict[str, Any]]:
        """Polling de mensagens recebidas (para processar comandos).

        Args:
            offset: ID do último update processado + 1.
            timeout: Long-poll timeout em segundos.

        Returns:
            Lista de update objects do Telegram.

        Raises:
            niquests.exceptions.RequestException: Se a requisição falhar.
        """
        resp = self.session.get(
            f"{self._base}/getUpdates",
            params={"offset": offset, "timeout": timeout},
            timeout=timeout + 5,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data.get("result", [])

    def test_connection(self) -> bool:
        """Verifica se token e chat_id estão válidos via getMe.

        Returns:
            True se a API responder com ok=True.
        """
        try:
            resp = self.session.get(f"{self._base}/getMe", timeout=5)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            ok: bool = bool(result.get("ok"))
            if ok:
                bot_name = result["result"].get("username", "?")
                logger.success(f"Telegram OK — bot: @{bot_name}")
            return ok
        except niquests.exceptions.RequestException as exc:
            logger.error(f"Telegram conexao falhou: {exc}")
            return False


# ---------------------------------------------------------------------------
# Carregamento de configuração
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Carrega config.yaml.

    Returns:
        Dicionário com seções rios e alertas.

    Raises:
        FileNotFoundError: Se config.yaml não for encontrado.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml não encontrado: {_CONFIG_PATH}")
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Throttle anti-spam
# ---------------------------------------------------------------------------

class AlertThrottle:
    """Controla o intervalo mínimo entre alertas do mesmo key.

    Args:
        interval_s: Intervalo mínimo em segundos entre alertas.
    """

    def __init__(self, interval_s: int = _THROTTLE_S) -> None:
        self._interval_s = interval_s
        self._last: dict[str, datetime] = {}

    def can_send(self, key: str) -> bool:
        """Verifica se pode enviar alerta para a chave dada.

        Args:
            key: Chave única do alerta (ex.: 'river:Guaíba:ALERTA').

        Returns:
            True se o intervalo desde o último envio já passou.
        """
        now  = datetime.now(timezone.utc)
        last = self._last.get(key)
        if last is None or (now - last).total_seconds() >= self._interval_s:
            return True
        return False

    def mark_sent(self, key: str) -> None:
        """Registra o envio de um alerta.

        Args:
            key: Chave única do alerta.
        """
        self._last[key] = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Formatação de mensagens
# ---------------------------------------------------------------------------

def _fmt_river_alert(
    river: str,
    segment: str,
    level_m: float,
    status: str,
    trend: str,
    pct_cota: float,
    cota_alerta_m: float,
    municipios: list[str],
) -> str:
    """Formata mensagem de alerta de nível de rio para Telegram (Markdown).

    Args:
        river: Nome do rio.
        segment: Código da estação/segmento.
        level_m: Nível atual em metros.
        status: NORMAL | ATENCAO | ALERTA | EMERGENCIA.
        trend: SUBINDO | ESTAVEL | DESCENDO.
        pct_cota: Percentual da cota de alerta.
        cota_alerta_m: Cota de alerta em metros.
        municipios: Lista de municípios em risco.

    Returns:
        String formatada para envio via Telegram Markdown.
    """
    emoji  = _EMOJI.get(status, "")
    trend_icon = {"SUBINDO": "↑", "DESCENDO": "↓", "ESTAVEL": "→"}.get(trend, "")
    mun_str = ", ".join(municipios[:4]) if municipios else "—"
    ts_str  = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")

    return (
        f"{emoji} *Alerta Hidro — Rio {river}*\n"
        f"Nível: *{level_m:.2f} m* {trend_icon} ({pct_cota:.0f}% da cota)\n"
        f"Status: *{status}* | Cota alerta: {cota_alerta_m:.1f} m\n"
        f"Estação: `{segment}`\n"
        f"Municípios: {mun_str}\n"
        f"_{ts_str}_"
    )


def _fmt_rain_alert(
    station_id: str,
    rain_1h: float,
    rain_24h: float,
    threshold_1h: float,
    threshold_24h: float,
) -> str:
    """Formata mensagem de alerta de chuva intensa para Telegram.

    Args:
        station_id: Código da estação.
        rain_1h: Acumulado 1h em mm.
        rain_24h: Acumulado 24h em mm.
        threshold_1h: Limiar crítico 1h em mm.
        threshold_24h: Limiar crítico 24h em mm.

    Returns:
        String formatada para Telegram Markdown.
    """
    ts_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
    lines  = [
        f"{_EMOJI['ALERTA']} *Chuva Intensa — Estação `{station_id}`*",
    ]
    if rain_1h >= threshold_1h:
        lines.append(f"Acum. 1h:  *{rain_1h:.1f} mm* (limiar: {threshold_1h} mm)")
    if rain_24h >= threshold_24h:
        lines.append(f"Acum. 24h: *{rain_24h:.1f} mm* (limiar: {threshold_24h} mm)")
    lines.append(f"_{ts_str}_")
    return "\n".join(lines)


def _fmt_status_summary(df_status: pd.DataFrame) -> str:
    """Formata resumo geral do status dos rios para o comando /status.

    Args:
        df_status: DataFrame com colunas river, segment, level_m, status, trend.

    Returns:
        String formatada para Telegram Markdown.
    """
    if df_status.empty:
        return "_Nenhum dado de nível disponível._"

    ts_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
    lines  = [f"*Monitor Hidro RS* — _{ts_str}_\n"]

    for _, row in df_status.iterrows():
        emoji      = _EMOJI.get(str(row.get("status", "NORMAL")), "")
        trend_icon = {"SUBINDO": "↑", "DESCENDO": "↓", "ESTAVEL": "→"}.get(
            str(row.get("trend", "")), ""
        )
        level = row.get("level_m")
        pct   = row.get("pct_cota_alerta")
        level_str = f"{level:.2f}m" if level is not None else "—"
        pct_str   = f" ({pct:.0f}%)" if pct is not None else ""
        lines.append(
            f"{emoji} *{row['river']}* [{row['segment']}]: "
            f"{level_str}{pct_str} {trend_icon}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verificação de alertas
# ---------------------------------------------------------------------------

def check_river_alerts(
    client: TelegramClient,
    db: ClimateDB,
    throttle: AlertThrottle,
    config: dict[str, Any],
) -> int:
    """Verifica níveis dos rios e envia alertas para status ≥ ATENCAO.

    Lê river_status do DuckDB; para cada segmento fora de NORMAL, verifica
    o throttle e envia mensagem Telegram se necessário.

    Args:
        client: TelegramClient configurado.
        db: ClimateDB com conexão ativa.
        throttle: AlertThrottle para controle de spam.
        config: Dicionário do config.yaml (seção 'rios').

    Returns:
        Número de alertas enviados.
    """
    df = db.get_river_status()
    if df.empty:
        logger.debug("river_status: sem dados.")
        return 0

    cfg_rios = config.get("rios", {})
    sent = 0

    for _, row in df.iterrows():
        status = str(row.get("status", "NORMAL"))
        if status == "NORMAL":
            continue

        river   = str(row["river"])
        segment = str(row["segment"])
        key     = f"river:{river}:{status}"

        if not throttle.can_send(key):
            logger.debug(f"Throttle ativo: {key}")
            continue

        rio_cfg        = cfg_rios.get(river, {})
        cota_alerta_m  = float(rio_cfg.get("cota_alerta_m", 0))
        municipios     = rio_cfg.get("municipios_risco", [])

        level_m  = float(row.get("level_m") or 0)
        trend    = str(row.get("trend", "ESTAVEL"))
        pct_cota = float(row.get("pct_cota_alerta") or 0)

        msg = _fmt_river_alert(
            river, segment, level_m, status, trend,
            pct_cota, cota_alerta_m, municipios,
        )

        try:
            client.send_message(msg)
            throttle.mark_sent(key)
            db.log_alert(
                alert_type    = "NIVEL_RIO",
                severity      = status,
                message       = msg,
                location      = municipios[0] if municipios else None,
                river         = river,
                level_m       = level_m,
                threshold_m   = cota_alerta_m,
                telegram_sent = True,
            )
            sent += 1
            logger.warning(f"Alerta enviado: [{status}] Rio {river} [{segment}]")
        except niquests.exceptions.RequestException as exc:
            logger.error(f"Falha ao enviar alerta {river}: {exc}")

    return sent


def check_rain_alerts(
    client: TelegramClient,
    db: ClimateDB,
    throttle: AlertThrottle,
    config: dict[str, Any],
) -> int:
    """Verifica acumulados de chuva e envia alertas se limiares forem excedidos.

    Lê rain_accumulated do DuckDB; usa o limiar padrão da região mais crítica
    (Vale_Taquari) como referência global — pode ser parametrizado por região.

    Args:
        client: TelegramClient configurado.
        db: ClimateDB com conexão ativa.
        throttle: AlertThrottle para controle de spam.
        config: Dicionário do config.yaml (seção 'alertas').

    Returns:
        Número de alertas de chuva enviados.
    """
    # Busca acumulados mais recentes por estação
    df: pd.DataFrame = db._con.execute("""
        SELECT station_id, rain_1h, rain_24h
        FROM rain_accumulated
        QUALIFY ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY date DESC) = 1
    """).df()

    if df.empty:
        logger.debug("rain_accumulated: sem dados.")
        return 0

    # Limiares: usa média das regiões do config
    cfg_alertas = config.get("alertas", {})
    t1h_vals  = [v.get("rain_1h_critico_mm",  25) for v in cfg_alertas.values()]
    t24h_vals = [v.get("rain_24h_critico_mm", 60) for v in cfg_alertas.values()]
    thr_1h    = min(t1h_vals)   if t1h_vals  else 25.0
    thr_24h   = min(t24h_vals)  if t24h_vals else 60.0

    sent = 0
    for _, row in df.iterrows():
        sid    = str(row["station_id"])
        r1h    = float(row.get("rain_1h")  or 0)
        r24h   = float(row.get("rain_24h") or 0)

        if r1h < thr_1h and r24h < thr_24h:
            continue

        key = f"rain:{sid}"
        if not throttle.can_send(key):
            continue

        msg = _fmt_rain_alert(sid, r1h, r24h, thr_1h, thr_24h)
        try:
            client.send_message(msg)
            throttle.mark_sent(key)
            db.log_alert(
                alert_type    = "CHUVA_INTENSA",
                severity      = "ALERTA",
                message       = msg,
                telegram_sent = True,
            )
            sent += 1
            logger.warning(f"Alerta chuva enviado: {sid} 1h={r1h}mm 24h={r24h}mm")
        except niquests.exceptions.RequestException as exc:
            logger.error(f"Falha ao enviar alerta chuva {sid}: {exc}")

    return sent


# ---------------------------------------------------------------------------
# Processamento de comandos (/status)
# ---------------------------------------------------------------------------

def process_commands(
    client: TelegramClient,
    db: ClimateDB,
    last_update_id: int,
) -> int:
    """Processa comandos recebidos via Telegram polling.

    Comandos suportados:
    - /status — resumo atual dos rios monitorados
    - /ping   — responde 'pong' (health check)

    Args:
        client: TelegramClient configurado.
        db: ClimateDB com conexão ativa.
        last_update_id: Último update_id processado (para offset).

    Returns:
        Novo last_update_id após processar os updates.
    """
    try:
        updates = client.get_updates(offset=last_update_id + 1, timeout=5)
    except niquests.exceptions.RequestException as exc:
        logger.warning(f"get_updates falhou: {exc}")
        return last_update_id

    for upd in updates:
        uid = int(upd.get("update_id", 0))
        if uid > last_update_id:
            last_update_id = uid

        msg = upd.get("message", {})
        text = str(msg.get("text", "")).strip().lower()

        if text in ("/status", "/status@"):
            df_status = db.get_river_status()
            reply     = _fmt_status_summary(df_status)
            client.send_message(reply)
            logger.info("Comando /status respondido.")
        elif text == "/ping":
            client.send_message("_pong_ — Monitor Hidro RS ativo.")

    return last_update_id


# ---------------------------------------------------------------------------
# Orquestrador: ciclo único de alertas
# ---------------------------------------------------------------------------

def run_alert_cycle(
    db: ClimateDB | None = None,
    token: str | None = None,
    chat_id: str | None = None,
    throttle: AlertThrottle | None = None,
    last_update_id: int = 0,
) -> dict[str, Any]:
    """Executa um ciclo de verificação e disparo de alertas.

    Verifica rios + chuva, processa comandos, retorna métricas do ciclo.

    Args:
        db: ClimateDB. Se None, abre conexão temporária.
        token: Token do bot Telegram. Se None, lê de TELEGRAM_TOKEN env.
        chat_id: Chat destino. Se None, lê de TELEGRAM_CHAT_ID env.
        throttle: AlertThrottle compartilhado. Se None, cria novo.
        last_update_id: Último update Telegram processado.

    Returns:
        Dict com river_alerts (int), rain_alerts (int),
        last_update_id (int), duration_s (float).

    Raises:
        ValueError: Se token ou chat_id não forem fornecidos nem encontrados.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    token   = token   or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError(
            "TELEGRAM_TOKEN e TELEGRAM_CHAT_ID devem estar no .env ou serem passados."
        )

    t_start = datetime.now(timezone.utc)
    client  = TelegramClient(token, chat_id)
    if throttle is None:
        throttle = AlertThrottle()

    config = _load_config()

    _owns_db = db is None
    if _owns_db:
        db = ClimateDB()

    try:
        river_n = check_river_alerts(client, db, throttle, config)
        rain_n  = check_rain_alerts(client,  db, throttle, config)
        last_update_id = process_commands(client, db, last_update_id)
    finally:
        if _owns_db:
            db.close()

    duration = (datetime.now(timezone.utc) - t_start).total_seconds()
    if river_n + rain_n > 0:
        logger.info(
            f"Ciclo alertas: {river_n} rios + {rain_n} chuva enviados "
            f"em {duration:.1f}s."
        )

    return {
        "river_alerts":    river_n,
        "rain_alerts":     rain_n,
        "last_update_id":  last_update_id,
        "duration_s":      round(duration, 2),
    }


# ---------------------------------------------------------------------------
# Loop contínuo
# ---------------------------------------------------------------------------

def run_loop(
    interval_s: int = 600,
    token: str | None = None,
    chat_id: str | None = None,
) -> None:
    """Executa o bot em loop contínuo com intervalo fixo.

    Cria ThrottleAlert e ClimateDB compartilhados entre ciclos.
    Captura KeyboardInterrupt para encerrar graciosamente.

    Args:
        interval_s: Intervalo entre ciclos em segundos (padrão 600 = 10min).
        token: Token do bot. Se None, usa TELEGRAM_TOKEN do .env.
        chat_id: Chat destino. Se None, usa TELEGRAM_CHAT_ID do .env.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    token   = token   or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error("TELEGRAM_TOKEN e TELEGRAM_CHAT_ID nao encontrados no .env")
        return

    client = TelegramClient(token, chat_id)
    if not client.test_connection():
        logger.error("Nao foi possivel conectar ao Telegram — abortando loop.")
        return

    throttle       = AlertThrottle()
    last_update_id = 0

    logger.info(f"Bot iniciado — ciclos a cada {interval_s}s. Ctrl+C para parar.")

    try:
        while True:
            with ClimateDB() as db:
                result = run_alert_cycle(
                    db=db,
                    token=token,
                    chat_id=chat_id,
                    throttle=throttle,
                    last_update_id=last_update_id,
                )
                last_update_id = result["last_update_id"]

            time.sleep(interval_s)

    except KeyboardInterrupt:
        logger.info("Bot encerrado pelo usuário.")


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

    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Bot Telegram — Monitor Hidro RS")
    parser.add_argument("--loop",     action="store_true",
                        help="Roda em loop continuo (intervalo 10min)")
    parser.add_argument("--interval", type=int, default=600,
                        help="Intervalo em segundos (padrao 600)")
    parser.add_argument("--test",     action="store_true",
                        help="Testa conexao e envia mensagem de teste")
    args = parser.parse_args()

    token   = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ERRO: TELEGRAM_TOKEN e TELEGRAM_CHAT_ID nao definidos no .env")
        sys.exit(1)

    client = TelegramClient(token, chat_id)

    if args.test:
        ok = client.test_connection()
        if ok:
            ts = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
            client.send_message(
                f"{_EMOJI['INFO']} *Monitor Hidro RS*\n"
                f"Sistema online — _{ts}_"
            )
            print("Mensagem de teste enviada.")
        sys.exit(0 if ok else 1)

    if args.loop:
        run_loop(interval_s=args.interval, token=token, chat_id=chat_id)
    else:
        result = run_alert_cycle(token=token, chat_id=chat_id)
        print("\n--- Resultado -----------------------------------")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("-------------------------------------------------")
