"""
Agente 3 — Engenheiro de Software
Bot Telegram para alertas hidrometeorológicos do RS.

Fonte de dados: Supabase PostgreSQL (arquitetura híbrida v4 — sem DuckDB).

Funcionalidades:
  - Alertas de nível de rio quando status ∈ (ALERTA, EMERGENCIA)
    lendo a tabela live_river_levels
  - Alertas de chuva intensa (1h/24h acima dos limiares do config.yaml)
    lendo a tabela live_rain_readings
  - Throttling anti-spam PERSISTENTE: 1 alerta/chave/hora via alerts_log
    (o runner do GitHub Actions é efêmero — memória não sobrevive entre runs)
  - Mensagens ricas em Markdown com emoji de severidade
  - Comando /status para consulta on-demand via polling
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import niquests
import psycopg2
import psycopg2.extras
import yaml
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_CONFIG_PATH  = _ROOT / "config" / "config.yaml"

_TELEGRAM_API = "https://api.telegram.org/bot{token}"

# Emoji por severidade
_EMOJI: dict[str, str] = {
    "EMERGENCIA": "\U0001f6a8",   # 🚨
    "ALERTA":     "\U000026a0",   # ⚠️
    "ATENCAO":    "\U0001f7e1",   # 🟡
    "NORMAL":     "\U0001f7e2",   # 🟢
    "INFO":       "\U0001f4cb",   # 📋
}

# Throttle: intervalo mínimo em segundos entre alertas da mesma chave
_THROTTLE_S = 3600   # 1 hora

# Status que disparam alerta de rio (ATENCAO não notifica — só ALERTA+)
_ALERT_STATUSES = ("ALERTA", "EMERGENCIA")


# ---------------------------------------------------------------------------
# Conexão Supabase
# ---------------------------------------------------------------------------

def _pg_connect():
    """Conexão Supabase (canônico em utils.comum)."""
    from utils.comum import pg_connect
    return pg_connect(connect_timeout=15)


def _query(
    conn: "psycopg2.extensions.connection",
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Executa SELECT e retorna lista de dicts.

    Args:
        conn: Conexão psycopg2 aberta.
        sql: Query SQL com placeholders %s.
        params: Parâmetros da query.

    Returns:
        Lista de linhas como dicionários (RealDictCursor).

    Raises:
        psycopg2.Error: Se a query falhar.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _fetch_river_levels(conn: "psycopg2.extensions.connection") -> list[dict[str, Any]]:
    """Lê o snapshot atual dos rios monitorados no Supabase.

    Args:
        conn: Conexão psycopg2 aberta.

    Returns:
        Lista de dicts com rio_id, rio_nome, nivel_atual_m, status,
        percentual_cota, cota_alerta_m e timestamp.
    """
    return _query(conn, """
        SELECT rio_id, rio_nome, nivel_atual_m, status,
               percentual_cota, cota_alerta_m, "timestamp"
        FROM live_river_levels
        ORDER BY rio_nome
    """)


def _fetch_rain_readings(conn: "psycopg2.extensions.connection") -> list[dict[str, Any]]:
    """Lê o snapshot de chuva por estação no Supabase.

    Args:
        conn: Conexão psycopg2 aberta.

    Returns:
        Lista de dicts com station_id, precip_1h e precip_24h.
    """
    return _query(conn, """
        SELECT station_id, precip_1h, precip_24h
        FROM live_rain_readings
        WHERE precip_1h IS NOT NULL OR precip_24h IS NOT NULL
    """)


# ---------------------------------------------------------------------------
# Throttle anti-spam persistente (alerts_log)
# ---------------------------------------------------------------------------

def _can_send(conn: "psycopg2.extensions.connection", key: str) -> bool:
    """Verifica na alerts_log se a janela de 1h da chave já passou.

    O throttle é persistido no banco porque o runner do GitHub Actions
    é destruído após cada ciclo — estado em memória não sobrevive.

    Args:
        conn: Conexão psycopg2 aberta.
        key: Chave única do alerta (ex.: 'river:guaiba:ALERTA').

    Returns:
        True se nenhum alerta da mesma chave foi enviado na última 1h.
    """
    rows = _query(conn, """
        SELECT MAX(sent_at) AS last_sent
        FROM alerts_log
        WHERE alert_key = %s AND telegram_sent = TRUE
    """, (key,))
    last = rows[0]["last_sent"] if rows else None
    if last is None:
        return True
    elapsed = (datetime.now(timezone.utc) - last).total_seconds()
    return elapsed >= _THROTTLE_S


def _log_alert(
    conn: "psycopg2.extensions.connection",
    key: str,
    alert_type: str,
    severity: str,
    message: str,
    rio_id: Optional[str] = None,
    level_m: Optional[float] = None,
    threshold_m: Optional[float] = None,
    telegram_sent: bool = True,
) -> None:
    """Registra alerta enviado na alerts_log (histórico + base do throttle).

    Args:
        conn: Conexão psycopg2 aberta.
        key: Chave única do alerta.
        alert_type: NIVEL_RIO | CHUVA_INTENSA.
        severity: ATENCAO | ALERTA | EMERGENCIA.
        message: Texto enviado ao Telegram.
        rio_id: Slug do rio (se aplicável).
        level_m: Nível atual em metros (se aplicável).
        threshold_m: Cota/limiar de referência (se aplicável).
        telegram_sent: Se a mensagem foi efetivamente entregue.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO alerts_log
                (alert_key, alert_type, severity, rio_id, message,
                 level_m, threshold_m, telegram_sent)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (key, alert_type, severity, rio_id, message,
              level_m, threshold_m, telegram_sent))
    conn.commit()


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


def _municipios_do_rio(config: dict[str, Any], rio_nome: str) -> list[str]:
    """Busca municípios de risco do rio no config.yaml (chave tolerante).

    O banco guarda 'Guaíba'/'Lagoa_Patos' etc.; o config usa as mesmas
    chaves — a comparação ignora acentos/caixa para robustez.

    Args:
        config: Dicionário do config.yaml.
        rio_nome: Nome do rio como está no banco.

    Returns:
        Lista de municípios de risco (vazia se não encontrado).
    """
    import unicodedata

    def _slug(s: str) -> str:
        return (
            unicodedata.normalize("NFD", s)
            .encode("ascii", "ignore").decode()
            .lower().strip()
        )

    alvo = _slug(rio_nome)
    for nome, cfg in config.get("rios", {}).items():
        if _slug(nome) == alvo:
            return list(cfg.get("municipios_risco", []))
    return []


# ---------------------------------------------------------------------------
# Formatação de mensagens
# ---------------------------------------------------------------------------

def _fmt_river_alert(
    rio_nome: str,
    rio_id: str,
    level_m: float,
    status: str,
    pct_cota: float,
    cota_alerta_m: float,
    municipios: list[str],
) -> str:
    """Formata mensagem de alerta de nível de rio para Telegram (Markdown).

    Args:
        rio_nome: Nome do rio.
        rio_id: Slug/identificador do rio.
        level_m: Nível atual em metros.
        status: ALERTA | EMERGENCIA.
        pct_cota: Percentual da cota de alerta.
        cota_alerta_m: Cota de alerta em metros.
        municipios: Lista de municípios em risco.

    Returns:
        String formatada para envio via Telegram Markdown.
    """
    emoji   = _EMOJI.get(status, "")
    mun_str = ", ".join(municipios[:4]) if municipios else "—"
    ts_str  = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")

    return (
        f"{emoji} *Alerta Hidro — Rio {rio_nome}*\n"
        f"Nível: *{level_m:.2f} m* ({pct_cota:.0f}% da cota)\n"
        f"Status: *{status}* | Cota alerta: {cota_alerta_m:.1f} m\n"
        f"Rio ID: `{rio_id}`\n"
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


def _fmt_status_summary(rios: list[dict[str, Any]]) -> str:
    """Formata resumo geral do status dos rios para o comando /status.

    Args:
        rios: Linhas de live_river_levels como dicts.

    Returns:
        String formatada para Telegram Markdown.
    """
    if not rios:
        return "_Nenhum dado de nível disponível._"

    ts_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
    lines  = [f"*Monitor Hidro RS* — _{ts_str}_\n"]

    for row in rios:
        status = str(row.get("status") or "NORMAL")
        emoji  = _EMOJI.get(status, "")
        level  = row.get("nivel_atual_m")
        pct    = row.get("percentual_cota")
        level_str = f"{float(level):.2f}m" if level is not None else "—"
        pct_str   = f" ({float(pct):.0f}%)" if pct is not None else ""
        lines.append(
            f"{emoji} *{row['rio_nome']}*: {level_str}{pct_str} — {status}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verificação de alertas
# ---------------------------------------------------------------------------

def check_river_alerts(
    client: TelegramClient,
    conn: "psycopg2.extensions.connection",
    config: dict[str, Any],
) -> int:
    """Verifica níveis dos rios no Supabase e alerta status ALERTA/EMERGENCIA.

    Lê live_river_levels; para cada rio em status crítico, verifica o
    throttle persistente (alerts_log, janela 1h) e envia Telegram.

    Args:
        client: TelegramClient configurado.
        conn: Conexão psycopg2 com o Supabase.
        config: Dicionário do config.yaml (seção 'rios').

    Returns:
        Número de alertas enviados.
    """
    rios = _fetch_river_levels(conn)
    if not rios:
        logger.debug("live_river_levels: sem dados.")
        return 0

    sent = 0
    for row in rios:
        status = str(row.get("status") or "NORMAL")
        if status not in _ALERT_STATUSES:
            continue

        rio_id   = str(row["rio_id"])
        rio_nome = str(row.get("rio_nome") or rio_id)
        key      = f"river:{rio_id}:{status}"

        if not _can_send(conn, key):
            logger.debug(f"Throttle ativo (1h): {key}")
            continue

        level_m       = float(row.get("nivel_atual_m") or 0)
        pct_cota      = float(row.get("percentual_cota") or 0)
        cota_alerta_m = float(row.get("cota_alerta_m") or 0)
        municipios    = _municipios_do_rio(config, rio_nome)

        msg = _fmt_river_alert(
            rio_nome, rio_id, level_m, status,
            pct_cota, cota_alerta_m, municipios,
        )

        try:
            client.send_message(msg)
            _log_alert(
                conn, key,
                alert_type="NIVEL_RIO", severity=status, message=msg,
                rio_id=rio_id, level_m=level_m, threshold_m=cota_alerta_m,
            )
            sent += 1
            logger.warning(f"Alerta enviado: [{status}] Rio {rio_nome}")
        except niquests.exceptions.RequestException as exc:
            logger.error(f"Falha ao enviar alerta {rio_nome}: {exc}")

    return sent


def check_rain_alerts(
    client: TelegramClient,
    conn: "psycopg2.extensions.connection",
    config: dict[str, Any],
) -> int:
    """Verifica acumulados de chuva no Supabase e alerta acima dos limiares.

    Lê live_rain_readings (precip_1h/precip_24h); usa o limiar mais
    conservador entre as regiões do config como referência global.

    Args:
        client: TelegramClient configurado.
        conn: Conexão psycopg2 com o Supabase.
        config: Dicionário do config.yaml (seção 'alertas').

    Returns:
        Número de alertas de chuva enviados.
    """
    leituras = _fetch_rain_readings(conn)
    if not leituras:
        logger.debug("live_rain_readings: sem dados de chuva.")
        return 0

    cfg_alertas = config.get("alertas", {})
    t1h_vals  = [v.get("rain_1h_critico_mm",  25) for v in cfg_alertas.values()]
    t24h_vals = [v.get("rain_24h_critico_mm", 60) for v in cfg_alertas.values()]
    thr_1h    = min(t1h_vals)   if t1h_vals  else 25.0
    thr_24h   = min(t24h_vals)  if t24h_vals else 60.0

    sent = 0
    for row in leituras:
        sid  = str(row["station_id"])
        r1h  = float(row.get("precip_1h")  or 0)
        r24h = float(row.get("precip_24h") or 0)

        if r1h < thr_1h and r24h < thr_24h:
            continue

        key = f"rain:{sid}"
        if not _can_send(conn, key):
            continue

        msg = _fmt_rain_alert(sid, r1h, r24h, thr_1h, thr_24h)
        try:
            client.send_message(msg)
            _log_alert(
                conn, key,
                alert_type="CHUVA_INTENSA", severity="ALERTA", message=msg,
                level_m=r24h, threshold_m=thr_24h,
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
    conn: "psycopg2.extensions.connection",
    last_update_id: int,
) -> int:
    """Processa comandos recebidos via Telegram polling.

    Comandos suportados:
    - /status — resumo atual dos rios monitorados (Supabase)
    - /ping   — responde 'pong' (health check)

    Args:
        client: TelegramClient configurado.
        conn: Conexão psycopg2 com o Supabase.
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

        msg  = upd.get("message", {})
        text = str(msg.get("text", "")).strip().lower()

        if text.startswith("/status"):
            rios  = _fetch_river_levels(conn)
            reply = _fmt_status_summary(rios)
            client.send_message(reply)
            logger.info("Comando /status respondido.")
        elif text == "/ping":
            client.send_message("_pong_ — Monitor Hidro RS ativo.")

    return last_update_id


# ---------------------------------------------------------------------------
# Orquestrador: ciclo único de alertas
# ---------------------------------------------------------------------------

def run_alert_cycle(
    token: str | None = None,
    chat_id: str | None = None,
    last_update_id: int = 0,
) -> dict[str, Any]:
    """Executa um ciclo de verificação e disparo de alertas.

    Conecta ao Supabase, verifica rios + chuva, processa comandos e
    retorna métricas do ciclo. O throttle de 1h é persistente (alerts_log).

    Args:
        token: Token do bot Telegram. Se None, lê de TELEGRAM_TOKEN env.
        chat_id: Chat destino. Se None, lê de TELEGRAM_CHAT_ID env.
        last_update_id: Último update Telegram processado.

    Returns:
        Dict com river_alerts (int), rain_alerts (int),
        last_update_id (int), duration_s (float).

    Raises:
        ValueError: Se token ou chat_id não forem fornecidos nem encontrados.
    """
    token   = token   or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError(
            "TELEGRAM_TOKEN e TELEGRAM_CHAT_ID devem estar no .env ou serem passados."
        )

    t_start = datetime.now(timezone.utc)
    client  = TelegramClient(token, chat_id)
    config  = _load_config()

    river_n = rain_n = 0
    conn = _pg_connect()
    if conn is None:
        logger.error("Sem conexão Supabase — ciclo de alertas abortado.")
    else:
        try:
            river_n = check_river_alerts(client, conn, config)
            rain_n  = check_rain_alerts(client, conn, config)
            last_update_id = process_commands(client, conn, last_update_id)
        except psycopg2.Error as exc:
            logger.error(f"Erro Supabase no ciclo de alertas: {exc}")
            conn.rollback()
        finally:
            conn.close()

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

    Captura KeyboardInterrupt para encerrar graciosamente.

    Args:
        interval_s: Intervalo entre ciclos em segundos (padrão 600 = 10min).
        token: Token do bot. Se None, usa TELEGRAM_TOKEN do .env.
        chat_id: Chat destino. Se None, usa TELEGRAM_CHAT_ID do .env.
    """
    token   = token   or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error("TELEGRAM_TOKEN e TELEGRAM_CHAT_ID nao encontrados no .env")
        return

    client = TelegramClient(token, chat_id)
    if not client.test_connection():
        logger.error("Nao foi possivel conectar ao Telegram — abortando loop.")
        return

    last_update_id = 0
    logger.info(f"Bot iniciado — ciclos a cada {interval_s}s. Ctrl+C para parar.")

    try:
        while True:
            result = run_alert_cycle(
                token=token,
                chat_id=chat_id,
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
