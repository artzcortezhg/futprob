# -*- coding: utf-8 -*-
"""
Watchdog do futprob (Bloco de estabilização): sobe bot.py e dashboard.py
como subprocessos, monitora se algum caiu e reinicia sozinho, avisando no
Telegram quando um restart acontece. Roda pra sempre (até ser encerrado) —
é isso que o iniciar_futprob.bat chama, em vez de subir bot/painel direto.

Limite conhecido: se o PRÓPRIO watchdog cair, nada o reinicia sozinho
durante a sessão (só no próximo boot, via Startup). O pedido original era
"se o bot ou o painel caírem" — este script cobre exatamente esses dois.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB = RAIZ / "db" / "previsoes.sqlite"
PYTHON_EXE = sys.executable
INTERVALO_CHECAGEM_SEGUNDOS = 10
ESPERA_ANTES_DE_REINICIAR_SEGUNDOS = 3

CAMINHO_LOG = RAIZ / "logs" / "watchdog.log"
CAMINHO_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s watchdog %(levelname)s: %(message)s",
    handlers=[
        RotatingFileHandler(CAMINHO_LOG, maxBytes=2_000_000, backupCount=3, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("futprob.watchdog")

load_dotenv(RAIZ / ".env")


def _chat_id_salvo() -> str | None:
    try:
        with sqlite3.connect(CAMINHO_DB) as conn:
            row = conn.execute("SELECT valor FROM bot_estado WHERE chave='chat_id'").fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _notificar_telegram(texto: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = _chat_id_salvo()
    if not token or not chat_id:
        logger.info("sem token/chat_id salvo ainda — não dá pra notificar o restart no Telegram")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        dados = json.dumps({"chat_id": chat_id, "text": texto}).encode("utf-8")
        req = urllib.request.Request(url, data=dados, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        logger.exception("falha ao notificar o restart no Telegram (o restart em si já aconteceu)")


class ProcessoMonitorado:
    def __init__(self, nome: str, script_relativo: str):
        self.nome = nome
        self.script_relativo = script_relativo
        self.processo: subprocess.Popen | None = None
        self.n_restarts = 0

    def iniciar(self) -> None:
        caminho_log_stdout = RAIZ / "logs" / f"{self.nome}_stdout.log"
        arquivo_log = open(caminho_log_stdout, "a", encoding="utf-8")
        self.processo = subprocess.Popen(
            [PYTHON_EXE, str(RAIZ / self.script_relativo)],
            cwd=str(RAIZ), stdout=arquivo_log, stderr=subprocess.STDOUT,
        )
        logger.info(f"{self.nome}: iniciado (pid={self.processo.pid})")

    def vivo(self) -> bool:
        return self.processo is not None and self.processo.poll() is None

    def reiniciar(self) -> None:
        self.n_restarts += 1
        logger.warning(f"{self.nome}: processo caiu (código {self.processo.returncode}) — "
                        f"reiniciando em {ESPERA_ANTES_DE_REINICIAR_SEGUNDOS}s (restart #{self.n_restarts})")
        time.sleep(ESPERA_ANTES_DE_REINICIAR_SEGUNDOS)
        self.iniciar()
        _notificar_telegram(
            f"⚠️ Watchdog: {self.nome} caiu e foi reiniciado automaticamente (restart #{self.n_restarts}). "
            f"Veja logs/{self.nome}.log se continuar acontecendo."
        )


def main() -> None:
    processos = [
        ProcessoMonitorado("bot", "src/bot.py"),
        ProcessoMonitorado("painel", "src/dashboard.py"),
    ]
    for p in processos:
        p.iniciar()

    logger.info("watchdog rodando — monitorando bot e painel a cada "
                f"{INTERVALO_CHECAGEM_SEGUNDOS}s")
    try:
        while True:
            time.sleep(INTERVALO_CHECAGEM_SEGUNDOS)
            for p in processos:
                if not p.vivo():
                    p.reiniciar()
    except KeyboardInterrupt:
        logger.info("watchdog encerrado (Ctrl+C) — processos filhos continuam rodando por conta própria")


if __name__ == "__main__":
    main()
