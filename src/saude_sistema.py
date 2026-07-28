# -*- coding: utf-8 -*-
"""
Saúde do sistema (Bloco de estabilização) — fonte única de verdade sobre os
horários fixos do agendador e sobre "o sistema está ok?", usada tanto pelo
/status do bot quanto pelo /api/status-sistema do painel, pra nunca
divergirem sobre o que é normal.

O bot escreve um heartbeat em bot_estado a cada HEARTBEAT_INTERVALO_MIN
minutos (ver src/bot.py); se esse heartbeat estiver mais velho que
HEARTBEAT_TOLERANCIA_MIN, o bot é considerado fora do ar (processo morto,
travado, ou o PC desligou sem o watchdog reiniciar ainda).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from painel_db import carregar_estado_bot, inicializar_db_painel

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB_PADRAO = RAIZ / "db" / "previsoes.sqlite"
FUSO_BR = ZoneInfo("America/Sao_Paulo")

# horários fixos do agendador (bot.py) — fonte única, painel só LÊ pra exibir
HORARIO_MATUTINA = "09:00"
HORARIO_FECHAMENTO = "23:30"
MINUTOS_ANTES_PREJOGO = 60
CUTOFF_CATCHUP_MATUTINA = 20  # depois dessa hora local não vale mais rodar a matutina atrasada

HEARTBEAT_INTERVALO_MIN = 5
HEARTBEAT_TOLERANCIA_MIN = HEARTBEAT_INTERVALO_MIN * 3  # 15min sem sinal = bot considerado fora do ar


def _proxima_rotina(caminho_db: Path, agora_br: datetime) -> str:
    """Descrição em texto de qual rotina deve rodar em seguida — aproximação
    de leitura (não é o agendador de verdade), montada a partir do que o bot
    já gravou em bot_estado."""
    hoje = agora_br.date().isoformat()
    matutina_feita = carregar_estado_bot(caminho_db, "data_ultima_matutina") == hoje
    if not matutina_feita:
        if agora_br.hour < CUTOFF_CATCHUP_MATUTINA:
            return f"matutina de hoje ({HORARIO_MATUTINA}) ainda não rodou"
        return f"matutina de amanhã às {HORARIO_MATUTINA}"

    prejogo_feita = carregar_estado_bot(caminho_db, "data_ultima_prejogo") == hoje
    jogos_json = carregar_estado_bot(caminho_db, "jogos_hoje_json")
    data_jogos = carregar_estado_bot(caminho_db, "data_jogos_hoje")
    if not prejogo_feita and jogos_json and data_jogos == hoje:
        try:
            jogos_hoje = json.loads(jogos_json)
        except Exception:
            jogos_hoje = []
        if jogos_hoje:
            try:
                primeiro = min(pd.Timestamp(j["commence_time"]) for j in jogos_hoje)
                if primeiro.tzinfo is None:
                    primeiro = primeiro.tz_localize("UTC")
                momento = primeiro.tz_convert(FUSO_BR) - timedelta(minutes=MINUTOS_ANTES_PREJOGO)
                return f"pré-jogo (fechamento/CLV) às {momento.strftime('%H:%M')}"
            except Exception:
                pass

    return f"fechamento diário às {HORARIO_FECHAMENTO} (e matutina de amanhã às {HORARIO_MATUTINA})"


def calcular_status_sistema(caminho_db: Path = CAMINHO_DB_PADRAO) -> dict:
    """Resumo de saúde: heartbeat do bot, última coleta, próxima rotina
    esperada e quantos registros estão em aberto. Nunca lança exceção — se
    algo faltar, o campo correspondente vem vazio/None em vez de quebrar."""
    inicializar_db_painel(caminho_db)
    agora_br = datetime.now(FUSO_BR)

    heartbeat_raw = carregar_estado_bot(caminho_db, "heartbeat_bot")
    bot_ok, minutos_desde_heartbeat = False, None
    if heartbeat_raw:
        try:
            dt = datetime.fromisoformat(heartbeat_raw)
            minutos_desde_heartbeat = (agora_br - dt).total_seconds() / 60
            bot_ok = minutos_desde_heartbeat <= HEARTBEAT_TOLERANCIA_MIN
        except Exception:
            pass

    with sqlite3.connect(caminho_db) as conn:
        conn.row_factory = sqlite3.Row
        ultima_coleta = conn.execute("SELECT * FROM coletas ORDER BY id DESC LIMIT 1").fetchone()
        n_registros_abertos = conn.execute("SELECT COUNT(*) FROM registros WHERE status='aberto'").fetchone()[0]

    return {
        "agora": agora_br.isoformat(),
        "bot_ok": bot_ok,
        "minutos_desde_heartbeat": minutos_desde_heartbeat,
        "ultima_coleta": dict(ultima_coleta) if ultima_coleta else None,
        "proxima_rotina": _proxima_rotina(caminho_db, agora_br),
        "n_registros_abertos": n_registros_abertos,
        "chat_id_configurado": carregar_estado_bot(caminho_db, "chat_id") is not None,
    }
