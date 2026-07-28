# -*- coding: utf-8 -*-
"""Testes de src/saude_sistema.py: heartbeat do bot, próxima rotina esperada
e o resumo geral usado tanto pelo /status do bot quanto pelo
/api/status-sistema do painel."""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from painel_db import salvar_estado_bot, registrar_coleta
from saude_sistema import calcular_status_sistema, HEARTBEAT_TOLERANCIA_MIN

FUSO_BR = ZoneInfo("America/Sao_Paulo")


def test_sem_heartbeat_bot_nao_ok(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    status = calcular_status_sistema(caminho)
    assert status["bot_ok"] is False
    assert status["minutos_desde_heartbeat"] is None


def test_heartbeat_recente_bot_ok(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    salvar_estado_bot(caminho, "heartbeat_bot", datetime.now(FUSO_BR).isoformat())
    status = calcular_status_sistema(caminho)
    assert status["bot_ok"] is True
    assert status["minutos_desde_heartbeat"] < 1


def test_heartbeat_velho_bot_nao_ok(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    velho = datetime.now(FUSO_BR) - timedelta(minutes=HEARTBEAT_TOLERANCIA_MIN + 5)
    salvar_estado_bot(caminho, "heartbeat_bot", velho.isoformat())
    status = calcular_status_sistema(caminho)
    assert status["bot_ok"] is False


def test_ultima_coleta_e_registros_abertos_no_resumo(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    registrar_coleta(caminho, "betano", sucesso=True, tipo="manha", n_jogos_capturados=3)
    status = calcular_status_sistema(caminho)
    assert status["ultima_coleta"]["fonte"] == "betano"
    assert status["n_registros_abertos"] == 0


def test_chat_id_configurado_reflete_estado(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    assert calcular_status_sistema(caminho)["chat_id_configurado"] is False
    salvar_estado_bot(caminho, "chat_id", "12345")
    assert calcular_status_sistema(caminho)["chat_id_configurado"] is True


def test_proxima_rotina_antes_da_matutina_avisa_que_nao_rodou(tmp_path, monkeypatch):
    caminho = tmp_path / "teste.sqlite"
    import saude_sistema
    monkeypatch.setattr(saude_sistema, "CUTOFF_CATCHUP_MATUTINA", 23)  # garante que "agora" < cutoff
    status = calcular_status_sistema(caminho)
    assert "matutina" in status["proxima_rotina"]
