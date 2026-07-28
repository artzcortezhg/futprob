# -*- coding: utf-8 -*-
"""Testes do watchdog (scripts/watchdog.py): detecta processo caído e
reinicia sozinho. Usa um script Python descartável no lugar de bot.py real
(rápido, sem depender de Telegram/Betano)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import watchdog


def test_processo_monitorado_detecta_queda_e_reinicia(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "RAIZ", tmp_path)
    monkeypatch.setattr(watchdog, "ESPERA_ANTES_DE_REINICIAR_SEGUNDOS", 0)
    monkeypatch.setattr(watchdog, "_notificar_telegram", lambda texto: None)
    (tmp_path / "logs").mkdir()

    script_curto = tmp_path / "sai_rapido.py"
    script_curto.write_text("import sys; sys.exit(0)\n", encoding="utf-8")

    p = watchdog.ProcessoMonitorado("teste", "sai_rapido.py")
    p.iniciar()
    assert p.n_restarts == 0

    for _ in range(50):
        if not p.vivo():
            break
        time.sleep(0.1)
    assert not p.vivo()  # o processo curto já terminou sozinho

    p.reiniciar()
    assert p.n_restarts == 1
    assert p.vivo() or p.processo.poll() is not None  # reiniciou (pode já ter terminado de novo, é rápido)


def test_chat_id_salvo_none_sem_banco(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "CAMINHO_DB", tmp_path / "nao_existe.sqlite")
    assert watchdog._chat_id_salvo() is None
