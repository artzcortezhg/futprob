# -*- coding: utf-8 -*-
"""Testes do schema/helpers do SQLite do painel (painel_db.py)."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from painel_db import (
    inicializar_db_painel, inserir_registro, fechar_registro,
    marcar_resultado_manual, registrar_coleta, registrar_uso_oddspapi,
    salvar_estado_bot, carregar_estado_bot, salvar_mensagem_jogo, carregar_mensagem_jogo,
)


def test_inicializar_cria_as_tres_tabelas(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    inicializar_db_painel(caminho)
    with sqlite3.connect(caminho) as conn:
        tabelas = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"registros", "coletas", "oddspapi_uso"} <= tabelas


def test_inserir_e_fechar_registro(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    rid = inserir_registro(
        caminho, "Premier League", "Arsenal", "Liverpool", "1X2", "Casa",
        prob_modelo=0.5, odd_registrada=2.1, ev=0.05, origem="bot",
    )
    with sqlite3.connect(caminho) as conn:
        row = conn.execute("SELECT status, origem FROM registros WHERE id=?", (rid,)).fetchone()
    assert row == ("aberto", "bot")

    fechar_registro(caminho, rid, odd_fechamento=2.0, clv=0.02, resultado="ganhou")
    with sqlite3.connect(caminho) as conn:
        row = conn.execute("SELECT status, clv, resultado FROM registros WHERE id=?", (rid,)).fetchone()
    assert row == ("fechado", 0.02, "ganhou")


def test_marcar_resultado_manual_nao_apaga_nada(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    rid = inserir_registro(caminho, "La Liga", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    marcar_resultado_manual(caminho, rid, "ganhou")
    with sqlite3.connect(caminho) as conn:
        n = conn.execute("SELECT COUNT(*) FROM registros").fetchone()[0]
        resultado = conn.execute("SELECT resultado FROM registros WHERE id=?", (rid,)).fetchone()[0]
    assert n == 1
    assert resultado == "ganhou"


def test_marcar_resultado_invalido_gera_erro(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    rid = inserir_registro(caminho, "La Liga", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    with pytest.raises(ValueError):
        marcar_resultado_manual(caminho, rid, "empatou")


def test_registrar_coleta_e_uso_oddspapi(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    registrar_coleta(caminho, "betano", sucesso=True, tipo="manha", n_jogos_capturados=10, n_mercados_capturados=40)
    registrar_uso_oddspapi(caminho, "/odds", sucesso=False, observacao="timeout")
    with sqlite3.connect(caminho) as conn:
        coleta = conn.execute("SELECT fonte, sucesso, n_jogos_capturados FROM coletas").fetchone()
        uso = conn.execute("SELECT endpoint, sucesso FROM oddspapi_uso").fetchone()
    assert coleta == ("betano", 1, 10)
    assert uso == ("/odds", 0)


def test_estado_bot_salva_e_atualiza(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    assert carregar_estado_bot(caminho, "chat_id") is None
    salvar_estado_bot(caminho, "chat_id", "12345")
    assert carregar_estado_bot(caminho, "chat_id") == "12345"
    salvar_estado_bot(caminho, "chat_id", "67890")  # atualiza, não duplica
    assert carregar_estado_bot(caminho, "chat_id") == "67890"


def test_mensagem_jogo_salva_e_recupera(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    salvar_mensagem_jogo(caminho, 999, "Premier League", "Arsenal", "Liverpool", "2026-08-01", '{"1X2":{"Casa":0.5}}')
    dados = carregar_mensagem_jogo(caminho, 999)
    assert dados["liga"] == "Premier League"
    assert dados["time_casa"] == "Arsenal"
    assert dados["probs_json"] == '{"1X2":{"Casa":0.5}}'


def test_mensagem_jogo_inexistente_retorna_none(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    assert carregar_mensagem_jogo(caminho, 123) is None
