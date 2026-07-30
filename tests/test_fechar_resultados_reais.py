# -*- coding: utf-8 -*-
"""Testes de scripts/fechar_resultados_reais.py: fecha registros abertos a
partir do placar real pesquisado (data/resultados_reais_pendentes.json)."""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fechar_resultados_reais as fechar
from painel_db import inserir_registro


def test_listar_jogos_pendentes_so_traz_jogos_com_data_passada(tmp_path):
    caminho_db = tmp_path / "teste.sqlite"
    inserir_registro(caminho_db, "brasileirao", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05, data_jogo="2020-01-01")
    inserir_registro(caminho_db, "brasileirao", "C", "D", "1X2", "Casa", 0.5, 2.0, 0.05, data_jogo="2099-01-01")
    jogos = fechar.listar_jogos_pendentes(caminho_db)
    assert len(jogos) == 1
    assert jogos[0]["time_casa"] == "A"


def test_aplicar_resultados_fecha_mercados_resolviveis_pelo_placar(tmp_path, monkeypatch):
    caminho_db = tmp_path / "teste.sqlite"
    r_1x2 = inserir_registro(caminho_db, "brasileirao", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05, data_jogo="2020-01-01")
    r_btts = inserir_registro(caminho_db, "brasileirao", "A", "B", "Ambas marcam", "Sim", 0.5, 1.9, 0.05, data_jogo="2020-01-01")
    r_escanteios = inserir_registro(caminho_db, "brasileirao", "A", "B", "Escanteios Over/Under 9.5", "Over", 0.5, 1.9, 0.05, data_jogo="2020-01-01")

    caminho_pendentes = tmp_path / "pendentes.json"
    caminho_pendentes.write_text(json.dumps({"brasileirao|A|B|2020-01-01": [2, 0]}), encoding="utf-8")

    resultado = fechar.aplicar_resultados(caminho_db, caminho_pendentes)
    assert resultado["fechados"] == 2  # 1X2 e ambas marcam, resolvíveis pelo placar
    assert resultado["pulados_mercado_sem_placar_suficiente"] == 1  # escanteios, não dá pra saber

    with sqlite3.connect(caminho_db) as conn:
        status_1x2, res_1x2 = conn.execute("SELECT status, resultado FROM registros WHERE id=?", (r_1x2,)).fetchone()
        status_btts, res_btts = conn.execute("SELECT status, resultado FROM registros WHERE id=?", (r_btts,)).fetchone()
        status_esc, res_esc = conn.execute("SELECT status, resultado FROM registros WHERE id=?", (r_escanteios,)).fetchone()

    assert status_1x2 == "fechado" and res_1x2 == "ganhou"  # 2x0 -> casa ganhou
    assert status_btts == "fechado" and res_btts == "perdeu"  # 2x0 -> só um marcou, ambas marcam=Sim perdeu
    assert status_esc == "aberto" and res_esc is None  # nunca inventa resultado de escanteios


def test_aplicar_resultados_sem_arquivo_pendentes_da_erro_claro(tmp_path):
    caminho_db = tmp_path / "teste.sqlite"
    inserir_registro(caminho_db, "brasileirao", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05, data_jogo="2020-01-01")
    try:
        fechar.aplicar_resultados(caminho_db, tmp_path / "nao_existe.json")
        assert False, "devia ter levantado FileNotFoundError"
    except FileNotFoundError:
        pass
