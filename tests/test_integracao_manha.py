# -*- coding: utf-8 -*-
"""Testes da função pura de mapeamento mercado-coletado -> mercado-futprob
(integracao_manha.py). Não roda coleta nem Telegram de verdade."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from integracao_manha import _mercados_para_jogo


def test_mapeia_over_under_gols():
    resultado = _mercados_para_jogo("ou_2.5_goals", {"Over": 2.1, "Under": 1.75})
    pares = {(m, s) for m, s, _ in resultado}
    assert ("Over/Under 2.5", "Over") in pares
    assert ("Over/Under 2.5", "Under") in pares


def test_mapeia_escanteios():
    resultado = _mercados_para_jogo("ou_9.5_corners", {"Over": 1.9, "Under": 1.85})
    pares = {(m, s) for m, s, _ in resultado}
    assert ("Escanteios Over/Under 9.5", "Over") in pares


def test_mapeia_cartoes():
    resultado = _mercados_para_jogo("ou_3.5_cards", {"Over": 1.95, "Under": 1.80})
    pares = {(m, s) for m, s, _ in resultado}
    assert ("Cartões Over/Under 3.5", "Over") in pares


def test_mapeia_btts():
    resultado = _mercados_para_jogo("btts", {"Sim": 1.85, "Não": 1.95})
    pares = {(m, s) for m, s, _ in resultado}
    assert ("Ambas marcam", "Sim") in pares
    assert ("Ambas marcam", "Não") in pares


def test_mercado_fora_do_escopo_retorna_vazio():
    assert _mercados_para_jogo("ou_5.5_shots", {"Over": 2.0, "Under": 1.8}) == []
    assert _mercados_para_jogo("double_chance", {"1X": 1.3}) == []  # tratado à parte (precisa nome do time)
