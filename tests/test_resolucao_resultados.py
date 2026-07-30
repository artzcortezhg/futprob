# -*- coding: utf-8 -*-
"""Testes de resolucao_resultados.py: dado o placar real, diz se a
seleção registrada bateu -- base do fechamento automático de resultados
(scripts/fechar_resultados_reais.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resolucao_resultados import resultado_bate_selecao


def test_1x2_casa_vitoria_da_casa():
    assert resultado_bate_selecao("1X2", "Casa", 2, 1) is True
    assert resultado_bate_selecao("1X2", "Fora", 2, 1) is False
    assert resultado_bate_selecao("1X2", "Empate", 2, 1) is False


def test_1x2_empate():
    assert resultado_bate_selecao("1X2", "Empate", 1, 1) is True
    assert resultado_bate_selecao("1X2", "Casa", 1, 1) is False


def test_ambas_marcam():
    assert resultado_bate_selecao("Ambas marcam", "Sim", 2, 1) is True
    assert resultado_bate_selecao("Ambas marcam", "Sim", 2, 0) is False
    assert resultado_bate_selecao("Ambas marcam", "Não", 2, 0) is True


def test_over_under_gols():
    assert resultado_bate_selecao("Over/Under 2.5", "Over", 2, 1) is True  # total 3 > 2.5
    assert resultado_bate_selecao("Over/Under 2.5", "Under", 1, 0) is True  # total 1 < 2.5
    assert resultado_bate_selecao("Over/Under 2.5", "Over", 1, 0) is False


def test_dupla_chance():
    assert resultado_bate_selecao("Dupla chance", "1X (casa ou empate)", 1, 1) is True
    assert resultado_bate_selecao("Dupla chance", "1X (casa ou empate)", 0, 1) is False


def test_mercado_de_escanteios_retorna_none_nunca_inventa():
    """Placar sozinho não diz quantos escanteios/cartões/faltas rolaram --
    nunca inventa um resultado pra esses mercados."""
    assert resultado_bate_selecao("Escanteios Over/Under 9.5", "Over", 3, 2) is None
    assert resultado_bate_selecao("Cartões Over/Under 3.5", "Under", 3, 2) is None
