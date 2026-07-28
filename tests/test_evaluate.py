# -*- coding: utf-8 -*-
"""Testes das métricas de avaliação (evaluate.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import remover_margem_odds, log_loss_multiclasse, brier_score_multiclasse, tabela_calibracao


def test_remover_margem_soma_um():
    p_casa, p_empate, p_fora = remover_margem_odds(2.0, 3.5, 4.0)
    assert abs(p_casa + p_empate + p_fora - 1.0) < 1e-9
    # a margem foi removida: prob implícita bruta de casa (1/2.0=0.5) deve cair
    assert p_casa < 0.5


def test_remover_margem_com_odds_ausente():
    resultado = remover_margem_odds(np.nan, 3.5, 4.0)
    assert all(np.isnan(v) for v in resultado)


def test_log_loss_previsao_perfeita_tende_a_zero():
    y_idx = np.array([0, 1, 2])
    probs = np.array([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005], [0.0005, 0.0005, 0.999]])
    assert log_loss_multiclasse(y_idx, probs) < 0.01


def test_log_loss_pior_que_uniforme_quando_errado():
    y_idx = np.array([0, 0, 0])
    probs_errado = np.array([[0.01, 0.01, 0.98]] * 3)
    probs_uniforme = np.array([[1 / 3, 1 / 3, 1 / 3]] * 3)
    assert log_loss_multiclasse(y_idx, probs_errado) > log_loss_multiclasse(y_idx, probs_uniforme)


def test_brier_score_previsao_perfeita_e_zero():
    y_idx = np.array([0, 1])
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert brier_score_multiclasse(y_idx, probs) < 1e-9


def test_tabela_calibracao_formato():
    rng = np.random.default_rng(0)
    n = 200
    y_idx = rng.integers(0, 3, n)
    probs = rng.dirichlet([1, 1, 1], n)
    tabela = tabela_calibracao(y_idx, probs, n_bins=5)
    assert tabela["n"].sum() == n * 3  # 3 probabilidades (casa/empate/fora) por jogo
