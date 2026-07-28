# -*- coding: utf-8 -*-
"""Testes dos mercados de gols (markets.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from markets import calcular_mercados, mercado_1x2, mercado_handicap_europeu


def _matriz_exemplo():
    """Matriz de placares sintética (Poisson independente, lambda=1.4 mu=1.1)
    apenas para validar as somas dos mercados."""
    from scipy.stats import poisson
    gols = np.arange(0, 11)
    p_casa = poisson.pmf(gols, 1.4)
    p_fora = poisson.pmf(gols, 1.1)
    mat = np.outer(p_casa, p_fora)
    return mat / mat.sum()


def test_1x2_soma_um():
    mat = _matriz_exemplo()
    probs = mercado_1x2(mat)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_over_under_soma_um_em_todas_as_linhas():
    mat = _matriz_exemplo()
    m = calcular_mercados(mat)
    for linha, p in m["over_under"].items():
        assert abs(p["over"] + p["under"] - 1.0) < 1e-9


def test_ambas_marcam_soma_um():
    mat = _matriz_exemplo()
    m = calcular_mercados(mat)
    assert abs(sum(m["ambas_marcam"].values()) - 1.0) < 1e-9


def test_placar_exato_soma_um():
    mat = _matriz_exemplo()
    m = calcular_mercados(mat)
    assert abs(sum(m["placar_exato"].values()) - 1.0) < 1e-9
    assert abs(m["placar_exato"]["1x0"] - mat[1, 0]) < 1e-12


def test_handicap_zero_igual_1x2():
    mat = _matriz_exemplo()
    probs_1x2 = mercado_1x2(mat)
    handicaps = mercado_handicap_europeu(mat)
    for k in ("casa", "empate", "fora"):
        assert abs(handicaps[0][k] - probs_1x2[k]) < 1e-9


def test_handicaps_somam_um():
    mat = _matriz_exemplo()
    handicaps = mercado_handicap_europeu(mat)
    for h, probs in handicaps.items():
        assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_dupla_chance_consistente():
    mat = _matriz_exemplo()
    m = calcular_mercados(mat)
    assert abs(m["dupla_chance"]["12 (casa ou fora)"] - (1 - m["1x2"]["empate"])) < 1e-9
