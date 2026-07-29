# -*- coding: utf-8 -*-
"""Testes dos mercados de gols (markets.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from markets import (
    calcular_mercados, mercado_1x2, mercado_handicap_europeu,
    mercado_escanteios, mercado_cartoes, mercado_faltas,
    para_linhas_tabela_escanteios, para_linhas_tabela_cartoes, para_linhas_tabela_faltas,
    mascara_selecao, prob_conjunta,
)


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


def test_mascara_selecao_1x2_bate_com_mercado_1x2():
    """A soma da matriz na máscara tem que bater com o que mercado_1x2 já
    calcula agregado — senão a máscara está errada."""
    mat = _matriz_exemplo()
    probs = mercado_1x2(mat)
    assert mat[mascara_selecao("1X2", "Casa", mat)].sum() == pytest.approx(probs["casa"])
    assert mat[mascara_selecao("1X2", "Empate", mat)].sum() == pytest.approx(probs["empate"])
    assert mat[mascara_selecao("1X2", "Fora", mat)].sum() == pytest.approx(probs["fora"])


def test_mascara_selecao_desconhecida_retorna_none():
    mat = _matriz_exemplo()
    assert mascara_selecao("Escanteios Over/Under 9.5", "Over", mat) is None
    assert mascara_selecao("Placar exato", "1x0", mat) is None


def test_prob_conjunta_1x2_e_over_under_e_menor_que_cada_marginal():
    """Combinar duas seleções sempre reduz (ou mantém) a probabilidade —
    nunca pode dar um número maior que qualquer uma das marginais
    isoladas, senão a conta está errada."""
    mat = _matriz_exemplo()
    p_casa = mercado_1x2(mat)["casa"]
    conjunta = prob_conjunta(mat, [("1X2", "Casa"), ("Over/Under 1.5", "Over")])
    assert conjunta is not None
    assert conjunta <= p_casa + 1e-9


def test_prob_conjunta_nao_e_o_produto_ingenuo_das_marginais():
    """Gols e resultado são correlacionados (não independentes) — a
    probabilidade conjunta calculada na matriz de verdade precisa ser
    DIFERENTE do produto ingênuo das duas probabilidades marginais."""
    mat = _matriz_exemplo()
    p_casa = mercado_1x2(mat)["casa"]
    m = calcular_mercados(mat)
    p_over15 = m["over_under"]["1.5"]["over"]
    produto_ingenuo = p_casa * p_over15
    conjunta = prob_conjunta(mat, [("1X2", "Casa"), ("Over/Under 1.5", "Over")])
    assert abs(conjunta - produto_ingenuo) > 1e-6


def test_prob_conjunta_mercado_nao_suportado_retorna_none():
    mat = _matriz_exemplo()
    assert prob_conjunta(mat, [("1X2", "Casa"), ("Escanteios Over/Under 9.5", "Over")]) is None


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


def _matriz_escanteios_exemplo():
    from scipy.stats import nbinom
    n, p_casa, p_fora = 20.0, 20.0 / (20.0 + 5.5), 20.0 / (20.0 + 4.2)
    valores = np.arange(0, 25)
    pmf_casa = nbinom.pmf(valores, n, p_casa)
    pmf_fora = nbinom.pmf(valores, n, p_fora)
    mat = np.outer(pmf_casa, pmf_fora)
    return mat / mat.sum()


def test_mercado_escanteios_soma_um_em_todas_as_linhas():
    mat = _matriz_escanteios_exemplo()
    m = mercado_escanteios(mat)
    for grupo in ("total", "casa", "fora"):
        for linha, p in m[grupo].items():
            assert abs(p["over"] + p["under"] - 1.0) < 1e-9


def test_mercado_cartoes_e_faltas_soma_um():
    mat = _matriz_escanteios_exemplo()  # mesma forma de matriz conjunta, serve para o teste
    m_cartoes = mercado_cartoes(mat)
    m_faltas = mercado_faltas(mat)
    for m in (m_cartoes, m_faltas):
        for linha, p in m["total"].items():
            assert abs(p["over"] + p["under"] - 1.0) < 1e-9


def test_para_linhas_tabela_escanteios_cartoes_faltas():
    mat = _matriz_escanteios_exemplo()
    linhas_esc = para_linhas_tabela_escanteios(mercado_escanteios(mat))
    linhas_cart = para_linhas_tabela_cartoes(mercado_cartoes(mat))
    linhas_falt = para_linhas_tabela_faltas(mercado_faltas(mat))
    # total + casa + fora, 2 linhas (over/under) por linha de aposta
    assert len(linhas_esc) == 2 * (6 + 6 + 6)
    assert len(linhas_cart) == 2 * 4
    assert len(linhas_falt) == 2 * 9
