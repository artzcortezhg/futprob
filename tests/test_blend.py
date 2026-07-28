# -*- coding: utf-8 -*-
"""Testes da mistura modelo+mercado em log-odds (blend.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from blend import (
    logit, sigmoide, combinar_probs, avaliar_grade_w, escolher_w_otimo,
    salvar_peso, carregar_pesos, w_da_liga, GRADE_W,
)


def test_logit_sigmoide_sao_inversas():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(sigmoide(logit(p)), p)


def test_combinar_w_zero_e_o_modelo_puro():
    probs_modelo = np.array([[0.5, 0.2, 0.3], [0.1, 0.1, 0.8]])
    probs_mercado = np.array([[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]])
    combinado = combinar_probs(probs_modelo, probs_mercado, w=0.0)
    assert np.allclose(combinado, probs_modelo, atol=1e-8)


def test_combinar_w_um_e_o_mercado_puro():
    probs_modelo = np.array([[0.5, 0.2, 0.3]])
    probs_mercado = np.array([[0.2, 0.5, 0.3]])
    combinado = combinar_probs(probs_modelo, probs_mercado, w=1.0)
    assert np.allclose(combinado, probs_mercado, atol=1e-8)


def test_combinar_sempre_soma_um():
    rng = np.random.default_rng(0)
    probs_modelo = rng.dirichlet([1, 1, 1], size=20)
    probs_mercado = rng.dirichlet([1, 1, 1], size=20)
    for w in GRADE_W:
        combinado = combinar_probs(probs_modelo, probs_mercado, w)
        assert np.allclose(combinado.sum(axis=1), 1.0)


def test_avaliar_grade_w_ignora_jogos_sem_prejogo_e_acha_otimo():
    # mercado sempre acerta perfeitamente -> w=1.0 deve minimizar o log loss
    n = 50
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "FTR": rng.choice(["H", "D", "A"], size=n),
        "prob_casa_modelo": rng.uniform(0.2, 0.5, n),
        "prob_empate_modelo": rng.uniform(0.2, 0.4, n),
        "prob_fora_modelo": rng.uniform(0.2, 0.4, n),
    })
    # normaliza as probs do modelo pra somar 1
    soma = df[["prob_casa_modelo", "prob_empate_modelo", "prob_fora_modelo"]].sum(axis=1)
    for c in ["prob_casa_modelo", "prob_empate_modelo", "prob_fora_modelo"]:
        df[c] = df[c] / soma
    # mercado "perfeito": 0.9 no resultado real, resto dividido
    df["prob_casa_prejogo"] = np.where(df["FTR"] == "H", 0.9, 0.05)
    df["prob_empate_prejogo"] = np.where(df["FTR"] == "D", 0.9, 0.05)
    df["prob_fora_prejogo"] = np.where(df["FTR"] == "A", 0.9, 0.05)

    tabela = avaliar_grade_w(df)
    assert list(tabela["w"]) == GRADE_W
    assert escolher_w_otimo(tabela) == 1.0


def test_avaliar_grade_w_erro_se_nenhum_jogo_tem_prejogo():
    df = pd.DataFrame({
        "FTR": ["H"], "prob_casa_modelo": [0.5], "prob_empate_modelo": [0.3], "prob_fora_modelo": [0.2],
        "prob_casa_prejogo": [np.nan], "prob_empate_prejogo": [np.nan], "prob_fora_prejogo": [np.nan],
    })
    with pytest.raises(ValueError):
        avaliar_grade_w(df)


def test_salvar_e_carregar_pesos_roundtrip(tmp_path):
    caminho = tmp_path / "pesos_blend.json"
    salvar_peso("Premier League", 0.4, caminho=caminho)
    salvar_peso("La Liga", 0.6, caminho=caminho)
    pesos = carregar_pesos(caminho)
    assert pesos == {"Premier League": 0.4, "La Liga": 0.6}
    assert w_da_liga("Premier League", caminho=caminho) == 0.4
    assert w_da_liga("brasileirao", padrao=0.0, caminho=caminho) == 0.0  # liga sem peso salvo


def test_w_da_liga_sem_arquivo_retorna_padrao(tmp_path):
    caminho = tmp_path / "nao_existe.json"
    assert w_da_liga("Premier League", padrao=0.0, caminho=caminho) == 0.0
