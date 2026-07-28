# -*- coding: utf-8 -*-
"""Testes do modelo de disciplina (cartões/faltas) com efeito de árbitro."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_cards import (
    _ajustar_generico, _preparar_indices_arbitro, matriz_contagem,
    ajustar_modelo_cartoes, ajustar_modelo_faltas,
)


def _amostra_nb(mu, alpha, rng):
    n = 1.0 / alpha
    p = n / (n + mu)
    return rng.negative_binomial(n, p)


def _dados_sinteticos_com_arbitro(seed=5, n=3000):
    rng = np.random.default_rng(seed)
    times = ["A", "B", "C", "D", "E", "F"]
    arbitros = ["Arb1", "Arb2", "Arb3", "ArbRaro"]
    ataque_real = {"A": 0.2, "B": 0.1, "C": 0.0, "D": -0.05, "E": -0.1, "F": -0.15}
    m = np.mean(list(ataque_real.values()))
    ataque_real = {k: v - m for k, v in ataque_real.items()}
    defesa_real = {"A": 0.1, "B": 0.05, "C": 0.0, "D": -0.05, "E": -0.05, "F": -0.05}
    home_adv_real = -0.15  # cartões: mandante costuma receber MENOS cartões
    alpha_real = 0.3
    efeito_arb_real = {"Arb1": 0.2, "Arb2": -0.1, "Arb3": -0.1, "ArbRaro": 0.5}

    linhas = []
    datas = pd.date_range("2015-01-01", periods=n, freq="1D")
    for d in datas:
        casa, fora = rng.choice(times, size=2, replace=False)
        arb = rng.choice(arbitros, p=[0.35, 0.35, 0.25, 0.05])
        ef = efeito_arb_real[arb]
        mu_c = np.exp(ataque_real[casa] - defesa_real[fora] + home_adv_real + ef)
        mu_f = np.exp(ataque_real[fora] - defesa_real[casa] + ef)
        linhas.append({
            "liga": "Teste", "Date": d, "HomeTeam": casa, "AwayTeam": fora, "Referee": arb,
            "HY": _amostra_nb(mu_c, alpha_real, rng), "AY": _amostra_nb(mu_f, alpha_real, rng),
        })
    return pd.DataFrame(linhas), ataque_real, defesa_real, home_adv_real, alpha_real, efeito_arb_real


def test_arbitro_com_poucos_jogos_fica_de_fora():
    df = pd.DataFrame({"Referee": ["A"] * 15 + ["B"] * 8 + ["C"] * 20})
    validos, idx = _preparar_indices_arbitro(df, min_jogos=10)
    assert validos == ["A", "C"]
    assert (idx[15:23] == -1).all()


def test_media_efeito_arbitro_e_zero():
    df, *_ = _dados_sinteticos_com_arbitro()
    modelo = _ajustar_generico(df, "Teste", "2024-01-01", "HY", "AY", "cartoes",
                                xi=0.0001, usar_arbitro=True, min_jogos_arbitro=10)
    assert abs(np.mean(list(modelo.efeito_arbitro.values()))) < 1e-6


def test_recupera_efeitos_relativos_do_arbitro():
    df, ataque_real, defesa_real, home_adv_real, alpha_real, efeito_real = _dados_sinteticos_com_arbitro()
    modelo = _ajustar_generico(df, "Teste", "2024-01-01", "HY", "AY", "cartoes",
                                xi=0.0001, usar_arbitro=True, min_jogos_arbitro=10)

    m_real = np.mean(list(efeito_real.values()))
    for arb in efeito_real:
        esperado = efeito_real[arb] - m_real
        assert abs(modelo.efeito_arbitro[arb] - esperado) < 0.15
    assert abs(modelo.home_adv - home_adv_real) < 0.1
    assert abs(modelo.alpha - alpha_real) < 0.1


def test_sem_efeito_arbitro_quando_usar_arbitro_false():
    df, *_ = _dados_sinteticos_com_arbitro()
    modelo = _ajustar_generico(df, "Teste", "2024-01-01", "HY", "AY", "cartoes",
                                xi=0.0001, usar_arbitro=False, min_jogos_arbitro=10)
    assert modelo.efeito_arbitro == {}
    assert modelo.usar_arbitro is False


def test_matriz_contagem_soma_um_com_e_sem_arbitro():
    df, *_ = _dados_sinteticos_com_arbitro()
    modelo = _ajustar_generico(df, "Teste", "2024-01-01", "HY", "AY", "cartoes",
                                xi=0.0001, usar_arbitro=True, min_jogos_arbitro=10)
    mat_com_arb = matriz_contagem(modelo, "A", "F", arbitro="Arb1", max_valor=15)
    mat_sem_arb = matriz_contagem(modelo, "A", "F", arbitro=None, max_valor=15)
    assert abs(mat_com_arb.sum() - 1.0) < 1e-9
    assert abs(mat_sem_arb.sum() - 1.0) < 1e-9
    assert not np.allclose(mat_com_arb, mat_sem_arb)  # arbitro com efeito != 0 deve mudar a matriz


def test_ajustar_modelo_cartoes_usa_arbitro_apenas_nas_ligas_certas():
    df, *_ = _dados_sinteticos_com_arbitro()
    df_pl = df.copy(); df_pl["liga"] = "Premier League"
    df_ll = df.copy(); df_ll["liga"] = "La Liga"

    modelo_pl = ajustar_modelo_cartoes(df_pl, "Premier League", "2024-01-01", xi=0.0001)
    modelo_ll = ajustar_modelo_faltas(df_ll.rename(columns={"HY": "HF", "AY": "AF"}), "La Liga", "2024-01-01", xi=0.0001)

    assert modelo_pl.usar_arbitro is True
    assert len(modelo_pl.efeito_arbitro) > 0
    assert modelo_ll.usar_arbitro is False
    assert modelo_ll.efeito_arbitro == {}


def test_erro_se_time_desconhecido():
    df, *_ = _dados_sinteticos_com_arbitro()
    modelo = _ajustar_generico(df, "Teste", "2024-01-01", "HY", "AY", "cartoes",
                                xi=0.0001, usar_arbitro=True, min_jogos_arbitro=10)
    with pytest.raises(ValueError):
        matriz_contagem(modelo, "A", "TimeInexistente")
