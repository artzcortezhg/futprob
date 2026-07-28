# -*- coding: utf-8 -*-
"""Testes do modelo de escanteios (binomial negativa)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_corners import ajustar_modelo_escanteios, matriz_escanteios, distribuicao_total


def _amostra_nb(mu, alpha, rng):
    n = 1.0 / alpha
    p = n / (n + mu)
    return rng.negative_binomial(n, p)


def _dados_sinteticos(seed=1, n=6000, escala_media=5.5):
    """Gera dados sintéticos numa escala realista de escanteios (média ~5-6
    por time), justamente a escala que expôs o bug de inicialização/overflow
    numérico no ajuste do Championship (24+ times, média alta)."""
    rng = np.random.default_rng(seed)
    times = ["A", "B", "C", "D", "E", "F", "G", "H"]
    ataque_real = {t: v for t, v in zip(times, [0.3, 0.2, 0.1, 0.0, -0.1, -0.15, -0.2, -0.25])}
    m = np.mean(list(ataque_real.values()))
    ataque_real = {k: v - m for k, v in ataque_real.items()}
    defesa_real = {t: v for t, v in zip(times, [0.15, 0.1, 0.05, 0.0, -0.05, -0.05, -0.1, -0.1])}
    home_adv_real = np.log(escala_media) + 0.15  # mantém a média perto de escala_media
    alpha_real = 0.08

    linhas = []
    datas = pd.date_range("2015-01-01", periods=n, freq="1D")
    for d in datas:
        casa, fora = rng.choice(times, size=2, replace=False)
        mu_c = np.exp(ataque_real[casa] - defesa_real[fora] + home_adv_real)
        mu_f = np.exp(ataque_real[fora] - defesa_real[casa])
        linhas.append({
            "liga": "Teste", "Date": d, "HomeTeam": casa, "AwayTeam": fora,
            "HC": _amostra_nb(mu_c, alpha_real, rng), "AC": _amostra_nb(mu_f, alpha_real, rng),
        })
    return pd.DataFrame(linhas), ataque_real, defesa_real, home_adv_real, alpha_real


def test_media_ataque_e_zero():
    df, *_ = _dados_sinteticos()
    modelo = ajustar_modelo_escanteios(df, "Teste", data_corte="2033-01-01", xi=0.0001)
    assert abs(np.mean(list(modelo.ataque.values()))) < 1e-6


def test_recupera_parametros_em_escala_realista_de_escanteios():
    """Regressão do bug: com médias altas (~5-6 escanteios/time), o chute
    inicial ingênuo (tudo zero) gerava gradiente tão grande que o L-BFGS-B
    'convergia' na própria iteração 0, sem se mover. Este teste garante que
    o ajuste realmente aprende os parâmetros nessa escala."""
    df, ataque_real, defesa_real, home_adv_real, alpha_real = _dados_sinteticos()
    modelo = ajustar_modelo_escanteios(df, "Teste", data_corte="2033-01-01", xi=0.0001)

    assert abs(modelo.home_adv - home_adv_real) < 0.1
    assert abs(modelo.alpha - alpha_real) < 0.05
    for t in ataque_real:
        assert abs(modelo.ataque[t] - ataque_real[t]) < 0.15
        assert abs(modelo.defesa[t] - defesa_real[t]) < 0.15


def test_matriz_soma_um_e_media_condiz_com_mu_esperado():
    df, ataque_real, defesa_real, home_adv_real, _ = _dados_sinteticos()
    modelo = ajustar_modelo_escanteios(df, "Teste", data_corte="2033-01-01", xi=0.0001)

    mat = matriz_escanteios(modelo, "A", "H", max_escanteios=40)
    assert abs(mat.sum() - 1.0) < 1e-9
    assert (mat >= 0).all()

    mu_casa_esperado = np.exp(modelo.ataque["A"] - modelo.defesa["H"] + modelo.home_adv)
    media_casa_matriz = np.sum(np.arange(mat.shape[0]) * mat.sum(axis=1))
    assert abs(media_casa_matriz - mu_casa_esperado) < 0.5


def test_distribuicao_total_soma_um_e_e_convolucao_das_marginais():
    df, *_ = _dados_sinteticos()
    modelo = ajustar_modelo_escanteios(df, "Teste", data_corte="2033-01-01", xi=0.0001)
    mat = matriz_escanteios(modelo, "A", "H")
    total = distribuicao_total(mat)
    assert abs(total.sum() - 1.0) < 1e-9
    assert len(total) == 2 * mat.shape[0] - 1


def test_erro_se_time_desconhecido():
    df, *_ = _dados_sinteticos()
    modelo = ajustar_modelo_escanteios(df, "Teste", data_corte="2033-01-01")
    with pytest.raises(ValueError):
        matriz_escanteios(modelo, "A", "TimeInexistente")


def test_ajuste_nunca_usa_dados_posteriores_ao_corte():
    df, *_ = _dados_sinteticos()
    data_corte = pd.Timestamp("2020-06-15")
    modelo = ajustar_modelo_escanteios(df, "Teste", data_corte=data_corte)
    assert modelo.n_jogos_usados == len(df[df["Date"] < data_corte])
