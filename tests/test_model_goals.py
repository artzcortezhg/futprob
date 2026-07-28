# -*- coding: utf-8 -*-
"""Testes do modelo Dixon-Coles."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_goals import ajustar_modelo, matriz_placares, calcular_pesos_temporais


def _dados_sinteticos(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    times = ["A", "B", "C", "D", "E", "F"]
    ataque_real = {"A": 0.5, "B": 0.2, "C": 0.0, "D": -0.1, "E": -0.2, "F": -0.4}
    m = np.mean(list(ataque_real.values()))
    ataque_real = {k: v - m for k, v in ataque_real.items()}
    defesa_real = {"A": 0.3, "B": 0.1, "C": 0.0, "D": -0.1, "E": -0.2, "F": -0.1}
    home_adv_real = 0.25

    linhas = []
    datas = pd.date_range("2015-01-01", periods=n, freq="1D")
    for d in datas:
        casa, fora = rng.choice(times, size=2, replace=False)
        lam = np.exp(ataque_real[casa] - defesa_real[fora] + home_adv_real)
        mu = np.exp(ataque_real[fora] - defesa_real[casa])
        linhas.append({
            "liga": "Teste", "Date": d, "HomeTeam": casa, "AwayTeam": fora,
            "FTHG": rng.poisson(lam), "FTAG": rng.poisson(mu),
        })
    return pd.DataFrame(linhas), ataque_real, defesa_real, home_adv_real


def test_media_ataque_e_zero():
    df, *_ = _dados_sinteticos()
    modelo = ajustar_modelo(df, "Teste", data_corte="2030-01-01", xi=0.0005)
    assert abs(np.mean(list(modelo.ataque.values()))) < 1e-8


def test_recupera_parametros_conhecidos():
    df, ataque_real, defesa_real, home_adv_real = _dados_sinteticos()
    modelo = ajustar_modelo(df, "Teste", data_corte="2030-01-01", xi=0.0001)
    assert abs(modelo.home_adv - home_adv_real) < 0.05
    for t in ataque_real:
        assert abs(modelo.ataque[t] - ataque_real[t]) < 0.1
        assert abs(modelo.defesa[t] - defesa_real[t]) < 0.1


def test_matriz_placares_soma_um():
    df, *_ = _dados_sinteticos()
    modelo = ajustar_modelo(df, "Teste", data_corte="2030-01-01")
    mat = matriz_placares(modelo, "A", "F")
    assert mat.shape == (11, 11)
    assert abs(mat.sum() - 1.0) < 1e-9
    assert (mat >= 0).all()


def test_ajuste_nunca_usa_dados_posteriores_ao_corte():
    df, *_ = _dados_sinteticos()
    data_corte = pd.Timestamp("2020-06-15")
    modelo = ajustar_modelo(df, "Teste", data_corte=data_corte)
    assert modelo.n_jogos_usados == len(df[df["Date"] < data_corte])


def test_pesos_temporais_decrescem_com_o_tempo():
    datas = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]))
    pesos = calcular_pesos_temporais(datas, data_corte="2021-06-01", xi=0.0018)
    assert pesos[0] < pesos[1] < pesos[2]


def test_erro_se_time_desconhecido():
    df, *_ = _dados_sinteticos()
    modelo = ajustar_modelo(df, "Teste", data_corte="2030-01-01")
    with pytest.raises(ValueError):
        matriz_placares(modelo, "A", "TimeInexistente")


def test_ajuste_com_xg_usa_xg_arredondado_como_pseudo_gols():
    df, *_ = _dados_sinteticos()
    # cria colunas de xG idênticas aos gols observados -> o ajuste com
    # fonte='xg' deve reproduzir exatamente o ajuste com fonte='gols'
    df_xg = df.copy()
    df_xg["xG_casa"] = df_xg["FTHG"].astype(float)
    df_xg["xG_fora"] = df_xg["FTAG"].astype(float)

    modelo_gols = ajustar_modelo(df_xg, "Teste", data_corte="2030-01-01", fonte="gols")
    modelo_xg = ajustar_modelo(df_xg, "Teste", data_corte="2030-01-01", fonte="xg")

    assert modelo_xg.fonte == "xg"
    assert modelo_gols.fonte == "gols"
    assert abs(modelo_xg.home_adv - modelo_gols.home_adv) < 1e-6
    for t in modelo_gols.ataque:
        assert abs(modelo_xg.ataque[t] - modelo_gols.ataque[t]) < 1e-6


def test_ajuste_xg_exige_colunas_de_xg():
    df, *_ = _dados_sinteticos()
    with pytest.raises(ValueError):
        ajustar_modelo(df, "Teste", data_corte="2030-01-01", fonte="xg")


def test_fonte_invalida_gera_erro():
    df, *_ = _dados_sinteticos()
    with pytest.raises(ValueError):
        ajustar_modelo(df, "Teste", data_corte="2030-01-01", fonte="chutometro")
