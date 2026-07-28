# -*- coding: utf-8 -*-
"""Testes do modelo Dixon-Coles."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_goals import ajustar_modelo, matriz_placares, calcular_pesos_temporais, _log_poisson_continuo


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


def test_log_poisson_continuo_bate_com_scipy_para_inteiros():
    """A extensão contínua via log-gama deve reproduzir exatamente a
    log-pmf de Poisson padrão quando avaliada em contagens inteiras."""
    from scipy.stats import poisson as poisson_scipy
    k = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    lam = np.array([1.3, 1.3, 2.7, 0.8, 4.2])
    esperado = poisson_scipy.logpmf(k, lam)
    obtido = _log_poisson_continuo(k, lam)
    assert np.allclose(obtido, esperado)


def test_log_poisson_continuo_aceita_k_fracionario():
    """Diferente de scipy.stats.poisson (que dá -inf para k não-inteiro),
    a extensão contínua deve dar um valor finito e suave para xG fracionário."""
    valor = _log_poisson_continuo(np.array([1.7]), np.array([1.4]))
    assert np.isfinite(valor).all()


def test_ajuste_com_alpha_xg_zero_e_igual_ao_padrao_sem_colunas_xg():
    """alpha_xg=0.0 (padrão) não deve exigir nem usar as colunas de xG."""
    df, *_ = _dados_sinteticos()
    modelo = ajustar_modelo(df, "Teste", data_corte="2030-01-01", alpha_xg=0.0)
    assert modelo.alpha_xg == 0.0


def test_ajuste_com_xg_continuo_nao_arredonda():
    """Com alpha_xg=1, o alvo de treino é o xG exato (fracionário), não mais
    arredondado -- por isso um xG diferente do resultado real deve mover os
    parâmetros de forma proporcional à diferença, não em saltos de inteiro."""
    df, *_ = _dados_sinteticos()
    df_xg = df.copy()
    rng = np.random.default_rng(99)
    # xG com ruído fracionário em torno dos gols observados (nunca um inteiro exato)
    df_xg["xG_casa"] = df_xg["FTHG"].astype(float) + rng.uniform(-0.4, 0.4, len(df_xg))
    df_xg["xG_fora"] = df_xg["FTAG"].astype(float) + rng.uniform(-0.4, 0.4, len(df_xg))

    modelo_xg = ajustar_modelo(df_xg, "Teste", data_corte="2030-01-01", xi=0.0001, alpha_xg=1.0)
    assert modelo_xg.alpha_xg == 1.0
    # ainda deve recuperar parâmetros na vizinhança dos reais (o ruído tem média 0)
    assert abs(np.mean(list(modelo_xg.ataque.values()))) < 1e-6


def test_correcao_rho_usa_gols_observados_mesmo_treinando_com_xg():
    """Mesmo com alpha_xg=1 (treino 100% em xG), a correção de Dixon-Coles
    deve ser calculada sobre os placares REALMENTE observados. Construímos
    um xG que não tem nenhuma relação com placares baixos (sempre alto,
    ~3.0) e um placar observado sempre 0x0: se a correção usasse o xG (que
    nunca é 0 ou 1), rho ficaria sem gradiente nenhum vindo das 4 células
    especiais; usando os gols observados (sempre 0x0), o rho DEVE se mover
    para longe do valor inicial, evidenciando que a correção olha para os
    gols e não para o xG."""
    linhas = []
    datas = pd.date_range("2015-01-01", periods=500, freq="1D")
    rng = np.random.default_rng(7)
    times = ["A", "B", "C", "D"]
    for d in datas:
        casa, fora = rng.choice(times, size=2, replace=False)
        linhas.append({
            "liga": "Teste", "Date": d, "HomeTeam": casa, "AwayTeam": fora,
            "FTHG": 0, "FTAG": 0,  # sempre 0x0 -> força a célula (0,0) do tau
            "xG_casa": 3.0 + rng.uniform(-0.1, 0.1), "xG_fora": 3.0 + rng.uniform(-0.1, 0.1),
        })
    df = pd.DataFrame(linhas)
    modelo = ajustar_modelo(df, "Teste", data_corte="2030-01-01", xi=0.0001, alpha_xg=1.0)
    # tau(0,0) = 1 - lam*mu*rho: com 100% dos jogos 0x0, maximizar P(0,0)
    # empurra rho para o mais NEGATIVO possível (perto do piso -0.9) -- bem
    # longe do -0.05 inicial, e só possível se a correção usar os gols
    # observados (sempre 0x0), não o xG (sempre ~3.0, nunca nas 4 células).
    assert modelo.rho < -0.3


def test_ajuste_xg_exige_colunas_de_xg():
    df, *_ = _dados_sinteticos()
    with pytest.raises(ValueError):
        ajustar_modelo(df, "Teste", data_corte="2030-01-01", alpha_xg=0.5)


def test_alpha_xg_fora_do_intervalo_gera_erro():
    df, *_ = _dados_sinteticos()
    with pytest.raises(ValueError):
        ajustar_modelo(df, "Teste", data_corte="2030-01-01", alpha_xg=1.5)
    with pytest.raises(ValueError):
        ajustar_modelo(df, "Teste", data_corte="2030-01-01", alpha_xg=-0.1)
