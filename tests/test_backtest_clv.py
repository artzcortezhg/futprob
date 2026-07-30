# -*- coding: utf-8 -*-
"""Testes do backtest de CLV (backtest_clv.py)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest_clv import (
    calcular_ev, calcular_clv, simular_backtest_clv, resumo_backtest,
    calibracao_apostas_selecionadas, varredura_limiar_ev,
)


def test_calcular_ev():
    assert calcular_ev(0.5, 2.2) == pytest.approx(0.1)
    assert calcular_ev(0.5, 1.8) == pytest.approx(-0.1)


def test_calcular_clv_positivo_quando_odd_melhor_que_fechamento():
    # peguei odd 2.2, "valor justo" do fechamento era 1/0.5=2.0 -> CLV positivo
    assert calcular_clv(2.2, 0.5) == pytest.approx(0.1)


def test_calcular_clv_negativo_quando_odd_pior_que_fechamento():
    assert calcular_clv(1.8, 0.5) == pytest.approx(-0.1)


def _jogo_exemplo(prob_casa_modelo, psh, psch, ftr="H", psd=2.0, psa=2.0):
    # psd/psa baixas o bastante pra não gerar EV "fantasma" em empate/fora
    # sem querer, já que sobra probabilidade nessas duas seleções também
    return {
        "Date": pd.Timestamp("2024-01-01"), "liga": "Teste", "HomeTeam": "A", "AwayTeam": "B",
        "prob_casa_modelo": prob_casa_modelo, "prob_empate_modelo": (1 - prob_casa_modelo) / 2,
        "prob_fora_modelo": (1 - prob_casa_modelo) / 2,
        "PSH": psh, "PSD": psd, "PSA": psa,
        "PSCH": psch, "PSCD": psd, "PSCA": psa,
        "FTR": ftr,
    }


def test_backtest_registra_apenas_ev_acima_do_limiar():
    # EV casa = 0.6*2.5-1 = 0.5 (bem acima do limiar) -> deve entrar
    # EV casa = 0.3*1.5-1 = -0.55 (bem abaixo) -> não deve entrar
    df = pd.DataFrame([
        _jogo_exemplo(prob_casa_modelo=0.6, psh=2.5, psch=2.3),
        _jogo_exemplo(prob_casa_modelo=0.3, psh=1.5, psch=1.5),
    ])
    apostas = simular_backtest_clv(df, limiar_ev=0.05)
    assert len(apostas) == 1
    assert apostas.iloc[0]["selecao"] == "casa"


def test_backtest_no_maximo_uma_aposta_por_jogo():
    # monta um jogo onde tanto casa quanto fora teriam EV alto
    jogo = _jogo_exemplo(prob_casa_modelo=0.6, psh=2.5, psch=2.3)
    jogo["prob_fora_modelo"] = 0.35
    jogo["PSA"] = 3.5  # EV fora = 0.35*3.5-1 = 0.225, também > limiar
    df = pd.DataFrame([jogo])
    apostas = simular_backtest_clv(df, limiar_ev=0.05)
    assert len(apostas) == 1  # só a de maior EV (casa, EV=0.5 > 0.225)
    assert apostas.iloc[0]["selecao"] == "casa"


def test_backtest_pula_jogos_sem_odds():
    df = pd.DataFrame([_jogo_exemplo(prob_casa_modelo=0.6, psh=2.5, psch=2.3)])
    df.loc[0, "PSH"] = float("nan")
    apostas = simular_backtest_clv(df, limiar_ev=0.05)
    assert apostas.empty


def test_resumo_backtest_vazio():
    assert resumo_backtest(pd.DataFrame()) == {"n_apostas": 0}


def test_resumo_backtest_calcula_metricas():
    df = pd.DataFrame([
        {"ganhou": True, "clv": 0.1, "retorno_papel": 1.5},
        {"ganhou": False, "clv": -0.05, "retorno_papel": -1.0},
    ])
    resumo = resumo_backtest(df)
    assert resumo["n_apostas"] == 2
    assert resumo["taxa_acerto"] == pytest.approx(0.5)
    assert resumo["clv_medio"] == pytest.approx(0.025)
    assert resumo["clv_pct_positivo"] == pytest.approx(0.5)
    assert resumo["roi_papel"] == pytest.approx(0.25)


def test_calibracao_apostas_selecionadas_vazio():
    tabela = calibracao_apostas_selecionadas(pd.DataFrame(columns=["prob_modelo", "ganhou"]))
    assert tabela.empty


def test_calibracao_apostas_selecionadas_agrupa_por_faixa_de_probabilidade():
    df = pd.DataFrame([
        {"prob_modelo": 0.55, "ganhou": True},
        {"prob_modelo": 0.58, "ganhou": False},
        {"prob_modelo": 0.20, "ganhou": False},
    ])
    tabela = calibracao_apostas_selecionadas(df, n_bins=5)
    # as duas primeiras (0.55, 0.58) caem na mesma faixa (0.4-0.6)
    linha_alta = tabela[tabela["prob_media_prevista"] > 0.5].iloc[0]
    assert linha_alta["n"] == 2
    assert linha_alta["taxa_real_acerto"] == pytest.approx(0.5)  # 1 de 2 ganhou


def test_varredura_limiar_ev_reporta_roi_por_limiar():
    # jogo com EV bem alto na casa (bem acima de todos os limiares testados)
    df = pd.DataFrame([_jogo_exemplo(prob_casa_modelo=0.9, psh=3.0, psch=2.3, ftr="H")])
    resultado = varredura_limiar_ev(df, limiares=[0.05, 0.50])
    assert list(resultado["limiar_ev"]) == [0.05, 0.50]
    assert (resultado["n_apostas"] == 1).all()  # o mesmo jogo passa nos dois limiares
    for valor in resultado["roi_papel"]:
        assert valor == pytest.approx(2.0)  # ganhou com odd 3.0 -> retorno 2.0


def test_varredura_limiar_ev_sem_apostas_no_limiar_retorna_none():
    df = pd.DataFrame([_jogo_exemplo(prob_casa_modelo=0.3, psh=1.5, psch=1.5, ftr="A")])
    resultado = varredura_limiar_ev(df, limiares=[0.05])
    assert resultado.iloc[0]["n_apostas"] == 0
    assert resultado.iloc[0]["roi_papel"] is None
