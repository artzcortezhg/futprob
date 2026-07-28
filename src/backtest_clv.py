# -*- coding: utf-8 -*-
"""
Backtest histórico de CLV (Closing Line Value) para o mercado 1X2.

Regra de decisão: "registrei na odd pré-jogo (PSH/PSD/PSA) quando EV > 5%".
O EV é calculado com o MODELO PURO (nunca a mistura modelo+mercado) contra a
odd pré-jogo — comparar a mistura contra a mesma odd de onde ela veio seria
circular (a mistura já incorpora o próprio mercado, então o "EV" resultante
seria só a margem da casa, sempre negativo, e nunca dispararia uma aposta).
A mistura (ver blend.py) é reservada para a probabilidade "oficial" exibida
ao usuário (predict.py, painel) — aqui usamos sempre o modelo puro.

CLV = odd_conseguida * prob_justa_de_fechamento - 1. CLV > 0 significa que a
odd pré-jogo conseguida era melhor que o valor justo do fechamento (sinal de
que a aposta tinha valor, independente do resultado da partida).

Máximo UMA aposta por jogo: entre as 3 seleções (casa/empate/fora), só a de
maior EV entre as que passam do limiar é registrada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SELECOES = ["casa", "empate", "fora"]
LIMIAR_EV_PADRAO = 0.05


def calcular_ev(prob_modelo: float, odd: float) -> float:
    """Valor esperado de uma aposta de 1 unidade: prob*odd - 1."""
    return prob_modelo * odd - 1.0


def calcular_clv(odd_conseguida: float, prob_fechamento_sem_margem: float) -> float:
    """CLV = odd conseguida * probabilidade justa (sem margem) de
    fechamento - 1. Positivo = odd melhor que o valor justo do fechamento."""
    return odd_conseguida * prob_fechamento_sem_margem - 1.0


def simular_backtest_clv(df_avaliacao: pd.DataFrame, limiar_ev: float = LIMIAR_EV_PADRAO) -> pd.DataFrame:
    """`df_avaliacao` vem de evaluate.avaliar_walkforward (precisa ter as
    colunas prob_*_modelo, PSH/PSD/PSA, PSCH/PSCD/PSCA e FTR). Retorna um
    registro por jogo em que o EV (modelo puro x odd pré-jogo) passou do
    limiar em pelo menos uma seleção — a de maior EV entre elas."""
    colunas_necessarias = [
        "prob_casa_modelo", "prob_empate_modelo", "prob_fora_modelo",
        "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "FTR",
    ]
    df = df_avaliacao.dropna(subset=colunas_necessarias).copy()

    odds_prejogo = {"casa": "PSH", "empate": "PSD", "fora": "PSA"}
    odds_fechamento = {"casa": "PSCH", "empate": "PSCD", "fora": "PSCA"}
    probs_modelo = {"casa": "prob_casa_modelo", "empate": "prob_empate_modelo", "fora": "prob_fora_modelo"}
    resultado_para_selecao = {"H": "casa", "D": "empate", "A": "fora"}

    registros = []
    for _, jogo in df.iterrows():
        # probabilidade de fechamento sem margem (só validação/CLV)
        inv = np.array([1.0 / jogo["PSCH"], 1.0 / jogo["PSCD"], 1.0 / jogo["PSCA"]])
        probs_fechamento_sem_margem = dict(zip(SELECOES, inv / inv.sum()))

        candidatos = []
        for selecao in SELECOES:
            prob_mod = jogo[probs_modelo[selecao]]
            odd_prejogo = jogo[odds_prejogo[selecao]]
            ev = calcular_ev(prob_mod, odd_prejogo)
            if ev > limiar_ev:
                candidatos.append((selecao, prob_mod, odd_prejogo, ev))

        if not candidatos:
            continue

        selecao, prob_mod, odd_prejogo, ev = max(candidatos, key=lambda c: c[3])
        odd_fechamento = jogo[odds_fechamento[selecao]]
        clv = calcular_clv(odd_prejogo, probs_fechamento_sem_margem[selecao])
        ganhou = resultado_para_selecao[jogo["FTR"]] == selecao

        registros.append({
            "Date": jogo["Date"], "liga": jogo["liga"],
            "HomeTeam": jogo["HomeTeam"], "AwayTeam": jogo["AwayTeam"],
            "selecao": selecao, "prob_modelo": prob_mod,
            "odd_prejogo": odd_prejogo, "odd_fechamento": odd_fechamento,
            "ev": ev, "clv": clv, "ganhou": ganhou,
            "retorno_papel": (odd_prejogo - 1.0) if ganhou else -1.0,
        })

    return pd.DataFrame(registros)


def resumo_backtest(df_apostas: pd.DataFrame) -> dict:
    """Resumo agregado: nº apostas, taxa de acerto, CLV médio/mediano, %
    positivo, ROI de papel (retorno/aposta, unidade de stake=1)."""
    if df_apostas.empty:
        return {"n_apostas": 0}
    return {
        "n_apostas": len(df_apostas),
        "taxa_acerto": float(df_apostas["ganhou"].mean()),
        "clv_medio": float(df_apostas["clv"].mean()),
        "clv_mediano": float(df_apostas["clv"].median()),
        "clv_pct_positivo": float((df_apostas["clv"] > 0).mean()),
        "roi_papel": float(df_apostas["retorno_papel"].mean()),
    }
