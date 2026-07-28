# -*- coding: utf-8 -*-
"""
Mistura modelo + mercado para o 1X2, combinando em log-odds (logit).

    logit_combinado_k = w * logit(prob_mercado_k) + (1-w) * logit(prob_modelo_k)

para cada resultado k (casa/empate/fora), depois volta para probabilidade
(sigmoide) e renormaliza para somar 1 (a soma das sigmoides não fecha em 1
automaticamente). w=0 é o modelo puro, w=1 é o mercado puro.

Insumo do mercado: SEMPRE a odd pré-jogo (PSH/PSD/PSA), nunca a de
fechamento (PSCH/PSCD/PSCA) — a de fechamento é reservada para validação
(CLV), nunca entra como insumo de decisão (ver evaluate.avaliar_walkforward
e src/backtest_clv.py).

w é escolhido por liga via walk-forward (grade 0.0 a 1.0, passo 0.1,
minimizando log loss) e fixado como padrão em data/processed/modelos/
pesos_blend.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PESOS = RAIZ / "data" / "processed" / "modelos" / "pesos_blend.json"

GRADE_W = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def combinar_probs(probs_modelo: np.ndarray, probs_mercado: np.ndarray, w: float) -> np.ndarray:
    """probs_modelo, probs_mercado: array (n, 3) na ordem [casa, empate, fora].
    Retorna as probabilidades combinadas, já renormalizadas para somar 1
    em cada linha."""
    logit_combinado = w * logit(probs_mercado) + (1 - w) * logit(probs_modelo)
    probs = sigmoide(logit_combinado)
    return probs / probs.sum(axis=1, keepdims=True)


def avaliar_grade_w(df_avaliacao: pd.DataFrame, grade_w: list[float] = GRADE_W) -> pd.DataFrame:
    """Calcula o log loss da mistura modelo+mercado para cada w da grade,
    usando SOMENTE os jogos com odd pré-jogo disponível (prob_*_prejogo).
    `df_avaliacao` deve vir de evaluate.avaliar_walkforward."""
    from evaluate import log_loss_multiclasse, brier_score_multiclasse, _resultado_para_indice

    com_prejogo = df_avaliacao.dropna(subset=["prob_casa_prejogo", "prob_empate_prejogo", "prob_fora_prejogo"])
    if com_prejogo.empty:
        raise ValueError("Nenhum jogo avaliado tem odd pré-jogo (PSH/PSD/PSA) disponível.")

    y_idx = com_prejogo["FTR"].map(_resultado_para_indice).to_numpy()
    probs_modelo = com_prejogo[["prob_casa_modelo", "prob_empate_modelo", "prob_fora_modelo"]].to_numpy()
    probs_mercado = com_prejogo[["prob_casa_prejogo", "prob_empate_prejogo", "prob_fora_prejogo"]].to_numpy()

    linhas = []
    for w in grade_w:
        probs_combinadas = combinar_probs(probs_modelo, probs_mercado, w)
        linhas.append({
            "w": w,
            "log_loss": log_loss_multiclasse(y_idx, probs_combinadas),
            "brier_score": brier_score_multiclasse(y_idx, probs_combinadas),
        })
    return pd.DataFrame(linhas)


def escolher_w_otimo(tabela_grade: pd.DataFrame) -> float:
    return float(tabela_grade.loc[tabela_grade["log_loss"].idxmin(), "w"])


def carregar_pesos(caminho: Path = CAMINHO_PESOS) -> dict[str, float]:
    if not caminho.exists():
        return {}
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_peso(liga: str, w: float, caminho: Path = CAMINHO_PESOS) -> None:
    pesos = carregar_pesos(caminho)
    pesos[liga] = w
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(pesos, ensure_ascii=False, indent=2), encoding="utf-8")


def w_da_liga(liga: str, padrao: float = 0.0, caminho: Path = CAMINHO_PESOS) -> float:
    """w=0.0 (modelo puro) por padrão se a liga não tiver peso calibrado
    (ex.: não tem odd pré-jogo na fonte, como brasileirao/mls)."""
    return carregar_pesos(caminho).get(liga, padrao)
