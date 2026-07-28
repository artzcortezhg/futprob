# -*- coding: utf-8 -*-
"""
Cálculo de mercados de apostas de gols a partir da matriz de placares
(saída de model_goals.matriz_placares). Cada mercado é obtido por soma de
células da matriz — nenhuma probabilidade é recalculada fora dela.

Convenção da matriz: matriz[i, j] = P(mandante marca i gols e visitante
marca j gols), com i, j de 0 até max_gols.
"""
from __future__ import annotations

import numpy as np

# linhas de over/under de 0.5 a 5.5
LINHAS_OVER_UNDER = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
# handicaps europeus de -2 a +2 (aplicados ao mandante)
HANDICAPS_EUROPEUS = [-2, -1, 0, 1, 2]
MAX_GOLS_PLACAR_EXATO = 4


def mercado_1x2(matriz: np.ndarray) -> dict[str, float]:
    n = matriz.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return {
        "casa": float(matriz[i > j].sum()),
        "empate": float(matriz[i == j].sum()),
        "fora": float(matriz[i < j].sum()),
    }


def mercado_dupla_chance(probs_1x2: dict[str, float]) -> dict[str, float]:
    return {
        "1X (casa ou empate)": probs_1x2["casa"] + probs_1x2["empate"],
        "12 (casa ou fora)": probs_1x2["casa"] + probs_1x2["fora"],
        "X2 (empate ou fora)": probs_1x2["empate"] + probs_1x2["fora"],
    }


def mercado_over_under(matriz: np.ndarray, linhas: list[float] = LINHAS_OVER_UNDER) -> dict[str, dict[str, float]]:
    n = matriz.shape[0]
    total_gols = np.add.outer(np.arange(n), np.arange(n))
    resultado = {}
    for linha in linhas:
        over = float(matriz[total_gols > linha].sum())
        resultado[f"{linha}"] = {"over": over, "under": 1.0 - over}
    return resultado


def mercado_ambas_marcam(matriz: np.ndarray) -> dict[str, float]:
    n = matriz.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sim = float(matriz[(i >= 1) & (j >= 1)].sum())
    return {"sim": sim, "nao": 1.0 - sim}


def mercado_placar_exato(matriz: np.ndarray, max_gols: int = MAX_GOLS_PLACAR_EXATO) -> dict[str, float]:
    placares = {}
    for i in range(max_gols + 1):
        for j in range(max_gols + 1):
            placares[f"{i}x{j}"] = float(matriz[i, j])
    soma_listados = sum(placares.values())
    placares["outro placar"] = float(max(0.0, 1.0 - soma_listados))
    return placares


def mercado_handicap_europeu(matriz: np.ndarray, handicaps: list[int] = HANDICAPS_EUROPEUS) -> dict[int, dict[str, float]]:
    """Handicap europeu (de 3 vias): aplica `h` ao placar do mandante e
    reavalia 1X2 sobre o placar ajustado. Ex.: h=-1 => mandante precisa
    vencer por 2+ gols para 'cobrir' o handicap."""
    n = matriz.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    resultado = {}
    for h in handicaps:
        ajustado = i + h - j
        resultado[h] = {
            "casa": float(matriz[ajustado > 0].sum()),
            "empate": float(matriz[ajustado == 0].sum()),
            "fora": float(matriz[ajustado < 0].sum()),
        }
    return resultado


def calcular_mercados(matriz: np.ndarray) -> dict:
    """Calcula todos os mercados suportados a partir da matriz de placares."""
    probs_1x2 = mercado_1x2(matriz)
    return {
        "1x2": probs_1x2,
        "dupla_chance": mercado_dupla_chance(probs_1x2),
        "over_under": mercado_over_under(matriz),
        "ambas_marcam": mercado_ambas_marcam(matriz),
        "placar_exato": mercado_placar_exato(matriz),
        "handicap_europeu": mercado_handicap_europeu(matriz),
    }


def para_linhas_tabela(mercados: dict) -> list[tuple[str, str, float]]:
    """Achata o dicionário de mercados em linhas (mercado, seleção, probabilidade)
    prontas para exibição em tabela."""
    linhas: list[tuple[str, str, float]] = []

    linhas.append(("1X2", "Casa", mercados["1x2"]["casa"]))
    linhas.append(("1X2", "Empate", mercados["1x2"]["empate"]))
    linhas.append(("1X2", "Fora", mercados["1x2"]["fora"]))

    for selecao, prob in mercados["dupla_chance"].items():
        linhas.append(("Dupla chance", selecao, prob))

    for linha, probs in mercados["over_under"].items():
        linhas.append((f"Over/Under {linha}", "Over", probs["over"]))
        linhas.append((f"Over/Under {linha}", "Under", probs["under"]))

    linhas.append(("Ambas marcam", "Sim", mercados["ambas_marcam"]["sim"]))
    linhas.append(("Ambas marcam", "Não", mercados["ambas_marcam"]["nao"]))

    for placar, prob in mercados["placar_exato"].items():
        linhas.append(("Placar exato", placar, prob))

    for h, probs in mercados["handicap_europeu"].items():
        sinal = f"+{h}" if h > 0 else str(h)
        linhas.append((f"Handicap europeu ({sinal})", "Casa", probs["casa"]))
        linhas.append((f"Handicap europeu ({sinal})", "Empate", probs["empate"]))
        linhas.append((f"Handicap europeu ({sinal})", "Fora", probs["fora"]))

    return linhas
