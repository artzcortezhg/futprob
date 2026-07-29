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

from model_corners import distribuicao_total

# linhas de over/under de 0.5 a 5.5
LINHAS_OVER_UNDER = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
# handicaps europeus de -2 a +2 (aplicados ao mandante)
HANDICAPS_EUROPEUS = [-2, -1, 0, 1, 2]
MAX_GOLS_PLACAR_EXATO = 4

# escanteios, cartões e faltas: matriz conjunta casa x fora (mesma convenção
# de model_corners.matriz_escanteios / model_cards.matriz_contagem)
LINHAS_ESCANTEIOS_TOTAL = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
LINHAS_ESCANTEIOS_TIME = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
LINHAS_CARTOES = [2.5, 3.5, 4.5, 5.5]
LINHAS_FALTAS = [19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5]


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


def _mercado_over_under_total(distribuicao: np.ndarray, linhas: list[float]) -> dict[str, dict[str, float]]:
    """Over/under a partir de uma distribuição 1D do total (índice = valor)."""
    valores = np.arange(len(distribuicao))
    resultado = {}
    for linha in linhas:
        over = float(distribuicao[valores > linha].sum())
        resultado[f"{linha}"] = {"over": over, "under": 1.0 - over}
    return resultado


def _mercado_over_under_marginal(matriz: np.ndarray, eixo: int, linhas: list[float]) -> dict[str, dict[str, float]]:
    """Over/under de UM lado (casa ou fora) a partir da marginal da matriz
    conjunta. eixo=0 -> mandante (soma nas linhas), eixo=1 -> visitante."""
    marginal = matriz.sum(axis=1) if eixo == 0 else matriz.sum(axis=0)
    valores = np.arange(len(marginal))
    resultado = {}
    for linha in linhas:
        over = float(marginal[valores > linha].sum())
        resultado[f"{linha}"] = {"over": over, "under": 1.0 - over}
    return resultado


def mercado_escanteios(
    matriz: np.ndarray,
    linhas_total: list[float] = LINHAS_ESCANTEIOS_TOTAL,
    linhas_time: list[float] = LINHAS_ESCANTEIOS_TIME,
) -> dict:
    """Mercados de escanteios: over/under do total do jogo e over/under por
    time, a partir da matriz conjunta (model_corners.matriz_escanteios)."""
    total = distribuicao_total(matriz)
    return {
        "total": _mercado_over_under_total(total, linhas_total),
        "casa": _mercado_over_under_marginal(matriz, eixo=0, linhas=linhas_time),
        "fora": _mercado_over_under_marginal(matriz, eixo=1, linhas=linhas_time),
    }


def mercado_cartoes(matriz: np.ndarray, linhas: list[float] = LINHAS_CARTOES) -> dict:
    """Over/under de cartões (total do jogo), a partir da matriz conjunta
    (model_cards.matriz_contagem com o modelo de cartões)."""
    total = distribuicao_total(matriz)
    return {"total": _mercado_over_under_total(total, linhas)}


def mercado_faltas(matriz: np.ndarray, linhas: list[float] = LINHAS_FALTAS) -> dict:
    """Over/under de faltas (total do jogo), a partir da matriz conjunta
    (model_cards.matriz_contagem com o modelo de faltas)."""
    total = distribuicao_total(matriz)
    return {"total": _mercado_over_under_total(total, linhas)}


def para_linhas_tabela_escanteios(mercados: dict) -> list[tuple[str, str, float]]:
    linhas: list[tuple[str, str, float]] = []
    for linha, probs in mercados["total"].items():
        linhas.append((f"Escanteios Over/Under {linha}", "Over", probs["over"]))
        linhas.append((f"Escanteios Over/Under {linha}", "Under", probs["under"]))
    for lado, rotulo in (("casa", "Escanteios time da casa"), ("fora", "Escanteios time visitante")):
        for linha, probs in mercados[lado].items():
            linhas.append((f"{rotulo} Over/Under {linha}", "Over", probs["over"]))
            linhas.append((f"{rotulo} Over/Under {linha}", "Under", probs["under"]))
    return linhas


def para_linhas_tabela_cartoes(mercados: dict) -> list[tuple[str, str, float]]:
    linhas: list[tuple[str, str, float]] = []
    for linha, probs in mercados["total"].items():
        linhas.append((f"Cartões Over/Under {linha}", "Over", probs["over"]))
        linhas.append((f"Cartões Over/Under {linha}", "Under", probs["under"]))
    return linhas


def para_linhas_tabela_faltas(mercados: dict) -> list[tuple[str, str, float]]:
    linhas: list[tuple[str, str, float]] = []
    for linha, probs in mercados["total"].items():
        linhas.append((f"Faltas Over/Under {linha}", "Over", probs["over"]))
        linhas.append((f"Faltas Over/Under {linha}", "Under", probs["under"]))
    return linhas


def mascara_selecao(mercado: str, selecao: str, matriz: np.ndarray) -> np.ndarray | None:
    """Máscara booleana (mesmo shape da matriz de placares) marcando as
    células em que a seleção (mercado, selecao) se realiza — usada pra
    calcular a probabilidade CONJUNTA de combinar seleções do MESMO jogo
    num bilhete, em vez de multiplicar probabilidades marginais (que
    ignoraria a correlação real entre placar e cada mercado: um time que
    vence tende a marcar mais gols também, por exemplo). Só cobre os
    mercados de gols usados no bilhete (1X2, dupla chance, ambas marcam,
    over/under) — retorna None pra qualquer outro (escanteios/cartões
    vêm de outra matriz, não dá pra combinar aqui)."""
    n = matriz.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    if mercado == "1X2":
        if selecao == "Casa":
            return i > j
        if selecao == "Empate":
            return i == j
        if selecao == "Fora":
            return i < j
        return None

    if mercado == "Dupla chance":
        if selecao.startswith("1X"):
            return i >= j
        if selecao.startswith("12"):
            return i != j
        if selecao.startswith("X2"):
            return i <= j
        return None

    if mercado == "Ambas marcam":
        ambas = (i >= 1) & (j >= 1)
        return ambas if selecao == "Sim" else ~ambas

    if mercado.startswith("Over/Under"):
        try:
            linha = float(mercado.rsplit(maxsplit=1)[-1])
        except ValueError:
            return None
        total = i + j
        acima = total > linha
        return acima if selecao == "Over" else ~acima

    return None


def prob_conjunta(matriz: np.ndarray, selecoes: list[tuple[str, str]]) -> float | None:
    """Probabilidade de TODAS as seleções acontecerem juntas no mesmo
    jogo — soma a matriz de placares na interseção das máscaras de cada
    seleção. None se alguma seleção não tiver máscara suportada."""
    mascara_conjunta = None
    for mercado, selecao in selecoes:
        m = mascara_selecao(mercado, selecao, matriz)
        if m is None:
            return None
        mascara_conjunta = m if mascara_conjunta is None else (mascara_conjunta & m)
    if mascara_conjunta is None:
        return None
    return float(matriz[mascara_conjunta].sum())
