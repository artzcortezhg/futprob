# -*- coding: utf-8 -*-
"""
Resolve se uma seleção (mercado, seleção) registrada bateu ou não, a partir
do placar REAL da partida — reaproveita markets.mascara_selecao (a MESMA
lógica já usada pra calcular probabilidade conjunta do bilhete, ver
catalogo.montar_aposta_e_bilhete) em vez de duplicar as regras de 1X2/
dupla chance/ambas marcam/over-under numa segunda cópia.

Só resolve mercados de GOLS (1X2, dupla chance, ambas marcam, over/under
gols) — escanteios/cartões/faltas exigem contagem que não dá pra saber só
pelo placar final, e retornam None (nunca inventa um resultado pra eles).
"""
from __future__ import annotations

import numpy as np

from markets import mascara_selecao


def resultado_bate_selecao(mercado: str, selecao: str, gols_casa: int, gols_fora: int) -> bool | None:
    """True/False se dá pra resolver só com o placar, None se o mercado
    exige dado que o placar sozinho não tem (escanteios/cartões/faltas)."""
    tamanho = max(gols_casa, gols_fora) + 2
    matriz_indicadora = np.zeros((tamanho, tamanho))
    mascara = mascara_selecao(mercado, selecao, matriz_indicadora)
    if mascara is None:
        return None
    return bool(mascara[gols_casa, gols_fora])
