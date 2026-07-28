# -*- coding: utf-8 -*-
"""
Guarda-corpos aplicados em todo cálculo de EV do bot (Bloco 3.4e):

- Modo papel: NUNCA sugere valor em dinheiro (nenhuma função aqui formata
  valores monetários — só odds, probabilidades e EV em %).
- No máximo UMA seleção marcada como "apostaria" por jogo.
- Nunca marca placar exato como seleção de aposta.
- Nunca considera seleção com probabilidade do modelo < 8%.
- EV > 15% vira "suspeito: provável erro do modelo" — nunca é a apostaria.
"""
from __future__ import annotations

LIMIAR_EV_APOSTARIA_PADRAO = 0.05
LIMIAR_EV_SUSPEITO_PADRAO = 0.15
PROB_MINIMA_PADRAO = 0.08


def aplicar_guardrails(
    candidatos: list[dict],
    limiar_ev_apostaria: float = LIMIAR_EV_APOSTARIA_PADRAO,
    limiar_ev_suspeito: float = LIMIAR_EV_SUSPEITO_PADRAO,
    prob_minima: float = PROB_MINIMA_PADRAO,
) -> list[dict]:
    """candidatos: lista de dicts com pelo menos {mercado, selecao,
    prob_modelo, odd, ev}. Retorna o ranking (ordenado por EV desc) já
    filtrado, com os campos 'suspeito' e 'apostaria' adicionados. No máximo
    um item tem apostaria=True."""
    filtrados = []
    for c in candidatos:
        if c["mercado"].strip().lower() == "placar exato":
            continue  # nunca placar exato
        if c["prob_modelo"] < prob_minima:
            continue  # nunca prob < 8%
        item = dict(c)
        item["suspeito"] = item["ev"] > limiar_ev_suspeito
        item["aviso"] = "suspeito: provável erro do modelo" if item["suspeito"] else None
        filtrados.append(item)

    filtrados.sort(key=lambda c: c["ev"], reverse=True)

    apostou = False
    for item in filtrados:
        pode_apostar = (not apostou) and (not item["suspeito"]) and (item["ev"] > limiar_ev_apostaria)
        item["apostaria"] = pode_apostar
        apostou = apostou or pode_apostar

    return filtrados


def formatar_ranking(ranking: list[dict]) -> str:
    """Formata o ranking em texto — NUNCA inclui valor monetário, só odds,
    probabilidades e EV em percentual (modo papel)."""
    if not ranking:
        return "Nenhuma seleção passou pelos filtros (prob. mínima 8%, sem placar exato)."
    linhas = []
    for item in ranking:
        marca = " 🎯 APOSTARIA" if item["apostaria"] else ""
        aviso = f" ⚠️ {item['aviso']}" if item.get("aviso") else ""
        linhas.append(
            f"{item['mercado']} / {item['selecao']}: odd {item['odd']:.2f} | "
            f"prob. modelo {item['prob_modelo'] * 100:.1f}% | EV {item['ev'] * 100:+.1f}%{marca}{aviso}"
        )
    return "\n".join(linhas)
