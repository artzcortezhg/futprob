# -*- coding: utf-8 -*-
"""Testes dos guarda-corpos de EV (guardrails.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guardrails import aplicar_guardrails, formatar_ranking


def _cand(mercado, selecao, prob_modelo, odd):
    ev = prob_modelo * odd - 1.0
    return {"mercado": mercado, "selecao": selecao, "prob_modelo": prob_modelo, "odd": odd, "ev": ev}


def test_nunca_inclui_placar_exato():
    candidatos = [_cand("Placar exato", "1x0", 0.15, 8.0), _cand("1X2", "Casa", 0.5, 2.2)]
    ranking = aplicar_guardrails(candidatos)
    assert all(c["mercado"] != "Placar exato" for c in ranking)


def test_nunca_inclui_prob_abaixo_de_8_por_cento():
    candidatos = [_cand("1X2", "Empate", 0.07, 3.0), _cand("1X2", "Casa", 0.5, 2.2)]
    ranking = aplicar_guardrails(candidatos)
    assert len(ranking) == 1
    assert ranking[0]["selecao"] == "Casa"


def test_ev_acima_de_15_por_cento_fica_suspeito_e_nunca_e_apostaria():
    # EV = 0.9*2.5-1 = 1.25 (125%, bem suspeito)
    candidatos = [_cand("1X2", "Casa", 0.9, 2.5)]
    ranking = aplicar_guardrails(candidatos)
    assert ranking[0]["suspeito"] is True
    assert ranking[0]["apostaria"] is False
    assert "erro do modelo" in ranking[0]["aviso"]


def test_no_maximo_uma_apostaria_por_jogo():
    candidatos = [
        _cand("1X2", "Casa", 0.5, 2.3),   # ev = 0.15
        _cand("Over/Under 2.5", "Over", 0.55, 2.0),  # ev = 0.10
        _cand("Ambas marcam", "Sim", 0.5, 2.2),  # ev = 0.10
    ]
    ranking = aplicar_guardrails(candidatos)
    apostarias = [c for c in ranking if c["apostaria"]]
    assert len(apostarias) == 1
    assert apostarias[0]["selecao"] == "Casa"  # maior EV


def test_apostaria_pula_o_suspeito_e_marca_o_proximo_valido():
    candidatos = [
        _cand("1X2", "Casa", 0.9, 2.5),  # ev=1.25, suspeito -> nunca apostaria
        _cand("Over/Under 2.5", "Over", 0.55, 2.0),  # ev=0.10, válido
    ]
    ranking = aplicar_guardrails(candidatos)
    apostarias = [c for c in ranking if c["apostaria"]]
    assert len(apostarias) == 1
    assert apostarias[0]["selecao"] == "Over"


def test_ev_abaixo_do_limiar_nunca_vira_apostaria():
    candidatos = [_cand("1X2", "Casa", 0.4, 2.4)]  # ev = -0.04
    ranking = aplicar_guardrails(candidatos)
    assert ranking[0]["apostaria"] is False


def test_ranking_ordenado_por_ev_decrescente():
    candidatos = [
        _cand("1X2", "Fora", 0.3, 2.0),   # ev=-0.4
        _cand("1X2", "Casa", 0.5, 2.3),   # ev=0.15
        _cand("1X2", "Empate", 0.3, 3.0),  # ev=-0.1
    ]
    ranking = aplicar_guardrails(candidatos)
    evs = [c["ev"] for c in ranking]
    assert evs == sorted(evs, reverse=True)


def test_formatar_ranking_nunca_menciona_dinheiro():
    candidatos = [_cand("1X2", "Casa", 0.5, 2.3)]
    texto = formatar_ranking(aplicar_guardrails(candidatos))
    for palavra_proibida in ("R$", "reais", "stake", "valor"):
        assert palavra_proibida not in texto.lower()


def test_formatar_ranking_vazio():
    assert "Nenhuma seleção" in formatar_ranking([])
