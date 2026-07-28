# -*- coding: utf-8 -*-
"""Testes das funções puras do coletor (collector.py) — parsers, inferência
de chave de mercado e filtros. Não sobem browser nenhum aqui."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collector import (
    _infer_market_key, _mercado_permitido, _parse_betano,
    salvar_snapshot, EventOdds, MarketOdds, OutcomeOdd,
)


def test_infer_market_key_h2h():
    assert _infer_market_key("Resultado Final", ["1", "X", "2"]) == "h2h"


def test_infer_market_key_dupla_chance():
    assert _infer_market_key("Dupla Chance", ["1X", "12", "X2"]) == "double_chance"


def test_infer_market_key_ambas_marcam():
    assert _infer_market_key("Ambas Equipes Marcam", ["Sim", "Não"]) == "btts"


def test_infer_market_key_over_under_gols():
    assert _infer_market_key("Total de Golos Mais/Menos", ["Acima 2.5", "Abaixo 2.5"]) == "ou_2.5_goals"


def test_infer_market_key_cartoes():
    assert _infer_market_key("Total de Cartões", ["Acima 3.5", "Abaixo 3.5"]) == "ou_3.5_cards"


def test_infer_market_key_cartoes_vermelhos_nao_e_cartoes_normal():
    chave = _infer_market_key("Cartões Vermelhos Acima/Abaixo", ["Acima 0.5", "Abaixo 0.5"])
    assert chave == "ou_0.5_redcards"


def test_infer_market_key_escanteios_por_time():
    chave = _infer_market_key("Escanteios do Flamengo RJ", ["Acima 4.5", "Abaixo 4.5"], home_name="Flamengo RJ")
    assert chave == "ou_4.5_corners_team1"


def test_infer_market_key_empate_anula_nao_e_dupla_chance():
    """Regressão: 'Empate Anula a Aposta' (Draw No Bet, 2 seleções nomeadas
    pelo time) é um mercado DIFERENTE de dupla chance (3 seleções 1X/12/X2)
    — batiam na mesma categoria antes e cruzavam errado com o modelo de
    'Dupla chance' (visto ao vivo: 100% dos double_chance coletados eram na
    verdade Draw No Bet, com só 2 seleções nomeadas com o time)."""
    chave = _infer_market_key("Empate Anula a Aposta", ["Fluminense", "Bahia"])
    assert chave == "draw_no_bet"
    assert chave != "double_chance"


def test_draw_no_bet_nao_e_mercado_permitido():
    """draw_no_bet não tem modelo no futprob — melhor descartar do que
    coletar e cruzar com o modelo errado (Dupla chance)."""
    assert _mercado_permitido("draw_no_bet") is False


def test_dupla_chance_genuina_continua_permitida():
    assert _mercado_permitido("double_chance") is True


def test_mercado_permitido_aceita_apenas_o_que_o_futprob_modela():
    assert _mercado_permitido("h2h")
    assert _mercado_permitido("double_chance")
    assert _mercado_permitido("btts")
    assert _mercado_permitido("ou_2.5_goals")
    assert _mercado_permitido("ou_9.5_corners")
    assert _mercado_permitido("ou_3.5_cards")
    # fora do escopo do futprob (Bloco 4): chutes, faltas, tempo, handicap
    assert not _mercado_permitido("ou_5.5_shots")
    assert not _mercado_permitido("ou_10.5_fouls")
    assert not _mercado_permitido("ou_0.5_redcards")
    assert not _mercado_permitido("ou_2.5_goals_ht")
    assert not _mercado_permitido("ah_-1")
    assert not _mercado_permitido("misc")


def _betano_json_exemplo(start_offset_horas=24):
    inicio = (datetime.utcnow() + timedelta(hours=start_offset_horas)).isoformat() + "Z"
    return {
        "events": {
            "1": {
                "participants": [{"name": "Flamengo RJ"}, {"name": "Palmeiras"}],
                "startDate": inicio,
                "marketIdList": ["10", "11"],
            }
        },
        "markets": {
            "10": {"name": "Resultado Final", "selectionIdList": ["100", "101", "102"]},
            "11": {"name": "Total de Cartões", "selectionIdList": ["110", "111"]},
        },
        "selections": {
            "100": {"name": "1", "price": 2.1}, "101": {"name": "X", "price": 3.2}, "102": {"name": "2", "price": 3.5},
            "110": {"name": "Acima 3.5", "price": 1.9}, "111": {"name": "Abaixo 3.5", "price": 1.85},
        },
    }


def test_parse_betano_extrai_eventos_e_filtra_mercados():
    eventos = _parse_betano(_betano_json_exemplo(), "betano")
    assert len(eventos) == 1
    ev = eventos[0]
    assert ev.home_team == "Flamengo RJ" and ev.away_team == "Palmeiras"
    chaves = {m.market_key for m in ev.markets}
    assert chaves == {"h2h", "ou_3.5_cards"}


def test_parse_betano_ignora_eventos_virtuais():
    dados = _betano_json_exemplo()
    dados["events"]["1"]["url"] = "/virtuals/algum-jogo"
    eventos = _parse_betano(dados, "betano")
    assert eventos == []


def test_salvar_snapshot_grava_linhas(tmp_path):
    caminho_db = tmp_path / "teste.sqlite"
    resultado = {
        "betano": [EventOdds(
            external_id="soccer_flamengo_vs_palmeiras", home_team="Flamengo RJ", away_team="Palmeiras",
            commence_time=datetime.utcnow(), bookmaker="betano",
            markets=[MarketOdds(market_key="h2h", outcomes=[
                OutcomeOdd(name="1", price=2.1, bookmaker="betano"),
                OutcomeOdd(name="2", price=3.5, bookmaker="betano"),
            ])],
        )],
    }
    n = salvar_snapshot(resultado, "manha", caminho_db)
    assert n == 2

    import sqlite3
    with sqlite3.connect(caminho_db) as conn:
        total = conn.execute("SELECT COUNT(*) FROM odds_coletadas").fetchone()[0]
    assert total == 2
