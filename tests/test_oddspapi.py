# -*- coding: utf-8 -*-
"""Testes de src/oddspapi.py — mocka toda chamada de rede (nunca gasta
cota de verdade num teste automatizado)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import oddspapi
from painel_db import registrar_uso_oddspapi


def test_uso_atual_conta_registros(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    assert oddspapi.uso_atual(caminho) == 0
    registrar_uso_oddspapi(caminho, "odds-by-tournaments", sucesso=True)
    assert oddspapi.uso_atual(caminho) == 1


def test_buscar_melhores_odds_bloqueia_quando_cota_estourada(tmp_path, monkeypatch):
    caminho = tmp_path / "teste.sqlite"
    monkeypatch.setattr(oddspapi, "LIMITE_USOS", 2)
    registrar_uso_oddspapi(caminho, "x", sucesso=True)
    registrar_uso_oddspapi(caminho, "x", sucesso=True)
    resultado = oddspapi.buscar_melhores_odds(caminho)
    assert resultado["sucesso"] is False
    assert "cota" in resultado["erro"]


def test_buscar_melhores_odds_monta_jogos_a_partir_da_resposta_mockada(tmp_path, monkeypatch):
    caminho = tmp_path / "teste.sqlite"

    def _chamar_falso(endpoint, params, caminho_db):
        if endpoint == "odds-by-tournaments":
            return [{
                "tournamentId": 325, "participant1Id": 111, "participant2Id": 222,
                "startTime": "2026-07-29T22:30:00.000Z",
                "bookmakerOdds": {"pinnacle": {"markets": {
                    "101": {"outcomes": {
                        "101": {"players": {"0": {"price": 1.8}}},
                        "102": {"players": {"0": {"price": 3.4}}},
                        "103": {"players": {"0": {"price": 4.5}}},
                    }},
                    "999999": {"outcomes": {"999999": {"players": {"0": {"price": 1.5}}}}},  # mercado desconhecido, ignorado
                }}},
            }]
        if endpoint == "participants":
            return {"111": "Mirassol FC SP", "222": "Clube do Remo PA"}
        raise AssertionError(f"endpoint inesperado: {endpoint}")

    monkeypatch.setattr(oddspapi, "_chamar", _chamar_falso)
    resultado = oddspapi.buscar_melhores_odds(caminho)
    assert resultado["sucesso"] is True
    assert len(resultado["jogos"]) == 1
    jogo = resultado["jogos"][0]
    assert jogo["liga"] == "brasileirao"
    assert jogo["casa"] == "Mirassol FC SP"
    assert jogo["fora"] == "Clube do Remo PA"
    mercados = {(m["mercado"], m["selecao"]): m["odd"] for m in jogo["mercados"]}
    assert mercados[("1X2", "Casa")] == 1.8
    assert mercados[("1X2", "Empate")] == 3.4
    assert mercados[("1X2", "Fora")] == 4.5
    assert len(jogo["mercados"]) == 3  # o mercado desconhecido (999999) não entra


def test_nomes_participantes_usa_cache_local_sem_chamar_api(tmp_path):
    caminho = tmp_path / "teste.sqlite"
    import sqlite3
    with sqlite3.connect(caminho) as conn:
        conn.executescript(oddspapi.SQL_CRIAR_TABELA_PARTICIPANTES)
        conn.execute("INSERT INTO oddspapi_participantes (participant_id, nome) VALUES (111, 'Mirassol FC SP')")
        conn.commit()

    def _chamar_que_nao_deveria_ser_chamada(*a, **k):
        raise AssertionError("não devia precisar chamar a API — o id já estava no cache")

    import oddspapi as mod
    original = mod._chamar
    mod._chamar = _chamar_que_nao_deveria_ser_chamada
    try:
        nomes = mod._nomes_participantes({111}, caminho)
    finally:
        mod._chamar = original
    assert nomes == {111: "Mirassol FC SP"}
