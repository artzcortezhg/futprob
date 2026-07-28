# -*- coding: utf-8 -*-
"""Testes do painel web (dashboard.py) via TestClient do FastAPI."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dashboard
from painel_db import inserir_registro, fechar_registro, registrar_coleta, registrar_uso_oddspapi


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    caminho_db = tmp_path / "teste.sqlite"
    monkeypatch.setattr(dashboard, "CAMINHO_DB", caminho_db)
    return TestClient(dashboard.app), caminho_db


def test_pagina_inicial_carrega(cliente):
    client, _ = cliente
    resposta = client.get("/")
    assert resposta.status_code == 200
    assert "futprob" in resposta.text


def test_jogos_do_dia_vazio(cliente):
    client, _ = cliente
    resposta = client.get("/api/jogos-do-dia")
    assert resposta.status_code == 200
    assert resposta.json() == {"jogos": []}


def test_registros_vazio(cliente):
    client, _ = cliente
    resposta = client.get("/api/registros")
    assert resposta.json() == {"registros": []}


def test_registros_lista_apos_inserir(cliente):
    client, caminho_db = cliente
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05, origem="bot")
    resposta = client.get("/api/registros")
    dados = resposta.json()["registros"]
    assert len(dados) == 1
    assert dados[0]["status"] == "aberto"


def test_marcar_resultado_valido(cliente):
    client, caminho_db = cliente
    rid = inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    resposta = client.post(f"/api/registros/{rid}/resultado?resultado=ganhou")
    assert resposta.status_code == 200
    dados = client.get("/api/registros").json()["registros"]
    assert dados[0]["resultado"] == "ganhou"


def test_marcar_resultado_invalido_retorna_400(cliente):
    client, caminho_db = cliente
    rid = inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    resposta = client.post(f"/api/registros/{rid}/resultado?resultado=empatou")
    assert resposta.status_code == 400


def test_marcar_resultado_nao_apaga_outros_registros(cliente):
    client, caminho_db = cliente
    rid1 = inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    rid2 = inserir_registro(caminho_db, "La Liga", "C", "D", "1X2", "Fora", 0.4, 2.5, 0.05)
    client.post(f"/api/registros/{rid1}/resultado?resultado=ganhou")
    dados = client.get("/api/registros").json()["registros"]
    assert len(dados) == 2  # nenhum registro sumiu


def test_clv_serie_vazio(cliente):
    client, _ = cliente
    resposta = client.get("/api/clv-serie")
    assert resposta.json() == {"pontos": []}


def test_clv_serie_com_registros_fechados(cliente):
    client, caminho_db = cliente
    rid = inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    fechar_registro(caminho_db, rid, odd_fechamento=1.9, clv=0.02, resultado="ganhou")
    resposta = client.get("/api/clv-serie")
    pontos = resposta.json()["pontos"]
    assert len(pontos) == 1
    assert pontos[0]["clv_medio"] == pytest.approx(0.02)


def test_status_coleta_sem_coletas(cliente):
    client, _ = cliente
    dados = client.get("/api/status-coleta").json()
    assert dados["ultima_coleta"] is None
    assert dados["oddspapi_gasto"] == 0
    assert dados["oddspapi_restante"] == 250


def test_status_coleta_com_coleta_e_uso(cliente):
    client, caminho_db = cliente
    registrar_coleta(caminho_db, "betano", sucesso=True, tipo="manha", n_jogos_capturados=5)
    registrar_uso_oddspapi(caminho_db, "/odds", sucesso=True)
    dados = client.get("/api/status-coleta").json()
    assert dados["ultima_coleta"]["fonte"] == "betano"
    assert dados["oddspapi_gasto"] == 1
    assert dados["oddspapi_restante"] == 249
