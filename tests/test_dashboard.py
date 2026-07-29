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
    dados = resposta.json()
    assert dados["dia"] == "hoje"
    assert dados["jogos"] == []


def test_jogos_do_dia_amanha_vazio(cliente):
    client, _ = cliente
    resposta = client.get("/api/jogos-do-dia?dia=amanha")
    assert resposta.json() == {"dia": "amanha", "jogos": []}


def test_maiores_probabilidades_vazio_sem_jogos_reais(cliente):
    client, _ = cliente
    dados = client.get("/api/maiores-probabilidades").json()
    assert "EV" in dados["aviso"] or "ev" in dados["aviso"].lower()
    assert dados["top"] == []


def test_apostarias_hoje_vazio_tem_mensagem_explicativa(cliente):
    client, _ = cliente
    dados = client.get("/api/apostarias-hoje").json()
    assert dados["apostarias"] == []
    assert "sistema funcionando" in dados["mensagem_vazio"]


def test_apostarias_hoje_so_traz_registros_com_flag_apostaria(cliente):
    client, caminho_db = cliente
    from datetime import date
    hoje = date.today().isoformat()
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.1,
                      data_jogo=hoje, origem="auto_manha", apostaria=True)
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Empate", 0.3, 3.0, -0.05,
                      data_jogo=hoje, origem="auto_manha", apostaria=False)
    dados = client.get("/api/apostarias-hoje").json()
    assert len(dados["apostarias"]) == 1
    assert dados["apostarias"][0]["selecao"] == "Casa"
    assert "sem edge no backtest" in dados["apostarias"][0]["etiqueta"]


def test_apostarias_hoje_mostra_registro_de_jogo_amanha_se_registrado_hoje(cliente):
    """Regressão: a rotina da manhã registra apostarias pros jogos reais
    mais próximos, que podem ser de AMANHÃ (kickoff tarde da noite, já
    virou o dia em UTC) — filtrar por data_jogo fazia a seção aparecer
    vazia mesmo com apostarias recém-registradas."""
    client, caminho_db = cliente
    from datetime import date, timedelta
    amanha = (date.today() + timedelta(days=1)).isoformat()
    inserir_registro(caminho_db, "brasileirao", "A", "B", "1X2", "Fora", 0.3, 4.7, 0.1,
                      data_jogo=amanha, origem="auto_manha", apostaria=True)
    dados = client.get("/api/apostarias-hoje").json()
    assert len(dados["apostarias"]) == 1
    assert dados["apostarias"][0]["data_jogo"] == amanha


def test_status_sistema_responde(cliente):
    client, _ = cliente
    dados = client.get("/api/status-sistema").json()
    assert dados["bot_ok"] is False  # sem heartbeat gravado no banco de teste
    assert "proxima_rotina" in dados


def test_oddspapi_status_sem_uso(cliente):
    client, _ = cliente
    dados = client.get("/api/oddspapi/status").json()
    assert dados == {"gasto": 0, "limite": 250, "restante": 250}


def test_oddspapi_buscar_usa_a_funcao_de_verdade_mas_mockada(cliente, monkeypatch):
    """Nunca deve chamar a rede de verdade num teste — mocka
    oddspapi.buscar_melhores_odds inteira."""
    import dashboard
    client, _ = cliente
    monkeypatch.setattr(
        dashboard.oddspapi, "buscar_melhores_odds",
        lambda caminho_db: {"sucesso": True, "jogos": [{"liga": "brasileirao", "casa": "A", "fora": "B",
                                                          "commence_time": "x", "mercados": []}], "uso_apos": 1},
    )
    resposta = client.post("/api/oddspapi/buscar")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["sucesso"] is True
    assert dados["jogos"][0]["casa"] == "A"


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


def test_registros_preenche_data_a_partir_de_criado_em_quando_nulo(cliente):
    client, caminho_db = cliente
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)  # sem data_jogo
    dados = client.get("/api/registros").json()["registros"]
    assert dados[0]["data_jogo"] is not None  # preenchido com a data de criado_em


def test_registros_filtra_por_liga(cliente):
    client, caminho_db = cliente
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    inserir_registro(caminho_db, "La Liga", "C", "D", "1X2", "Fora", 0.4, 2.5, 0.05)
    dados = client.get("/api/registros?liga=La Liga").json()["registros"]
    assert len(dados) == 1
    assert dados[0]["liga"] == "La Liga"


def test_registros_filtra_por_familia(cliente):
    client, caminho_db = cliente
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05)
    inserir_registro(caminho_db, "Premier League", "A", "B", "Ambas marcam", "Sim", 0.5, 1.9, 0.05)
    dados = client.get("/api/registros?familia=Ambas marcam").json()["registros"]
    assert len(dados) == 1
    assert dados[0]["mercado"] == "Ambas marcam"


def test_registros_secao_ao_vivo_exclui_backtest_por_padrao(cliente):
    client, caminho_db = cliente
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05, origem="bot")
    inserir_registro(caminho_db, "Premier League", "C", "D", "1X2", "Fora", 0.4, 2.5, 0.05, origem="backtest")
    dados = client.get("/api/registros").json()["registros"]
    assert len(dados) == 1
    assert dados[0]["time_casa"] == "A"


def test_registros_secao_backtest_so_traz_backtest(cliente):
    client, caminho_db = cliente
    inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05, origem="bot")
    inserir_registro(caminho_db, "Premier League", "C", "D", "1X2", "Fora", 0.4, 2.5, 0.05, origem="backtest")
    dados = client.get("/api/registros?secao=backtest").json()["registros"]
    assert len(dados) == 1
    assert dados[0]["time_casa"] == "C"


def test_registros_elimina_duplicatas_exatas(cliente):
    client, caminho_db = cliente
    for _ in range(2):
        inserir_registro(caminho_db, "Premier League", "A", "B", "1X2", "Casa", 0.5, 2.0, 0.05,
                          data_jogo="2026-08-01", origem="auto_manha")
    dados = client.get("/api/registros").json()["registros"]
    assert len(dados) == 1  # mesma liga/times/mercado/selecao/data/status -> um só na listagem


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
