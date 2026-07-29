# -*- coding: utf-8 -*-
"""Testes do catálogo do dia (src/catalogo.py): agrupamento visual do card
completo, ranking de maiores probabilidades e o card em si (com aviso
explícito de "sem modelo" pra escanteios/cartões em ligas sem essas
estatísticas na fonte, ex.: MLS)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predict
from catalogo import (
    _grupo_exibicao, maiores_probabilidades, card_completo, card_sem_modelo, GRUPOS_CARD,
    _mercados_para_jogo, combinar_modelo_e_odds, _selecao_h2h, _selecao_over_under,
    descobrir_jogos_do_dia_completo, resolver_fixture_para_liga, enriquecer_odds_externas_com_modelo,
)
from resolucao_times import LIGA_SERIE_B


def test_grupo_exibicao_agrupa_1x2_e_dupla_chance_juntos():
    assert _grupo_exibicao("1X2") == "1X2 e dupla chance"
    assert _grupo_exibicao("Dupla chance") == "1X2 e dupla chance"


def test_grupo_exibicao_agrupa_cartoes_e_faltas_juntos():
    assert _grupo_exibicao("Cartões Over/Under 3.5") == "Cartões e faltas"
    assert _grupo_exibicao("Faltas Over/Under 20.5") == "Cartões e faltas"


def test_grupo_exibicao_escanteios_e_gols_separados():
    assert _grupo_exibicao("Escanteios Over/Under 9.5") == "Escanteios"
    assert _grupo_exibicao("Over/Under 2.5") == "Over/Under gols"
    assert _grupo_exibicao("Ambas marcam") == "Ambas marcam"


def test_grupo_exibicao_placar_exato_fica_fora_do_card():
    assert _grupo_exibicao("Placar exato") is None
    assert _grupo_exibicao("Handicap europeu (+1)") is None


def test_selecao_h2h_formato_real_betano_abreviado():
    """Regressão: a Betano nomeia as seleções do 1X2 como '1'/'X'/'2' (visto
    ao vivo), não pelo nome do time — o código antigo só reconhecia nome de
    time/'Empate' e por isso NENHUMA odd de 1X2 batia (Odd '—' no painel)."""
    assert _selecao_h2h("1", "Juventude-RS", "Avaí") == "Casa"
    assert _selecao_h2h("X", "Juventude-RS", "Avaí") == "Empate"
    assert _selecao_h2h("2", "Juventude-RS", "Avaí") == "Fora"


def test_selecao_h2h_ainda_aceita_nome_de_time():
    assert _selecao_h2h("Juventude-RS", "Juventude-RS", "Avaí") == "Casa"
    assert _selecao_h2h("Empate", "Juventude-RS", "Avaí") == "Empate"


def test_selecao_over_under_formato_real_betano_em_portugues():
    """Regressão: a Betano usa 'Mais de X'/'Menos de X' (visto ao vivo), não
    'Over'/'Under' — o código antigo não reconhecia isso e nenhuma odd de
    over/under batia."""
    assert _selecao_over_under("Mais de 2.5") == "Over"
    assert _selecao_over_under("Menos de 2.5") == "Under"


def test_selecao_over_under_ainda_aceita_ingles():
    assert _selecao_over_under("Over") == "Over"
    assert _selecao_over_under("Under") == "Under"


def test_mercados_para_jogo_over_under_com_rotulo_em_portugues():
    resultado = _mercados_para_jogo("ou_2.5_goals", {"Mais de 2.5": 2.42, "Menos de 2.5": 1.53})
    pares = {(m, s) for m, s, _ in resultado}
    assert ("Over/Under 2.5", "Over") in pares
    assert ("Over/Under 2.5", "Under") in pares


def test_mercados_para_jogo_escanteios_total():
    resultado = _mercados_para_jogo("ou_9.5_corners", {"Mais de 9.5": 1.7, "Menos de 9.5": 2.07})
    pares = {(m, s) for m, s, _ in resultado}
    assert ("Escanteios Over/Under 9.5", "Over") in pares
    assert ("Escanteios Over/Under 9.5", "Under") in pares


def test_mercados_para_jogo_escanteios_por_time_nao_colide_com_total():
    """Regressão: escanteios por time (ou_X_corners_team1/team2) caíam no
    MESMO nome de mercado que o total ('Escanteios Over/Under X'), que não
    bate com as chaves que o modelo calcula pra por-time ('Escanteios time
    da casa/visitante Over/Under X') — as odds por time nunca casavam."""
    casa = _mercados_para_jogo("ou_4.5_corners_team1", {"Mais de 4.5": 1.9, "Menos de 4.5": 1.85})
    fora = _mercados_para_jogo("ou_4.5_corners_team2", {"Mais de 4.5": 2.0, "Menos de 4.5": 1.75})
    assert ("Escanteios time da casa Over/Under 4.5", "Over") in {(m, s) for m, s, _ in casa}
    assert ("Escanteios time visitante Over/Under 4.5", "Over") in {(m, s) for m, s, _ in fora}
    # não pode ter virado o nome genérico do total
    assert not any(m == "Escanteios Over/Under 4.5" for m, _, _ in casa + fora)


def test_mercados_para_jogo_cartoes_por_time_sem_modelo_e_descartado():
    """markets.py só tem cartões TOTAL (sem por-time) — se algum dia vier
    'ou_X_cards_team1', não pode virar um 'Cartões Over/Under X' genérico
    (colidiria com o total de verdade)."""
    resultado = _mercados_para_jogo("ou_3.5_cards_team1", {"Mais de 3.5": 1.95, "Menos de 3.5": 1.80})
    assert resultado == []


def test_combinar_modelo_e_odds_h2h_com_rotulo_abreviado_da_betano():
    probs = {"1X2": {"Casa": 0.5, "Empate": 0.25, "Fora": 0.25}}
    odds = {"casa_coletado": "Juventude-RS", "fora_coletado": "Avaí",
            "mercados": {"h2h": {"1": 1.78, "X": 3.4, "2": 6.0}}}
    candidatos = combinar_modelo_e_odds(probs, odds, "Juventude", "Avai")
    selecoes = {c["selecao"]: c["odd"] for c in candidatos}
    assert selecoes == {"Casa": 1.78, "Empate": 3.4, "Fora": 6.0}


def test_maiores_probabilidades_ordena_desc_e_respeita_top_n():
    cards = [
        {"liga": "brasileirao", "time_casa": "A", "time_fora": "B",
         "grupos": {"1X2 e dupla chance": [
             {"mercado": "1X2", "selecao": "Casa", "prob_modelo": 0.4, "odd": 2.0, "ev": 0.1},
             {"mercado": "1X2", "selecao": "Empate", "prob_modelo": 0.3, "odd": 3.0, "ev": None},
         ]}},
        {"liga": "brasileirao", "time_casa": "C", "time_fora": "D",
         "grupos": {"Ambas marcam": [
             {"mercado": "Ambas marcam", "selecao": "Sim", "prob_modelo": 0.85, "odd": None, "ev": None},
         ]}},
    ]
    top = maiores_probabilidades(cards, top_n=2)
    assert len(top) == 2
    assert top[0]["selecao"] == "Sim"  # 0.85 é a maior prob
    assert top[0]["prob_modelo"] == 0.85
    assert top[1]["prob_modelo"] == 0.4


def _dados_sinteticos_liga(liga: str, n: int = 400, seed: int = 0, com_estatisticas_extras: bool = False) -> pd.DataFrame:
    """com_estatisticas_extras=True preenche HC/AC/HF/AF/HY/AY/HR/AR/Referee
    com valores plausíveis -- necessário pra testar uma liga que NÃO está em
    LIGAS_SEM_ESTATISTICAS_EXTRAS (ex.: brasileirao, desde que passou a ter
    escanteios/cartões/faltas via súmula da CBF + ge.globo.com), senão
    ajustar_modelo_escanteios/cartoes/faltas quebra com dataframe vazio."""
    rng = np.random.default_rng(seed)
    times = ["A", "B", "C", "D", "E", "F"]
    linhas = []
    datas = pd.date_range("2020-01-01", periods=n, freq="3D")
    for d in datas:
        casa, fora = rng.choice(times, size=2, replace=False)
        linhas.append({
            "liga": liga, "Date": d, "HomeTeam": casa, "AwayTeam": fora,
            "FTHG": rng.poisson(1.4), "FTAG": rng.poisson(1.1),
            "FTR": "H",
            "HC": rng.poisson(5.5) if com_estatisticas_extras else np.nan,
            "AC": rng.poisson(4.2) if com_estatisticas_extras else np.nan,
            "HF": rng.poisson(12.0) if com_estatisticas_extras else np.nan,
            "AF": rng.poisson(13.0) if com_estatisticas_extras else np.nan,
            "HY": rng.poisson(1.8) if com_estatisticas_extras else np.nan,
            "AY": rng.poisson(2.0) if com_estatisticas_extras else np.nan,
            "HR": rng.poisson(0.1) if com_estatisticas_extras else np.nan,
            "AR": rng.poisson(0.1) if com_estatisticas_extras else np.nan,
            "Referee": "Arbitro Teste" if com_estatisticas_extras else None,
            "PSCH": np.nan, "PSCD": np.nan, "PSCA": np.nan,
        })
    df = pd.DataFrame(linhas)
    df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    return df


def test_card_completo_liga_sem_estatisticas_avisa_escanteios_e_cartoes(tmp_path, monkeypatch):
    # mls: única liga que ainda não tem escanteios/cartões/faltas na fonte
    # (Brasileirão Série A passou a ter, via súmula da CBF + ge.globo.com)
    df = _dados_sinteticos_liga("mls")
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)
    monkeypatch.setattr(predict, "CAMINHO_DADOS_PADRAO", caminho_csv)

    caminho_db = tmp_path / "vazio.sqlite"  # sem odds_coletadas -> tem_odds_coletadas=False

    card = card_completo("mls", "A", "B", "2026-01-01", times_por_liga={}, caminho_db=caminho_db)

    assert card["tem_odds_coletadas"] is False
    assert set(card["avisos_grupo"].keys()) == {"Escanteios", "Cartões e faltas"}
    assert "sem modelo" in card["avisos_grupo"]["Escanteios"]
    assert set(card["grupos"].keys()) == set(GRUPOS_CARD)
    assert card["grupos"]["Escanteios"] == []  # nenhuma linha, mas o grupo existe (nunca some em silêncio)
    assert len(card["grupos"]["1X2 e dupla chance"]) > 0
    assert all(item["odd"] is None and item["ev"] is None for linhas in card["grupos"].values() for item in linhas)


def _criar_odds_coletadas(caminho, linhas):
    import sqlite3
    with sqlite3.connect(caminho) as conn:
        conn.execute("""CREATE TABLE odds_coletadas (
            id INTEGER PRIMARY KEY, coletado_em TEXT, tipo_foto TEXT, casa_apostas TEXT,
            time_casa_coletado TEXT, time_fora_coletado TEXT, commence_time TEXT,
            mercado TEXT, selecao TEXT, odd REAL
        )""")
        conn.executemany(
            "INSERT INTO odds_coletadas (coletado_em, tipo_foto, casa_apostas, time_casa_coletado, "
            "time_fora_coletado, commence_time, mercado, selecao, odd) VALUES (?,?,?,?,?,?,?,?,?)",
            linhas,
        )
        conn.commit()


def test_descobrir_jogos_do_dia_completo_inclui_serie_b_e_esconde_fora_de_escopo(tmp_path):
    """Jogo real de liga DE INTERESSE (incluindo Série B, que passou a ter
    modelo próprio via súmula da CBF + ge.globo.com) precisa aparecer
    marcado 'modelado: True' (nunca sumir em silêncio — senão o painel dá a
    entender que não há jogo nenhum). Jogo de campeonato FORA de escopo não
    entra na lista nem como placeholder — só polui, e o pedido foi focar
    nas ligas citadas."""
    import datetime
    caminho = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", "2026-07-28T22:30:00+00:00", "h2h", "1", 2.1),
        ("2026-07-28T09:00:00", "manha", "betano", "Ponte Preta", "Athletic Club MG", "2026-07-28T22:30:00+00:00", "h2h", "1", 2.0),
        ("2026-07-28T09:00:00", "manha", "betano", "Time Aleatorio XPTO", "Outro Time XPTO", "2026-07-28T22:30:00+00:00", "h2h", "1", 2.0),
    ])
    times = {
        "brasileirao": ["Flamengo RJ", "Palmeiras", "Ponte Preta", "Ceara"],
        "brasileirao_b": ["Ponte Preta", "Athletic Club"],
    }
    jogos = descobrir_jogos_do_dia_completo(datetime.date(2026, 7, 28), times, caminho)
    assert len(jogos) == 2  # o "Time Aleatorio XPTO" fora de escopo não entra
    assert all(j["modelado"] for j in jogos)
    por_casa = {j["casa"]: j for j in jogos}
    assert por_casa["Flamengo RJ"]["liga"] == "brasileirao"
    assert por_casa["Ponte Preta"]["liga"] == "brasileirao_b"
    assert por_casa["Ponte Preta"]["fora"] == "Athletic Club"


def test_resolver_fixture_para_liga_desambigua_fortaleza_botafogo_sp():
    """Fortaleza jogou a Série A no histórico (roster de treino) E está na
    Série B 2026; Botafogo-SP só existe na Série B. O confronto real entre
    os dois só pode ser a Série B — é isso que a interseção de ligas
    decide, sem precisar de nenhuma regra especial."""
    times = {
        "brasileirao": ["Fortaleza", "Ceara"],
        LIGA_SERIE_B: ["Fortaleza", "Botafogo-SP"],
    }
    resolvido = resolver_fixture_para_liga("Fortaleza", "Botafogo-SP", times)
    assert resolvido == (LIGA_SERIE_B, "Fortaleza", "Botafogo-SP")


def test_card_sem_modelo_mostra_odd_sem_prob_nem_ev(tmp_path):
    caminho_db = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho_db, [
        ("2026-07-28T09:00:00", "manha", "betano", "Fortaleza", "Botafogo-SP", "2026-07-29T00:35:00+00:00", "h2h", "1", 1.32),
        ("2026-07-28T09:00:00", "manha", "betano", "Fortaleza", "Botafogo-SP", "2026-07-29T00:35:00+00:00", "h2h", "2", 3.2),
    ])
    times = {LIGA_SERIE_B: ["Fortaleza", "Botafogo-SP"]}
    card = card_sem_modelo(LIGA_SERIE_B, "Fortaleza", "Botafogo-SP", "2026-07-29", times, caminho_db)
    assert card["modelado"] is False
    assert card["tem_odds_coletadas"] is True
    mercados = {(item["mercado"], item["selecao"]): item["odd"] for item in card["linhas"]}
    assert mercados[("1X2", "Casa")] == 1.32
    assert mercados[("1X2", "Fora")] == 3.2
    assert "prob_modelo" not in card["linhas"][0]
    assert "ev" not in card["linhas"][0]


def test_enriquecer_odds_externas_serie_b_calcula_prob_e_ev(tmp_path, monkeypatch):
    """Série B passou a ter modelo próprio (súmula da CBF + ge.globo.com,
    ver ingest_cbf.py/ingest_geglobo.py) -- o cruzamento com odds externas
    (OddsPapi) tem que calcular prob/ev de verdade pra ela, igual a
    qualquer outra liga modelada, nunca mais deixar None por padrão."""
    df = _dados_sinteticos_liga(LIGA_SERIE_B, com_estatisticas_extras=True)
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)
    monkeypatch.setattr(predict, "CAMINHO_DADOS_PADRAO", caminho_csv)

    times = {LIGA_SERIE_B: ["A", "B"]}
    jogos = [{
        "liga": LIGA_SERIE_B, "casa": "A", "fora": "B", "commence_time": "x",
        "mercados": [{"mercado": "1X2", "selecao": "Casa", "odd": 1.87}],
    }]
    resultado = enriquecer_odds_externas_com_modelo(jogos, times)
    assert resultado[0]["tem_modelo"] is True
    assert resultado[0]["mercados"][0]["prob_modelo"] is not None
    assert resultado[0]["mercados"][0]["ev"] is not None


def test_enriquecer_odds_externas_time_nao_resolvido_fica_sem_modelo():
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras"]}
    jogos = [{
        "liga": "brasileirao", "casa": "Time Que Nao Existe XPTO", "fora": "Palmeiras", "commence_time": "x",
        "mercados": [{"mercado": "1X2", "selecao": "Casa", "odd": 2.0}],
    }]
    resultado = enriquecer_odds_externas_com_modelo(jogos, times)
    assert resultado[0]["tem_modelo"] is False


def test_enriquecer_odds_externas_liga_modelada_calcula_prob_e_ev(tmp_path, monkeypatch):
    df = _dados_sinteticos_liga("brasileirao", com_estatisticas_extras=True)
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)
    monkeypatch.setattr(predict, "CAMINHO_DADOS_PADRAO", caminho_csv)

    times = {"brasileirao": ["A", "B"]}
    jogos = [{
        "liga": "brasileirao", "casa": "A", "fora": "B", "commence_time": "x",
        "mercados": [{"mercado": "1X2", "selecao": "Casa", "odd": 2.0},
                     {"mercado": "Mercado Que Nao Existe", "selecao": "X", "odd": 1.5}],
    }]
    resultado = enriquecer_odds_externas_com_modelo(jogos, times)
    assert resultado[0]["tem_modelo"] is True
    m_1x2 = resultado[0]["mercados"][0]
    assert m_1x2["prob_modelo"] is not None
    assert m_1x2["ev"] == pytest.approx(m_1x2["prob_modelo"] * 2.0 - 1.0)
    m_desconhecido = resultado[0]["mercados"][1]
    assert m_desconhecido["prob_modelo"] is None  # mercado que o modelo não calcula
    assert m_desconhecido["suspeito"] is False  # sem ev, nunca marca suspeito


def test_enriquecer_odds_externas_propaga_aviso_de_historico_curto(tmp_path, monkeypatch):
    """O aviso de 'poucos jogos no histórico' (prever(), predict.py) não pode
    se perder ao passar pelo enriquecimento de odds externas — é o que
    explica pro usuário POR QUE um EV alto pode ser suspeito em vez de
    valor real, em vez de só marcar 'suspeito' sem dizer o motivo."""
    df = _dados_sinteticos_liga("brasileirao", com_estatisticas_extras=True)
    rng = np.random.default_rng(1)
    linhas_novo_time = []
    datas_recentes = pd.date_range("2021-11-01", periods=10, freq="3D")
    for d in datas_recentes:
        linhas_novo_time.append({
            "liga": "brasileirao", "Date": d, "HomeTeam": "Novato", "AwayTeam": "A",
            "FTHG": rng.poisson(1.4), "FTAG": rng.poisson(1.1),
            "FTR": "H", "HC": rng.poisson(5.5), "AC": rng.poisson(4.2),
            "HF": rng.poisson(12.0), "AF": rng.poisson(13.0),
            "HY": rng.poisson(1.8), "AY": rng.poisson(2.0),
            "HR": rng.poisson(0.1), "AR": rng.poisson(0.1), "Referee": "Arbitro Teste",
            "PSCH": np.nan, "PSCD": np.nan, "PSCA": np.nan,
        })
    df = pd.concat([df, pd.DataFrame(linhas_novo_time)], ignore_index=True)
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)
    monkeypatch.setattr(predict, "CAMINHO_DADOS_PADRAO", caminho_csv)

    times = {"brasileirao": ["Novato", "A"]}
    jogos = [{
        "liga": "brasileirao", "casa": "Novato", "fora": "A", "commence_time": "x",
        "mercados": [{"mercado": "1X2", "selecao": "Casa", "odd": 2.0}],
    }]
    resultado = enriquecer_odds_externas_com_modelo(jogos, times)
    avisos = resultado[0]["avisos_modelo"]
    assert any("Novato" in a and "10 jogo" in a for a in avisos)


def test_enriquecer_odds_externas_monta_aposta_e_bilhete(tmp_path, monkeypatch):
    df = _dados_sinteticos_liga("brasileirao", com_estatisticas_extras=True)
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)
    monkeypatch.setattr(predict, "CAMINHO_DADOS_PADRAO", caminho_csv)

    times = {"brasileirao": ["A", "B"]}
    jogos = [{
        "liga": "brasileirao", "casa": "A", "fora": "B", "commence_time": "x",
        "mercados": [
            {"mercado": "1X2", "selecao": "Casa", "odd": 2.0},
            {"mercado": "1X2", "selecao": "Empate", "odd": 3.2},
            {"mercado": "Over/Under 1.5", "selecao": "Over", "odd": 1.6},
            {"mercado": "Over/Under 1.5", "selecao": "Under", "odd": 2.3},
        ],
    }]
    resultado = enriquecer_odds_externas_com_modelo(jogos, times)
    jogo = resultado[0]

    # bilhete: 2 pernas de famílias DIFERENTES (nunca duas de 1X2 juntas)
    bilhete = jogo["melhor_bilhete"]
    assert bilhete is not None
    assert len(bilhete["pernas"]) == 2
    grupos_das_pernas = {_grupo_exibicao(p["mercado"]) for p in bilhete["pernas"]}
    assert len(grupos_das_pernas) == 2  # duas famílias diferentes, nunca a mesma

    odd_combinada_esperada = bilhete["pernas"][0]["odd"] * bilhete["pernas"][1]["odd"]
    assert bilhete["odd_combinada"] == pytest.approx(odd_combinada_esperada)
    assert bilhete["ev_combinado"] == pytest.approx(bilhete["prob_conjunta"] * bilhete["odd_combinada"] - 1.0)

    # a probabilidade conjunta calculada na matriz NUNCA pode ser o produto
    # ingênuo das marginais (isso ignoraria a correlação real gols/resultado)
    prob_marginal_perna1 = next(m["prob_modelo"] for m in jogo["mercados"]
                                 if m["mercado"] == bilhete["pernas"][0]["mercado"] and m["selecao"] == bilhete["pernas"][0]["selecao"])
    prob_marginal_perna2 = next(m["prob_modelo"] for m in jogo["mercados"]
                                 if m["mercado"] == bilhete["pernas"][1]["mercado"] and m["selecao"] == bilhete["pernas"][1]["selecao"])
    produto_ingenuo = prob_marginal_perna1 * prob_marginal_perna2
    assert abs(bilhete["prob_conjunta"] - produto_ingenuo) > 1e-9


def test_enriquecer_odds_externas_marca_ev_suspeito_acima_de_15_por_cento():
    """MESMA regra do resto do sistema (guardrails.py): EV > 15% é mais
    provável ser instabilidade do modelo do que valor real — não pode
    passar sem aviso só porque veio de uma fonte de odds diferente."""
    from catalogo import LIMIAR_EV_SUSPEITO_PADRAO
    times = {LIGA_SERIE_B: ["Fortaleza", "Botafogo-SP"]}  # sem modelo -> ev fica None de qualquer forma
    # usa Série B só pra testar sem depender de fit real; testa a regra
    # suspeito=ev>limiar diretamente via a lógica de enriquecimento com um
    # jogo modelado sintético
    import predict as predict_mod

    def _prever_falso(liga, casa, fora, gravar=False):
        return {"linhas_mercados": [("1X2", "Casa", 0.9)], "matriz": None, "avisos": []}  # prob bem alta -> ev bem alto com odd 3.0

    import catalogo as mod
    original = mod.prever
    mod.prever = _prever_falso
    try:
        jogos = [{
            "liga": "brasileirao", "casa": "A", "fora": "B", "commence_time": "x",
            "mercados": [{"mercado": "1X2", "selecao": "Casa", "odd": 3.0}],
        }]
        resultado = mod.enriquecer_odds_externas_com_modelo(jogos, {"brasileirao": ["A", "B"]})
    finally:
        mod.prever = original
    m = resultado[0]["mercados"][0]
    assert m["ev"] > LIMIAR_EV_SUSPEITO_PADRAO
    assert m["suspeito"] is True
