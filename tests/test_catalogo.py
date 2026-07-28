# -*- coding: utf-8 -*-
"""Testes do catálogo do dia (src/catalogo.py): agrupamento visual do card
completo, ranking de maiores probabilidades e o card em si (com aviso
explícito de "sem modelo" pra escanteios/cartões em ligas sem essas
estatísticas na fonte, ex.: Brasileirão/MLS)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predict
from catalogo import (
    _grupo_exibicao, maiores_probabilidades, card_completo, GRUPOS_CARD,
    _mercados_para_jogo, combinar_modelo_e_odds, _selecao_h2h, _selecao_over_under,
    descobrir_jogos_do_dia_completo,
)


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


def _dados_sinteticos_liga(liga: str, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = ["A", "B", "C", "D", "E", "F"]
    linhas = []
    datas = pd.date_range("2020-01-01", periods=n, freq="3D")
    for d in datas:
        casa, fora = rng.choice(times, size=2, replace=False)
        linhas.append({
            "liga": liga, "Date": d, "HomeTeam": casa, "AwayTeam": fora,
            "FTHG": rng.poisson(1.4), "FTAG": rng.poisson(1.1),
            "FTR": "H", "HC": np.nan, "AC": np.nan, "HF": np.nan, "AF": np.nan,
            "HY": np.nan, "AY": np.nan, "HR": np.nan, "AR": np.nan, "Referee": None,
            "PSCH": np.nan, "PSCD": np.nan, "PSCA": np.nan,
        })
    df = pd.DataFrame(linhas)
    df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    return df


def test_card_completo_liga_sem_estatisticas_avisa_escanteios_e_cartoes(tmp_path, monkeypatch):
    df = _dados_sinteticos_liga("brasileirao")
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)
    monkeypatch.setattr(predict, "CAMINHO_DADOS_PADRAO", caminho_csv)

    caminho_db = tmp_path / "vazio.sqlite"  # sem odds_coletadas -> tem_odds_coletadas=False

    card = card_completo("brasileirao", "A", "B", "2026-01-01", times_por_liga={}, caminho_db=caminho_db)

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


def test_descobrir_jogos_do_dia_completo_inclui_jogo_sem_modelo(tmp_path):
    """Regressão: um jogo real (ex.: Série B) sem os dois times na lista
    fechada de nenhuma liga modelada não pode sumir em silêncio da listagem
    de 'jogos de hoje' — precisa aparecer marcado como 'modelado: False',
    senão o painel dá a entender que não há jogo nenhum naquele dia."""
    import datetime
    caminho = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", "2026-07-28T22:30:00+00:00", "h2h", "1", 2.1),
        ("2026-07-28T09:00:00", "manha", "betano", "Ponte Preta", "Athletic Club MG", "2026-07-28T22:30:00+00:00", "h2h", "1", 2.0),
    ])
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras", "Ponte Preta", "Ceara"]}
    jogos = descobrir_jogos_do_dia_completo(datetime.date(2026, 7, 28), times, caminho)
    assert len(jogos) == 2
    modelados = {j["casa"]: j for j in jogos if j["modelado"]}
    sem_modelo = [j for j in jogos if not j["modelado"]]
    assert "Flamengo RJ" in modelados and modelados["Flamengo RJ"]["liga"] == "brasileirao"
    assert len(sem_modelo) == 1
    assert sem_modelo[0]["casa"] == "Ponte Preta"
    assert sem_modelo[0]["fora"] == "Athletic Club MG"
    assert sem_modelo[0]["liga"] is None
