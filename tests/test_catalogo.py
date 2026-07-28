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
