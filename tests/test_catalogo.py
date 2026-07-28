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
from catalogo import _grupo_exibicao, maiores_probabilidades, card_completo, GRUPOS_CARD


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
