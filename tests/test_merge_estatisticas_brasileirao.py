# -*- coding: utf-8 -*-
"""Testes de scripts/merge_estatisticas_brasileirao.py: preenche
HC/AC/HF/AF/HY/AY/HR/AR/Referee nas linhas 'brasileirao' já existentes em
partidas.csv, sem nunca sobrescrever Date/HomeTeam/AwayTeam/FTHG/FTAG."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import merge_estatisticas_brasileirao as merge


def _partidas_exemplo(tmp_path):
    df = pd.DataFrame([
        {"Div": "BRA", "Date": "2026-07-26", "HomeTeam": "Bahia", "AwayTeam": "Corinthians",
         "FTHG": 1, "FTAG": 1, "FTR": "D", "HC": None, "AC": None, "HF": None, "AF": None,
         "HY": None, "AY": None, "HR": None, "AR": None, "Referee": None,
         "PSH": None, "PSD": None, "PSA": None, "PSCH": None, "PSCD": None, "PSCA": None,
         "liga": "brasileirao", "temporada": "2026"},
        {"Div": "E0", "Date": "2026-07-26", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
         "FTHG": 2, "FTAG": 0, "FTR": "H", "HC": 5, "AC": 3, "HF": 10, "AF": 12,
         "HY": 1, "AY": 2, "HR": 0, "AR": 0, "Referee": "Michael Oliver",
         "PSH": 1.8, "PSD": 3.5, "PSA": 4.2, "PSCH": 1.7, "PSCD": 3.6, "PSCA": 4.5,
         "liga": "Premier League", "temporada": "2526"},
    ])
    caminho = tmp_path / "partidas.csv"
    df.to_csv(caminho, index=False)
    return caminho


def _coleta_exemplo(tmp_path, **overrides):
    linha = {
        "id_jogo_cbf": "832088", "liga": "brasileirao", "temporada": 2026, "rodada": 20,
        "data_iso": "2026-07-26", "time_casa_slug": "bahia", "time_fora_slug": "corinthians",
        "gols_casa": 1, "gols_fora": 1,
        "time_casa_sumula": "Bahia", "time_fora_sumula": "Corinthians", "arbitro": "Ramon Abatti Abel",
        "HY": 2, "AY": 5, "HR": 0, "AR": 0,
        "escanteios_casa": 7, "escanteios_fora": 2, "faltas_casa": 17, "faltas_fora": 17,
    }
    linha.update(overrides)
    caminho = tmp_path / "coleta.csv"
    pd.DataFrame([linha]).to_csv(caminho, index=False)
    return caminho


def test_mescla_preenche_estatisticas_sem_alterar_placar_ou_times(tmp_path, monkeypatch):
    monkeypatch.setattr(merge, "CAMINHO_PARTIDAS", _partidas_exemplo(tmp_path))
    monkeypatch.setattr(merge, "CAMINHO_COLETA", _coleta_exemplo(tmp_path))

    resultado = merge.mesclar()
    assert resultado["mesclados"] == 1

    df = pd.read_csv(merge.CAMINHO_PARTIDAS)
    linha = df[df["liga"] == "brasileirao"].iloc[0]
    assert linha["HomeTeam"] == "Bahia" and linha["AwayTeam"] == "Corinthians"
    assert linha["FTHG"] == 1 and linha["FTAG"] == 1  # placar intocado
    assert linha["HC"] == 7 and linha["AC"] == 2
    assert linha["HF"] == 17 and linha["AF"] == 17
    assert linha["HY"] == 2 and linha["AY"] == 5
    assert linha["Referee"] == "Ramon Abatti Abel"

    # a linha de outra liga não pode ser tocada
    outra = df[df["liga"] == "Premier League"].iloc[0]
    assert outra["Referee"] == "Michael Oliver"


def test_mescla_nao_aplica_quando_placar_diverge(tmp_path, monkeypatch):
    """Se o placar coletado não bate com o já confiável do
    football-data.co.uk, é sinal de jogo errado casado -- nunca aplica as
    estatísticas nesse caso (prefere faltar a inventar)."""
    monkeypatch.setattr(merge, "CAMINHO_PARTIDAS", _partidas_exemplo(tmp_path))
    monkeypatch.setattr(merge, "CAMINHO_COLETA", _coleta_exemplo(tmp_path, gols_casa=9, gols_fora=9))

    resultado = merge.mesclar()
    assert resultado["placar_diverge"] == 1
    assert resultado["mesclados"] == 0

    df = pd.read_csv(merge.CAMINHO_PARTIDAS)
    linha = df[df["liga"] == "brasileirao"].iloc[0]
    assert pd.isna(linha["HC"])


def test_mescla_tenta_data_mais_um_dia_pra_jogo_tarde_da_noite(tmp_path, monkeypatch):
    """Jogo com bola rolando tarde da noite pode cair no dia seguinte em
    UTC no football-data.co.uk -- confirmado manualmente (Fluminense x
    Bragantino: CBF registrou 17/07, partidas.csv tem 18/07). A busca
    precisa tentar a data exata primeiro e cair pra +1 dia só se não achar."""
    monkeypatch.setattr(merge, "CAMINHO_PARTIDAS", _partidas_exemplo(tmp_path))
    monkeypatch.setattr(merge, "CAMINHO_COLETA", _coleta_exemplo(tmp_path, data_iso="2026-07-25"))
    # a linha de partidas.csv de exemplo já usa 2026-07-26 -- 1 dia à frente
    resultado = merge.mesclar()
    assert resultado["mesclados"] == 1


def test_mescla_time_nao_mapeado_nao_quebra(tmp_path, monkeypatch):
    monkeypatch.setattr(merge, "CAMINHO_PARTIDAS", _partidas_exemplo(tmp_path))
    monkeypatch.setattr(merge, "CAMINHO_COLETA", _coleta_exemplo(tmp_path, time_casa_slug="time-desconhecido-xpto"))

    resultado = merge.mesclar()
    assert resultado["time_nao_mapeado"] == 1
    assert resultado["mesclados"] == 0
