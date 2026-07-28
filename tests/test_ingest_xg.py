# -*- coding: utf-8 -*-
"""Testes da ingestão de xG (ingest_xg.py)."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest_xg import gerar_anos_temporadas, casar_com_partidas


def test_gera_dez_anos():
    anos = gerar_anos_temporadas(10, date(2026, 7, 28))
    assert anos == list(range(2016, 2026))


def test_casamento_exato():
    df_partidas = pd.DataFrame([
        {"liga": "Premier League", "Date": pd.Timestamp("2020-01-01"), "HomeTeam": "Arsenal",
         "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 1, "temporada": "1920"},
    ])
    df_understat = pd.DataFrame([
        {"liga": "Premier League", "Date": pd.Timestamp("2020-01-01"), "HomeTeam": "Arsenal",
         "AwayTeam": "Chelsea", "xG_casa": 1.8, "xG_fora": 0.9, "temporada": "1920"},
    ])
    base = casar_com_partidas(df_understat, df_partidas)
    assert len(base) == 1
    assert base.iloc[0]["xG_casa"] == 1.8


def test_casamento_por_proximidade_quando_data_diverge_um_dia():
    df_partidas = pd.DataFrame([
        {"liga": "La Liga", "Date": pd.Timestamp("2020-03-02"), "HomeTeam": "Barcelona",
         "AwayTeam": "Betis", "FTHG": 3, "FTAG": 0, "temporada": "1920"},
    ])
    df_understat = pd.DataFrame([
        # Understat registrou um dia antes (fuso horário)
        {"liga": "La Liga", "Date": pd.Timestamp("2020-03-01"), "HomeTeam": "Barcelona",
         "AwayTeam": "Betis", "xG_casa": 2.5, "xG_fora": 0.4, "temporada": "1920"},
    ])
    base = casar_com_partidas(df_understat, df_partidas)
    assert len(base) == 1
    assert base.iloc[0]["xG_casa"] == 2.5
    assert base.iloc[0]["Date"] == pd.Timestamp("2020-03-02")  # mantém a data do football-data


def test_nao_casa_times_diferentes_mesmo_com_data_proxima():
    df_partidas = pd.DataFrame([
        {"liga": "La Liga", "Date": pd.Timestamp("2020-03-02"), "HomeTeam": "Barcelona",
         "AwayTeam": "Betis", "FTHG": 3, "FTAG": 0, "temporada": "1920"},
    ])
    df_understat = pd.DataFrame([
        {"liga": "La Liga", "Date": pd.Timestamp("2020-03-01"), "HomeTeam": "Real Madrid",
         "AwayTeam": "Sevilla", "xG_casa": 2.5, "xG_fora": 0.4, "temporada": "1920"},
    ])
    base = casar_com_partidas(df_understat, df_partidas)
    assert len(base) == 1
    assert pd.isna(base.iloc[0]["xG_casa"])
