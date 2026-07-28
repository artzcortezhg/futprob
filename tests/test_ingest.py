# -*- coding: utf-8 -*-
"""Testes de geração de códigos de temporada e extra leagues (ingest.py)."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest import gerar_codigos_temporadas, carregar_e_normalizar_extra, COLUNAS_DESEJADAS


def test_gera_dez_codigos():
    codigos = gerar_codigos_temporadas(10, date(2026, 7, 28))
    assert codigos == ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]


def test_apos_agosto_inclui_temporada_corrente():
    codigos = gerar_codigos_temporadas(3, date(2026, 9, 1))
    assert codigos[-1] == "2627"


def test_antes_de_agosto_nao_inclui_temporada_ainda_nao_iniciada():
    codigos = gerar_codigos_temporadas(3, date(2026, 7, 31))
    assert codigos[-1] == "2526"


def _bruto_extra_exemplo(tmp_path, nome="BRA.csv"):
    linhas = [
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA",
        "Brazil,Serie A,2016,10/05/2016,22:00,Flamengo RJ,Palmeiras,1,2,A,2.5,3.2,2.9",
        "Brazil,Serie A,2020,10/08/2020,20:00,Corinthians,Santos,0,0,D,2.1,3.1,3.6",
        "Brazil,Serie A,2025,10/08/2025,20:00,Corinthians,Santos,1,0,H,,,",  # sem odds (jogo recente)
        "Brazil,Serie A,2025,15/08/2025,,,,,,,,,",  # linha vazia (sem times) -> deve ser descartada
    ]
    caminho = tmp_path / nome
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho


def test_carregar_extra_mapeia_colunas_para_esquema_unificado(tmp_path):
    caminho = _bruto_extra_exemplo(tmp_path)
    df = carregar_e_normalizar_extra(caminho, "BRA", n_temporadas=10)
    assert list(df.columns) == COLUNAS_DESEJADAS + ["liga", "temporada"]
    assert set(df["liga"]) == {"brasileirao"}
    assert df["HomeTeam"].tolist() == ["Flamengo RJ", "Corinthians", "Corinthians"]
    assert df["FTHG"].tolist() == [1.0, 0.0, 1.0]


def test_carregar_extra_deixa_escanteios_cartoes_faltas_arbitro_vazios(tmp_path):
    caminho = _bruto_extra_exemplo(tmp_path)
    df = carregar_e_normalizar_extra(caminho, "BRA", n_temporadas=10)
    for col in ("HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "Referee"):
        assert df[col].isna().all()


def test_carregar_extra_descarta_linhas_sem_times(tmp_path):
    caminho = _bruto_extra_exemplo(tmp_path)  # 4 linhas de dados, a última sem times/data
    df = carregar_e_normalizar_extra(caminho, "BRA", n_temporadas=10)
    assert len(df) == 3


def test_carregar_extra_temporada_e_ano_calendario(tmp_path):
    caminho = _bruto_extra_exemplo(tmp_path)
    df = carregar_e_normalizar_extra(caminho, "BRA", n_temporadas=10)
    assert set(df["temporada"]) == {"2016", "2020", "2025"}


def test_carregar_extra_filtra_pelas_ultimas_n_temporadas(tmp_path):
    caminho = _bruto_extra_exemplo(tmp_path)
    df = carregar_e_normalizar_extra(caminho, "BRA", n_temporadas=2)
    # ano_max=2025, n=2 -> só 2024 e 2025 (o jogo de 2016 e 2020 ficam de fora)
    assert set(df["temporada"]) == {"2025"}


def test_carregar_extra_odds_ausentes_viram_nan(tmp_path):
    caminho = _bruto_extra_exemplo(tmp_path)
    df = carregar_e_normalizar_extra(caminho, "BRA", n_temporadas=10)
    jogo_recente = df[df["Date"] == pd.Timestamp("2025-08-10")].iloc[0]
    assert pd.isna(jogo_recente["PSCH"])
