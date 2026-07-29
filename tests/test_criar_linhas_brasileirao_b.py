# -*- coding: utf-8 -*-
"""Testes de scripts/criar_linhas_brasileirao_b.py: cria linhas NOVAS pra
liga 'brasileirao_b' em partidas.csv (essa liga não tinha nenhum histórico
antes) a partir da coleta CBF+ge.globo."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import criar_linhas_brasileirao_b as criar


def _partidas_vazia(tmp_path):
    colunas = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
               "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "Referee",
               "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "liga", "temporada"]
    caminho = tmp_path / "partidas.csv"
    pd.DataFrame(columns=colunas).to_csv(caminho, index=False)
    return caminho


def _coleta_exemplo(tmp_path):
    linhas = [
        {"id_jogo_cbf": "1", "liga": "brasileirao_b", "temporada": 2026, "rodada": 1,
         "data_iso": "2026-03-21", "time_casa_slug": "vila-nova", "time_fora_slug": "crb",
         "gols_casa": 2, "gols_fora": 2, "arbitro": "Fulano de Tal",
         "HY": 3, "AY": 2, "HR": 0, "AR": 1,
         "escanteios_casa": 6, "escanteios_fora": 4, "faltas_casa": 12, "faltas_fora": 15},
        {"id_jogo_cbf": "2", "liga": "brasileirao_b", "temporada": 2026, "rodada": 1,
         "data_iso": "2026-03-22", "time_casa_slug": "gremio-novorizontino", "time_fora_slug": "londrina",
         "gols_casa": 1, "gols_fora": 3, "arbitro": "Ciclano da Silva",
         "HY": 1, "AY": 1, "HR": 0, "AR": 0,
         "escanteios_casa": 3, "escanteios_fora": 8, "faltas_casa": 10, "faltas_fora": 9},
    ]
    caminho = tmp_path / "coleta_b.csv"
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    return caminho


def test_cria_linhas_com_nomes_canonicos_do_roster_atual(tmp_path, monkeypatch):
    monkeypatch.setattr(criar, "CAMINHO_PARTIDAS", _partidas_vazia(tmp_path))
    monkeypatch.setattr(criar, "CAMINHO_COLETA", _coleta_exemplo(tmp_path))

    resultado = criar.aplicar()
    assert resultado["linhas_adicionadas"] == 2

    df = pd.read_csv(criar.CAMINHO_PARTIDAS)
    linha = df[df["HomeTeam"] == "Vila Nova"].iloc[0]
    assert linha["AwayTeam"] == "CRB"
    assert linha["liga"] == "brasileirao_b"
    assert linha["FTHG"] == 2 and linha["FTAG"] == 2 and linha["FTR"] == "D"
    assert linha["HC"] == 6 and linha["AF"] == 15
    assert linha["Referee"] == "Fulano de Tal"
    assert pd.isna(linha["PSH"])  # sem fonte de odds históricas pra Série B


def test_nome_canonico_desambigua_botafogo_sp_do_slug_puro():
    """A CBF usa o slug 'botafogo' puro (sem '-sp') pro Botafogo-SP na
    Série B -- tem que bater com o nome EXATO do roster estático
    (resolucao_times.TIMES_SERIE_B_2026: 'Botafogo-SP'), senão a resolução
    de fixture ao vivo não reconhece o time."""
    assert criar._nome_canonico("botafogo") == "Botafogo-SP"


def test_cria_linhas_deriva_nome_pra_time_fora_do_roster_atual():
    """Time que não está no roster estático 2026 (promovido/rebaixado em
    outra temporada) recebe um nome derivado do slug -- não precisa bater
    com nenhum roster, só ser consistente."""
    assert criar._nome_canonico("gremio-novorizontino") == "Novorizontino"
    assert criar._nome_canonico("time-generico-xyz") == "Time Generico Xyz"


def test_rodar_de_novo_nao_duplica_linhas_ja_existentes(tmp_path, monkeypatch):
    monkeypatch.setattr(criar, "CAMINHO_PARTIDAS", _partidas_vazia(tmp_path))
    monkeypatch.setattr(criar, "CAMINHO_COLETA", _coleta_exemplo(tmp_path))
    criar.aplicar()
    resultado2 = criar.aplicar()
    assert resultado2["linhas_adicionadas"] == 0
    assert resultado2["total_brasileirao_b"] == 2
