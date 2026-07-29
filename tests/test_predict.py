# -*- coding: utf-8 -*-
"""Testes do CLI de previsão (predict.py): alpha_xg padrão por liga, aviso
para ligas sem escanteios/cartões/faltas, e migração do SQLite."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import (
    alpha_xg_padrao, LIGAS_SEM_ESTATISTICAS_EXTRAS, LIGAS_COM_XG,
    prever, gravar_previsao, inicializar_db,
)
from model_goals import ModeloDixonColes


def test_alpha_xg_padrao_premier_league_e_la_liga():
    assert alpha_xg_padrao("Premier League") == 0.5
    assert alpha_xg_padrao("La Liga") == 0.5


def test_alpha_xg_padrao_demais_ligas_e_zero():
    assert alpha_xg_padrao("Championship") == 0.0
    assert alpha_xg_padrao("brasileirao") == 0.0
    assert alpha_xg_padrao("mls") == 0.0


def test_ligas_sem_estatisticas_extras_contem_so_mls():
    """Brasileirão (Série A) saiu desse conjunto -- escanteios/cartões/
    faltas/árbitro agora vêm da súmula da CBF + ge.globo.com (ver
    ingest_cbf.py/ingest_geglobo.py), não mais só de football-data.co.uk."""
    assert LIGAS_SEM_ESTATISTICAS_EXTRAS == {"mls"}


def test_ligas_com_xg_nao_inclui_ligas_sem_estatisticas():
    assert LIGAS_COM_XG.isdisjoint(LIGAS_SEM_ESTATISTICAS_EXTRAS)


def _dados_sinteticos_liga(liga: str, n: int = 500, seed: int = 0) -> pd.DataFrame:
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


def test_prever_liga_sem_estatisticas_gera_aviso_e_pula_mercados_extras(tmp_path):
    df = _dados_sinteticos_liga("mls")
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)

    resultado = prever(
        "mls", "A", "B",
        caminho_dados=caminho_csv, data_corte="2022-01-01", gravar=False,
    )

    assert len(resultado["avisos"]) == 1
    assert "mls" in resultado["avisos"][0]
    mercados_presentes = {m for m, _, _ in resultado["linhas_mercados"]}
    assert not any("Escanteios" in m or "Cartões" in m or "Faltas" in m for m in mercados_presentes)
    assert any(m == "1X2" for m in mercados_presentes)  # mercados de gols continuam presentes


def test_prever_time_com_poucos_jogos_gera_aviso_de_historico_curto(tmp_path):
    """Time recém promovido (poucos jogos na base) precisa de um aviso
    explícito — sem isso, o EV alto vindo de uma estimativa instável parece
    (mas não é) uma vantagem real contra o mercado."""
    df = _dados_sinteticos_liga("mls")
    rng = np.random.default_rng(1)
    linhas_novo_time = []
    datas_recentes = pd.date_range("2021-11-01", periods=10, freq="3D")
    for d in datas_recentes:
        adversario = rng.choice(["A", "B", "C"])
        linhas_novo_time.append({
            "liga": "mls", "Date": d, "HomeTeam": "Novato", "AwayTeam": adversario,
            "FTHG": rng.poisson(1.4), "FTAG": rng.poisson(1.1),
            "FTR": "H", "HC": np.nan, "AC": np.nan, "HF": np.nan, "AF": np.nan,
            "HY": np.nan, "AY": np.nan, "HR": np.nan, "AR": np.nan, "Referee": None,
            "PSCH": np.nan, "PSCD": np.nan, "PSCA": np.nan,
        })
    df = pd.concat([df, pd.DataFrame(linhas_novo_time)], ignore_index=True)
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)

    resultado = prever(
        "mls", "Novato", "A",
        caminho_dados=caminho_csv, data_corte="2022-01-01", gravar=False,
    )
    avisos_historico = [a for a in resultado["avisos"] if "Novato" in a]
    assert len(avisos_historico) == 1
    assert "10 jogo" in avisos_historico[0]
    # o adversário tem histórico de sobra (~130+ jogos) -> nenhum aviso pra ele
    assert not any("'A'" in a for a in resultado["avisos"])


def test_prever_brasileirao_nao_esta_mais_em_ligas_sem_estatisticas():
    """Confirma a mudança de comportamento: Brasileirão Série A saiu do
    conjunto de ligas sem escanteios/cartões/faltas -- não é mais tratado
    como MLS (a única liga que ainda não tem essa fonte)."""
    assert "brasileirao" not in LIGAS_SEM_ESTATISTICAS_EXTRAS


def test_prever_liga_com_estatisticas_nao_gera_aviso(tmp_path):
    df = _dados_sinteticos_liga("Teste")
    df["HC"] = 5.0
    df["AC"] = 4.0
    df["HY"] = 1.5
    df["AY"] = 2.0
    df["HF"] = 10.0
    df["AF"] = 11.0
    caminho_csv = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho_csv, index=False)

    resultado = prever(
        "Teste", "A", "B",
        caminho_dados=caminho_csv, data_corte="2022-01-01", gravar=False,
    )
    assert resultado["avisos"] == []
    mercados_presentes = {m for m, _, _ in resultado["linhas_mercados"]}
    assert any("Escanteios" in m for m in mercados_presentes)


def _modelo_exemplo(alpha_xg=0.5) -> ModeloDixonColes:
    return ModeloDixonColes(
        liga="Premier League", data_corte="2026-01-01", xi=0.0018,
        times=["A", "B"], ataque={"A": 0.1, "B": -0.1}, defesa={"A": 0.0, "B": 0.0},
        home_adv=0.2, rho=-0.05, n_jogos_usados=100, alpha_xg=alpha_xg,
    )


def test_gravar_previsao_migra_banco_sem_coluna_alpha_xg(tmp_path):
    caminho_db = tmp_path / "previsoes_antigo.sqlite"
    # simula um banco criado antes do campo alpha_xg existir
    with sqlite3.connect(caminho_db) as conn:
        conn.execute("""
            CREATE TABLE previsoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                criado_em TEXT NOT NULL, liga TEXT NOT NULL, time_casa TEXT NOT NULL,
                time_fora TEXT NOT NULL, data_corte_modelo TEXT NOT NULL, xi REAL NOT NULL,
                home_adv REAL NOT NULL, rho REAL NOT NULL, n_jogos_modelo INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE previsoes_mercados (
                previsao_id INTEGER NOT NULL REFERENCES previsoes(id),
                mercado TEXT NOT NULL, selecao TEXT NOT NULL, probabilidade REAL NOT NULL
            )
        """)
        conn.commit()

    modelo = _modelo_exemplo(alpha_xg=0.5)
    previsao_id = gravar_previsao(caminho_db, "Premier League", "A", "B", modelo, [("1X2", "Casa", 0.5)])

    with sqlite3.connect(caminho_db) as conn:
        colunas = [r[1] for r in conn.execute("PRAGMA table_info(previsoes)")]
        assert "alpha_xg" in colunas
        row = conn.execute("SELECT alpha_xg FROM previsoes WHERE id=?", (previsao_id,)).fetchone()
        assert row[0] == 0.5


def test_gravar_previsao_registra_alpha_xg_correto(tmp_path):
    caminho_db = tmp_path / "previsoes.sqlite"
    modelo = _modelo_exemplo(alpha_xg=0.0)
    previsao_id = gravar_previsao(caminho_db, "Championship", "A", "B", modelo, [("1X2", "Casa", 0.5)])
    with sqlite3.connect(caminho_db) as conn:
        row = conn.execute("SELECT alpha_xg FROM previsoes WHERE id=?", (previsao_id,)).fetchone()
        assert row[0] == 0.0
