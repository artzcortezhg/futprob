# -*- coding: utf-8 -*-
"""Popula a tabela `registros` do painel com o resultado do backtest
historico de CLV (Bloco 1.2), como dados de demonstracao do painel antes
do bot/coletor (Blocos 3/4) estarem gerando registros ao vivo. Insercao em
lote (uma conexao so) por performance -- ~2400 linhas."""
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from painel_db import inicializar_db_painel, CAMINHO_DB_PADRAO

RAIZ = Path(__file__).resolve().parent.parent
mapa = {
    "premier_league": "Premier League",
    "la_liga": "La Liga",
    "championship": "Championship",
}

inicializar_db_painel(CAMINHO_DB_PADRAO)
criado_em = datetime.now().isoformat(timespec="seconds")

linhas_para_inserir = []
for arq, liga in mapa.items():
    caminho = RAIZ / "data" / "processed" / f"backtest_clv_{arq}.csv"
    if not caminho.exists():
        continue
    df = pd.read_csv(caminho, parse_dates=["Date"])
    for _, linha in df.iterrows():
        resultado = "ganhou" if linha["ganhou"] else "perdeu"
        linhas_para_inserir.append((
            criado_em, liga, linha["Date"].date().isoformat(), linha["HomeTeam"], linha["AwayTeam"],
            "1X2", linha["selecao"].capitalize(), "backtest", float(linha["prob_modelo"]),
            float(linha["odd_prejogo"]), float(linha["ev"]), float(linha["odd_fechamento"]),
            float(linha["clv"]), "fechado", resultado, "backtest",
        ))

with sqlite3.connect(CAMINHO_DB_PADRAO) as conn:
    conn.executemany(
        """INSERT INTO registros
           (criado_em, liga, data_jogo, time_casa, time_fora, mercado, selecao,
            casa_apostas, prob_modelo, odd_registrada, ev, odd_fechamento, clv, status, resultado, origem)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        linhas_para_inserir,
    )
    conn.commit()

print(f"{len(linhas_para_inserir)} registros historicos inseridos.")
