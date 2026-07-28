# -*- coding: utf-8 -*-
"""
CLI de previsão: recebe liga e dois times, ajusta o modelo Dixon-Coles
(usando apenas jogos anteriores à data de corte), imprime uma tabela com
todos os mercados de gols e grava a previsão em db/previsoes.sqlite.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from markets import (
    calcular_mercados, para_linhas_tabela,
    mercado_escanteios, mercado_cartoes, mercado_faltas,
    para_linhas_tabela_escanteios, para_linhas_tabela_cartoes, para_linhas_tabela_faltas,
)
from model_goals import XI_PADRAO, ajustar_modelo, matriz_placares
from model_corners import ajustar_modelo_escanteios, matriz_escanteios
from model_cards import ajustar_modelo_cartoes, ajustar_modelo_faltas, matriz_contagem

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS_PADRAO = RAIZ / "data" / "processed" / "partidas.csv"
CAMINHO_DB_PADRAO = RAIZ / "db" / "previsoes.sqlite"

SQL_CRIAR_TABELAS = """
CREATE TABLE IF NOT EXISTS previsoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em TEXT NOT NULL,
    liga TEXT NOT NULL,
    time_casa TEXT NOT NULL,
    time_fora TEXT NOT NULL,
    data_corte_modelo TEXT NOT NULL,
    xi REAL NOT NULL,
    home_adv REAL NOT NULL,
    rho REAL NOT NULL,
    n_jogos_modelo INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS previsoes_mercados (
    previsao_id INTEGER NOT NULL REFERENCES previsoes(id),
    mercado TEXT NOT NULL,
    selecao TEXT NOT NULL,
    probabilidade REAL NOT NULL
);
"""


def inicializar_db(caminho_db: Path) -> None:
    caminho_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(caminho_db) as conn:
        conn.executescript(SQL_CRIAR_TABELAS)


def gravar_previsao(
    caminho_db: Path,
    liga: str,
    time_casa: str,
    time_fora: str,
    modelo,
    linhas_mercados: list[tuple[str, str, float]],
) -> int:
    """Grava a previsão (cabeçalho + todas as linhas de mercado) no SQLite
    no momento em que é gerada. Retorna o id da previsão."""
    inicializar_db(caminho_db)
    criado_em = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(caminho_db) as conn:
        cursor = conn.execute(
            """INSERT INTO previsoes
               (criado_em, liga, time_casa, time_fora, data_corte_modelo, xi, home_adv, rho, n_jogos_modelo)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (criado_em, liga, time_casa, time_fora, modelo.data_corte, modelo.xi,
             modelo.home_adv, modelo.rho, modelo.n_jogos_usados),
        )
        previsao_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO previsoes_mercados (previsao_id, mercado, selecao, probabilidade) VALUES (?, ?, ?, ?)",
            [(previsao_id, mercado, selecao, prob) for mercado, selecao, prob in linhas_mercados],
        )
        conn.commit()
    return previsao_id


def formatar_tabela(linhas_mercados: list[tuple[str, str, float]]) -> str:
    linhas_fmt = [(mercado, selecao, f"{prob * 100:.1f}%") for mercado, selecao, prob in linhas_mercados]
    return tabulate(linhas_fmt, headers=["Mercado", "Seleção", "Probabilidade"], tablefmt="simple")


def prever(
    liga: str,
    time_casa: str,
    time_fora: str,
    caminho_dados: Path = CAMINHO_DADOS_PADRAO,
    data_corte: str | None = None,
    xi: float = XI_PADRAO,
    max_gols: int = 10,
    fonte_gols: str = "gols",
    arbitro: str | None = None,
    caminho_db: Path = CAMINHO_DB_PADRAO,
    gravar: bool = True,
):
    df = pd.read_csv(caminho_dados, parse_dates=["Date"])

    if liga not in set(df["liga"]):
        raise ValueError(f"Liga '{liga}' não encontrada na base. Ligas disponíveis: {sorted(df['liga'].unique())}")

    data_corte = data_corte or datetime.now().date().isoformat()
    modelo = ajustar_modelo(df, liga, data_corte, xi=xi, fonte=fonte_gols)

    matriz = matriz_placares(modelo, time_casa, time_fora, max_gols=max_gols)
    mercados = calcular_mercados(matriz)
    linhas_mercados = para_linhas_tabela(mercados)

    # escanteios/cartões/faltas sempre a partir dos gols observados (não
    # dependem da fonte escolhida para o modelo de gols, que só se aplica a
    # este último). Se --dados já é a base padrão, reaproveita o mesmo df.
    df_padrao = df if fonte_gols == "gols" else pd.read_csv(CAMINHO_DADOS_PADRAO, parse_dates=["Date"])

    modelo_escanteios = ajustar_modelo_escanteios(df_padrao, liga, data_corte, xi=xi)
    matriz_esc = matriz_escanteios(modelo_escanteios, time_casa, time_fora)
    linhas_mercados += para_linhas_tabela_escanteios(mercado_escanteios(matriz_esc))

    modelo_cartoes = ajustar_modelo_cartoes(df_padrao, liga, data_corte, xi=xi)
    matriz_cart = matriz_contagem(modelo_cartoes, time_casa, time_fora, arbitro=arbitro, max_valor=10)
    linhas_mercados += para_linhas_tabela_cartoes(mercado_cartoes(matriz_cart))

    modelo_faltas = ajustar_modelo_faltas(df_padrao, liga, data_corte, xi=xi)
    matriz_falt = matriz_contagem(modelo_faltas, time_casa, time_fora, arbitro=arbitro, max_valor=30)
    linhas_mercados += para_linhas_tabela_faltas(mercado_faltas(matriz_falt))

    previsao_id = None
    if gravar:
        previsao_id = gravar_previsao(caminho_db, liga, time_casa, time_fora, modelo, linhas_mercados)

    return {
        "modelo": modelo,
        "matriz": matriz,
        "mercados": mercados,
        "linhas_mercados": linhas_mercados,
        "previsao_id": previsao_id,
    }


def main():
    parser = argparse.ArgumentParser(description="Previsão de mercados de gols (Dixon-Coles)")
    parser.add_argument("--liga", required=True, help="Ex.: 'Premier League', 'La Liga', 'Championship'")
    parser.add_argument("--casa", required=True, help="Time mandante")
    parser.add_argument("--fora", required=True, help="Time visitante")
    parser.add_argument("--data-corte", default=None, help="YYYY-MM-DD (padrão: hoje). Usa só jogos anteriores a essa data.")
    parser.add_argument("--dados", default=str(CAMINHO_DADOS_PADRAO))
    parser.add_argument("--xi", type=float, default=XI_PADRAO)
    parser.add_argument("--max-gols", type=int, default=10)
    parser.add_argument("--fonte-gols", choices=["gols", "xg"], default="gols",
                         help="Insumo de treino do modelo de gols (Premier League/La Liga apenas para 'xg')")
    parser.add_argument("--arbitro", default=None, help="Nome do árbitro (para os mercados de cartões/faltas); sem árbitro = média da liga")
    parser.add_argument("--db", default=str(CAMINHO_DB_PADRAO))
    parser.add_argument("--no-gravar", action="store_true", help="Não grava a previsão no SQLite (uso em testes)")
    args = parser.parse_args()

    resultado = prever(
        liga=args.liga,
        time_casa=args.casa,
        time_fora=args.fora,
        caminho_dados=Path(args.dados),
        data_corte=args.data_corte,
        xi=args.xi,
        max_gols=args.max_gols,
        fonte_gols=args.fonte_gols,
        arbitro=args.arbitro,
        caminho_db=Path(args.db),
        gravar=not args.no_gravar,
    )

    modelo = resultado["modelo"]
    print(f"\n{args.casa} (casa) x {args.fora} (fora) — {args.liga}")
    print(f"Modelo ajustado com dados até {modelo.data_corte} | {modelo.n_jogos_usados} jogos | "
          f"home_adv={modelo.home_adv:.3f} rho={modelo.rho:.3f}\n")
    print(formatar_tabela(resultado["linhas_mercados"]))
    if resultado["previsao_id"] is not None:
        print(f"\n[gravado em {args.db} | previsao_id={resultado['previsao_id']}]")


if __name__ == "__main__":
    main()
