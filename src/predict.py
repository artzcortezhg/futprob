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
from ingest import LIGAS_EXTRA

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS_PADRAO = RAIZ / "data" / "processed" / "partidas.csv"
CAMINHO_DADOS_XG_PADRAO = RAIZ / "data" / "processed" / "partidas_xg.csv"
CAMINHO_DB_PADRAO = RAIZ / "db" / "previsoes.sqlite"

# ligas cuja fonte não tem escanteios/cartões/faltas/árbitro (extra leagues
# do football-data.co.uk, ver src/ingest.py) — nelas só saem mercados de gols
LIGAS_SEM_ESTATISTICAS_EXTRAS = set(LIGAS_EXTRA.values())

# ligas com xG disponível (src/ingest_xg.py): recebem alpha_xg=0.5 por padrão
LIGAS_COM_XG = {"Premier League", "La Liga"}
ALPHA_XG_PADRAO_COM_XG = 0.5

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
    n_jogos_modelo INTEGER NOT NULL,
    alpha_xg REAL NOT NULL DEFAULT 0.0
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
        # migração: bancos criados antes do campo alpha_xg existir
        colunas = {linha[1] for linha in conn.execute("PRAGMA table_info(previsoes)")}
        if "alpha_xg" not in colunas:
            conn.execute("ALTER TABLE previsoes ADD COLUMN alpha_xg REAL NOT NULL DEFAULT 0.0")
        conn.commit()


def alpha_xg_padrao(liga: str) -> float:
    """alpha_xg=0.5 por padrão para as ligas com xG disponível na fonte
    (Premier League, La Liga); 0.0 (gols puros) para as demais."""
    return ALPHA_XG_PADRAO_COM_XG if liga in LIGAS_COM_XG else 0.0


def gravar_previsao(
    caminho_db: Path,
    liga: str,
    time_casa: str,
    time_fora: str,
    modelo,
    linhas_mercados: list[tuple[str, str, float]],
) -> int:
    """Grava a previsão (cabeçalho + todas as linhas de mercado) no SQLite
    no momento em que é gerada, incluindo o alpha_xg usado no modelo de
    gols. Retorna o id da previsão."""
    inicializar_db(caminho_db)
    criado_em = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(caminho_db) as conn:
        cursor = conn.execute(
            """INSERT INTO previsoes
               (criado_em, liga, time_casa, time_fora, data_corte_modelo, xi, home_adv, rho, n_jogos_modelo, alpha_xg)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (criado_em, liga, time_casa, time_fora, modelo.data_corte, modelo.xi,
             modelo.home_adv, modelo.rho, modelo.n_jogos_usados, modelo.alpha_xg),
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
    caminho_dados: Path | None = None,
    data_corte: str | None = None,
    xi: float = XI_PADRAO,
    max_gols: int = 10,
    alpha_xg: float | None = None,
    arbitro: str | None = None,
    caminho_db: Path = CAMINHO_DB_PADRAO,
    gravar: bool = True,
):
    """alpha_xg=None (padrão) usa o valor recomendado por liga: 0.5 para
    Premier League/La Liga (única fonte com xG), 0.0 (gols puros) para as
    demais. Para ligas sem escanteios/cartões/faltas/árbitro na fonte
    (brasileirao, mls), esses mercados são pulados com um aviso em vez de
    erro — ver LIGAS_SEM_ESTATISTICAS_EXTRAS."""
    if alpha_xg is None:
        alpha_xg = alpha_xg_padrao(liga)

    caminho_dados = caminho_dados or (CAMINHO_DADOS_XG_PADRAO if alpha_xg > 0 else CAMINHO_DADOS_PADRAO)
    df = pd.read_csv(caminho_dados, parse_dates=["Date"])

    if liga not in set(df["liga"]):
        raise ValueError(f"Liga '{liga}' não encontrada na base. Ligas disponíveis: {sorted(df['liga'].unique())}")

    data_corte = data_corte or datetime.now().date().isoformat()
    modelo = ajustar_modelo(df, liga, data_corte, xi=xi, alpha_xg=alpha_xg)

    matriz = matriz_placares(modelo, time_casa, time_fora, max_gols=max_gols)
    mercados = calcular_mercados(matriz)
    linhas_mercados = para_linhas_tabela(mercados)

    avisos = []
    if liga in LIGAS_SEM_ESTATISTICAS_EXTRAS:
        avisos.append(
            f"Liga '{liga}': a fonte não tem escanteios/cartões/faltas/árbitro — "
            "só os mercados de gols estão disponíveis para esta liga."
        )
    else:
        # escanteios/cartões/faltas sempre a partir dos gols observados (não
        # dependem de alpha_xg, que só se aplica ao modelo de gols). Se
        # --dados já é a base padrão, reaproveita o mesmo df.
        df_padrao = df if alpha_xg == 0.0 else pd.read_csv(CAMINHO_DADOS_PADRAO, parse_dates=["Date"])

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
        "avisos": avisos,
    }


def main():
    parser = argparse.ArgumentParser(description="Previsão de mercados de gols (Dixon-Coles)")
    parser.add_argument("--liga", required=True, help="Ex.: 'Premier League', 'La Liga', 'Championship'")
    parser.add_argument("--casa", required=True, help="Time mandante")
    parser.add_argument("--fora", required=True, help="Time visitante")
    parser.add_argument("--data-corte", default=None, help="YYYY-MM-DD (padrão: hoje). Usa só jogos anteriores a essa data.")
    parser.add_argument("--dados", default=None, help="Padrão: escolhido automaticamente a partir do alpha_xg resolvido")
    parser.add_argument("--xi", type=float, default=XI_PADRAO)
    parser.add_argument("--max-gols", type=int, default=10)
    parser.add_argument("--alpha-xg", type=float, default=None,
                         help="Peso do xG no alvo de treino do modelo de gols, 0-1. Padrão: 0.5 para "
                              "Premier League/La Liga (única fonte com xG), 0.0 para as demais ligas.")
    parser.add_argument("--arbitro", default=None, help="Nome do árbitro (para os mercados de cartões/faltas); sem árbitro = média da liga")
    parser.add_argument("--db", default=str(CAMINHO_DB_PADRAO))
    parser.add_argument("--no-gravar", action="store_true", help="Não grava a previsão no SQLite (uso em testes)")
    args = parser.parse_args()

    resultado = prever(
        liga=args.liga,
        time_casa=args.casa,
        time_fora=args.fora,
        caminho_dados=Path(args.dados) if args.dados else None,
        data_corte=args.data_corte,
        xi=args.xi,
        max_gols=args.max_gols,
        alpha_xg=args.alpha_xg,
        arbitro=args.arbitro,
        caminho_db=Path(args.db),
        gravar=not args.no_gravar,
    )

    modelo = resultado["modelo"]
    print(f"\n{args.casa} (casa) x {args.fora} (fora) — {args.liga}")
    print(f"Modelo ajustado com dados até {modelo.data_corte} | {modelo.n_jogos_usados} jogos | "
          f"home_adv={modelo.home_adv:.3f} rho={modelo.rho:.3f} alpha_xg={modelo.alpha_xg:.2f}\n")
    for aviso in resultado["avisos"]:
        print(f"[aviso] {aviso}")
    print(formatar_tabela(resultado["linhas_mercados"]))
    if resultado["previsao_id"] is not None:
        print(f"\n[gravado em {args.db} | previsao_id={resultado['previsao_id']}]")


if __name__ == "__main__":
    main()
