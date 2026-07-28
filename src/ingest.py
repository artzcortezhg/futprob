# -*- coding: utf-8 -*-
"""
Ingestão de dados históricos do football-data.co.uk para as ligas
E0 (Premier League), SP1 (La Liga), E1 (Championship) e as "extra leagues"
Brasileirão (BRA) e MLS (USA).

Baixa os CSVs brutos de cada temporada para data/raw/, normaliza datas,
nomes de times e colunas, e unifica tudo em um único arquivo em
data/processed/partidas.csv com uma coluna 'liga'.

As extra leagues vêm num formato diferente das ligas principais: um único
CSV com TODO o histórico (colunas Country, League, Season, Date, Time, Home,
Away, HG, AG, Res + odds), em vez de um arquivo por temporada. Além disso,
não têm escanteios, cartões, faltas nem árbitro na fonte — essas colunas
ficam NaN, e o restante do sistema (model_corners, model_cards, predict)
já lida com isso sem quebrar (predict.py avisa em vez de tentar ajustar
esses modelos para essas ligas).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw"
PASTA_PROCESSED = RAIZ / "data" / "processed"
ARQUIVO_SAIDA = PASTA_PROCESSED / "partidas.csv"

BASE_URL = "https://www.football-data.co.uk/mmz4281"
BASE_URL_EXTRA = "https://www.football-data.co.uk/new"

# Códigos de liga usados pelo football-data.co.uk -> nome amigável
LIGAS = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "E1": "Championship",
}

# "extra leagues": código do arquivo -> nome amigável (liga='brasileirao'/'mls').
# Formato de arquivo diferente das ligas principais (ver carregar_e_normalizar_extra).
LIGAS_EXTRA = {
    "BRA": "brasileirao",
    "USA": "mls",
}

CABECALHOS = {"User-Agent": "Mozilla/5.0 (compatible; futprob/1.0)"}

# Colunas que queremos manter na base final (quando existirem no CSV bruto)
COLUNAS_DESEJADAS = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HC", "AC",
    "HF", "AF",
    "HY", "AY", "HR", "AR",
    "Referee",
    "PSCH", "PSCD", "PSCA",
]

# Aliases conhecidos de nomes de times (para eventuais inconsistências
# de grafia entre temporadas). Mantido pequeno e extensível.
ALIASES_TIMES: dict[str, str] = {}


def gerar_codigos_temporadas(n: int = 10, hoje: date | None = None) -> list[str]:
    """Gera os códigos das últimas n temporadas no formato usado pelo site
    (ex.: '2324' para a temporada 2023/2024).

    Considera que a temporada nova começa em agosto: se hoje é antes de
    agosto, a temporada "corrente" ainda é a que começou no ano anterior.
    """
    hoje = hoje or date.today()
    ano_inicio_atual = hoje.year if hoje.month >= 8 else hoje.year - 1
    anos = range(ano_inicio_atual - n + 1, ano_inicio_atual + 1)
    return [f"{str(a)[2:].zfill(2)}{str(a + 1)[2:].zfill(2)}" for a in anos]


def baixar_csv(codigo_temporada: str, codigo_liga: str, forcar: bool = False) -> Path | None:
    """Baixa (ou reutiliza cache local) o CSV de uma temporada/liga.

    Retorna o caminho do arquivo local, ou None se o download falhar
    (ex.: temporada ainda não iniciada / código inexistente).
    """
    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    destino = PASTA_RAW / f"{codigo_liga}_{codigo_temporada}.csv"

    if destino.exists() and not forcar:
        return destino

    url = f"{BASE_URL}/{codigo_temporada}/{codigo_liga}.csv"
    try:
        resposta = requests.get(url, headers=CABECALHOS, timeout=30)
    except requests.RequestException as exc:
        print(f"  [aviso] falha de rede em {url}: {exc}", file=sys.stderr)
        return None

    if resposta.status_code != 200 or not resposta.content:
        print(f"  [aviso] indisponível: {url} (HTTP {resposta.status_code})", file=sys.stderr)
        return None

    destino.write_bytes(resposta.content)
    return destino


def baixar_csv_extra(codigo_arquivo: str, forcar: bool = True) -> Path | None:
    """Baixa o CSV de uma 'extra league' (um único arquivo com TODO o
    histórico, atualizado continuamente pela fonte — por isso o padrão é
    sempre buscar uma versão fresca, diferente do cache por temporada das
    ligas principais)."""
    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    destino = PASTA_RAW / f"{codigo_arquivo}.csv"

    if destino.exists() and not forcar:
        return destino

    url = f"{BASE_URL_EXTRA}/{codigo_arquivo}.csv"
    try:
        resposta = requests.get(url, headers=CABECALHOS, timeout=30)
    except requests.RequestException as exc:
        print(f"  [aviso] falha de rede em {url}: {exc}", file=sys.stderr)
        return None

    if resposta.status_code != 200 or not resposta.content:
        print(f"  [aviso] indisponível: {url} (HTTP {resposta.status_code})", file=sys.stderr)
        return None

    destino.write_bytes(resposta.content)
    return destino


def carregar_e_normalizar_extra(caminho_csv: Path, codigo_arquivo: str, n_temporadas: int) -> pd.DataFrame:
    """Lê o CSV de uma extra league (Country, League, Season, Date, Time,
    Home, Away, HG, AG, Res + odds) e normaliza para o esquema unificado de
    partidas.csv. Escanteios/cartões/faltas/árbitro ficam NaN (não existem
    na fonte). A coluna 'Season' já é o ano-calendário da temporada
    (Brasileirão abr-dez, MLS fev-nov) — diferente do código 'ano+ano+1'
    das ligas principais —, então filtramos as últimas n_temporadas por
    esse ano diretamente, sem a lógica de corte em agosto."""
    liga_nome = LIGAS_EXTRA[codigo_arquivo]
    bruto = pd.read_csv(caminho_csv, encoding="utf-8-sig", low_memory=False)

    df = pd.DataFrame({
        "Div": codigo_arquivo,
        "Date": pd.to_datetime(bruto["Date"], format="mixed", dayfirst=True, errors="coerce"),
        "HomeTeam": bruto["Home"].astype(str).str.strip(),
        "AwayTeam": bruto["Away"].astype(str).str.strip(),
        "FTHG": pd.to_numeric(bruto["HG"], errors="coerce"),
        "FTAG": pd.to_numeric(bruto["AG"], errors="coerce"),
        "FTR": bruto["Res"],
        "HC": np.nan, "AC": np.nan,
        "HF": np.nan, "AF": np.nan,
        "HY": np.nan, "AY": np.nan, "HR": np.nan, "AR": np.nan,
        "Referee": None,
        "PSCH": pd.to_numeric(bruto.get("PSCH"), errors="coerce"),
        "PSCD": pd.to_numeric(bruto.get("PSCD"), errors="coerce"),
        "PSCA": pd.to_numeric(bruto.get("PSCA"), errors="coerce"),
    })
    df = df.reindex(columns=COLUNAS_DESEJADAS)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"], how="any")

    df["liga"] = liga_nome
    df["temporada"] = bruto.loc[df.index, "Season"].astype(int).astype(str)

    ano_max = df["temporada"].astype(int).max()
    anos_alvo = {str(a) for a in range(ano_max - n_temporadas + 1, ano_max + 1)}
    df = df[df["temporada"].isin(anos_alvo)]

    return df.reset_index(drop=True)


def _normalizar_nome_time(nome: str) -> str:
    nome = str(nome).strip()
    return ALIASES_TIMES.get(nome, nome)


def carregar_e_normalizar(caminho_csv: Path, codigo_liga: str) -> pd.DataFrame:
    """Lê um CSV bruto de uma temporada e retorna um DataFrame normalizado
    apenas com as colunas de interesse."""
    df = pd.read_csv(caminho_csv, encoding="utf-8-sig", low_memory=False)

    # garante presença de todas as colunas desejadas (preenche com NaN se faltar)
    df = df.reindex(columns=COLUNAS_DESEJADAS)

    # remove linhas totalmente vazias (comuns no fim de alguns CSVs)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"], how="any")

    # datas: football-data usa dd/mm/yy em temporadas antigas e dd/mm/yyyy
    # nas recentes. dayfirst=True cobre ambos os formatos.
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    df["HomeTeam"] = df["HomeTeam"].map(_normalizar_nome_time)
    df["AwayTeam"] = df["AwayTeam"].map(_normalizar_nome_time)

    df["liga"] = LIGAS[codigo_liga]

    # tipagem numérica das colunas de estatísticas/odds
    colunas_numericas = [
        "FTHG", "FTAG", "HC", "AC", "HF", "AF",
        "HY", "AY", "HR", "AR", "PSCH", "PSCD", "PSCA",
    ]
    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def construir_base(n_temporadas: int = 10, forcar_atualizacao: bool = False) -> pd.DataFrame:
    """Baixa e consolida as últimas n_temporadas de cada liga em um único
    DataFrame, salvando o resultado em data/processed/partidas.csv."""
    codigos_temporadas = gerar_codigos_temporadas(n_temporadas)
    print(f"Temporadas alvo: {codigos_temporadas}")

    partes = []
    for codigo_liga in LIGAS:
        for i, codigo_temp in enumerate(codigos_temporadas):
            # a última temporada da lista pode estar em andamento: força
            # atualização dela para pegar jogos novos, a não ser que
            # forcar_atualizacao já force tudo.
            eh_temporada_corrente = i == len(codigos_temporadas) - 1
            forcar = forcar_atualizacao or eh_temporada_corrente

            caminho = baixar_csv(codigo_temp, codigo_liga, forcar=forcar)
            if caminho is None:
                continue
            try:
                df_temp = carregar_e_normalizar(caminho, codigo_liga)
            except Exception as exc:
                print(f"  [aviso] falha ao processar {caminho}: {exc}", file=sys.stderr)
                continue
            df_temp["temporada"] = codigo_temp
            partes.append(df_temp)
            time.sleep(0.1)  # gentileza com o servidor

    for codigo_arquivo in LIGAS_EXTRA:
        caminho = baixar_csv_extra(codigo_arquivo, forcar=True)
        if caminho is None:
            continue
        try:
            df_extra = carregar_e_normalizar_extra(caminho, codigo_arquivo, n_temporadas)
        except Exception as exc:
            print(f"  [aviso] falha ao processar {caminho}: {exc}", file=sys.stderr)
            continue
        print(f"  [{LIGAS_EXTRA[codigo_arquivo]}] {len(df_extra)} partidas carregadas")
        partes.append(df_extra)
        time.sleep(0.1)

    if not partes:
        raise RuntimeError("Nenhum dado foi baixado/processado com sucesso.")

    base = pd.concat(partes, ignore_index=True)
    base = base.sort_values(["liga", "Date"]).reset_index(drop=True)

    PASTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    base.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"Base consolidada salva em {ARQUIVO_SAIDA} ({len(base)} partidas).")
    return base


def main():
    parser = argparse.ArgumentParser(description="Ingestão de dados do football-data.co.uk")
    parser.add_argument("--temporadas", type=int, default=10, help="Número de temporadas a baixar (padrão: 10)")
    parser.add_argument("--forcar", action="store_true", help="Força novo download de todas as temporadas (ignora cache)")
    args = parser.parse_args()

    construir_base(n_temporadas=args.temporadas, forcar_atualizacao=args.forcar)


if __name__ == "__main__":
    main()
