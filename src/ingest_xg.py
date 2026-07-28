# -*- coding: utf-8 -*-
"""
Ingestão de xG (gols esperados) por partida do Understat, para Premier
League e La Liga (Championship não tem xG no Understat, fica de fora).

Casa cada partida do Understat com a linha correspondente em
data/processed/partidas.csv por data + nomes de times normalizados (as duas
fontes grafam os times de forma diferente; ALIASES_TIMES faz a ponte).
Algumas partidas têm o horário registrado em fusos diferentes entre as
fontes, deslocando a data em +-1 dia: por isso o casamento é feito em duas
passadas — exata primeiro, depois por proximidade de data (mesmos times,
mesma temporada, data mais próxima) só para o resíduo não casado.

Saída: data/processed/partidas_xg.csv, com as mesmas colunas de
partidas.csv (restrito a Premier League e La Liga) mais xG_casa e xG_fora.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw" / "understat"
CAMINHO_PARTIDAS = RAIZ / "data" / "processed" / "partidas.csv"
ARQUIVO_SAIDA = RAIZ / "data" / "processed" / "partidas_xg.csv"

BASE_URL = "https://understat.com/main/getLeagueData"
CABECALHOS = {
    "User-Agent": "Mozilla/5.0 (compatible; futprob/1.0)",
    "X-Requested-With": "XMLHttpRequest",
}

# código usado pelo Understat -> nome amigável da liga (igual ao de ingest.py)
LIGAS_UNDERSTAT = {
    "EPL": "Premier League",
    "La liga": "La Liga",
}

# nome do time no Understat -> nome equivalente no football-data.co.uk,
# identificado comparando os conjuntos de nomes das duas fontes.
ALIASES_TIMES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Celta Vigo": "Celta",
    "Deportivo La Coruna": "La Coruna",
    "Espanyol": "Espanol",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "SD Huesca": "Huesca",
    "Sporting Gijon": "Sp Gijon",
}

TOLERANCIA_DIAS_FALLBACK = 3  # janela de busca para o casamento por proximidade


def gerar_anos_temporadas(n: int = 10, hoje: date | None = None) -> list[int]:
    """Anos de início das últimas n temporadas (2016 = temporada 2016/17),
    com a mesma regra de corte de agosto usada em ingest.py."""
    hoje = hoje or date.today()
    ano_inicio_atual = hoje.year if hoje.month >= 8 else hoje.year - 1
    return list(range(ano_inicio_atual - n + 1, ano_inicio_atual + 1))


def baixar_temporada(codigo_understat: str, ano: int, forcar: bool = False) -> Path | None:
    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    slug = codigo_understat.replace(" ", "_").lower()
    destino = PASTA_RAW / f"{slug}_{ano}.json"

    if destino.exists() and not forcar:
        return destino

    url = f"{BASE_URL}/{quote(codigo_understat)}/{ano}"
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


def carregar_temporada(caminho_json: Path, liga: str, ano: int) -> pd.DataFrame:
    """Lê o JSON de uma temporada/liga do Understat e retorna um DataFrame
    de partidas já finalizadas, com times normalizados para o padrão
    football-data."""
    import json

    dados = json.loads(caminho_json.read_text(encoding="utf-8"))
    linhas = []
    for jogo in dados["dates"]:
        if not jogo.get("isResult"):
            continue
        linhas.append({
            "Date": pd.Timestamp(jogo["datetime"]).normalize(),
            "HomeTeam": ALIASES_TIMES.get(jogo["h"]["title"], jogo["h"]["title"]),
            "AwayTeam": ALIASES_TIMES.get(jogo["a"]["title"], jogo["a"]["title"]),
            "gols_casa_understat": int(jogo["goals"]["h"]),
            "gols_fora_understat": int(jogo["goals"]["a"]),
            "xG_casa": float(jogo["xG"]["h"]),
            "xG_fora": float(jogo["xG"]["a"]),
        })
    df = pd.DataFrame(linhas)
    df["liga"] = liga
    codigo_temporada = f"{str(ano)[2:].zfill(2)}{str(ano + 1)[2:].zfill(2)}"
    df["temporada"] = codigo_temporada
    return df


def _casar_por_proximidade(nao_casados_us: pd.DataFrame, nao_casados_fd: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Casamento residual: mesmos times + mesma temporada, pega a data mais
    próxima dentro da tolerância (cobre jogos com fuso horário deslocando a
    data em +-1 dia entre as fontes). Retorna os casados e o conjunto de
    índices de nao_casados_fd que foram consumidos."""
    casados = []
    usados_fd: set = set()
    for _, linha_us in nao_casados_us.iterrows():
        candidatos = nao_casados_fd[
            (nao_casados_fd["HomeTeam"] == linha_us["HomeTeam"])
            & (nao_casados_fd["AwayTeam"] == linha_us["AwayTeam"])
            & (nao_casados_fd["temporada"].astype(str) == str(linha_us["temporada"]))
            & (~nao_casados_fd.index.isin(usados_fd))
        ].copy()
        if candidatos.empty:
            continue
        candidatos["diff_dias"] = (candidatos["Date"] - linha_us["Date"]).abs().dt.days
        candidatos = candidatos[candidatos["diff_dias"] <= TOLERANCIA_DIAS_FALLBACK]
        if candidatos.empty:
            continue
        melhor = candidatos.sort_values("diff_dias").iloc[0]
        usados_fd.add(melhor.name)
        registro = melhor.to_dict()
        registro["xG_casa"] = linha_us["xG_casa"]
        registro["xG_fora"] = linha_us["xG_fora"]
        casados.append(registro)
    if not casados:
        return pd.DataFrame(columns=list(nao_casados_fd.columns) + ["xG_casa", "xG_fora"]), usados_fd
    return pd.DataFrame(casados).drop(columns=["diff_dias"], errors="ignore"), usados_fd


def casar_com_partidas(df_understat: pd.DataFrame, df_partidas: pd.DataFrame) -> pd.DataFrame:
    """Casa as partidas do Understat (com xG) com as linhas de
    data/processed/partidas.csv, por (liga, Date, HomeTeam, AwayTeam) com
    fallback de proximidade de data para o resíduo."""
    resultado_final = []

    for liga in df_understat["liga"].unique():
        us_liga = df_understat[df_understat["liga"] == liga]
        fd_liga = df_partidas[df_partidas["liga"] == liga].copy()

        exato = fd_liga.merge(
            us_liga[["Date", "HomeTeam", "AwayTeam", "xG_casa", "xG_fora"]],
            on=["Date", "HomeTeam", "AwayTeam"], how="left",
        )

        casados_mask = exato["xG_casa"].notna()
        casados = exato[casados_mask]
        nao_casados_fd = exato[~casados_mask].drop(columns=["xG_casa", "xG_fora"])

        chaves_casadas = set(zip(casados["Date"], casados["HomeTeam"], casados["AwayTeam"]))
        nao_casados_us = us_liga[~us_liga.apply(
            lambda r: (r["Date"], r["HomeTeam"], r["AwayTeam"]) in chaves_casadas, axis=1
        )]

        casados_fallback, indices_usados = _casar_por_proximidade(nao_casados_us, nao_casados_fd)
        residual_sem_xg = nao_casados_fd[~nao_casados_fd.index.isin(indices_usados)].copy()
        residual_sem_xg["xG_casa"] = pd.NA
        residual_sem_xg["xG_fora"] = pd.NA

        resultado_final.append(casados)
        if not casados_fallback.empty:
            resultado_final.append(casados_fallback)
        if not residual_sem_xg.empty:
            resultado_final.append(residual_sem_xg)

    base = pd.concat(resultado_final, ignore_index=True)
    return base.sort_values(["liga", "Date"]).reset_index(drop=True)


def construir_base_xg(n_temporadas: int = 10, forcar_atualizacao: bool = False) -> pd.DataFrame:
    df_partidas = pd.read_csv(CAMINHO_PARTIDAS, parse_dates=["Date"])
    anos = gerar_anos_temporadas(n_temporadas)
    print(f"Temporadas alvo (ano de início): {anos}")

    partes = []
    for codigo_understat, liga in LIGAS_UNDERSTAT.items():
        for i, ano in enumerate(anos):
            eh_temporada_corrente = i == len(anos) - 1
            forcar = forcar_atualizacao or eh_temporada_corrente
            caminho = baixar_temporada(codigo_understat, ano, forcar=forcar)
            if caminho is None:
                continue
            try:
                df_temp = carregar_temporada(caminho, liga, ano)
            except Exception as exc:
                print(f"  [aviso] falha ao processar {caminho}: {exc}", file=sys.stderr)
                continue
            partes.append(df_temp)
            time.sleep(0.3)

    if not partes:
        raise RuntimeError("Nenhum dado de xG foi baixado/processado com sucesso.")

    df_understat = pd.concat(partes, ignore_index=True)
    base = casar_com_partidas(df_understat, df_partidas)

    n_total_understat = len(df_understat)
    n_casados = base["xG_casa"].notna().sum()
    print(f"Partidas do Understat: {n_total_understat} | casadas com partidas.csv: {n_casados} "
          f"({n_casados / n_total_understat:.1%})")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"Base com xG salva em {ARQUIVO_SAIDA} ({len(base)} partidas).")
    return base


def main():
    parser = argparse.ArgumentParser(description="Ingestão de xG do Understat (Premier League e La Liga)")
    parser.add_argument("--temporadas", type=int, default=10)
    parser.add_argument("--forcar", action="store_true")
    args = parser.parse_args()

    construir_base_xg(n_temporadas=args.temporadas, forcar_atualizacao=args.forcar)


if __name__ == "__main__":
    main()
