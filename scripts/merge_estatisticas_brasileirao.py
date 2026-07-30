# -*- coding: utf-8 -*-
"""
Funde data/raw/cbf_geglobo_brasileirao.csv (saída de
scripts/coletar_estatisticas_cbf.py) nas linhas 'brasileirao' JÁ existentes
em data/processed/partidas.csv, preenchendo HC/AC/HF/AF/HY/AY/HR/AR/Referee
(hoje NaN pra essa liga — ver ingest.py). NUNCA mexe em FTHG/FTAG/FTR/Date
(o resultado já vem confiável do football-data.co.uk) — só valida que o
placar bate como conferência de integridade, e avisa (não sobrescreve) se
não bater.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from resolucao_times import NOMES_HISTORICOS_CBF  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PARTIDAS = RAIZ / "data" / "processed" / "partidas.csv"
CAMINHO_COLETA = RAIZ / "data" / "raw" / "cbf_geglobo_brasileirao.csv"

# Slug normalizado da CBF (mesmo usado em ingest_geglobo.TIME_CBF_PARA_GLOBO,
# depois de tirar o sufixo -saf) -> nome canônico usado em partidas.csv
# (nomenclatura football-data.co.uk, ver ingest.py). Lista de times
# conferida diretamente contra os valores reais de HomeTeam/AwayTeam da
# liga 'brasileirao' em partidas.csv. NOMES_HISTORICOS_CBF (compartilhado
# com criar_linhas_brasileirao_b.py, ver resolucao_times.py) cobre os slugs
# de nome completo que a CBF usava em temporadas mais antigas pro MESMO
# clube que também jogou a Série B -- nunca duplicar essas entradas aqui.
SLUG_PARA_CANONICO: dict[str, str] = {
    **NOMES_HISTORICOS_CBF,
    "america": "America MG",
    "atletico-goianiense": "Atletico GO", "atletico-mineiro": "Atletico-MG",
    "avai": "Avai", "bahia": "Bahia", "botafogo": "Botafogo RJ",
    "red-bull-bragantino": "Bragantino", "ceara": "Ceara",
    "corinthians": "Corinthians",
    "coritiba": "Coritiba", "criciuma": "Criciuma", "cruzeiro": "Cruzeiro",
    "cuiaba": "Cuiaba", "flamengo": "Flamengo RJ", "fluminense": "Fluminense",
    "fortaleza": "Fortaleza", "goias": "Goias", "gremio": "Gremio",
    "internacional": "Internacional", "juventude": "Juventude",
    "mirassol": "Mirassol", "palmeiras": "Palmeiras", "parana": "Parana",
    "ponte-preta": "Ponte Preta", "remo": "Remo",
    "santos": "Santos", "sao-paulo": "Sao Paulo", "sport-recife": "Sport Recife",
    "sport": "Sport Recife", "vitoria": "Vitoria",
}


def _normalizar_slug(slug: str) -> str:
    return slug[:-4] if slug.endswith("-saf") else slug


def mesclar() -> dict:
    df = pd.read_csv(CAMINHO_PARTIDAS, parse_dates=["Date"])
    coleta = pd.read_csv(CAMINHO_COLETA, dtype={"id_jogo_cbf": str})

    stats = {"mesclados": 0, "time_nao_mapeado": 0, "jogo_nao_encontrado": 0, "placar_diverge": 0}

    for _, linha in coleta.iterrows():
        casa = SLUG_PARA_CANONICO.get(_normalizar_slug(linha["time_casa_slug"]))
        fora = SLUG_PARA_CANONICO.get(_normalizar_slug(linha["time_fora_slug"]))
        if casa is None or fora is None:
            stats["time_nao_mapeado"] += 1
            continue

        # jogos com bola rolando tarde da noite (comum no Brasileirão)
        # podem cair no dia seguinte em UTC -- football-data.co.uk registra
        # a data 1 dia à frente da própria listagem da CBF em vários casos
        # confirmados manualmente; tenta a data exata primeiro, senão +1 dia
        data_cbf = pd.Timestamp(linha["data_iso"])
        indices = pd.Index([])
        for data in (data_cbf, data_cbf + pd.Timedelta(days=1)):
            mascara = (
                (df["liga"] == "brasileirao") & (df["Date"] == data)
                & (df["HomeTeam"] == casa) & (df["AwayTeam"] == fora)
            )
            indices = df.index[mascara]
            if len(indices):
                break
        if len(indices) == 0:
            stats["jogo_nao_encontrado"] += 1
            continue

        idx = indices[0]
        if pd.notna(linha["gols_casa"]) and (df.at[idx, "FTHG"] != linha["gols_casa"] or df.at[idx, "FTAG"] != linha["gols_fora"]):
            stats["placar_diverge"] += 1
            continue  # não confia nas estatísticas dessa linha se nem o placar bate

        if pd.notna(linha["escanteios_casa"]):
            df.at[idx, "HC"] = linha["escanteios_casa"]
            df.at[idx, "AC"] = linha["escanteios_fora"]
        if pd.notna(linha["faltas_casa"]):
            df.at[idx, "HF"] = linha["faltas_casa"]
            df.at[idx, "AF"] = linha["faltas_fora"]
        if pd.notna(linha["HY"]):
            df.at[idx, "HY"] = linha["HY"]
            df.at[idx, "AY"] = linha["AY"]
            df.at[idx, "HR"] = linha["HR"]
            df.at[idx, "AR"] = linha["AR"]
        if pd.notna(linha["arbitro"]):
            df.at[idx, "Referee"] = linha["arbitro"]
        stats["mesclados"] += 1

    df.to_csv(CAMINHO_PARTIDAS, index=False)
    return stats


if __name__ == "__main__":
    if not CAMINHO_COLETA.exists():
        print(f"Nada pra mesclar: {CAMINHO_COLETA} não existe ainda (rode coletar_estatisticas_cbf.py antes).")
        sys.exit(1)
    resultado = mesclar()
    print(resultado)
