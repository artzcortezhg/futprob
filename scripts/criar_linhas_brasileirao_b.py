# -*- coding: utf-8 -*-
"""
Cria as linhas de 'brasileirao_b' em data/processed/partidas.csv a partir de
data/raw/cbf_geglobo_brasileirao_b.csv — ESSA liga não tinha NENHUM
histórico antes disso (football-data.co.uk não cobre Série B), então
diferente do merge da Série A, aqui é preencher linha nova inteira: Date,
HomeTeam/AwayTeam, FTHG/FTAG/FTR (resultado, direto da própria listagem de
rodada da CBF — nunca de outra fonte) + HC/AC/HF/AF/HY/AY/HR/AR/Referee.
PSH/PSD/PSA/PSCH/PSCD/PSCA (odds) ficam NaN — não existe fonte de odds
históricas pra Série B.

Os nomes canônicos dos times do roster atual (2026, ver
resolucao_times.TIMES_SERIE_B_2026) são usados literalmente, pra bater com
o mesmo mecanismo de resolução de nomes já usado pro resto do sistema. Times
de temporadas passadas que não estão mais na Série B (promovidos/rebaixados)
recebem um nome derivado do próprio slug — não precisam bater com nenhum
roster, só precisam ser consistentes entre si (é só treino de modelo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from resolucao_times import NOMES_HISTORICOS_CBF  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PARTIDAS = RAIZ / "data" / "processed" / "partidas.csv"
CAMINHO_COLETA = RAIZ / "data" / "raw" / "cbf_geglobo_brasileirao_b.csv"

# Roster atual (2026) -- ver resolucao_times.TIMES_SERIE_B_2026. Nomes AQUI
# têm que bater literalmente com aquela lista pra resolução de fixture ao
# vivo funcionar sem precisar de fuzzy matching extra.
SLUG_PARA_CANONICO_ATUAL: dict[str, str] = {
    "america": "America MG", "athletic": "Athletic", "atletico-goianiense": "Atletico GO",
    "avai": "Avai", "botafogo-sp": "Botafogo-SP",
    # a CBF usa o slug "botafogo" puro (sem "-sp") pro Botafogo-SP na Série B
    # -- mesma ambiguidade de nome já tratada em ingest_geglobo.py
    "botafogo": "Botafogo-SP",
    "ceara": "Ceara", "crb": "CRB",
    "criciuma": "Criciuma", "cuiaba": "Cuiaba", "fortaleza": "Fortaleza",
    "goias": "Goias", "juventude": "Juventude", "londrina": "Londrina",
    "nautico": "Nautico", "gremio-novorizontino": "Novorizontino",
    "operario": "Operario Ferroviario", "ponte-preta": "Ponte Preta",
    "sao-bernardo": "Sao Bernardo", "sport-recife": "Sport", "vila-nova": "Vila Nova",
}

# Times que NUNCA jogaram a Série A (não entram em NOMES_HISTORICOS_CBF,
# que é só pra identidade compartilhada entre as duas ligas), mas cuja CBF
# usou slugs diferentes em temporadas diferentes pro MESMO clube dentro da
# própria Série B -- achado auditando: "amazonas-fc" (2024) e "amazonas-saf"
# (2025, normaliza pra "amazonas") viravam DOIS nomes internos diferentes
# ("Amazonas Fc" e "Amazonas"), fragmentando o histórico do mesmo time.
_VARIANTES_MESMO_CLUBE_SERIE_B: dict[str, str] = {
    "amazonas-fc": "Amazonas",
}


def _normalizar_slug(slug: str) -> str:
    return slug[:-4] if slug.endswith("-saf") else slug


def _nome_canonico(slug: str) -> str:
    """Ordem de prioridade: roster ATUAL (nomes que têm que bater com
    TIMES_SERIE_B_2026 pra resolução ao vivo funcionar) -> identidade
    histórica compartilhada com a Série A (NOMES_HISTORICOS_CBF -- um clube
    com spells nas duas divisões precisa ter o MESMO nome nos dois
    datasets, senão resolver_time_todas_ligas nunca reconhece que é o
    mesmo time) -> variante de slug do mesmo clube só dentro da Série B
    (_VARIANTES_MESMO_CLUBE_SERIE_B) -> nome derivado do slug como último
    recurso."""
    slug_norm = _normalizar_slug(slug)
    if slug_norm in SLUG_PARA_CANONICO_ATUAL:
        return SLUG_PARA_CANONICO_ATUAL[slug_norm]
    if slug_norm in NOMES_HISTORICOS_CBF:
        return NOMES_HISTORICOS_CBF[slug_norm]
    if slug_norm in _VARIANTES_MESMO_CLUBE_SERIE_B:
        return _VARIANTES_MESMO_CLUBE_SERIE_B[slug_norm]
    return slug_norm.replace("-", " ").title()


def construir_linhas() -> pd.DataFrame:
    coleta = pd.read_csv(CAMINHO_COLETA, dtype={"id_jogo_cbf": str})
    coleta = coleta.dropna(subset=["gols_casa", "gols_fora"])

    linhas = []
    for _, r in coleta.iterrows():
        casa, fora = int(r["gols_casa"]), int(r["gols_fora"])
        linhas.append({
            "Div": "BRB", "Date": r["data_iso"],
            "HomeTeam": _nome_canonico(r["time_casa_slug"]), "AwayTeam": _nome_canonico(r["time_fora_slug"]),
            "FTHG": casa, "FTAG": fora,
            "FTR": "H" if casa > fora else ("A" if casa < fora else "D"),
            "HC": r["escanteios_casa"], "AC": r["escanteios_fora"],
            "HF": r["faltas_casa"], "AF": r["faltas_fora"],
            "HY": r["HY"], "AY": r["AY"], "HR": r["HR"], "AR": r["AR"],
            "Referee": r["arbitro"],
            "PSH": None, "PSD": None, "PSA": None, "PSCH": None, "PSCD": None, "PSCA": None,
            "liga": "brasileirao_b", "temporada": str(r["temporada"]),
        })
    return pd.DataFrame(linhas)


def aplicar() -> dict:
    df = pd.read_csv(CAMINHO_PARTIDAS, parse_dates=["Date"])
    novas = construir_linhas()
    novas["Date"] = pd.to_datetime(novas["Date"])

    existentes = df[df["liga"] == "brasileirao_b"]
    chave_existente = set(zip(existentes["Date"], existentes["HomeTeam"], existentes["AwayTeam"]))
    novas = novas[~novas.apply(lambda r: (r["Date"], r["HomeTeam"], r["AwayTeam"]) in chave_existente, axis=1)]

    df_final = pd.concat([df, novas], ignore_index=True)
    df_final = df_final.reindex(columns=df.columns)
    df_final.to_csv(CAMINHO_PARTIDAS, index=False)
    return {"linhas_adicionadas": len(novas), "total_brasileirao_b": len(existentes) + len(novas)}


if __name__ == "__main__":
    if not CAMINHO_COLETA.exists():
        print(f"Nada pra criar: {CAMINHO_COLETA} não existe ainda (rode coletar_estatisticas_cbf.py antes).")
        sys.exit(1)
    print(aplicar())
