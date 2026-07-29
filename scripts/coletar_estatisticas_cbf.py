# -*- coding: utf-8 -*-
"""
Coleta escanteios/faltas (ge.globo) + cartões/árbitro (súmula CBF) +
resultado (CBF, só necessário pra Série B — Série A já tem gols em
partidas.csv) de uma ou mais temporadas do Brasileirão A/B.

Grava incrementalmente em data/raw/cbf_geglobo_{liga}.csv (uma linha por
jogo já processado) — resumível: reruns pulam jogos já coletados (usa o
cache em disco de ingest_cbf/ingest_geglobo, e o próprio CSV de saída como
checkpoint). Feito pra rodar em background por muito tempo (milhares de
jogos, cada um com 1-2 requisições de rede).

Uso:
    python scripts/coletar_estatisticas_cbf.py --liga brasileirao --temporadas 2024 2025 2026
    python scripts/coletar_estatisticas_cbf.py --liga brasileirao_b --temporadas 2018-2026
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ingest_cbf as icbf  # noqa: E402
import ingest_geglobo as igg  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PASTA_SAIDA = RAIZ / "data" / "raw"


def _parse_temporadas(valores: list[str]) -> list[int]:
    anos: list[int] = []
    for v in valores:
        if "-" in v:
            ini, fim = v.split("-")
            anos += list(range(int(ini), int(fim) + 1))
        else:
            anos.append(int(v))
    return sorted(set(anos))


def _ja_coletados(caminho_saida: Path) -> set[str]:
    if not caminho_saida.exists():
        return set()
    df = pd.read_csv(caminho_saida, usecols=["id_jogo_cbf"], dtype=str)
    return set(df["id_jogo_cbf"])


def coletar(liga: str, temporadas: list[int]) -> None:
    caminho_saida = PASTA_SAIDA / f"cbf_geglobo_{liga}.csv"
    ja_feitos = _ja_coletados(caminho_saida)
    print(f"[{liga}] já coletados anteriormente: {len(ja_feitos)}", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page_cbf = browser.new_page(user_agent=icbf.USER_AGENT)
        page_globo = browser.new_page(user_agent=igg.USER_AGENT)

        for temporada in temporadas:
            print(f"[{liga}] enumerando temporada {temporada}...", flush=True)
            jogos = icbf.listar_jogos_temporada(page_cbf, liga, temporada)
            jogos_jogados = [j for j in jogos if j["data_iso"] is not None]
            print(f"[{liga}] {temporada}: {len(jogos_jogados)}/{len(jogos)} já disputados", flush=True)

            for i, jogo in enumerate(jogos_jogados):
                if jogo["id_jogo_cbf"] in ja_feitos:
                    continue

                sumula = icbf.obter_sumula_com_cache(page_cbf, liga, temporada, jogo)
                stats_globo = igg.obter_estatisticas_com_cache(
                    page_globo, liga, jogo["data_iso"],
                    jogo["time_casa_slug"], jogo["time_fora_slug"],
                )

                linha = {
                    "id_jogo_cbf": jogo["id_jogo_cbf"], "liga": liga, "temporada": temporada,
                    "rodada": jogo["rodada"], "data_iso": jogo["data_iso"],
                    "time_casa_slug": jogo["time_casa_slug"], "time_fora_slug": jogo["time_fora_slug"],
                    "gols_casa": jogo["gols_casa"], "gols_fora": jogo["gols_fora"],
                    "time_casa_sumula": (sumula or {}).get("time_casa"),
                    "time_fora_sumula": (sumula or {}).get("time_fora"),
                    "arbitro": (sumula or {}).get("arbitro"),
                    "HY": (sumula or {}).get("HY"), "AY": (sumula or {}).get("AY"),
                    "HR": (sumula or {}).get("HR"), "AR": (sumula or {}).get("AR"),
                    "escanteios_casa": (stats_globo or {}).get("escanteios_casa"),
                    "escanteios_fora": (stats_globo or {}).get("escanteios_fora"),
                    "faltas_casa": (stats_globo or {}).get("faltas_casa"),
                    "faltas_fora": (stats_globo or {}).get("faltas_fora"),
                }
                df_linha = pd.DataFrame([linha])
                df_linha.to_csv(caminho_saida, mode="a", header=not caminho_saida.exists(), index=False)

                if (i + 1) % 20 == 0:
                    print(f"[{liga}] {temporada}: {i+1}/{len(jogos_jogados)} processados", flush=True)
                time.sleep(0.3)

        browser.close()
    print(f"[{liga}] concluído.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liga", required=True, choices=["brasileirao", "brasileirao_b"])
    parser.add_argument("--temporadas", nargs="+", required=True, help="ex.: 2026 ou 2018-2026")
    args = parser.parse_args()
    coletar(args.liga, _parse_temporadas(args.temporadas))


if __name__ == "__main__":
    main()
