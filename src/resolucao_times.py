# -*- coding: utf-8 -*-
"""
Resolução aproximada de nomes de time (sem acento, parcial, com
sugestões) contra o roster do futprob. Módulo neutro (sem depender de
bot.py nem integracao_manha.py) justamente pra evitar import circular
entre os dois, que também usam essas funções.
"""
from __future__ import annotations

import difflib
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PARTIDAS_PADRAO = RAIZ / "data" / "processed" / "partidas.csv"


def normalizar_texto(s: str) -> str:
    """minúsculo, sem acento — pra casar 'Grêmio' com 'Gremio' etc."""
    nfkd = unicodedata.normalize("NFKD", s)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acento.lower().strip()


def carregar_times_por_liga(caminho_partidas: Path = CAMINHO_PARTIDAS_PADRAO) -> dict[str, list[str]]:
    df = pd.read_csv(caminho_partidas, usecols=["liga", "HomeTeam", "AwayTeam"])
    resultado = {}
    for liga in df["liga"].unique():
        d = df[df["liga"] == liga]
        resultado[liga] = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    return resultado


def score_nomes(a: str, b: str) -> float:
    """Similaridade entre dois nomes de time (sem acento, com bônus de
    substring parcial) — usado tanto pra casar contra o roster do modelo
    (candidatos_time) quanto pra casar contra nomes CRUS coletados (busca
    de fixture real, ver src/catalogo.py)."""
    a_norm, b_norm = normalizar_texto(a), normalizar_texto(b)
    score = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    if a_norm and b_norm and (a_norm in b_norm or b_norm in a_norm):
        score += 0.25
    return score


def candidatos_time(nome: str, times_por_liga: dict[str, list[str]], top_n: int = 5) -> list[tuple[str, str, float]]:
    """Retorna até top_n (liga, time, score) ordenados por score desc,
    usando busca aproximada sem acento e por substring parcial."""
    alvo = normalizar_texto(nome)
    if not alvo:
        return []
    candidatos = []
    for liga, times in times_por_liga.items():
        for t in times:
            candidatos.append((liga, t, score_nomes(nome, t)))
    candidatos.sort(key=lambda c: -c[2])
    return candidatos[:top_n]


def _aceitavel(nome: str, candidato: str, score: float, limiar: float) -> bool:
    """Um candidato só é aceito se, além do score mínimo, um dos nomes for
    substring do outro (após normalizar) OU o score for bem alto (>=0.90).
    Sem essa segunda condição, clubes DIFERENTES que só compartilham uma
    palavra (ex.: 'Botafogo-SP' e 'Botafogo RJ' — dois times reais e
    distintos do futebol brasileiro) batiam ~0.73 no SequenceMatcher e
    eram confundidos um com o outro."""
    if score < limiar:
        return False
    alvo_norm = normalizar_texto(nome)
    cand_norm = normalizar_texto(candidato)
    eh_substring = alvo_norm in cand_norm or cand_norm in alvo_norm
    return eh_substring or score >= 0.90


def resolver_time(nome: str, times_por_liga: dict[str, list[str]]) -> tuple[str, str] | None:
    """Casamento aproximado simples: retorna (liga, nome_interno) do melhor
    candidato, ou None se nada bateu um mínimo razoável. Usado internamente
    (coleta, catálogo) onde não faz sentido perguntar ao usuário."""
    cands = candidatos_time(nome, times_por_liga, top_n=1)
    if not cands or not _aceitavel(nome, cands[0][1], cands[0][2], limiar=0.55):
        return None
    return (cands[0][0], cands[0][1])


def resolver_time_ambiguo(nome: str, times_por_liga: dict[str, list[str]],
                          limiar: float = 0.55, margem_ambiguidade: float = 0.08) -> dict:
    """Versão com sugestões, usada pelo /jogo (interação com o usuário):
    {"status": "ok", "liga":..., "time":...}
    {"status": "ambiguo", "opcoes": [(liga,time), ...]}
    {"status": "nao_encontrado"}"""
    cands = [c for c in candidatos_time(nome, times_por_liga, top_n=5) if _aceitavel(nome, c[1], c[2], limiar)]
    if not cands:
        return {"status": "nao_encontrado"}
    melhor_score = cands[0][2]
    proximos = [c for c in cands if melhor_score - c[2] <= margem_ambiguidade]
    if len(proximos) > 1:
        return {"status": "ambiguo", "opcoes": [(c[0], c[1]) for c in proximos]}
    return {"status": "ok", "liga": cands[0][0], "time": cands[0][1]}
