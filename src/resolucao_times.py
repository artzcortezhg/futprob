# -*- coding: utf-8 -*-
"""
Resolução aproximada de nomes de time (sem acento, parcial, com
sugestões) contra o roster do futprob. Módulo neutro (sem depender de
bot.py nem integracao_manha.py) justamente pra evitar import circular
entre os dois, que também usam essas funções.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PARTIDAS_PADRAO = RAIZ / "data" / "processed" / "partidas.csv"

# Roster ESTÁTICO da Série B 2026 (fonte: Wikipedia — 20 participantes da
# temporada). SEM modelo próprio (FBref bloqueado por Cloudflare, ver
# investigação anterior — não dá pra treinar Dixon-Coles sem histórico de
# gols coletável) — mas reconhecido como "em escopo" do projeto, pra que um
# jogo real da Série B (ex.: Fortaleza x Botafogo-SP) apareça identificado
# e com as odds coletadas, em vez de cair no mesmo balaio de qualquer jogo
# aleatório de fora do projeto (liga estrangeira, outro esporte etc.).
LIGA_SERIE_B = "brasileirao_b"
TIMES_SERIE_B_2026 = [
    "America MG", "Athletic", "Atletico GO", "Avai", "Botafogo-SP", "Ceara",
    "CRB", "Criciuma", "Cuiaba", "Fortaleza", "Goias", "Juventude", "Londrina",
    "Nautico", "Novorizontino", "Operario Ferroviario", "Ponte Preta",
    "Sao Bernardo", "Sport", "Vila Nova",
]


def normalizar_texto(s: str) -> str:
    """minúsculo, sem acento — pra casar 'Grêmio' com 'Gremio' etc."""
    nfkd = unicodedata.normalize("NFKD", s)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acento.lower().strip()


JANELA_PERTENCIMENTO_DIAS = 730  # ~2 temporadas -- ver carregar_times_por_liga


def carregar_times_por_liga(caminho_partidas: Path = CAMINHO_PARTIDAS_PADRAO,
                             janela_dias: int = JANELA_PERTENCIMENTO_DIAS,
                             hoje: pd.Timestamp | None = None) -> dict[str, list[str]]:
    """Só considera um time "pertencente" a uma liga se ele jogou nela nos
    últimos `janela_dias` -- sem isso, um time que subiu de divisão anos
    atrás continua "pertencendo" à liga antiga pra sempre, e uma partida
    AMISTOSA contra um adversário de outra divisão (comum em pré-temporada)
    pode ser aceita como jogo real dessa liga antiga. Bug real encontrado
    ao vivo: 'Bristol City x Newcastle' (amistoso de pré-temporada, não
    Championship) foi tratado como jogo do Championship porque Newcastle
    aparece no roster histórico -- só que o último jogo dele lá foi em
    2017 (9 anos atrás; ele está na Premier League desde então). LIGA_SERIE_B
    combina o roster ATUAL (TIMES_SERIE_B_2026, estático — garante que os 20
    times de 2026 sempre resolvem fixture ao vivo, mesmo que a base
    histórica ainda não tenha colhido dados suficientes de algum deles) com
    o histórico recente — união, nunca substituição."""
    hoje = hoje if hoje is not None else pd.Timestamp.now()
    corte = hoje - pd.Timedelta(days=janela_dias)
    df = pd.read_csv(caminho_partidas, usecols=["liga", "Date", "HomeTeam", "AwayTeam"], parse_dates=["Date"])
    df = df[df["Date"] >= corte]
    resultado = {}
    for liga in df["liga"].unique():
        d = df[df["liga"] == liga]
        resultado[liga] = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    resultado[LIGA_SERIE_B] = sorted(set(resultado.get(LIGA_SERIE_B, [])) | set(TIMES_SERIE_B_2026))
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


# abreviações MUITO comuns de nome de clube — conservador de propósito
# (só entradas inequívocas no contexto de futebol), pra não perder matches
# legítimos só por causa de abreviação: 'Atlanta United FC' (nome cru da
# OddsPapi) x 'Atlanta Utd' (nome no nosso roster) são o MESMO clube, só
# que a comparação por palavra inteira rejeitava 'utd' != 'united'.
_ALIASES_TOKEN = {"utd": "united", "st": "saint"}


def _tokens(texto: str) -> set[str]:
    """Tokeniza pra comparação por palavra inteira: sem acento/maiúsculas,
    sem pontuação (pra 'St.' virar o token 'st', não 'st.'), com
    abreviações comuns canonicalizadas (ver _ALIASES_TOKEN)."""
    norm = normalizar_texto(texto).replace(".", "")
    tokens = {t for t in re.split(r"[\s\-]+", norm) if t}
    return {_ALIASES_TOKEN.get(t, t) for t in tokens}


def _aceitavel(nome: str, candidato: str, score: float, limiar: float) -> bool:
    """Um candidato só é aceito se, além do score mínimo, TODAS as palavras
    de um dos nomes existirem como palavra INTEIRA no outro (após
    normalizar, sem pontuação, com abreviações comuns canonicalizadas) OU
    o score for bem alto (>=0.90).

    Precisa ser por palavra inteira, não por trecho dentro de uma palavra:
    'Botafogo-SP' x 'Botafogo RJ' (dois clubes DISTINTOS que só
    compartilham uma palavra) batiam ~0.73 no SequenceMatcher e eram
    confundidos. Um substring bruto (sem respeitar fronteira de palavra)
    tem o MESMO problema de outro jeito: 'Parana' (Paraná Clube) é
    literalmente um trecho de 'Paranaense' (Athletico Paranaense, clube
    DIFERENTE) — 'parana' in 'ca paranaense pr' dá True, mas não é a
    mesma palavra. Comparando por tokens, 'parana' nunca aparece como
    palavra inteira em ['ca','paranaense','pr'], então é rejeitado."""
    if score < limiar:
        return False
    alvo_tokens = _tokens(nome)
    cand_tokens = _tokens(candidato)
    eh_substring_por_palavra = alvo_tokens <= cand_tokens or cand_tokens <= alvo_tokens
    return eh_substring_por_palavra or score >= 0.90


def resolver_time_todas_ligas(nome: str, times_por_liga: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Como resolver_time, mas retorna TODOS os matches aceitáveis (o
    melhor de cada liga), não só o global — necessário porque um time pode
    existir em mais de uma lista ao mesmo tempo (ex.: 'Fortaleza' jogou a
    Série A no histórico E está na Série B 2026). Decidir qual liga vale
    pro CONFRONTO cabe a quem cruza os dois lados de um jogo (ver
    resolver_fixture_para_liga em catalogo.py), não a essa função."""
    resultado = []
    for liga, times in times_por_liga.items():
        melhor_time, melhor_score = None, 0.0
        for t in times:
            score = score_nomes(nome, t)
            if score > melhor_score:
                melhor_time, melhor_score = t, score
        if melhor_time and _aceitavel(nome, melhor_time, melhor_score, limiar=0.55):
            resultado.append((liga, melhor_time))
    return resultado


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
