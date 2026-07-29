# -*- coding: utf-8 -*-
"""Testes da resolução aproximada de nomes de time — cobre especificamente o
bug encontrado ao vivo em 2026-07-28: clubes DIFERENTES que só compartilham
uma palavra (ex.: 'Botafogo-SP' e 'Botafogo RJ', dois times reais e
distintos do futebol brasileiro) não podem ser confundidos um com o outro só
por causa de um SequenceMatcher.ratio() alto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resolucao_times import (  # noqa: E402
    resolver_time, resolver_time_ambiguo, resolver_time_todas_ligas,
    carregar_times_por_liga, LIGA_SERIE_B, TIMES_SERIE_B_2026,
)

TIMES_POR_LIGA = {
    "brasileirao": ["Botafogo RJ", "Atletico GO", "Atletico-MG", "Flamengo RJ", "Juventude", "Avai", "Internacional"],
}


def test_nao_confunde_clubes_diferentes_com_palavra_em_comum():
    assert resolver_time("Botafogo-SP", TIMES_POR_LIGA) is None
    assert resolver_time("Athletic Club MG", TIMES_POR_LIGA) is None


def test_nao_confunde_substring_dentro_de_uma_so_palavra():
    """Regressão achada testando com nomes reais da OddsPapi: 'Parana'
    (Paraná Clube) é literalmente um TRECHO de 'Paranaense' (Athletico
    Paranaense, clube diferente) — 'parana' in 'ca paranaense pr' é True,
    mas não é a mesma palavra. O substring só pode valer por PALAVRA
    inteira, não por pedaço dentro de uma palavra maior."""
    times = {"brasileirao": ["Parana", "Athletico-PR"]}
    assert resolver_time("CA Paranaense PR", times) is None


def test_ainda_resolve_variacoes_legitimas_do_mesmo_clube():
    assert resolver_time("Flamengo", TIMES_POR_LIGA) == ("brasileirao", "Flamengo RJ")
    assert resolver_time("Botafogo RJ", TIMES_POR_LIGA) == ("brasileirao", "Botafogo RJ")
    assert resolver_time("Internacional", TIMES_POR_LIGA) == ("brasileirao", "Internacional")


def test_resolve_abreviacoes_comuns_de_clube():
    """Achado testando com nomes reais da OddsPapi: 'Atlanta United FC' e
    'Atlanta Utd' são o MESMO clube (329 jogos de histórico!), só que a
    checagem por palavra inteira rejeitava 'utd' != 'united' — mesma coisa
    pra 'Saint'/'St.'. Sem esse alias, o cruzamento com odds externas
    perdia jogos que na verdade têm histórico completo."""
    times = {"mls": ["Atlanta Utd", "St. Louis City", "Real Salt Lake"]}
    assert resolver_time("Atlanta United FC", times) == ("mls", "Atlanta Utd")
    assert resolver_time("Saint Louis City SC", times) == ("mls", "St. Louis City")


def test_ambiguo_tambem_rejeita_match_de_palavra_em_comum():
    resultado = resolver_time_ambiguo("Botafogo-SP", TIMES_POR_LIGA)
    assert resultado["status"] == "nao_encontrado"


def test_resolver_time_todas_ligas_acha_time_em_mais_de_uma_lista():
    """'Fortaleza' jogou a Série A no histórico de treino E está na Série B
    2026 (promoção/rebaixamento) — precisa aparecer nas DUAS listas, não só
    na 'melhor' isoladamente, senão o cruzamento de um confronto real
    (Fortaleza x Botafogo-SP) não consegue achar a liga certa."""
    times = {"brasileirao": ["Fortaleza", "Ceara"], LIGA_SERIE_B: TIMES_SERIE_B_2026}
    opcoes = dict(resolver_time_todas_ligas("Fortaleza", times))
    assert opcoes.get("brasileirao") == "Fortaleza"
    assert opcoes.get(LIGA_SERIE_B) == "Fortaleza"


def test_resolver_time_todas_ligas_time_exclusivo_de_uma_lista():
    times = {"brasileirao": ["Ceara"], LIGA_SERIE_B: TIMES_SERIE_B_2026}
    opcoes = dict(resolver_time_todas_ligas("Botafogo-SP", times))
    assert list(opcoes.keys()) == [LIGA_SERIE_B]


def test_carregar_times_por_liga_inclui_serie_b(tmp_path):
    import pandas as pd
    df = pd.DataFrame([{"liga": "brasileirao", "HomeTeam": "Fortaleza", "AwayTeam": "Ceara"}])
    caminho = tmp_path / "partidas_teste.csv"
    df.to_csv(caminho, index=False)
    resultado = carregar_times_por_liga(caminho)
    assert LIGA_SERIE_B in resultado
    assert "Botafogo-SP" in resultado[LIGA_SERIE_B]
