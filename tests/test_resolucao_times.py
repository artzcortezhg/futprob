# -*- coding: utf-8 -*-
"""Testes da resolução aproximada de nomes de time — cobre especificamente o
bug encontrado ao vivo em 2026-07-28: clubes DIFERENTES que só compartilham
uma palavra (ex.: 'Botafogo-SP' e 'Botafogo RJ', dois times reais e
distintos do futebol brasileiro) não podem ser confundidos um com o outro só
por causa de um SequenceMatcher.ratio() alto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resolucao_times import resolver_time, resolver_time_ambiguo  # noqa: E402

TIMES_POR_LIGA = {
    "brasileirao": ["Botafogo RJ", "Atletico GO", "Atletico-MG", "Flamengo RJ", "Juventude", "Avai", "Internacional"],
}


def test_nao_confunde_clubes_diferentes_com_palavra_em_comum():
    assert resolver_time("Botafogo-SP", TIMES_POR_LIGA) is None
    assert resolver_time("Athletic Club MG", TIMES_POR_LIGA) is None


def test_ainda_resolve_variacoes_legitimas_do_mesmo_clube():
    assert resolver_time("Flamengo", TIMES_POR_LIGA) == ("brasileirao", "Flamengo RJ")
    assert resolver_time("Botafogo RJ", TIMES_POR_LIGA) == ("brasileirao", "Botafogo RJ")
    assert resolver_time("Internacional", TIMES_POR_LIGA) == ("brasileirao", "Internacional")


def test_ambiguo_tambem_rejeita_match_de_palavra_em_comum():
    resultado = resolver_time_ambiguo("Botafogo-SP", TIMES_POR_LIGA)
    assert resultado["status"] == "nao_encontrado"
