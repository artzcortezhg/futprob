# -*- coding: utf-8 -*-
"""Testes de ingest_geglobo.py: construção de URL (mapa time->estado/slug)
e extração de escanteios/faltas do texto da página -- sem rede."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest_geglobo import construir_urls_candidatas, _extrair_estatisticas, _normalizar_slug_cbf


def test_normalizar_slug_cbf_tira_sufixo_saf():
    assert _normalizar_slug_cbf("coritiba-saf") == "coritiba"
    assert _normalizar_slug_cbf("vasco-da-gama-saf") == "vasco-da-gama"
    assert _normalizar_slug_cbf("corinthians") == "corinthians"


def test_construir_urls_time_conhecido_usa_estado_do_mandante():
    urls = construir_urls_candidatas("brasileirao", "2026-07-26", "bahia", "corinthians")
    assert urls == ["https://ge.globo.com/ba/futebol/brasileirao-serie-a/jogo/26-07-2026/bahia-corinthians.ghtml"]


def test_construir_urls_usa_slug_globo_quando_diferente_do_slug_cbf():
    """Athletico-PR: CBF usa o nome completo no slug, ge.globo usa o
    apelido 'atletico-pr' -- confirmado comparando com uma URL real."""
    urls = construir_urls_candidatas("brasileirao", "2026-07-30", "corinthians", "athletico-paranaense")
    assert urls == ["https://ge.globo.com/sp/futebol/brasileirao-serie-a/jogo/30-07-2026/corinthians-atletico-pr.ghtml"]


def test_construir_urls_desambigua_botafogo_por_liga():
    """A CBF usa o mesmo slug 'botafogo' pro Botafogo/RJ (Série A) e pro
    Botafogo-SP (Série B) -- só a liga da partida desambigua qual é."""
    url_serie_a = construir_urls_candidatas("brasileirao", "2026-07-26", "botafogo", "cruzeiro")
    assert "ge.globo.com/rj/futebol/brasileirao-serie-a" in url_serie_a[0]

    url_serie_b = construir_urls_candidatas("brasileirao_b", "2026-07-23", "botafogo", "juventude")
    assert "ge.globo.com/sp/ribeirao-preto-e-regiao/futebol/brasileirao-serie-b" in url_serie_b[0]
    assert "botafogo-sp-juventude" in url_serie_b[0]


def test_construir_urls_times_historicos_da_serie_b_confirmados_ao_vivo():
    """Confirmado com jogos reais (ver investigação de cobertura de
    escanteios/faltas da Série B: metade dos jogos falhava simplesmente
    porque o time nem estava no mapa, nunca tentava buscar)."""
    urls = construir_urls_candidatas("brasileirao_b", "2018-11-17", "brasil", "guarani")
    assert urls == ["https://ge.globo.com/rs/futebol/brasileirao-serie-b/jogo/17-11-2018/brasil-de-pelotas-guarani.ghtml"]

    urls = construir_urls_candidatas("brasileirao_b", "2018-11-13", "figueirense", "paysandu")
    assert urls == ["https://ge.globo.com/sc/futebol/brasileirao-serie-b/jogo/13-11-2018/figueirense-paysandu.ghtml"]


def test_construir_urls_time_desconhecido_retorna_lista_vazia():
    """Nunca inventa estado/slug pra time fora do mapa -- lista vazia é o
    sinal de 'não sei', não um palpite não verificado."""
    assert construir_urls_candidatas("brasileirao", "2026-07-26", "time-que-nao-existe", "bahia") == []


def test_extrair_estatisticas_encontra_escanteios_e_faltas():
    texto = "algum texto antes\n7\nEscanteios\n2\nmais texto\n17\nFaltas cometidas\n17\nfim"
    stats = _extrair_estatisticas(texto)
    assert stats == {"escanteios_casa": 7, "escanteios_fora": 2, "faltas_casa": 17, "faltas_fora": 17}


def test_extrair_estatisticas_sem_padrao_retorna_none():
    assert _extrair_estatisticas("página sem nenhuma estatística de jogo") is None
