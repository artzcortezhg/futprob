# -*- coding: utf-8 -*-
"""Testes do parsing da súmula da CBF (ingest_cbf.py) — cartões/árbitro,
sem depender de rede nem de gerar um PDF de verdade (o texto sintético
reproduz a estrutura real que a extração do PDF devolve, confirmada
manualmente numa súmula real de 2026)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest_cbf import _parsear_texto_sumula


_TEXTO_SUMULA_EXEMPLO = """CBF - CONFEDERAÇÃO BRASILEIRA DE FUTEBOL Jogo: 199
SÚMULA ON-LINE
Campeonato: Campeonato Brasileiro - Série A/2026 Rodada: 20
Jogo: Bahia / BA X Corinthians / SP
Data: 26/07/2026 Horário: 16:00 Estádio: Octávio Mangabeira / Salvador
Arbitragem
Arbitro: Ramon Abatti Abel (FIFA-PRO / SC)
Gols
Tempo 1T/2T Nº Tipo Nome do Jogador Equipe
07:00 1T 26 NR Fabrizio German Angileri Corinthians/SP
Cartões Amarelos
Tempo 1T/2T Nº Nome do Jogador Equipe
16:00 1T 26 Fabrizio German Angileri Corinthians/SP
Motivo: A1.3. Cometer uma falta tática.
19:00 1T 33 David de Duarte Macedo Bahia/BA
Motivo: A1.24. Outro motivo.
37:00 1T AT Charles Alexandre Patrice Francis Hembert Bahia/BA
Motivo: A2. Desaprovar com gestos.
+08:00 1T TC Rogerio Ceni Bahia/BA
Motivo: A2. Desaprovar com palavras.
21:00 2T 7 Breno de Souza Bidon Corinthians/SP
Motivo: A1.13. Entrada temerária.
Cartões Vermelhos
NÃO HOUVE EXPULSÕES
"""


def test_parsear_sumula_extrai_arbitro_e_times_do_cabecalho():
    resultado = _parsear_texto_sumula(_TEXTO_SUMULA_EXEMPLO)
    assert resultado["time_casa"] == "Bahia"
    assert resultado["time_fora"] == "Corinthians"
    assert resultado["arbitro"] == "Ramon Abatti Abel"


def test_parsear_sumula_conta_cartoes_so_de_jogadores_nunca_comissao_tecnica():
    """Charles Alexandre (AT) e Rogerio Ceni (TC) são comissão técnica, não
    jogadores -- o mercado de cartões de aposta é sempre sobre jogadores em
    campo, então essas duas linhas NUNCA podem entrar na contagem."""
    resultado = _parsear_texto_sumula(_TEXTO_SUMULA_EXEMPLO)
    assert resultado["HY"] == 1  # só David de Duarte Macedo (Bahia, jogador)
    assert resultado["AY"] == 2  # Fabrizio + Breno (Corinthians, jogadores)


def test_parsear_sumula_sem_expulsoes_da_zero_cartoes_vermelhos():
    resultado = _parsear_texto_sumula(_TEXTO_SUMULA_EXEMPLO)
    assert resultado["HR"] == 0
    assert resultado["AR"] == 0


def test_parsear_sumula_sem_cabecalho_reconhecivel_da_erro():
    import pytest
    with pytest.raises(ValueError):
        _parsear_texto_sumula("um texto qualquer sem o formato esperado")
