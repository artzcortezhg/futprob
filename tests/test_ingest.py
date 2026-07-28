# -*- coding: utf-8 -*-
"""Testes de geração de códigos de temporada (ingest.py)."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest import gerar_codigos_temporadas


def test_gera_dez_codigos():
    codigos = gerar_codigos_temporadas(10, date(2026, 7, 28))
    assert codigos == ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]


def test_apos_agosto_inclui_temporada_corrente():
    codigos = gerar_codigos_temporadas(3, date(2026, 9, 1))
    assert codigos[-1] == "2627"


def test_antes_de_agosto_nao_inclui_temporada_ainda_nao_iniciada():
    codigos = gerar_codigos_temporadas(3, date(2026, 7, 31))
    assert codigos[-1] == "2526"
