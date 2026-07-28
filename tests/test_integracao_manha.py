# -*- coding: utf-8 -*-
"""Testes de integracao_manha.py. Não roda coleta nem Telegram de verdade.

O mapeamento mercado-coletado -> mercado-futprob (_mercados_para_jogo,
_selecao_h2h etc.) mora em src/catalogo.py e é testado lá (test_catalogo.py)
— aqui testamos especificamente que processar_foto_manha_async usa essa
MESMA função (combinar_modelo_e_odds) pra montar os candidatos, e não uma
cópia própria. Regressão real: por um tempo, integracao_manha.py tinha sua
própria cópia da lógica de h2h (desatualizada, nunca reconhecia os rótulos
reais '1'/'X'/'2' da Betano), então o 1X2 nunca entrava nos registros
automáticos mesmo depois do mapeamento ter sido corrigido em catalogo.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from catalogo import combinar_modelo_e_odds


def test_combinar_modelo_e_odds_com_evento_no_formato_que_integracao_manha_monta():
    """Simula exatamente o dict que processar_foto_manha_async constrói a
    partir de um EventOdds recém-coletado (ver 'odds_do_evento' em
    integracao_manha.py), com rótulos REAIS da Betano ('1'/'X'/'2')."""
    probs = {"1X2": {"Casa": 0.5, "Empate": 0.25, "Fora": 0.25},
             "Ambas marcam": {"Sim": 0.55, "Não": 0.45}}
    odds_do_evento = {
        "casa_coletado": "Flamengo", "fora_coletado": "Palmeiras",
        "mercados": {
            "h2h": {"1": 2.1, "X": 3.2, "2": 3.5},
            "btts": {"Sim": 1.85, "Não": 1.95},
        },
    }
    candidatos = combinar_modelo_e_odds(probs, odds_do_evento, "Flamengo RJ", "Palmeiras")
    mercados = {(c["mercado"], c["selecao"]) for c in candidatos}
    assert ("1X2", "Casa") in mercados
    assert ("1X2", "Empate") in mercados
    assert ("1X2", "Fora") in mercados
    assert ("Ambas marcam", "Sim") in mercados
