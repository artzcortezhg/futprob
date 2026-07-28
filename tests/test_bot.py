# -*- coding: utf-8 -*-
"""Testes das funções puras do bot (bot.py): parsing de odds, resolução de
times (com acento/parcial/ambiguidade), cruzamento modelo+odds coletadas,
mapeamento pra mercados e cálculo de relatório. Nada aqui sobe o Telegram
de verdade."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bot import (
    parsear_odds_coladas, resolver_time, resolver_time_ambiguo, normalizar_texto,
    candidatos_time, montar_candidatos, probs_modelo_de_linhas, familia_mercado,
    calcular_report, buscar_fixture_real, resolver_fixture_para_liga, buscar_odds_coletadas_para_fixture,
    combinar_modelo_e_odds, descobrir_jogos_do_dia,
)


def test_parsear_odds_1x2_simples():
    validos, invalidos = parsear_odds_coladas("1=2.05 X=3.40 2=3.75")
    assert validos == {("1X2", "Casa"): 2.05, ("1X2", "Empate"): 3.40, ("1X2", "Fora"): 3.75}
    assert invalidos == []


def test_parsear_odds_over_under_gols():
    validos, _ = parsear_odds_coladas("O2.5=2.10 U2.5=1.75")
    assert validos == {("Over/Under 2.5", "Over"): 2.10, ("Over/Under 2.5", "Under"): 1.75}


def test_parsear_odds_escanteios_e_cartoes():
    validos, _ = parsear_odds_coladas("OC9.5=1.90 UC9.5=1.85 OA3.5=1.95 UA3.5=1.80")
    assert validos[("Escanteios Over/Under 9.5", "Over")] == 1.90
    assert validos[("Cartões Over/Under 3.5", "Under")] == 1.80


def test_parsear_odds_btts_e_dupla_chance():
    validos, _ = parsear_odds_coladas("BTTS=1.85 NBTTS=1.95 DC1X=1.25 DC12=1.10 DCX2=1.40")
    assert validos[("Ambas marcam", "Sim")] == 1.85
    assert validos[("Dupla chance", "12 (casa ou fora)")] == 1.10


def test_parsear_odds_aceita_virgula_decimal():
    validos, _ = parsear_odds_coladas("1=2,05")
    assert validos == {("1X2", "Casa"): 2.05}


def test_parsear_odds_token_nao_reconhecido_vai_para_invalidos():
    validos, invalidos = parsear_odds_coladas("1=2.05 ZZZ=abc futebol")
    assert ("1X2", "Casa") in validos
    assert "ZZZ=abc" in invalidos
    assert "futebol" in invalidos


def test_parsear_odds_vazio_retorna_vazio():
    validos, invalidos = parsear_odds_coladas("")
    assert validos == {} and invalidos == []


def test_parsear_odds_rejeita_odd_menor_ou_igual_a_um():
    validos, invalidos = parsear_odds_coladas("1=0.95")
    assert validos == {}
    assert "1=0.95" in invalidos


def test_normalizar_texto_remove_acento_e_baixa_caixa():
    assert normalizar_texto("Grêmio") == "gremio"
    assert normalizar_texto("São Paulo") == "sao paulo"
    assert normalizar_texto("ATLÉTICO-MG") == "atletico-mg"


def test_resolver_time_encontra_nome_exato():
    times = {"Premier League": ["Arsenal", "Liverpool"], "La Liga": ["Barcelona"]}
    assert resolver_time("Arsenal", times) == ("Premier League", "Arsenal")


def test_resolver_time_encontra_nome_aproximado():
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras"]}
    resolvido = resolver_time("Flamengo", times)
    assert resolvido == ("brasileirao", "Flamengo RJ")


def test_resolver_time_aceita_busca_sem_acento():
    times = {"brasileirao": ["Gremio", "Sao Paulo", "Atletico-MG"]}
    assert resolver_time("Grêmio", times) == ("brasileirao", "Gremio")
    assert resolver_time("São Paulo", times) == ("brasileirao", "Sao Paulo")


def test_resolver_time_nao_acha_nada_muito_diferente():
    times = {"Premier League": ["Arsenal"]}
    assert resolver_time("Corinthians", times) is None


def test_resolver_time_ambiguo_status_ok():
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras"]}
    r = resolver_time_ambiguo("Flamengo", times)
    assert r == {"status": "ok", "liga": "brasileirao", "time": "Flamengo RJ"}


def test_resolver_time_ambiguo_status_nao_encontrado():
    times = {"Premier League": ["Arsenal"]}
    assert resolver_time_ambiguo("Corinthians", times) == {"status": "nao_encontrado"}


def test_resolver_time_ambiguo_sugere_quando_scores_proximos():
    # dois times com nomes bem parecidos entre si e com o termo buscado
    times = {"brasileirao": ["Sport Recife", "Sport Huelva"]}
    r = resolver_time_ambiguo("Sport", times)
    assert r["status"] == "ambiguo"
    nomes = {t for _, t in r["opcoes"]}
    assert "Sport Recife" in nomes and "Sport Huelva" in nomes


def test_candidatos_time_ordenado_por_score():
    times = {"brasileirao": ["Flamengo RJ", "Corinthians", "Palmeiras"]}
    cands = candidatos_time("Flamengo", times, top_n=3)
    assert cands[0][1] == "Flamengo RJ"
    assert cands[0][2] >= cands[1][2] >= cands[2][2]


def test_montar_candidatos_cruza_odds_com_probs():
    odds = {("1X2", "Casa"): 2.2, ("1X2", "Empate"): 3.0}
    probs = {"1X2": {"Casa": 0.5, "Empate": 0.25, "Fora": 0.25}}
    candidatos, sem_prob = montar_candidatos(odds, probs)
    assert len(candidatos) == 2
    assert sem_prob == []
    casa = next(c for c in candidatos if c["selecao"] == "Casa")
    assert abs(casa["ev"] - (0.5 * 2.2 - 1.0)) < 1e-9


def test_montar_candidatos_sem_prob_disponivel():
    odds = {("Escanteios Over/Under 9.5", "Over"): 1.9}
    probs = {"1X2": {"Casa": 0.5}}
    candidatos, sem_prob = montar_candidatos(odds, probs)
    assert candidatos == []
    assert "Escanteios Over/Under 9.5/Over" in sem_prob


def test_probs_modelo_de_linhas_agrupa_por_mercado():
    linhas = [("1X2", "Casa", 0.5), ("1X2", "Empate", 0.3), ("Ambas marcam", "Sim", 0.6)]
    probs = probs_modelo_de_linhas(linhas)
    assert probs == {"1X2": {"Casa": 0.5, "Empate": 0.3}, "Ambas marcam": {"Sim": 0.6}}


def test_familia_mercado():
    assert familia_mercado("1X2") == "1X2"
    assert familia_mercado("Over/Under 2.5") == "Over/Under gols"
    assert familia_mercado("Escanteios Over/Under 9.5") == "Escanteios"
    assert familia_mercado("Cartões Over/Under 3.5") == "Cartões"
    assert familia_mercado("Ambas marcam") == "Ambas marcam"
    assert familia_mercado("Dupla chance") == "Dupla chance"


def test_calcular_report_vazio():
    assert calcular_report([])["n"] == 0


def test_calcular_report_ic_cruza_zero_quando_poucos_dados_mistos():
    registros = [
        {"mercado": "1X2", "clv": 0.05, "resultado": "ganhou", "odd_registrada": 2.0},
        {"mercado": "1X2", "clv": -0.05, "resultado": "perdeu", "odd_registrada": 2.0},
    ]
    r = calcular_report(registros)
    assert r["n"] == 2
    assert "ruído" in r["frase"]


def test_calcular_report_agrupa_por_familia():
    registros = [
        {"mercado": "1X2", "clv": 0.05, "resultado": "ganhou", "odd_registrada": 2.0},
        {"mercado": "Escanteios Over/Under 9.5", "clv": 0.02, "resultado": "perdeu", "odd_registrada": 1.9},
    ]
    r = calcular_report(registros)
    assert "1X2" in r["clv_por_familia"]
    assert "Escanteios" in r["clv_por_familia"]


def _criar_odds_coletadas(caminho, linhas):
    with sqlite3.connect(caminho) as conn:
        conn.execute("""CREATE TABLE odds_coletadas (
            id INTEGER PRIMARY KEY, coletado_em TEXT, tipo_foto TEXT, casa_apostas TEXT,
            time_casa_coletado TEXT, time_fora_coletado TEXT, commence_time TEXT,
            mercado TEXT, selecao TEXT, odd REAL
        )""")
        conn.executemany(
            "INSERT INTO odds_coletadas (coletado_em, tipo_foto, casa_apostas, time_casa_coletado, "
            "time_fora_coletado, commence_time, mercado, selecao, odd) VALUES (?,?,?,?,?,?,?,?,?)",
            linhas,
        )
        conn.commit()


def _futuro(horas: float = 24) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(hours=horas)).isoformat()


def test_buscar_fixture_real_sem_tabela_retorna_nao_encontrado(tmp_path):
    caminho = tmp_path / "vazio.sqlite"
    with sqlite3.connect(caminho) as conn:
        conn.execute("CREATE TABLE dummy (x INTEGER)")
        conn.commit()
    assert buscar_fixture_real("Arsenal", caminho) == {"status": "nao_encontrado"}


def test_buscar_fixture_real_acha_pelo_nome_cru_coletado(tmp_path):
    """Busca pelo nome CRU coletado, não pelo roster — encontra o jogo
    real mesmo antes de qualquer resolução pro modelo."""
    caminho = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", _futuro(), "h2h", "1", 2.1),
    ])
    resultado = buscar_fixture_real("Flamengo", caminho)
    assert resultado["status"] == "ok"
    assert resultado["casa_coletado"] == "Flamengo"
    assert resultado["fora_coletado"] == "Palmeiras"


def test_buscar_fixture_real_ignora_jogo_ja_comecado(tmp_path):
    caminho = tmp_path / "odds.sqlite"
    from datetime import datetime, timedelta, timezone
    passado = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", passado, "h2h", "1", 2.1),
    ])
    assert buscar_fixture_real("Flamengo", caminho) == {"status": "nao_encontrado"}


def test_buscar_fixture_real_nao_cai_pra_confronto_antigo_stale(tmp_path):
    """Regressão do bug real: 'Ponte Preta' bate no roster da Série A, e a
    busca ANTIGA (buscar_proxima_partida) procurava por qualquer coleta
    histórica em que 'Ponte Preta' aparecesse resolvido pro modelo,
    achando um confronto velho/stale ('Ponte Preta x Ceará') em vez do
    jogo real de hoje (Ponte Preta x Athletic Club, Série B, sem modelo).
    A busca nova encontra sempre o jogo REAL mais próximo pelo nome cru —
    aqui simulado com só o confronto real (sem coleta antiga nenhuma pra
    cair como fallback)."""
    caminho = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T18:41:00", "manha", "betano", "Ponte Preta", "Athletic Club MG", _futuro(), "h2h", "1", 2.0),
    ])
    resultado = buscar_fixture_real("ponte preta", caminho)
    assert resultado["status"] == "ok"
    assert resultado["casa_coletado"] == "Ponte Preta"
    assert resultado["fora_coletado"] == "Athletic Club MG"  # o adversário REAL, não "Ceará"


def test_resolver_fixture_para_liga_ok_quando_os_dois_times_existem_na_mesma_liga():
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras"]}
    assert resolver_fixture_para_liga("Flamengo", "Palmeiras", times) == ("brasileirao", "Flamengo RJ", "Palmeiras")


def test_resolver_fixture_para_liga_none_quando_adversario_sem_modelo():
    """A regra de pertencimento: mesmo o time da casa existindo no roster,
    se o adversário real (ex.: de outra divisão) não existir em NENHUMA
    liga modelada, o confronto inteiro não vira previsão."""
    times = {"brasileirao": ["Ponte Preta", "Ceara"]}
    assert resolver_fixture_para_liga("Ponte Preta", "Athletic Club MG", times) is None


def test_resolver_fixture_para_liga_none_quando_ligas_diferentes():
    times = {"brasileirao": ["Flamengo RJ"], "mls": ["Inter Miami"]}
    assert resolver_fixture_para_liga("Flamengo", "Inter Miami", times) is None


def test_buscar_odds_coletadas_para_fixture_encontra_e_agrupa(tmp_path):
    caminho = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", "2026-07-29T22:30:00", "h2h", "Flamengo", 2.1),
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", "2026-07-29T22:30:00", "h2h", "Empate", 3.2),
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", "2026-07-29T22:30:00", "h2h", "Palmeiras", 3.6),
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", "2026-07-29T22:30:00", "ou_2.5_goals", "Over", 1.9),
    ])
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras"]}
    odds = buscar_odds_coletadas_para_fixture("Flamengo RJ", "Palmeiras", times, caminho)
    assert odds is not None
    assert odds["casa_coletado"] == "Flamengo"
    assert set(odds["mercados"]["h2h"].keys()) == {"Flamengo", "Empate", "Palmeiras"}
    assert odds["mercados"]["ou_2.5_goals"]["Over"] == 1.9


def test_buscar_odds_coletadas_para_fixture_retorna_none_se_nao_coletado(tmp_path):
    caminho = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T09:00:00", "manha", "betano", "Corinthians", "Santos", "2026-07-29T22:30:00", "h2h", "Corinthians", 1.8),
    ])
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras", "Corinthians", "Santos"]}
    assert buscar_odds_coletadas_para_fixture("Flamengo RJ", "Palmeiras", times, caminho) is None


def test_combinar_modelo_e_odds_mapeia_h2h_corretamente():
    probs = {"1X2": {"Casa": 0.5, "Empate": 0.25, "Fora": 0.25}}
    odds = {"casa_coletado": "Flamengo", "fora_coletado": "Palmeiras",
            "mercados": {"h2h": {"Flamengo": 2.1, "Empate": 3.2, "Palmeiras": 3.6}}}
    candidatos = combinar_modelo_e_odds(probs, odds, "Flamengo RJ", "Palmeiras")
    selecoes = {c["selecao"] for c in candidatos}
    assert selecoes == {"Casa", "Empate", "Fora"}


def test_combinar_modelo_e_odds_mapeia_over_under():
    probs = {"Over/Under 2.5": {"Over": 0.55, "Under": 0.45}}
    odds = {"casa_coletado": "A", "fora_coletado": "B", "mercados": {"ou_2.5_goals": {"Over": 2.0, "Under": 1.8}}}
    candidatos = combinar_modelo_e_odds(probs, odds, "A", "B")
    assert len(candidatos) == 2
    assert {c["mercado"] for c in candidatos} == {"Over/Under 2.5"}


def test_descobrir_jogos_do_dia_filtra_por_data(tmp_path):
    import datetime
    caminho = tmp_path / "odds.sqlite"
    _criar_odds_coletadas(caminho, [
        ("2026-07-28T09:00:00", "manha", "betano", "Flamengo", "Palmeiras", "2026-07-28T22:30:00+00:00", "h2h", "Flamengo", 2.1),
        ("2026-07-28T09:00:00", "manha", "betano", "Corinthians", "Santos", "2026-08-05T22:30:00+00:00", "h2h", "Corinthians", 1.8),
    ])
    times = {"brasileirao": ["Flamengo RJ", "Palmeiras", "Corinthians", "Santos"]}
    jogos = descobrir_jogos_do_dia(datetime.date(2026, 7, 28), times, caminho)
    assert len(jogos) == 1
    assert jogos[0]["casa"] == "Flamengo RJ"
