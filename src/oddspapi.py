# -*- coding: utf-8 -*-
"""
Integração com a OddsPapi (api.oddspapi.io/v4) — fonte independente de
odds (Pinnacle) pra Brasileirão Série A/B e MLS. Gatilho SEMPRE manual
(botão no painel ou /oddspapi no bot) — nunca automático/agendado, pra não
gastar cota sozinho.

Cota do plano gratuito: 250 usos. A documentação não confirma renovação
mensal (não há campo de "reset" na resposta de /account — pode ser cota
total da conta, não mensal). Cada chamada a um endpoint FATURÁVEL conta 1
uso, não importa o tamanho da resposta; /account e /historical-odds são
sempre grátis. Ver registrar_uso_oddspapi (painel_db.py) pro contador local.

Cobertura confirmada ao vivo em 2026-07-28 (ver também a investigação
anterior, Bloco 5):
- Pinnacle tem odds pra Brasileirão Série A (tournamentId 325), Série B
  (390 — achado novo, não estava confirmado antes) e MLS (242).
- Só mercados de GOLS (1X2, over/under, ambas marcam, dupla chance) — o
  TIPO de mercado "totals-corners"/"totals-bookings" existe na legenda
  geral da OddsPapi (pra futebol em geral), mas não foi encontrado nas
  fixtures reais dessas 3 ligas — provável baixa prioridade/liquidez pra
  essas competições specificamente.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("futprob.oddspapi")

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB_PADRAO = RAIZ / "db" / "previsoes.sqlite"

BASE_URL = "https://api.oddspapi.io/v4"
LIMITE_USOS = 250
BOOKMAKER = "pinnacle"

# tournamentId confirmados por investigação ao vivo (GET /tournaments)
TOURNAMENT_IDS = {
    "brasileirao": 325,
    "brasileirao_b": 390,
    "mls": 242,
}

# legenda ESTÁTICA dos mercados de gols que nos interessam — decodificada
# ao vivo via GET /v4/markets (sportId=10/futebol, period=fulltime) em
# 2026-07-28. marketId -> (mercado_futprob, {outcomeId: selecao_futprob})
MERCADOS_RELEVANTES: dict[int, tuple[str, dict[int, str]]] = {
    101: ("1X2", {101: "Casa", 102: "Empate", 103: "Fora"}),
    104: ("Ambas marcam", {104: "Sim", 105: "Não"}),
    101902: ("Dupla chance", {101902: "1X (casa ou empate)", 101903: "12 (casa ou fora)", 101904: "X2 (empate ou fora)"}),
    106: ("Over/Under 0.5", {106: "Over", 107: "Under"}),
    108: ("Over/Under 1.5", {108: "Over", 109: "Under"}),
    1010: ("Over/Under 2.5", {1010: "Over", 1011: "Under"}),
    1012: ("Over/Under 3.5", {1012: "Over", 1013: "Under"}),
    1014: ("Over/Under 4.5", {1014: "Over", 1015: "Under"}),
    1016: ("Over/Under 5.5", {1016: "Over", 1017: "Under"}),
}

SQL_CRIAR_TABELA_PARTICIPANTES = """
CREATE TABLE IF NOT EXISTS oddspapi_participantes (
    participant_id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL
);
"""


def uso_atual(caminho_db: Path = CAMINHO_DB_PADRAO) -> int:
    with sqlite3.connect(caminho_db) as conn:
        try:
            return conn.execute("SELECT COUNT(*) FROM oddspapi_uso").fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def _chamar(endpoint: str, params: dict, caminho_db: Path) -> dict | list:
    """Chama um endpoint FATURÁVEL da OddsPapi (1 uso) e registra o
    consumo em oddspapi_uso (sucesso ou falha) antes de devolver."""
    from painel_db import registrar_uso_oddspapi

    api_key = os.environ.get("ODDSPAPI_KEY")
    if not api_key:
        raise RuntimeError("ODDSPAPI_KEY não configurada no .env")
    query = "&".join(f"{k}={v}" for k, v in {**params, "apiKey": api_key}.items())
    url = f"{BASE_URL}/{endpoint}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "futprob/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            corpo = json.loads(resp.read().decode("utf-8"))
        registrar_uso_oddspapi(caminho_db, endpoint, sucesso=True)
        return corpo
    except Exception as exc:
        registrar_uso_oddspapi(caminho_db, endpoint, sucesso=False, observacao=str(exc))
        raise


def _nomes_participantes(ids_necessarios: set[int], caminho_db: Path) -> dict[int, str]:
    """Resolve participant_id -> nome usando o cache local primeiro; só
    chama /v4/participants (faturável) se algum id realmente faltar —
    nomes de time são estáveis, então essa chamada tende a acontecer no
    máximo uma vez na vida do banco."""
    with sqlite3.connect(caminho_db) as conn:
        conn.executescript(SQL_CRIAR_TABELA_PARTICIPANTES)
        conn.commit()
        rows = conn.execute(
            f"SELECT participant_id, nome FROM oddspapi_participantes WHERE participant_id IN "
            f"({','.join('?' * len(ids_necessarios))})",
            list(ids_necessarios),
        ).fetchall() if ids_necessarios else []
    encontrados = {pid: nome for pid, nome in rows}
    faltando = ids_necessarios - set(encontrados)
    if not faltando:
        return encontrados

    logger.info(f"cache de participantes sem {len(faltando)} id(s) — buscando na OddsPapi (1 uso)")
    dados = _chamar("participants", {"sportId": 10}, caminho_db)
    if isinstance(dados, dict):
        with sqlite3.connect(caminho_db) as conn:
            conn.executemany(
                "INSERT INTO oddspapi_participantes (participant_id, nome) VALUES (?, ?) "
                "ON CONFLICT(participant_id) DO UPDATE SET nome=excluded.nome",
                [(int(pid), nome) for pid, nome in dados.items()],
            )
            conn.commit()
        for pid in faltando:
            if str(pid) in dados:
                encontrados[pid] = dados[str(pid)]
    return encontrados


def buscar_melhores_odds(caminho_db: Path = CAMINHO_DB_PADRAO) -> dict:
    """Busca as odds Pinnacle atuais pra Brasileirão A/B e MLS (1 chamada
    faturável a odds-by-tournaments, + eventualmente 1 a participants na
    primeira vez). Retorna:
    {"sucesso": True, "jogos": [{"liga":, "casa":, "fora":, "commence_time":,
     "mercados": [{"mercado":, "selecao":, "odd":}, ...]}, ...]}
    ou {"sucesso": False, "erro": "..."} — nunca lança exceção."""
    gasto = uso_atual(caminho_db)
    if gasto >= LIMITE_USOS:
        return {"sucesso": False, "erro": f"cota estimada esgotada ({gasto}/{LIMITE_USOS}) — não chamei a API"}

    ids_torneios = ",".join(str(t) for t in TOURNAMENT_IDS.values())
    liga_por_tournament_id = {v: k for k, v in TOURNAMENT_IDS.items()}

    try:
        fixtures = _chamar("odds-by-tournaments", {"bookmaker": BOOKMAKER, "tournamentIds": ids_torneios}, caminho_db)
    except Exception as exc:
        logger.exception("falha ao buscar odds na OddsPapi")
        return {"sucesso": False, "erro": str(exc)}

    if not isinstance(fixtures, list):
        return {"sucesso": False, "erro": "resposta em formato inesperado (não é lista de fixtures)"}

    ids_participantes = set()
    for fx in fixtures:
        ids_participantes.add(fx.get("participant1Id"))
        ids_participantes.add(fx.get("participant2Id"))
    ids_participantes.discard(None)
    nomes = _nomes_participantes(ids_participantes, caminho_db)

    jogos = []
    for fx in fixtures:
        pinnacle = (fx.get("bookmakerOdds") or {}).get(BOOKMAKER)
        if not pinnacle or not pinnacle.get("markets"):
            continue
        casa = nomes.get(fx.get("participant1Id"), f"time#{fx.get('participant1Id')}")
        fora = nomes.get(fx.get("participant2Id"), f"time#{fx.get('participant2Id')}")

        mercados_encontrados = []
        for market_id_str, mkt in pinnacle["markets"].items():
            legenda = MERCADOS_RELEVANTES.get(int(market_id_str))
            if not legenda:
                continue
            mercado_nome, mapa_selecao = legenda
            for outcome_id_str, outcome in (mkt.get("outcomes") or {}).items():
                selecao = mapa_selecao.get(int(outcome_id_str))
                if not selecao:
                    continue
                jogadores = outcome.get("players") or {}
                preco = (jogadores.get("0") or {}).get("price")
                if preco:
                    mercados_encontrados.append({"mercado": mercado_nome, "selecao": selecao, "odd": preco})

        if mercados_encontrados:
            jogos.append({
                "liga": liga_por_tournament_id.get(fx.get("tournamentId"), "?"),
                "casa": casa, "fora": fora, "commence_time": fx.get("startTime"),
                "mercados": mercados_encontrados,
            })

    return {"sucesso": True, "jogos": jogos, "uso_apos": uso_atual(caminho_db)}
