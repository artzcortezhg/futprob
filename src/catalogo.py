# -*- coding: utf-8 -*-
"""
Catálogo do dia — fonte ÚNICA de verdade sobre quais jogos são reais hoje e
amanhã, e sobre o card completo (todas as famílias de mercado) de cada um.
Usado tanto pelo bot (src/bot.py) quanto pelo painel (src/dashboard.py), pra
nunca divergirem sobre "o que é um jogo de hoje" nem duplicar a lógica de
casar modelo com odds coletadas.

"Jogos do dia" vem SEMPRE da coleta mais recente da Betano (odds_coletadas),
nunca de previsões avulsas salvas no banco em algum momento passado — ver o
bug de fase anterior em que o painel mostrava previsões de teste (ex.:
Premier League fora de temporada) como se fossem "jogos de hoje".
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from resolucao_times import resolver_time, score_nomes
from predict import prever, probs_modelo_de_linhas, LIGAS_SEM_ESTATISTICAS_EXTRAS

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB_PADRAO = RAIZ / "db" / "previsoes.sqlite"
FUSO_BR = ZoneInfo("America/Sao_Paulo")

# grupos de exibição do card completo (Bloco "reforma do painel") — cada
# grupo vira uma aba/bloco no card do jogo
GRUPOS_CARD = ["1X2 e dupla chance", "Over/Under gols", "Ambas marcam", "Escanteios", "Cartões e faltas"]

# mapeia o market_key do coletor (Betano) para (mercado, selecao) do futprob
_MARKET_KEY_PARA_FUTPROB = {
    "h2h": [("1X2", "Casa", 0), ("1X2", "Empate", 1), ("1X2", "Fora", 2)],
    "btts": [("Ambas marcam", "Sim", "sim"), ("Ambas marcam", "Não", "nao")],
    "double_chance": [
        ("Dupla chance", "1X (casa ou empate)", "1x"),
        ("Dupla chance", "12 (casa ou fora)", "12"),
        ("Dupla chance", "X2 (empate ou fora)", "x2"),
    ],
}


# rótulos reais observados na Betano pra over/under: em português
# ("Mais de 2.5"/"Menos de 2.5"), não "Over"/"Under" — mantemos os rótulos
# em inglês também por robustez (outra fonte/locale eventual)
_OVER_LABELS = ("over", "acima", "mais de")
_UNDER_LABELS = ("under", "abaixo", "menos de")

# a Betano identifica a seleção do 1X2 por "1"/"X"/"2" (visto ao vivo), não
# pelo nome do time — mas aceitamos nome de time/"Empate" também por
# robustez, caso outra fonte/formato apareça
_H2H_ABREV = {"1": "Casa", "x": "Empate", "2": "Fora"}


def _selecao_over_under(nome_odd: str) -> str | None:
    n = nome_odd.strip().lower()
    if any(rotulo in n for rotulo in _OVER_LABELS):
        return "Over"
    if any(rotulo in n for rotulo in _UNDER_LABELS):
        return "Under"
    return None


def _selecao_h2h(nome_odd: str, casa_coletado: str, fora_coletado: str) -> str | None:
    bruto = nome_odd.strip()
    chave = bruto.lower()
    if chave in _H2H_ABREV:
        return _H2H_ABREV[chave]
    if bruto == casa_coletado:
        return "Casa"
    if bruto == fora_coletado:
        return "Fora"
    if chave in ("empate", "draw"):
        return "Empate"
    return None


def _mercados_para_jogo(mercado_key: str, outcomes: dict[str, float]) -> list[tuple[str, str, float]]:
    """Converte {nome_outcome_betano: odd} de um market_key coletado em
    [(mercado_futprob, selecao_futprob, odd), ...]."""
    resultado = []
    if mercado_key.startswith("ou_"):
        partes = mercado_key.split("_")
        linha, stat = partes[1], partes[2]
        nome_mercado = {"goals": "Over/Under", "corners": "Escanteios Over/Under", "cards": "Cartões Over/Under"}.get(stat)
        if nome_mercado is None:
            return []
        for nome_odd, odd in outcomes.items():
            selecao = _selecao_over_under(nome_odd)
            if selecao:
                resultado.append((f"{nome_mercado} {linha}", selecao, odd))
        return resultado

    if mercado_key == "btts":
        for nome_odd, odd in outcomes.items():
            if nome_odd.lower() in ("sim", "yes", "1"):
                resultado.append(("Ambas marcam", "Sim", odd))
            elif nome_odd.lower() in ("não", "nao", "no", "0"):
                resultado.append(("Ambas marcam", "Não", odd))
        return resultado

    return []  # h2h e double_chance exigem nome do time (casa/fora), tratados à parte


def hoje_br() -> date:
    return datetime.now(FUSO_BR).date()


def familia_mercado(mercado: str) -> str:
    """Família pra fins de relatório/CLV (report do bot) — granularidade
    fina, NÃO é o mesmo agrupamento visual do card (ver `_grupo_exibicao`)."""
    m = mercado.lower()
    if m.startswith("1x2"):
        return "1X2"
    if m.startswith("dupla chance"):
        return "Dupla chance"
    if m.startswith("ambas marcam"):
        return "Ambas marcam"
    if m.startswith("escanteios"):
        return "Escanteios"
    if m.startswith(("cartões", "cartoes")):
        return "Cartões"
    if m.startswith("faltas"):
        return "Faltas"
    if m.startswith("over/under"):
        return "Over/Under gols"
    return "Outros"


def _grupo_exibicao(mercado: str) -> str | None:
    """Agrupamento visual do card completo (item 3 da reforma do painel).
    Retorna None para mercados que não aparecem no card (placar exato,
    handicap europeu — cabem mal num card e não são o foco aqui)."""
    m = mercado.lower()
    if m.startswith("1x2") or m.startswith("dupla chance"):
        return "1X2 e dupla chance"
    if m.startswith("over/under"):
        return "Over/Under gols"
    if m.startswith("ambas marcam"):
        return "Ambas marcam"
    if m.startswith("escanteios"):
        return "Escanteios"
    if m.startswith(("cartões", "cartoes", "faltas")):
        return "Cartões e faltas"
    return None


def _snapshot_mais_recente(caminho_db: Path) -> pd.DataFrame:
    with sqlite3.connect(caminho_db) as conn:
        try:
            return pd.read_sql_query(
                "SELECT DISTINCT time_casa_coletado, time_fora_coletado, commence_time FROM odds_coletadas "
                "WHERE coletado_em = (SELECT MAX(coletado_em) FROM odds_coletadas)",
                conn,
            )
        except Exception:
            return pd.DataFrame(columns=["time_casa_coletado", "time_fora_coletado", "commence_time"])


def resolver_fixture_para_liga(casa_cru: str, fora_cru: str,
                                times_por_liga: dict[str, list[str]]) -> tuple[str, str, str] | None:
    """REGRA DE PERTENCIMENTO: um confronto só casa com o modelo de uma
    liga se OS DOIS times existirem na lista fechada de times daquela liga
    (vinda dos dados de treino) — nunca um jogo com só um lado resolvido,
    nem os dois resolvidos em ligas diferentes. Retorna (liga, casa_interno,
    fora_interno) ou None ("campeonato sem modelo — sem previsão").

    Isso sozinho não basta pra evitar contaminação entre divisões: um time
    que já jogou a Série A em alguma temporada do histórico (ex.: Ponte
    Preta) continua "existindo" no roster mesmo jogando a Série B agora.
    A defesa real contra isso é o adversário: se o adversário REAL de hoje
    (ex.: Athletic Club) não está em NENHUMA lista fechada, a regra rejeita
    o confronto inteiro — é assim que se evita gerar previsão pra um jogo
    de outra divisão."""
    casa_resolvido = resolver_time(casa_cru, times_por_liga)
    fora_resolvido = resolver_time(fora_cru, times_por_liga)
    if not casa_resolvido or not fora_resolvido or casa_resolvido[0] != fora_resolvido[0]:
        return None
    return (casa_resolvido[0], casa_resolvido[1], fora_resolvido[1])


def descobrir_jogos_do_dia(dia: date, times_por_liga: dict[str, list[str]], caminho_db: Path = CAMINHO_DB_PADRAO) -> list[dict]:
    """A partir da coleta mais recente, resolve e filtra os jogos cujo
    commence_time cai no `dia` informado (horário de Brasília). Nomes
    resolvidos = nomes internos do futprob, nunca a grafia crua da Betano."""
    df = _snapshot_mais_recente(caminho_db)
    jogos = []
    for _, row in df.iterrows():
        try:
            dt = pd.Timestamp(row["commence_time"])
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            dt_br = dt.tz_convert(FUSO_BR)
        except Exception:
            continue
        if dt_br.date() != dia:
            continue
        resolvido = resolver_fixture_para_liga(row["time_casa_coletado"], row["time_fora_coletado"], times_por_liga)
        if resolvido:
            liga, casa, fora = resolvido
            jogos.append({"liga": liga, "casa": casa, "fora": fora, "commence_time": dt.isoformat()})
    return jogos


def buscar_fixture_real(nome_busca: str, caminho_db: Path = CAMINHO_DB_PADRAO,
                         limiar: float = 0.55, margem_ambiguidade: float = 0.08) -> dict:
    """Busca o time pelo NOME CRU coletado (não pelo roster do modelo) nas
    fixtures REAIS mais próximas (hoje -> futuro) da coleta MAIS RECENTE —
    nunca em coletas antigas. É essa ordem que importa: antes achava o time
    no roster do modelo e só DEPOIS procurava um jogo dele em qualquer
    coleta histórica, o que podia devolver um confronto ANTIGO e stale
    (ex.: 'Ponte Preta' bateu no roster da Série A e a busca antiga achou
    uma coleta velha com 'Ponte Preta x Ceará', quando o jogo real de hoje
    é 'Ponte Preta x Athletic Club' na Série B). Agora a busca sempre acha
    o jogo real primeiro; resolver pro modelo (resolver_fixture_para_liga)
    só acontece depois, e só decide SE dá pra prever, nunca QUAL é o jogo.

    Retorna:
      {"status": "ok", "casa_coletado":.., "fora_coletado":.., "commence_time":..}
      {"status": "ambiguo", "opcoes": [{"casa":.., "fora":.., "commence_time":..}, ...]}
      {"status": "nao_encontrado"}
    """
    with sqlite3.connect(caminho_db) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT time_casa_coletado, time_fora_coletado, commence_time FROM odds_coletadas "
                "WHERE coletado_em = (SELECT MAX(coletado_em) FROM odds_coletadas)",
                conn,
            )
        except Exception:
            return {"status": "nao_encontrado"}
    if df.empty:
        return {"status": "nao_encontrado"}

    agora = datetime.now(timezone.utc)
    candidatos = []
    for _, row in df.iterrows():
        try:
            dt = pd.Timestamp(row["commence_time"])
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
        except Exception:
            continue
        if dt.to_pydatetime() < agora:
            continue  # só pré-jogo: fixture já começada/passada não conta
        melhor_score = max(score_nomes(nome_busca, row["time_casa_coletado"]), score_nomes(nome_busca, row["time_fora_coletado"]))
        if melhor_score >= limiar:
            candidatos.append((melhor_score, dt, row))

    if not candidatos:
        return {"status": "nao_encontrado"}

    candidatos.sort(key=lambda c: (-c[0], c[1]))
    melhor_score = candidatos[0][0]
    proximos = [c for c in candidatos if melhor_score - c[0] <= margem_ambiguidade]

    vistos: set[tuple[str, str]] = set()
    unicos = []
    for c in proximos:
        chave = (c[2]["time_casa_coletado"], c[2]["time_fora_coletado"])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(c)

    if len(unicos) > 1:
        return {"status": "ambiguo", "opcoes": [
            {"casa": c[2]["time_casa_coletado"], "fora": c[2]["time_fora_coletado"], "commence_time": c[1].isoformat()}
            for c in unicos
        ]}

    _, dt, row = unicos[0]
    return {"status": "ok", "casa_coletado": row["time_casa_coletado"], "fora_coletado": row["time_fora_coletado"],
            "commence_time": dt.isoformat()}


def buscar_odds_coletadas_para_fixture(time_casa: str, time_fora: str, times_por_liga: dict[str, list[str]],
                                        caminho_db: Path = CAMINHO_DB_PADRAO) -> dict | None:
    """Procura, na coleta mais recente, as odds já capturadas pra esse
    confronto específico (nomes internos). Retorna None se ainda não há
    coleta pra esse jogo; senão {"casa_coletado":..., "fora_coletado":...,
    "mercados": {mercado_key: {outcome: odd}}}."""
    with sqlite3.connect(caminho_db) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM odds_coletadas WHERE coletado_em = (SELECT MAX(coletado_em) FROM odds_coletadas)",
                conn,
            )
        except Exception:
            return None
    if df.empty:
        return None

    pares = df[["time_casa_coletado", "time_fora_coletado"]].drop_duplicates()
    for _, par in pares.iterrows():
        casa_resolvido = resolver_time(par["time_casa_coletado"], times_por_liga)
        fora_resolvido = resolver_time(par["time_fora_coletado"], times_por_liga)
        if casa_resolvido and fora_resolvido and casa_resolvido[1] == time_casa and fora_resolvido[1] == time_fora:
            linhas = df[(df["time_casa_coletado"] == par["time_casa_coletado"]) & (df["time_fora_coletado"] == par["time_fora_coletado"])]
            mercados: dict[str, dict[str, float]] = {}
            for _, linha in linhas.iterrows():
                mercados.setdefault(linha["mercado"], {})[linha["selecao"]] = linha["odd"]
            return {"casa_coletado": par["time_casa_coletado"], "fora_coletado": par["time_fora_coletado"], "mercados": mercados}
    return None


def combinar_modelo_e_odds(probs_modelo: dict, odds: dict, time_casa: str, time_fora: str) -> list[dict]:
    """Cruza as probabilidades do modelo com as odds já coletadas pra esse
    jogo específico, retornando os candidatos (mercado/selecao/prob/odd/ev)
    prontos pros guarda-corpos."""
    candidatos = []
    for mercado_key, outcomes in odds["mercados"].items():
        if mercado_key == "h2h":
            for nome_odd, odd in outcomes.items():
                selecao = _selecao_h2h(nome_odd, odds["casa_coletado"], odds["fora_coletado"])
                prob = probs_modelo.get("1X2", {}).get(selecao) if selecao else None
                if prob is not None:
                    candidatos.append({"mercado": "1X2", "selecao": selecao, "prob_modelo": prob, "odd": odd, "ev": prob * odd - 1.0})
        else:
            for mercado, selecao, odd in _mercados_para_jogo(mercado_key, outcomes):
                prob = probs_modelo.get(mercado, {}).get(selecao)
                if prob is not None:
                    candidatos.append({"mercado": mercado, "selecao": selecao, "prob_modelo": prob, "odd": odd, "ev": prob * odd - 1.0})
    return candidatos


def card_completo(liga: str, time_casa: str, time_fora: str, data_jogo: str | None,
                   times_por_liga: dict[str, list[str]], caminho_db: Path = CAMINHO_DB_PADRAO) -> dict:
    """Roda o modelo, cruza com odds já coletadas (se houver) e devolve o
    card completo do jogo, agrupado por família de mercado (GRUPOS_CARD).
    Onde não existe modelo pra família (ex.: escanteios no Brasileirão/MLS),
    o grupo some da lista mas ganha um aviso explícito em `avisos_grupo` —
    nunca fica faltando em silêncio."""
    resultado_pred = prever(liga, time_casa, time_fora, gravar=False)
    linhas = resultado_pred["linhas_mercados"]
    probs_modelo = probs_modelo_de_linhas(linhas)

    odds = buscar_odds_coletadas_para_fixture(time_casa, time_fora, times_por_liga, caminho_db)
    odds_por_chave: dict[tuple[str, str], dict] = {}
    if odds is not None:
        for c in combinar_modelo_e_odds(probs_modelo, odds, time_casa, time_fora):
            odds_por_chave[(c["mercado"], c["selecao"])] = c

    grupos: dict[str, list[dict]] = {g: [] for g in GRUPOS_CARD}
    for mercado, selecao, prob in linhas:
        grupo = _grupo_exibicao(mercado)
        if grupo is None:
            continue  # placar exato / handicap europeu não entram no card
        c = odds_por_chave.get((mercado, selecao))
        grupos[grupo].append({
            "mercado": mercado, "selecao": selecao, "prob_modelo": prob,
            "odd": c["odd"] if c else None, "ev": c["ev"] if c else None,
        })

    avisos_grupo: dict[str, str] = {}
    if liga in LIGAS_SEM_ESTATISTICAS_EXTRAS:
        for grupo in ("Escanteios", "Cartões e faltas"):
            avisos_grupo[grupo] = "sem modelo para esta liga (fonte não tem histórico deste mercado)"

    return {
        "liga": liga, "time_casa": time_casa, "time_fora": time_fora, "data_jogo": data_jogo,
        "tem_odds_coletadas": odds is not None,
        "grupos": grupos,
        "avisos_grupo": avisos_grupo,
        "avisos_modelo": resultado_pred["avisos"],
    }


def maiores_probabilidades(cards: list[dict], top_n: int = 10) -> list[dict]:
    """Os `top_n` desfechos mais prováveis do dia entre TODOS os jogos e
    mercados dos cards já calculados (ver `card_completo`) — nunca refaz o
    ajuste do modelo, só reordena o que já foi calculado."""
    todos = []
    for card in cards:
        for grupo, linhas in card["grupos"].items():
            for item in linhas:
                todos.append({
                    "liga": card["liga"], "time_casa": card["time_casa"], "time_fora": card["time_fora"],
                    "mercado": item["mercado"], "selecao": item["selecao"],
                    "prob_modelo": item["prob_modelo"], "odd": item["odd"], "ev": item["ev"],
                })
    todos.sort(key=lambda x: -x["prob_modelo"])
    return todos[:top_n]
