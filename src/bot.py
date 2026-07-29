# -*- coding: utf-8 -*-
"""
Bot de Telegram do futprob (python-telegram-bot). Funções:

- Rotina matutina automática (9h, horário de Brasília): roda a coleta da
  Betano, calcula EV de todos os mercados contra o modelo, registra as
  "apostarias" (guarda-corpos de sempre) e manda o catálogo do dia + as
  apostarias no Telegram. Agenda sozinha a rotina pré-jogo do dia (ver
  abaixo). Fonte de "jogos do dia": o coletor da Betano — o fixtures.csv do
  football-data.co.uk existe mas está desatualizado (última rodada listada
  é de maio/2026) e não cobre nenhuma das 5 ligas do projeto agora.
- Rotina pré-jogo (agendada dinamicamente, 60min antes do 1º jogo do dia):
  coleta de novo (fechamento), calcula CLV dos registros de hoje, notifica.
- Rotina de fechamento (diária, mais tarde): tenta atualizar
  data/processed/partidas.csv e fechar resultado/ROI dos registros 1X2.
- Catch-up: se o PC estava desligado no horário, a rotina matutina/pré-jogo
  perdida roda assim que o bot sobe de novo, se ainda fizer sentido (ver
  `_catch_up_rotinas`).
- /jogo <time>: busca aproximada (sem acento, parcial, com sugestão se
  ambíguo). Se já existe coleta da Betano pra esse confronto, mostra
  probabilidade do modelo + odd coletada + EV por mercado, destacando a
  apostaria. Sem coleta ainda, mostra só as probabilidades do modelo e
  avisa "odds ainda não coletadas hoje" — nunca pede nada ao usuário.
- Colar odds em RESPOSTA a uma mensagem de jogo do bot: modo MANUAL,
  disponível como fallback só quando a coleta automática falhou (ver
  guarda-corpos em guardrails.py).
- /clv, /report, /desfazer: como antes.

Nenhum comando falha em silêncio: todo erro não tratado cai no handler de
erro global (tratar_erro), que responde "algo deu errado" no chat e grava o
traceback completo no log.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB = RAIZ / "db" / "previsoes.sqlite"
CAMINHO_PARTIDAS = RAIZ / "data" / "processed" / "partidas.csv"
FUSO_BR = ZoneInfo("America/Sao_Paulo")

CAMINHO_LOG = RAIZ / "logs" / "bot.log"
CAMINHO_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[
        RotatingFileHandler(CAMINHO_LOG, maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("futprob.bot")

load_dotenv(RAIZ / ".env")

from saude_sistema import (  # noqa: E402
    HORARIO_MATUTINA, HORARIO_FECHAMENTO, MINUTOS_ANTES_PREJOGO, CUTOFF_CATCHUP_MATUTINA,
    HEARTBEAT_INTERVALO_MIN, calcular_status_sistema,
)
from painel_db import (  # noqa: E402
    inserir_registro, fechar_registro, registrar_coleta,
    salvar_estado_bot, carregar_estado_bot, salvar_mensagem_jogo, carregar_mensagem_jogo,
)
from guardrails import aplicar_guardrails, formatar_ranking  # noqa: E402
from predict import prever, formatar_tabela, probs_modelo_de_linhas  # noqa: E402
from resolucao_times import (  # noqa: E402
    normalizar_texto, carregar_times_por_liga, candidatos_time, resolver_time, resolver_time_ambiguo, LIGA_SERIE_B,
)
from integracao_manha import processar_foto_manha_async  # noqa: E402
from catalogo import (  # noqa: E402
    _mercados_para_jogo, buscar_fixture_real, resolver_fixture_para_liga, buscar_odds_coletadas_para_fixture,
    combinar_modelo_e_odds, descobrir_jogos_do_dia, familia_mercado, hoje_br,
    card_completo, card_sem_modelo, maiores_probabilidades, GRUPOS_CARD, enriquecer_odds_externas_com_modelo,
)
import oddspapi  # noqa: E402


# ── Mercados/odds coladas manualmente (fallback) ────────────────────────────

_CODIGO_FIXO = {
    "1": ("1X2", "Casa"), "X": ("1X2", "Empate"), "2": ("1X2", "Fora"),
    "BTTS": ("Ambas marcam", "Sim"), "NBTTS": ("Ambas marcam", "Não"),
    "DC1X": ("Dupla chance", "1X (casa ou empate)"),
    "DC12": ("Dupla chance", "12 (casa ou fora)"),
    "DCX2": ("Dupla chance", "X2 (empate ou fora)"),
}
_CODIGO_LINHA = [
    (re.compile(r'^O(\d+\.?\d*)$'), "Over/Under {linha}", "Over"),
    (re.compile(r'^U(\d+\.?\d*)$'), "Over/Under {linha}", "Under"),
    (re.compile(r'^OC(\d+\.?\d*)$'), "Escanteios Over/Under {linha}", "Over"),
    (re.compile(r'^UC(\d+\.?\d*)$'), "Escanteios Over/Under {linha}", "Under"),
    (re.compile(r'^OA(\d+\.?\d*)$'), "Cartões Over/Under {linha}", "Over"),
    (re.compile(r'^UA(\d+\.?\d*)$'), "Cartões Over/Under {linha}", "Under"),
]

FORMATO_AJUDA_ODDS = (
    "Não entendi essas odds. Formato esperado: código=odd, separados por espaço.\n"
    "Códigos aceitos:\n"
    "  1 / X / 2 — resultado (1X2)\n"
    "  O<linha> / U<linha> — over/under de gols (ex.: O2.5=2.10)\n"
    "  BTTS / NBTTS — ambas marcam sim/não\n"
    "  DC1X / DC12 / DCX2 — dupla chance\n"
    "  OC<linha> / UC<linha> — over/under de escanteios (ex.: OC9.5=1.90)\n"
    "  OA<linha> / UA<linha> — over/under de cartões (ex.: OA3.5=1.95)\n"
    "Exemplo: 1=2.05 X=3.40 2=3.75 O2.5=2.10\n\n"
    "Lembrete: o normal é a coleta automática (9h + pré-jogo). Colar odds "
    "manualmente é só o fallback pra quando a coleta falha."
)


def _resolver_codigo(codigo: str) -> tuple[str, str] | None:
    if codigo in _CODIGO_FIXO:
        return _CODIGO_FIXO[codigo]
    for padrao, template_mercado, selecao in _CODIGO_LINHA:
        m = padrao.match(codigo)
        if m:
            return (template_mercado.format(linha=m.group(1)), selecao)
    return None


def parsear_odds_coladas(texto: str) -> tuple[dict[tuple[str, str], float], list[str]]:
    """Retorna ({(mercado, selecao): odd}, [tokens não reconhecidos])."""
    validos: dict[tuple[str, str], float] = {}
    invalidos: list[str] = []
    for token in texto.split():
        m = re.match(r'^([A-Za-z0-9.]+)=([\d.,]+)$', token)
        if not m:
            invalidos.append(token)
            continue
        codigo, valor_str = m.groups()
        resolvido = _resolver_codigo(codigo.upper())
        try:
            odd = float(valor_str.replace(",", "."))
        except ValueError:
            resolvido = None
        if resolvido is None or odd <= 1.0:
            invalidos.append(token)
            continue
        validos[resolvido] = odd
    return validos, invalidos


def montar_candidatos(odds_parseadas: dict[tuple[str, str], float], probs_modelo: dict) -> tuple[list[dict], list[str]]:
    candidatos = []
    sem_prob = []
    for (mercado, selecao), odd in odds_parseadas.items():
        prob = probs_modelo.get(mercado, {}).get(selecao)
        if prob is None:
            sem_prob.append(f"{mercado}/{selecao}")
            continue
        candidatos.append({"mercado": mercado, "selecao": selecao, "prob_modelo": prob, "odd": odd, "ev": prob * odd - 1.0})
    return candidatos, sem_prob


def calcular_report(registros_fechados: list[dict]) -> dict:
    fechados_com_clv = [r for r in registros_fechados if r.get("clv") is not None]
    if not fechados_com_clv:
        return {"n": 0, "frase": "Nenhum registro fechado com CLV ainda."}

    clvs = np.array([r["clv"] for r in fechados_com_clv])
    n = len(clvs)
    clv_medio = float(clvs.mean())
    erro_padrao = float(clvs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    ic_inf, ic_sup = clv_medio - 1.96 * erro_padrao, clv_medio + 1.96 * erro_padrao

    if ic_inf <= 0 <= ic_sup:
        frase = "O intervalo de confiança cruza o zero: ainda não dá pra dizer que o CLV é diferente de ruído."
    elif ic_sup < 0:
        frase = "O CLV negativo parece real (IC 95% inteiro abaixo de zero) — não é só ruído."
    else:
        frase = "O CLV positivo parece real (IC 95% inteiro acima de zero) — mas a amostra pode ser pequena, continue acompanhando."

    por_familia: dict[str, list[float]] = {}
    for r in fechados_com_clv:
        por_familia.setdefault(familia_mercado(r["mercado"]), []).append(r["clv"])
    clv_por_familia = {fam: float(np.mean(vs)) for fam, vs in por_familia.items()}

    retornos = [
        (r["odd_registrada"] - 1.0) if r.get("resultado") == "ganhou" else -1.0
        for r in registros_fechados if r.get("resultado") in ("ganhou", "perdeu")
    ]
    roi_papel = float(np.mean(retornos)) if retornos else None

    return {
        "n": n, "clv_medio": clv_medio, "ic": (ic_inf, ic_sup), "frase": frase,
        "clv_por_familia": clv_por_familia, "roi_papel": roi_papel, "n_com_resultado": len(retornos),
    }


# ── Handlers do Telegram ─────────────────────────────────────────────────────

async def tratar_erro(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler de erro GLOBAL: nenhum comando falha em silêncio. Loga o
    traceback completo e avisa o usuário no chat, quando dá pra saber qual."""
    logger.error("Erro não tratado processando um update", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Algo deu errado ao processar seu pedido. Já registrei o detalhe no log."
            )
        except Exception:
            logger.exception("Falha até ao tentar avisar o usuário do erro")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    salvar_estado_bot(CAMINHO_DB, "chat_id", str(update.effective_chat.id))
    await update.message.reply_text(
        "futprob bot — modo papel, nunca sugere valor em dinheiro.\n\n"
        "Automático: coleta às 9h + pré-jogo (60min antes do 1º jogo) + fechamento à noite. "
        "Colar odds manualmente é só fallback pra quando a coleta falha.\n\n"
        "/jogo <time> — mercados da próxima partida (com odd/EV se já coletado)\n"
        "/clv — fecha registros abertos com a odd de fechamento (1X2)\n"
        "/report — acumulado de CLV/ROI\n"
        "/desfazer — remove o último registro (mantém no histórico, só marca como desfeito)\n"
        "/status — bot, agendador, painel e última coleta estão ok?\n"
        "/oddspapi — busca manual das odds Pinnacle (Brasileirão A/B, MLS) — gasta 1 uso da cota"
    )


async def _enviar_jogo(update_message, liga: str, time_casa: str, time_fora: str,
                        data_jogo: str | None = None, horario_fmt: str | None = None) -> None:
    """Roda o modelo, cruza com odds já coletadas (se houver) e manda a
    tabela combinada. Cacheia as probabilidades pra reply manual (fallback)."""
    times_por_liga = carregar_times_por_liga()
    resultado = prever(liga, time_casa, time_fora, gravar=True)
    for aviso in resultado["avisos"]:
        await update_message.reply_text(f"⚠️ {aviso}")

    probs = probs_modelo_de_linhas(resultado["linhas_mercados"])
    odds = buscar_odds_coletadas_para_fixture(time_casa, time_fora, times_por_liga)

    sufixo_horario = f" — {horario_fmt}" if horario_fmt else ""
    cabecalho = f"{time_casa} x {time_fora} — {liga}{sufixo_horario}\n"
    if odds is None:
        texto = cabecalho + "(odds ainda não coletadas hoje pra esse jogo)\n\n" + formatar_tabela(resultado["linhas_mercados"])
    else:
        candidatos = combinar_modelo_e_odds(probs, odds, time_casa, time_fora)
        if not candidatos:
            texto = cabecalho + "(odds coletadas não bateram com nenhum mercado do modelo)\n\n" + formatar_tabela(resultado["linhas_mercados"])
        else:
            ranking = aplicar_guardrails(candidatos)
            texto = cabecalho + f"Odds coletadas (Betano) — {formatar_ranking(ranking)}"
            for item in ranking:
                inserir_registro(
                    CAMINHO_DB, liga, time_casa, time_fora, item["mercado"], item["selecao"],
                    prob_modelo=item["prob_modelo"], odd_registrada=item["odd"], ev=item["ev"],
                    casa_apostas="betano", data_jogo=data_jogo, origem="bot",
                    apostaria=item["apostaria"],
                )

    if len(texto) > 4000:
        texto = texto[:3990] + "\n(...)"
    enviada = await update_message.reply_text(texto)
    salvar_mensagem_jogo(CAMINHO_DB, enviada.message_id, liga, time_casa, time_fora, data_jogo, json.dumps(probs))


def _formatar_horario_br(commence_time_iso: str) -> str:
    try:
        dt = pd.Timestamp(commence_time_iso)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert(FUSO_BR).strftime("%d/%m %H:%M")
    except Exception:
        return commence_time_iso


async def cmd_jogo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Busca SEMPRE nas fixtures REAIS mais próximas (hoje -> futuro) da
    coleta mais recente — nunca no roster do modelo primeiro (esse era o
    bug: achar o time no roster da Série A e só depois procurar um jogo
    dele em qualquer coleta antiga podia devolver um confronto stale de
    outra época). Só depois de achar o jogo real é que tenta resolver pro
    modelo; se um dos lados (ou os dois) não tiver modelo, informa isso
    explicitamente em vez de inventar ou esconder o confronto."""
    if not context.args:
        await update.message.reply_text("Uso: /jogo <nome do time>")
        return
    nome = " ".join(context.args)
    busca = buscar_fixture_real(nome, CAMINHO_DB)

    if busca["status"] == "nao_encontrado":
        await update.message.reply_text(
            f"Não achei nenhuma partida futura coletada pra um time parecido com '{nome}' "
            "(a coleta automática roda às 9h — se já passou e não apareceu, pode ser que não haja jogo hoje pra esse time)."
        )
        return
    if busca["status"] == "ambiguo":
        opcoes = "\n".join(
            f"  - {o['casa']} x {o['fora']} ({_formatar_horario_br(o['commence_time'])})" for o in busca["opcoes"]
        )
        await update.message.reply_text(
            f"'{nome}' bateu com mais de uma partida — qual dessas?\n{opcoes}\n\n"
            "Mande /jogo com o nome mais específico (ex.: um dos dois times)."
        )
        return

    casa_cru, fora_cru, commence_time = busca["casa_coletado"], busca["fora_coletado"], busca["commence_time"]
    horario_fmt = _formatar_horario_br(commence_time)
    times_por_liga = carregar_times_por_liga()
    resolvido = resolver_fixture_para_liga(casa_cru, fora_cru, times_por_liga)

    if resolvido is None:
        await update.message.reply_text(
            f"{casa_cru} x {fora_cru} — {horario_fmt}\n"
            "⚠️ campeonato sem modelo — sem previsão (um dos dois times, ou os dois, não está na base de "
            "times modelados das 5 ligas do futprob)."
        )
        return

    liga, time_casa, time_fora = resolvido
    dt = pd.Timestamp(commence_time)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    data_jogo = dt.tz_convert(FUSO_BR).date().isoformat()

    if liga == LIGA_SERIE_B:
        # Série B: sem modelo treinado (FBref bloqueado) — mostra a odd
        # coletada crua, nunca probabilidade/EV inventados
        card = card_sem_modelo(liga, time_casa, time_fora, data_jogo, times_por_liga, CAMINHO_DB)
        cabecalho = f"{time_casa} x {time_fora} (Brasileirão Série B) — {horario_fmt}\n⚠️ sem modelo treinado pra esta liga\n\n"
        if not card["linhas"]:
            texto = cabecalho + "(odds ainda não coletadas hoje pra esse jogo)"
        else:
            linhas_fmt = [f"{item['mercado']}/{item['selecao']}: odd {item['odd']:.2f}" for item in card["linhas"]]
            texto = cabecalho + "\n".join(linhas_fmt)
        await update.message.reply_text(texto)
        return

    await _enviar_jogo(update.message, liga, time_casa, time_fora, data_jogo, horario_fmt)


async def responder_odds_coladas(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler de texto livre em RESPOSTA a uma mensagem de jogo do bot —
    modo MANUAL (fallback quando a coleta automática falhou)."""
    msg = update.message
    if not msg.reply_to_message:
        return
    jogo = carregar_mensagem_jogo(CAMINHO_DB, msg.reply_to_message.message_id)
    if jogo is None:
        return

    odds_parseadas, invalidos = parsear_odds_coladas(msg.text or "")
    if not odds_parseadas:
        await msg.reply_text(FORMATO_AJUDA_ODDS)
        return

    probs_modelo = json.loads(jogo["probs_json"])
    candidatos, sem_prob = montar_candidatos(odds_parseadas, probs_modelo)

    if invalidos:
        await msg.reply_text(f"Não reconheci: {', '.join(invalidos)}.\n\n{FORMATO_AJUDA_ODDS}")
    if sem_prob:
        await msg.reply_text(f"Sem probabilidade do modelo pra: {', '.join(sem_prob)} (não calculado nesse jogo).")
    if not candidatos:
        return

    ranking = aplicar_guardrails(candidatos)
    await msg.reply_text(formatar_ranking(ranking))

    for item in ranking:
        rid = inserir_registro(
            CAMINHO_DB, jogo["liga"], jogo["time_casa"], jogo["time_fora"], item["mercado"], item["selecao"],
            prob_modelo=item["prob_modelo"], odd_registrada=item["odd"], ev=item["ev"],
            data_jogo=jogo["data_jogo"], origem="bot_manual",
            apostaria=item["apostaria"],
        )
        if item["apostaria"]:
            logger.info(f"apostaria (manual) registrada: id={rid} {jogo['time_casa']} x {jogo['time_fora']} {item['mercado']}/{item['selecao']}")


async def cmd_clv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    fechados, sem_fechamento = _fechar_registros_1x2()
    await update.message.reply_text(f"{fechados} registro(s) fechado(s). {sem_fechamento} ainda sem fechamento disponível na base.")


def _fechar_registros_1x2(caminho_db: Path = CAMINHO_DB, caminho_partidas: Path = CAMINHO_PARTIDAS) -> tuple[int, int]:
    """Fecha registros 1X2 abertos usando o fechamento (PSCH/PSCD/PSCA) já
    presente em data/processed/partidas.csv. Retorna (fechados, sem_fechamento)."""
    with sqlite3.connect(caminho_db) as conn:
        conn.row_factory = sqlite3.Row
        abertos = conn.execute("SELECT * FROM registros WHERE status='aberto' AND mercado='1X2'").fetchall()
    if not abertos:
        return 0, 0

    df = pd.read_csv(caminho_partidas, parse_dates=["Date"])
    fechados, sem_fechamento = 0, 0
    coluna_por_selecao = {"Casa": "PSCH", "Empate": "PSCD", "Fora": "PSCA"}

    for reg in abertos:
        candidatos = df[
            (df["liga"] == reg["liga"]) & (df["HomeTeam"] == reg["time_casa"]) & (df["AwayTeam"] == reg["time_fora"])
        ]
        if reg["data_jogo"]:
            candidatos = candidatos[candidatos["Date"] == pd.Timestamp(reg["data_jogo"])]
        col = coluna_por_selecao.get(reg["selecao"])
        if candidatos.empty or col is None or pd.isna(candidatos.iloc[0].get(col)):
            sem_fechamento += 1
            continue

        linha = candidatos.iloc[0]
        odd_fechamento = float(linha[col])
        inv = np.array([1.0 / linha["PSCH"], 1.0 / linha["PSCD"], 1.0 / linha["PSCA"]])
        prob_justa = (1.0 / linha[col]) / inv.sum()
        clv = reg["odd_registrada"] * prob_justa - 1.0
        resultado = None
        if pd.notna(linha.get("FTR")):
            mapa = {"Casa": "H", "Empate": "D", "Fora": "A"}
            resultado = "ganhou" if linha["FTR"] == mapa.get(reg["selecao"]) else "perdeu"
        fechar_registro(caminho_db, reg["id"], odd_fechamento, clv, resultado)
        fechados += 1

    return fechados, sem_fechamento


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with sqlite3.connect(CAMINHO_DB) as conn:
        conn.row_factory = sqlite3.Row
        registros = [dict(r) for r in conn.execute("SELECT * FROM registros WHERE status != 'desfeito'")]

    r = calcular_report(registros)
    if r["n"] == 0:
        await update.message.reply_text(f"{r['frase']} (registros abertos: {sum(1 for x in registros if x['status']=='aberto')})")
        return

    linhas = [
        f"Registros fechados com CLV: {r['n']}",
        f"CLV médio: {r['clv_medio']*100:+.1f}%",
        r["frase"],
        "",
        "Por família de mercado:",
    ]
    for fam, clv in sorted(r["clv_por_familia"].items(), key=lambda x: -x[1]):
        linhas.append(f"  {fam}: {clv*100:+.1f}%")
    if r["roi_papel"] is not None:
        linhas.append(f"\nROI de papel ({r['n_com_resultado']} com resultado conhecido): {r['roi_papel']*100:+.1f}%")
    await update.message.reply_text("\n".join(linhas))


async def cmd_desfazer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with sqlite3.connect(CAMINHO_DB) as conn:
        ultimo = conn.execute("SELECT id, liga, time_casa, time_fora, mercado, selecao FROM registros "
                               "WHERE status != 'desfeito' ORDER BY id DESC LIMIT 1").fetchone()
        if ultimo is None:
            await update.message.reply_text("Nenhum registro pra desfazer.")
            return
        conn.execute("UPDATE registros SET status='desfeito' WHERE id=?", (ultimo[0],))
        conn.commit()
    await update.message.reply_text(
        f"Desfeito: {ultimo[2]} x {ultimo[3]} — {ultimo[4]}/{ultimo[5]} (registro mantido no histórico, só marcado como desfeito)."
    )


async def _checar_painel() -> tuple[bool, str]:
    """Ping rápido (2s de timeout) no painel web — nunca lança exceção."""
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = os.environ.get("DASHBOARD_PORT", "8000")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"http://{host}:{port}/api/status-coleta")
        return (r.status_code == 200, f"http://{host}:{port}")
    except Exception as exc:
        return (False, f"não respondeu em http://{host}:{port} ({exc.__class__.__name__}) — veja logs/painel.log")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Responde na hora se bot, agendador, painel e última coleta estão ok —
    Bloco de estabilização (o sistema se vigia sozinho)."""
    status = calcular_status_sistema(CAMINHO_DB)
    n_jobs = len(context.job_queue.jobs())
    painel_ok, painel_detalhe = await _checar_painel()

    linhas = ["📊 Status do futprob"]
    linhas.append("Bot: ok (respondendo agora)")
    linhas.append(f"Agendador: {'ok' if n_jobs else '⚠️ nenhuma rotina agendada — reinicie o bot'} ({n_jobs} job(s) internos)")
    linhas.append(f"Painel: {'ok — ' + painel_detalhe if painel_ok else '⚠️ ' + painel_detalhe}")

    uc = status["ultima_coleta"]
    if uc:
        resultado = "sucesso" if uc["sucesso"] else f"FALHA: {uc['mensagem'] or '?'}"
        linhas.append(f"Última coleta: {uc['tipo'] or uc['fonte']} em {uc['executado_em']} — {resultado}")
    else:
        linhas.append("Última coleta: nenhuma ainda")

    linhas.append(f"Próxima rotina: {status['proxima_rotina']}")
    linhas.append(f"Registros abertos: {status['n_registros_abertos']}")
    if not status["chat_id_configurado"]:
        linhas.append("⚠️ chat_id não configurado — mande /start")
    await update.message.reply_text("\n".join(linhas))


async def cmd_oddspapi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Busca manual (NUNCA automática) das odds Pinnacle atuais via OddsPapi
    — Brasileirão Série A/B e MLS — e cruza com o MODELO do futprob (prob.
    e EV) nas ligas que têm modelo. Cada chamada gasta 1 uso da cota
    gratuita (250 no total), por isso só roda quando pedido."""
    await update.message.reply_text("Buscando na OddsPapi e cruzando com o modelo (alguns segundos por jogo)…")
    resumo = await asyncio.to_thread(oddspapi.buscar_melhores_odds, CAMINHO_DB)
    status = oddspapi.uso_atual(CAMINHO_DB)

    if not resumo["sucesso"]:
        await update.message.reply_text(f"⚠️ {resumo['erro']} (cota usada: {status}/{oddspapi.LIMITE_USOS})")
        return
    if not resumo["jogos"]:
        await update.message.reply_text(f"Nenhum jogo com odds Pinnacle no momento (cota usada: {status}/{oddspapi.LIMITE_USOS}).")
        return

    times_por_liga = carregar_times_por_liga()
    jogos = await asyncio.to_thread(enriquecer_odds_externas_com_modelo, resumo["jogos"], times_por_liga)

    linhas = [f"🎲 OddsPapi (Pinnacle) — {len(jogos)} jogo(s), cota {status}/{oddspapi.LIMITE_USOS}:\n"]
    for j in jogos:
        rotulo = "" if j["tem_modelo"] else " ⚠️ sem modelo (Série B) — só a odd"
        linhas.append(f"{j['casa']} x {j['fora']} ({j['liga']}){rotulo} — {j['commence_time']}")
        for aviso in j.get("avisos_modelo") or []:
            linhas.append(f"  ⚠️ {aviso}")
        mercados_ordenados = sorted(j["mercados"], key=lambda m: m["ev"] if m["ev"] is not None else -99, reverse=True)
        for m in mercados_ordenados:
            if m["ev"] is not None:
                aviso = " ⚠️ suspeito: provável erro do modelo" if m.get("suspeito") else ""
                linhas.append(f"  {m['mercado']}/{m['selecao']}: odd {m['odd']:.2f} | prob. {m['prob_modelo']*100:.1f}% | EV {m['ev']*100:+.1f}%{aviso}")
            else:
                linhas.append(f"  {m['mercado']}/{m['selecao']}: odd {m['odd']:.2f}")

        aposta = j.get("melhor_aposta")
        if aposta:
            linhas.append(f"  🎯 Melhor aposta: {aposta['mercado']}/{aposta['selecao']} — odd {aposta['odd']:.2f} | prob. {aposta['prob_modelo']*100:.1f}% | EV {aposta['ev']*100:+.1f}%")

        bilhete = j.get("melhor_bilhete")
        if bilhete:
            pernas_txt = " + ".join(f"{p['mercado']}/{p['selecao']} (odd {p['odd']:.2f})" for p in bilhete["pernas"])
            linhas.append(f"  🎟️ Melhor bilhete: {pernas_txt} — odd combinada {bilhete['odd_combinada']:.2f} | prob. conjunta {bilhete['prob_conjunta']*100:.1f}% | EV {bilhete['ev_combinado']*100:+.1f}%")

        linhas.append("")
    texto = "\n".join(linhas)
    if len(texto) > 4000:
        texto = texto[:3990] + "\n(...)"
    await update.message.reply_text(texto)


async def _heartbeat(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Grava sinal de vida periódico — é isso que o painel e o /status usam
    pra saber que o bot está rodando (não travado, não morto)."""
    salvar_estado_bot(CAMINHO_DB, "heartbeat_bot", datetime.now(FUSO_BR).isoformat())


# ── Rotinas automáticas (scheduler interno) ─────────────────────────────────

async def rotina_matutina(context: ContextTypes.DEFAULT_TYPE) -> None:
    """9h: coleta (foto 1), EV, registra apostarias, notifica catálogo, e
    agenda a rotina pré-jogo de hoje (dinâmica, 60min antes do 1º jogo)."""
    hoje = datetime.now(FUSO_BR).date()
    if carregar_estado_bot(CAMINHO_DB, "data_ultima_matutina") == hoje.isoformat():
        logger.info("rotina matutina: já rodou hoje, pulando")
        return

    chat_id = carregar_estado_bot(CAMINHO_DB, "chat_id")
    logger.info("rotina matutina: iniciando coleta")
    resumo = await processar_foto_manha_async()
    salvar_estado_bot(CAMINHO_DB, "data_ultima_matutina", hoje.isoformat())

    if not resumo["sucesso"]:
        texto = f"⚠️ Coleta da manhã falhou: {resumo.get('erro', 'motivo desconhecido')}. Modo manual (colar odds) disponível como fallback até a próxima coleta."
        if chat_id:
            await context.bot.send_message(chat_id, texto)
        return

    times_por_liga = carregar_times_por_liga()
    jogos_hoje = descobrir_jogos_do_dia(hoje, times_por_liga)
    salvar_estado_bot(CAMINHO_DB, "jogos_hoje_json", json.dumps(jogos_hoje))
    salvar_estado_bot(CAMINHO_DB, "data_jogos_hoje", hoje.isoformat())

    if chat_id:
        if not jogos_hoje:
            linhas = [f"Coleta da manhã ok — nenhum jogo das 5 ligas hoje ({hoje.isoformat()})."]
        else:
            linhas = [f"📋 Catálogo de hoje ({hoje.isoformat()}) — {len(jogos_hoje)} jogo(s) — automático, sem ação manual"]
            for j in jogos_hoje:
                linhas.append(f"  {j['casa']} x {j['fora']} ({j['liga']})")
            if resumo["apostarias"]:
                linhas.append(f"\n🎯 {len(resumo['apostarias'])} apostaria(s) registrada(s):")
                for a in resumo["apostarias"]:
                    linhas.append(f"  {a['time_casa']} x {a['time_fora']} — {a['mercado']}/{a['selecao']} odd {a['odd']:.2f} EV {a['ev']*100:+.1f}%")
            else:
                linhas.append("\nNenhuma apostaria automática hoje.")

        # mensagem diária de saúde (item 6b) — vai junto do catálogo, ou
        # sozinha em dia sem jogos, uma vez por dia (mesma guarda da matutina)
        status = calcular_status_sistema(CAMINHO_DB)
        linhas.append(f"\n🩺 Sistema ok — última coleta {resumo['n_jogos_capturados']} jogo(s) capturados agora | "
                       f"próxima rotina: {status['proxima_rotina']} | {status['n_registros_abertos']} registro(s) aberto(s)")
        await context.bot.send_message(chat_id, "\n".join(linhas))

    if jogos_hoje:
        _agendar_prejogo(context, jogos_hoje)


def _agendar_prejogo(context: ContextTypes.DEFAULT_TYPE, jogos_hoje: list[dict]) -> None:
    primeiro_kickoff = min(pd.Timestamp(j["commence_time"]) for j in jogos_hoje)
    if primeiro_kickoff.tzinfo is None:
        primeiro_kickoff = primeiro_kickoff.tz_localize("UTC")
    momento_prejogo = primeiro_kickoff.to_pydatetime() - timedelta(minutes=MINUTOS_ANTES_PREJOGO)
    agora = datetime.now(timezone.utc)
    delay = (momento_prejogo - agora).total_seconds()
    if delay < 0:
        delay = 5  # já passou da hora ideal (bot ligou tarde) -> roda quase já, catch-up
    context.job_queue.run_once(rotina_prejogo, when=delay)
    logger.info(f"rotina pré-jogo agendada em {delay:.0f}s (1º jogo às {primeiro_kickoff})")


async def rotina_prejogo(context: ContextTypes.DEFAULT_TYPE) -> None:
    """~60min antes do 1º jogo do dia: coleta de novo (fechamento) e calcula
    o CLV dos registros de hoje contra essa nova foto."""
    hoje = datetime.now(FUSO_BR).date().isoformat()
    if carregar_estado_bot(CAMINHO_DB, "data_ultima_prejogo") == hoje:
        return
    chat_id = carregar_estado_bot(CAMINHO_DB, "chat_id")

    logger.info("rotina pré-jogo: iniciando coleta de fechamento")
    resumo = await processar_foto_manha_async(limiar_ev=1e9)  # aqui só queremos a foto/EV pra CLV, não novas apostarias
    salvar_estado_bot(CAMINHO_DB, "data_ultima_prejogo", hoje)

    if not resumo["sucesso"]:
        if chat_id:
            await context.bot.send_message(chat_id, f"⚠️ Coleta pré-jogo falhou: {resumo.get('erro')}. CLV de hoje fica pendente.")
        return

    if chat_id:
        with sqlite3.connect(CAMINHO_DB) as conn:
            n_hoje = conn.execute("SELECT COUNT(*) FROM registros WHERE data_jogo=? AND status='aberto'", (hoje,)).fetchone()[0]
        await context.bot.send_message(chat_id, f"Coleta pré-jogo (fechamento) concluída — {n_hoje} registro(s) de hoje aguardando resultado pro CLV/ROI.")


async def rotina_fechamento(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fim do dia: tenta fechar resultado/ROI dos registros 1X2 usando a
    base local (atualizada via src/ingest.py quando o football-data.co.uk
    publicar o resultado)."""
    fechados, sem_fechamento = _fechar_registros_1x2()
    chat_id = carregar_estado_bot(CAMINHO_DB, "chat_id")
    if chat_id and fechados:
        await context.bot.send_message(chat_id, f"Fechamento diário: {fechados} registro(s) 1X2 fechado(s), {sem_fechamento} ainda pendente(s).")


def _catch_up_rotinas(app: Application) -> None:
    """Ao subir o bot (ex.: depois do PC religar), roda a rotina matutina
    perdida se ainda fizer sentido (antes do horário de corte), e reagenda
    a pré-jogo de hoje se a matutina já rodou mas a pré-jogo ainda não."""
    agora_br = datetime.now(FUSO_BR)
    hoje = agora_br.date().isoformat()

    matutina_feita_hoje = carregar_estado_bot(CAMINHO_DB, "data_ultima_matutina") == hoje
    if not matutina_feita_hoje and agora_br.hour < CUTOFF_CATCHUP_MATUTINA:
        logger.info("catch-up: rotina matutina de hoje ainda não rodou, agendando em alguns segundos")
        app.job_queue.run_once(rotina_matutina, when=5)
        return  # a própria rotina matutina já agenda a pré-jogo em seguida

    if matutina_feita_hoje:
        jogos_json = carregar_estado_bot(CAMINHO_DB, "jogos_hoje_json")
        data_jogos = carregar_estado_bot(CAMINHO_DB, "data_jogos_hoje")
        prejogo_feita_hoje = carregar_estado_bot(CAMINHO_DB, "data_ultima_prejogo") == hoje
        if jogos_json and data_jogos == hoje and not prejogo_feita_hoje:
            jogos_hoje = json.loads(jogos_json)
            if jogos_hoje:
                logger.info("catch-up: reagendando rotina pré-jogo de hoje")

                async def _reagendar_prejogo(context: ContextTypes.DEFAULT_TYPE, _jogos=jogos_hoje) -> None:
                    # job_queue sempre espera (await) o callback — _agendar_prejogo é
                    # síncrona, por isso precisa desse wrapper async em vez de lambda
                    _agendar_prejogo(context, _jogos)

                app.job_queue.run_once(_reagendar_prejogo, when=5)


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN não configurado no .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("jogo", cmd_jogo))
    app.add_handler(CommandHandler("clv", cmd_clv))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("desfazer", cmd_desfazer))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("oddspapi", cmd_oddspapi))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder_odds_coladas))
    app.add_error_handler(tratar_erro)

    hora_m, min_m = map(int, HORARIO_MATUTINA.split(":"))
    hora_f, min_f = map(int, HORARIO_FECHAMENTO.split(":"))
    app.job_queue.run_daily(rotina_matutina, time=datetime.now(FUSO_BR).replace(hour=hora_m, minute=min_m, second=0, microsecond=0).timetz())
    app.job_queue.run_daily(rotina_fechamento, time=datetime.now(FUSO_BR).replace(hour=hora_f, minute=min_f, second=0, microsecond=0).timetz())
    app.job_queue.run_repeating(_heartbeat, interval=HEARTBEAT_INTERVALO_MIN * 60, first=10)

    _catch_up_rotinas(app)

    logger.info(f"bot iniciado (polling) — matutina {HORARIO_MATUTINA}, pré-jogo {MINUTOS_ANTES_PREJOGO}min antes do 1º jogo, fechamento {HORARIO_FECHAMENTO}")
    app.run_polling()


if __name__ == "__main__":
    main()
