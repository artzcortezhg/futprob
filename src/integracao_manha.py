# -*- coding: utf-8 -*-
"""
Integração da foto da manhã (Bloco 4.8): roda o coletor, casa cada jogo
coletado com os times do futprob, calcula EV com o modelo puro (mesma regra
do backtest — nunca a mistura modelo+mercado, ver backtest_clv.py) contra a
odd coletada, aplica os MESMOS guarda-corpos do bot (src/guardrails.py) e:

- registra automaticamente as "apostarias" (origem='auto_manha');
- notifica o usuário no Telegram com os destaques;
- grava o resultado da coleta (sucesso/falha) em `coletas`, pro painel
  mostrar no status.

Se a coleta falhar, avisa no Telegram e no status do painel, e segue sem
travar nada (nenhuma exceção sobe até quebrar o processo chamador).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import timezone
from pathlib import Path

from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("futprob.integracao_manha")

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB = RAIZ / "db" / "previsoes.sqlite"

load_dotenv(RAIZ / ".env")

from collector import coletar, salvar_snapshot  # noqa: E402
from painel_db import inserir_registro, registrar_coleta, carregar_estado_bot  # noqa: E402
from guardrails import aplicar_guardrails, formatar_ranking  # noqa: E402
from predict import prever, probs_modelo_de_linhas  # noqa: E402
from resolucao_times import carregar_times_por_liga, resolver_time  # noqa: E402
from catalogo import _mercados_para_jogo, FUSO_BR  # noqa: E402


async def processar_foto_manha_async(limiar_ev: float = 0.05) -> dict:
    """Versão ASYNC-nativa (usa `await coletar()` direto, sem asyncio.run) —
    é a que o bot chama de dentro do seu próprio event loop. `asyncio.run()`
    não pode ser chamado de dentro de um loop já rodando (é exatamente isso
    que quebrava quando o bot chamava a versão síncrona antiga direto).
    Roda a coleta da manhã, casa com o futprob, calcula EV e registra as
    apostarias automáticas. Retorna um resumo (nunca lança exceção — coleta
    que falha vira um resumo com sucesso=False)."""
    try:
        resultado_coleta = await coletar()
    except Exception as exc:
        logger.exception("falha na coleta da manhã")
        registrar_coleta(CAMINHO_DB, "betano", sucesso=False, tipo="manha", mensagem=str(exc))
        return {"sucesso": False, "erro": str(exc), "apostarias": []}

    eventos = resultado_coleta.get("betano", [])
    n_jogos = len(eventos)
    n_mercados = sum(len(m.outcomes) for ev in eventos for m in ev.markets)
    salvar_snapshot(resultado_coleta, "manha", CAMINHO_DB)
    registrar_coleta(CAMINHO_DB, "betano", sucesso=n_jogos > 0, tipo="manha",
                      mensagem="ok" if n_jogos > 0 else "nenhum jogo capturado",
                      n_jogos_capturados=n_jogos, n_mercados_capturados=n_mercados)

    if n_jogos == 0:
        return {"sucesso": False, "erro": "nenhum jogo capturado", "apostarias": []}

    times_por_liga = carregar_times_por_liga()
    apostarias = []

    for ev in eventos:
        casa_resolvido = resolver_time(ev.home_team, times_por_liga)
        fora_resolvido = resolver_time(ev.away_team, times_por_liga)
        if not casa_resolvido or not fora_resolvido or casa_resolvido[0] != fora_resolvido[0]:
            continue  # jogo de liga que o futprob não modela
        liga, time_casa = casa_resolvido
        _, time_fora = fora_resolvido

        try:
            resultado_pred = prever(liga, time_casa, time_fora, gravar=False)
        except Exception as exc:
            logger.warning(f"falha ao prever {time_casa} x {time_fora}: {exc}")
            continue
        probs = probs_modelo_de_linhas(resultado_pred["linhas_mercados"])

        candidatos = []
        for mkt in ev.markets:
            outcomes = {o.name: o.price for o in mkt.outcomes}
            if mkt.market_key == "h2h":
                mapa_nome = {ev.home_team: "Casa", "Empate": "Empate", ev.away_team: "Fora"}
                for nome_odd, odd in outcomes.items():
                    selecao = mapa_nome.get(nome_odd)
                    prob = probs.get("1X2", {}).get(selecao) if selecao else None
                    if prob is not None:
                        candidatos.append({"mercado": "1X2", "selecao": selecao, "prob_modelo": prob, "odd": odd, "ev": prob * odd - 1.0})
            else:
                for mercado, selecao, odd in _mercados_para_jogo(mkt.market_key, outcomes):
                    prob = probs.get(mercado, {}).get(selecao)
                    if prob is not None:
                        candidatos.append({"mercado": mercado, "selecao": selecao, "prob_modelo": prob, "odd": odd, "ev": prob * odd - 1.0})

        if not candidatos:
            continue

        data_jogo = None
        if ev.commence_time:
            dt = ev.commence_time if ev.commence_time.tzinfo else ev.commence_time.replace(tzinfo=timezone.utc)
            data_jogo = dt.astimezone(FUSO_BR).date().isoformat()
        ranking = aplicar_guardrails(candidatos, limiar_ev_apostaria=limiar_ev)
        for item in ranking:
            rid = inserir_registro(
                CAMINHO_DB, liga, time_casa, time_fora, item["mercado"], item["selecao"],
                prob_modelo=item["prob_modelo"], odd_registrada=item["odd"], ev=item["ev"],
                casa_apostas="betano", data_jogo=data_jogo, origem="auto_manha",
                apostaria=item["apostaria"],
            )
            if item["apostaria"]:
                apostarias.append({
                    "id": rid, "liga": liga, "time_casa": time_casa, "time_fora": time_fora,
                    "mercado": item["mercado"], "selecao": item["selecao"], "odd": item["odd"], "ev": item["ev"],
                })

    return {"sucesso": True, "n_jogos_capturados": n_jogos, "apostarias": apostarias}


def processar_foto_manha(limiar_ev: float = 0.05) -> dict:
    """Wrapper SÍNCRONO de processar_foto_manha_async — só pra uso em
    scripts/CLI de topo (ex.: `python src/integracao_manha.py`), onde ainda
    não existe nenhum event loop rodando. NUNCA chamar isso de dentro de um
    handler async do bot — use `await processar_foto_manha_async(...)`."""
    return asyncio.run(processar_foto_manha_async(limiar_ev))


async def _notificar_telegram(texto: str) -> None:
    chat_id = carregar_estado_bot(CAMINHO_DB, "chat_id")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not chat_id or not token:
        logger.info("sem chat_id/token salvo ainda — não dá pra notificar (usuário precisa mandar /start pro bot)")
        return
    from telegram import Bot
    bot = Bot(token)
    await bot.send_message(chat_id=int(chat_id), text=texto)


def rodar_e_notificar(limiar_ev: float = 0.05) -> dict:
    resumo = processar_foto_manha(limiar_ev)
    if not resumo["sucesso"]:
        texto = f"⚠️ Coleta da manhã falhou: {resumo.get('erro', 'motivo desconhecido')}. Seguindo em modo manual."
    elif not resumo["apostarias"]:
        texto = f"Coleta da manhã ok ({resumo['n_jogos_capturados']} jogos), nenhuma apostaria automática hoje."
    else:
        linhas = [f"Coleta da manhã: {len(resumo['apostarias'])} apostaria(s) automática(s):"]
        for a in resumo["apostarias"]:
            linhas.append(f"  🎯 {a['time_casa']} x {a['time_fora']} ({a['liga']}) — {a['mercado']}/{a['selecao']} "
                           f"odd {a['odd']:.2f} EV {a['ev']*100:+.1f}%")
        texto = "\n".join(linhas)

    try:
        asyncio.run(_notificar_telegram(texto))
    except Exception:
        logger.exception("falha ao notificar no Telegram (coleta/registro seguem válidos)")

    print(texto)
    return resumo


if __name__ == "__main__":
    rodar_e_notificar()
