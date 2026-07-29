# -*- coding: utf-8 -*-
"""
Painel web do futprob (FastAPI) — reforma final:

1. "Maiores probabilidades do dia" — top 10 desfechos mais prováveis entre
   todos os jogos/mercados reais de hoje, com odd/EV quando já coletados.
2. "Apostarias de hoje" — só registros que passaram os guarda-corpos de EV
   (coluna `apostaria` em `registros`), sem número mínimo/máximo.
3. Cada jogo real de hoje vira um card com todas as famílias de mercado
   (1X2+dupla chance / over-under gols / ambas marcam / escanteios /
   cartões e faltas), avisando explicitamente quando não há modelo pra uma
   família nessa liga — nunca some em silêncio.
4. Registros (histórico): coluna Data sempre preenchida, sem duplicatas na
   listagem, filtráveis por liga/família de mercado.
5. Status do sistema (bot/agendador/painel/coleta) — ver src/saude_sistema.py.

Só leitura, exceto marcar manualmente o resultado de um jogo já ocorrido —
nenhuma ação do painel apaga ou corrompe dados. "Jogos do dia" vem SEMPRE da
coleta mais recente da Betano (nunca de previsões avulsas salvas em algum
momento passado no banco — ver src/catalogo.py e o bug de fase anterior em
que o painel mostrava exemplos de teste como se fossem jogos reais).

Host/porta configuráveis via .env (padrão: localhost, sem exposição
externa). Rodar com: python src/dashboard.py
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB = RAIZ / "db" / "previsoes.sqlite"
LIGAS_PADRAO = ["Premier League", "La Liga", "Championship", "brasileirao", "mls"]
LIMITE_USOS_ODDSPAPI = 250
CACHE_TTL_SEGUNDOS = 120  # card completo recalcula o modelo do zero (~5s/jogo) — evita refazer a cada refresh

CAMINHO_LOG = RAIZ / "logs" / "painel.log"
CAMINHO_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[
        RotatingFileHandler(CAMINHO_LOG, maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("futprob.dashboard")

load_dotenv(RAIZ / ".env")

from painel_db import inicializar_db_painel, marcar_resultado_manual  # noqa: E402
from predict import inicializar_db as inicializar_db_previsoes  # noqa: E402
from resolucao_times import carregar_times_por_liga  # noqa: E402
from catalogo import (  # noqa: E402
    descobrir_jogos_do_dia_completo, card_completo, card_sem_modelo, maiores_probabilidades,
    hoje_br, familia_mercado, GRUPOS_CARD, enriquecer_odds_externas_com_modelo,
)
from saude_sistema import calcular_status_sistema  # noqa: E402
import oddspapi  # noqa: E402

app = FastAPI(title="futprob — painel")


@app.middleware("http")
async def _sem_cache(request, call_next):
    """Nunca deixa o navegador cachear página/API — sem isso, um refresh
    normal podia mostrar dados antigos mesmo depois de uma correção no
    backend, parecendo (sem ser) que a correção não tinha pego."""
    resposta = await call_next(request)
    resposta.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resposta

_cache_catalogo: dict[str, tuple[float, list[dict]]] = {}


def _sem_nan(registros: list[dict]) -> list[dict]:
    """JSON estrito não aceita NaN (FastAPI usa allow_nan=False) — troca
    NaN por None antes de responder, sem alterar o banco."""
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in registros]


def _conectar() -> sqlite3.Connection:
    # garante as tabelas do painel E as de previsoes.py (predict.py pode
    # nunca ter sido rodado nesta máquina ainda)
    inicializar_db_painel(CAMINHO_DB)
    inicializar_db_previsoes(CAMINHO_DB)
    conn = sqlite3.connect(CAMINHO_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _cards_hoje(forcar: bool = False) -> list[dict]:
    """Jogos reais de hoje das ligas DE INTERESSE (Premier League, La Liga,
    Championship, Brasileirão Série A, Série B, MLS — ver
    catalogo.LIGAS_DE_INTERESSE): card completo pros modelados, e odds cruas
    (sem EV) pra Série B, que não tem modelo treinado. Jogos de qualquer
    outro campeonato/esporte não aparecem aqui — ver
    descobrir_jogos_do_dia_completo. Cache curto porque cada card recalcula
    o modelo do zero."""
    agora = time.time()
    if not forcar and "hoje" in _cache_catalogo:
        ts, cards = _cache_catalogo["hoje"]
        if agora - ts < CACHE_TTL_SEGUNDOS:
            return cards

    times_por_liga = carregar_times_por_liga()
    dia = hoje_br()
    jogos = descobrir_jogos_do_dia_completo(dia, times_por_liga, CAMINHO_DB)
    cards = []
    for j in jogos:
        try:
            if j["modelado"]:
                card = card_completo(j["liga"], j["casa"], j["fora"], dia.isoformat(), times_por_liga, CAMINHO_DB)
                card["modelado"] = True
            else:
                card = card_sem_modelo(j["liga"], j["casa"], j["fora"], dia.isoformat(), times_por_liga, CAMINHO_DB)
            cards.append(card)
        except Exception:
            logger.exception(f"falha ao montar card de {j['casa']} x {j['fora']} — pulando esse jogo, sem quebrar o resto")
    _cache_catalogo["hoje"] = (agora, cards)
    return cards


@app.get("/api/jogos-do-dia")
def jogos_do_dia(dia: str = "hoje", forcar: bool = False):
    """dia='hoje' -> cards completos pros modelados + placeholder 'sem
    modelo' pros reais sem modelo (nunca omite um jogo real). dia='amanha'
    -> mesma lista, mas sem card (só liga/times/horário/modelado) —
    normalmente ainda não há odds coletadas pra jogos de amanhã."""
    if dia == "amanha":
        times_por_liga = carregar_times_por_liga()
        jogos = descobrir_jogos_do_dia_completo(hoje_br() + timedelta(days=1), times_por_liga, CAMINHO_DB)
        return {"dia": "amanha", "jogos": jogos}
    cards = _cards_hoje(forcar=forcar)
    return {"dia": "hoje", "jogos": cards, "grupos_ordem": GRUPOS_CARD}


@app.get("/api/maiores-probabilidades")
def maiores_probabilidades_do_dia():
    cards = [c for c in _cards_hoje() if c.get("modelado")]
    top = maiores_probabilidades(cards, top_n=10)
    return {
        "aviso": "probabilidade alta não significa boa aposta — o EV é quem diz isso",
        "top": _sem_nan(top),
    }


@app.get("/api/apostarias-hoje")
def apostarias_hoje():
    """Só registros que passaram os guarda-corpos de EV (apostaria=1) —
    sem número mínimo nem máximo. Vazio é um resultado válido.

    Filtra por quando o sistema REGISTROU (criado_em), não pela data do
    jogo (data_jogo) — a rotina da manhã roda uma vez e registra o que
    achou de real "hoje -> futuro" na coleta; o jogo em si pode ser amanhã
    (ex.: kickoff tarde da noite, já virou o dia em UTC). Filtrar por
    data_jogo fazia a seção aparecer vazia mesmo com apostarias recém
    registradas, só porque o jogo delas era amanhã."""
    hoje = hoje_br().isoformat()
    with _conectar() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM registros WHERE apostaria=1 AND substr(criado_em,1,10)=? "
            "ORDER BY criado_em DESC",
            conn, params=(hoje,),
        )
    registros_hoje = _sem_nan(df.to_dict("records"))
    familias_gols = {"1X2", "Dupla chance", "Over/Under gols", "Ambas marcam"}
    for r in registros_hoje:
        r["etiqueta"] = (
            "família sem edge no backtest histórico — registro para estudo"
            if familia_mercado(r["mercado"]) in familias_gols else None
        )
    mensagem_vazio = "nenhuma divergência hoje — isso é o sistema funcionando, não um defeito" if not registros_hoje else None
    return {"apostarias": registros_hoje, "mensagem_vazio": mensagem_vazio}


@app.get("/api/registros")
def registros(status: str | None = None, liga: str | None = None, familia: str | None = None):
    with _conectar() as conn:
        # substr(...,1,10) normaliza data_jogo pra só a parte YYYY-MM-DD antes
        # de agrupar — registros antigos (antes de uma correção de formato)
        # podiam ter data_jogo como datetime completo ("2026-07-29T22:30:00")
        # em vez de só a data, o que fazia duas linhas do MESMO jogo/mercado
        # parecerem grupos diferentes e escapar do dedup
        query = """SELECT * FROM registros WHERE id IN (
            SELECT MAX(id) FROM registros
            GROUP BY liga, time_casa, time_fora, mercado, selecao,
                     substr(COALESCE(data_jogo, criado_em), 1, 10), status
        )"""
        params: list = []
        if status in ("aberto", "fechado"):
            query += " AND status=?"
            params.append(status)
        if liga:
            query += " AND liga=?"
            params.append(liga)
        query += " ORDER BY substr(COALESCE(data_jogo, criado_em), 1, 10) DESC, criado_em DESC LIMIT 500"
        df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        df["data_jogo"] = df["data_jogo"].fillna(df["criado_em"]).str.slice(0, 10)
        if familia:
            df = df[df["mercado"].apply(familia_mercado) == familia]
    return {"registros": _sem_nan(df.to_dict("records"))}


@app.get("/api/clv-serie")
def clv_serie():
    """Evolução do CLV médio e do ROI de papel acumulados (janela expansiva),
    com intervalo de confiança de 95% (mean +/- 1.96*erro padrão)."""
    with _conectar() as conn:
        # retorno_papel não é uma coluna armazenada: é derivado aqui (ganhou
        # -> odd-1, perdeu -> -1), já que o resultado é o que muda com o tempo
        df = pd.read_sql_query(
            "SELECT criado_em, clv, (CASE WHEN resultado='ganhou' THEN odd_registrada-1 ELSE -1 END) AS retorno_papel "
            "FROM registros WHERE status='fechado' AND clv IS NOT NULL AND resultado IS NOT NULL "
            "ORDER BY data_jogo, criado_em",
            conn,
        )

    if df.empty:
        return {"pontos": []}

    n = np.arange(1, len(df) + 1)
    clv_medio_acum = df["clv"].expanding().mean()
    clv_std_acum = df["clv"].expanding().std().fillna(0)
    erro_padrao_clv = clv_std_acum / np.sqrt(n)

    roi_medio_acum = df["retorno_papel"].expanding().mean()
    roi_std_acum = df["retorno_papel"].expanding().std().fillna(0)
    erro_padrao_roi = roi_std_acum / np.sqrt(n)

    pontos = []
    for i in range(len(df)):
        pontos.append({
            "indice": i + 1,
            "clv_medio": float(clv_medio_acum.iloc[i]),
            "clv_ic_inferior": float(clv_medio_acum.iloc[i] - 1.96 * erro_padrao_clv.iloc[i]),
            "clv_ic_superior": float(clv_medio_acum.iloc[i] + 1.96 * erro_padrao_clv.iloc[i]),
            "roi_medio": float(roi_medio_acum.iloc[i]),
            "roi_ic_inferior": float(roi_medio_acum.iloc[i] - 1.96 * erro_padrao_roi.iloc[i]),
            "roi_ic_superior": float(roi_medio_acum.iloc[i] + 1.96 * erro_padrao_roi.iloc[i]),
        })
    return {"pontos": pontos}


@app.get("/api/status-coleta")
def status_coleta():
    with _conectar() as conn:
        ultima = conn.execute("SELECT * FROM coletas ORDER BY id DESC LIMIT 1").fetchone()
        gasto = conn.execute("SELECT COUNT(*) FROM oddspapi_uso").fetchone()[0]
    return {
        "ultima_coleta": dict(ultima) if ultima else None,
        "oddspapi_gasto": gasto,
        "oddspapi_restante": LIMITE_USOS_ODDSPAPI - gasto,
        "oddspapi_limite": LIMITE_USOS_ODDSPAPI,
    }


@app.get("/api/status-sistema")
def status_sistema():
    """Bloco de estabilização: bot (heartbeat), agendador/próxima rotina,
    última coleta e registros abertos — o painel se olhando no espelho."""
    return calcular_status_sistema(CAMINHO_DB)


@app.get("/api/oddspapi/status")
def oddspapi_status():
    """Só lê o contador local — nunca chama a API (gratuito, sem custo)."""
    gasto = oddspapi.uso_atual(CAMINHO_DB)
    return {"gasto": gasto, "limite": oddspapi.LIMITE_USOS, "restante": oddspapi.LIMITE_USOS - gasto}


@app.post("/api/oddspapi/buscar")
def oddspapi_buscar():
    """Botão manual (nunca automático/agendado) — busca as odds Pinnacle
    atuais pra Brasileirão A/B e MLS, cruza com o MODELO do futprob (prob.
    e EV) nas ligas que têm modelo (Série B fica só com a odd, sem
    modelo). Gasta 1 uso da cota da OddsPapi (+1 na primeiríssima vez, pra
    resolver nomes de time)."""
    resultado = oddspapi.buscar_melhores_odds(CAMINHO_DB)
    if resultado["sucesso"]:
        times_por_liga = carregar_times_por_liga()
        resultado["jogos"] = _sem_nan(enriquecer_odds_externas_com_modelo(resultado["jogos"], times_por_liga))
    return resultado


@app.post("/api/registros/{registro_id}/resultado")
def marcar_resultado(registro_id: int, resultado: str):
    """Única escrita manual permitida no painel: marcar o resultado
    ('ganhou'/'perdeu') de um jogo já ocorrido. Não apaga nada."""
    try:
        marcar_resultado_manual(CAMINHO_DB, registro_id, resultado)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True}


PAGINA_HTML = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>futprob — painel</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem; background: #0d1117; color: #e6edf3; }
  @media (prefers-color-scheme: light) { body { background: #f6f8fa; color: #1f2328; } }
  h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }
  h2 { font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #444; padding-bottom: 0.3rem; }
  h4 { font-size: 0.85rem; margin: 0.8rem 0 0.2rem; opacity: 0.9; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #333; }
  .badge { padding: 0.1rem 0.5rem; border-radius: 0.3rem; font-size: 0.75rem; }
  .apostaria { background: #2ea043; color: white; }
  .positivo { color: #3fb950; }
  .negativo { color: #f85149; }
  button, select { background: #238636; color: white; border: none; padding: 0.4rem 0.8rem; border-radius: 0.4rem; cursor: pointer; }
  select { background: #21262d; }
  #status-linha, #status-sistema { font-size: 0.8rem; opacity: 0.85; margin-top: 0.3rem; }
  svg { background: #161b22; border-radius: 0.4rem; }
  .card { background: #161b22; border-radius: 0.5rem; padding: 1rem; margin-top: 0.5rem; }
  .aviso-fixo { font-size: 0.8rem; background: #3d2a00; border: 1px solid #806000; border-radius: 0.4rem; padding: 0.5rem 0.8rem; }
  @media (prefers-color-scheme: light) { .aviso-fixo { background: #fff3cd; border-color: #e0c060; } }
  .jogo-card { background: #161b22; border-radius: 0.5rem; padding: 0.6rem 1rem; margin-top: 0.5rem; }
  .jogo-card summary { cursor: pointer; font-weight: 600; }
  .card-conteudo { margin-top: 0.5rem; }
  .etiqueta-sem-edge { display: block; font-size: 0.75rem; opacity: 0.75; }
  .filtros { margin: 0.5rem 0; display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .destaque-aposta, .destaque-bilhete { font-size: 0.85rem; border-radius: 0.4rem; padding: 0.4rem 0.7rem; margin-top: 0.5rem; }
  .destaque-aposta { background: #0d2e1a; border: 1px solid #2ea043; }
  .destaque-bilhete { background: #1a2340; border: 1px solid #4c6fdb; margin-bottom: 0.3rem; }
  @media (prefers-color-scheme: light) { .destaque-aposta { background: #e6f6ea; } .destaque-bilhete { background: #e8ecfb; } }
</style>
</head>
<body>
<h1>futprob — painel</h1>
<div id="status-linha">carregando status…</div>
<div id="status-sistema">carregando status do sistema…</div>
<button onclick="atualizarTudo(true)">Atualizar</button>

<h2>Maiores probabilidades do dia</h2>
<div id="maiores-prob" class="card">carregando…</div>

<h2>Apostarias de hoje</h2>
<div id="apostarias" class="card">carregando…</div>

<h2>Jogos de hoje</h2>
<div id="jogos-hoje">carregando…</div>

<h2>Jogos de amanhã</h2>
<div id="jogos-amanha" class="card">carregando…</div>

<h2>OddsPapi (Pinnacle) — sob demanda</h2>
<div class="card">
  <p style="font-size:0.85rem;opacity:0.85">Fonte independente de odds (Brasileirão Série A/B e MLS). Nunca busca sozinho —
  só quando você aperta o botão, porque cada busca gasta 1 uso da cota gratuita (250 no total).
  Prob./EV vêm do MESMO modelo Dixon-Coles do resto do painel, ajustado na hora com o histórico real —
  mas EV acima de 15% é marcado "suspeito" (mesma regra do resto do sistema: mais provável ser
  instabilidade do modelo — pouco histórico do time, por exemplo — do que valor real).</p>
  <div id="oddspapi-status" style="font-size:0.85rem;margin-bottom:0.5rem">carregando cota…</div>
  <button onclick="buscarOddspapi()">Buscar odds agora</button>
  <div id="oddspapi-resultado" style="margin-top:0.7rem">Ainda não buscado nesta sessão.</div>
</div>

<h2>Histórico (registros e evolução de CLV/ROI)</h2>
<p style="font-size:0.8rem;opacity:0.7">Auditoria/estudo — não é algo pra conferir todo dia, por isso fica no final.
"Aberto" = ainda não sabemos o resultado; "Fechado" = já sabemos e o CLV foi calculado. Só 1X2 fecha sozinho
hoje — Ambas marcam e Over/Under ficam abertos até você marcar ganhou/perdeu na mão.</p>
<div class="filtros">
  <select id="filtro-liga" onchange="carregarRegistros()">
    <option value="">Todas as ligas</option>
    <option>Premier League</option>
    <option>La Liga</option>
    <option>Championship</option>
    <option value="brasileirao">Brasileirão</option>
    <option value="mls">MLS</option>
  </select>
  <select id="filtro-familia" onchange="carregarRegistros()">
    <option value="">Todas as famílias</option>
    <option>1X2</option>
    <option>Dupla chance</option>
    <option>Ambas marcam</option>
    <option>Over/Under gols</option>
    <option>Escanteios</option>
    <option>Cartões</option>
    <option>Faltas</option>
  </select>
</div>
<div id="registros" class="card">carregando…</div>

<h3>Evolução do CLV médio e ROI de papel (com IC 95%)</h3>
<div id="grafico" class="card">carregando…</div>

<script>
async function buscar(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function fmtOdd(v) { return v != null ? v.toFixed(2) : '—'; }
function fmtPct(v) { return v != null ? (v*100).toFixed(1) + '%' : '—'; }

async function carregarStatus() {
  const s = await buscar('/api/status-coleta');
  const uc = s.ultima_coleta;
  const linhaColeta = uc
    ? `Última coleta: ${uc.fonte} (${uc.tipo || '-'}) em ${uc.executado_em} — ${uc.sucesso ? 'sucesso' : 'FALHA'} — ${uc.mensagem || ''}`
    : 'Nenhuma coleta registrada ainda.';
  document.getElementById('status-linha').innerHTML =
    `${linhaColeta} · OddsPapi: ${s.oddspapi_gasto}/${s.oddspapi_limite} usos (restam ${s.oddspapi_restante})`;
}

async function carregarStatusSistema() {
  const s = await buscar('/api/status-sistema');
  const botTxt = s.bot_ok
    ? `ok (heartbeat há ${s.minutos_desde_heartbeat.toFixed(1)}min)`
    : '⚠️ sem sinal recente do bot — verifique se está rodando';
  const chatTxt = s.chat_id_configurado ? '' : ' · ⚠️ chat_id não configurado (mande /start no bot)';
  document.getElementById('status-sistema').innerHTML =
    `Bot: ${botTxt} · Próxima rotina: ${s.proxima_rotina} · Registros abertos: ${s.n_registros_abertos}${chatTxt}`;
}

async function carregarMaioresProbabilidades() {
  const d = await buscar('/api/maiores-probabilidades');
  const el = document.getElementById('maiores-prob');
  let html = `<p class="aviso-fixo">⚠️ ${d.aviso}</p>`;
  if (!d.top.length) { html += '<i>Nenhum jogo real das 5 ligas hoje.</i>'; el.innerHTML = html; return; }
  html += '<table><tr><th>Jogo</th><th>Liga</th><th>Mercado</th><th>Seleção</th><th>Prob. modelo</th><th>Odd (Betano)</th><th>EV</th></tr>';
  for (const item of d.top) {
    html += `<tr><td>${item.time_casa} x ${item.time_fora}</td><td>${item.liga}</td><td>${item.mercado}</td>`
      + `<td>${item.selecao}</td><td>${fmtPct(item.prob_modelo)}</td><td>${fmtOdd(item.odd)}</td><td>${fmtPct(item.ev)}</td></tr>`;
  }
  el.innerHTML = html + '</table>';
}

async function carregarApostarias() {
  const d = await buscar('/api/apostarias-hoje');
  const el = document.getElementById('apostarias');
  if (d.mensagem_vazio) { el.innerHTML = `<i>${d.mensagem_vazio}</i>`; return; }
  let html = '<table><tr><th>Jogo</th><th>Liga</th><th>Mercado/Seleção</th><th>Odd</th><th>EV</th><th>Origem</th></tr>';
  for (const r of d.apostarias) {
    const etiqueta = r.etiqueta ? `<span class="etiqueta-sem-edge">⚠️ ${r.etiqueta}</span>` : '';
    html += `<tr><td>${r.time_casa} x ${r.time_fora}</td><td>${r.liga}</td><td>${r.mercado}/${r.selecao}${etiqueta}</td>`
      + `<td>${fmtOdd(r.odd_registrada)}</td><td>${fmtPct(r.ev)}</td><td>${r.origem}</td></tr>`;
  }
  el.innerHTML = html + '</table>';
}

function renderizarCard(j, gruposOrdem) {
  if (j.modelado === false) {
    const rotuloOdds = j.tem_odds_coletadas ? '' : ' — odds ainda não coletadas hoje';
    let html = `<div class="jogo-card">${j.time_casa} x ${j.time_fora} (Brasileirão Série B)${rotuloOdds}<br>`
      + `<span class="etiqueta-sem-edge">⚠️ sem modelo treinado pra esta liga — mostrando só a odd coletada, sem probabilidade nem EV</span>`;
    if (j.linhas && j.linhas.length) {
      html += '<table><tr><th>Mercado</th><th>Seleção</th><th>Odd</th></tr>';
      for (const item of j.linhas) {
        html += `<tr><td>${item.mercado}</td><td>${item.selecao}</td><td>${fmtOdd(item.odd)}</td></tr>`;
      }
      html += '</table>';
    }
    return html + '</div>';
  }
  const rotuloOdds = j.tem_odds_coletadas ? '' : ' — odds ainda não coletadas hoje';
  let html = `<details class="jogo-card"><summary>${j.time_casa} x ${j.time_fora} (${j.liga})${rotuloOdds}</summary><div class="card-conteudo">`;
  for (const grupo of gruposOrdem) {
    const linhas = (j.grupos && j.grupos[grupo]) || [];
    const aviso = j.avisos_grupo && j.avisos_grupo[grupo];
    html += `<h4>${grupo}</h4>`;
    if (aviso) {
      html += `<i>${aviso}</i>`;
    } else if (!linhas.length) {
      html += '<i>sem linhas calculadas</i>';
    } else {
      html += '<table><tr><th>Mercado</th><th>Seleção</th><th>Prob.</th><th>Odd</th><th>EV</th></tr>';
      for (const item of linhas) {
        html += `<tr><td>${item.mercado}</td><td>${item.selecao}</td><td>${fmtPct(item.prob_modelo)}</td>`
          + `<td>${fmtOdd(item.odd)}</td><td>${fmtPct(item.ev)}</td></tr>`;
      }
      html += '</table>';
    }
  }
  return html + '</div></details>';
}

async function carregarJogosHoje(forcar) {
  const d = await buscar('/api/jogos-do-dia?dia=hoje' + (forcar ? '&forcar=true' : ''));
  const el = document.getElementById('jogos-hoje');
  if (!d.jogos.length) { el.innerHTML = '<i class="card">Nenhum jogo real das ligas de interesse hoje (Premier League, La Liga, Championship, Brasileirão Série A/B, MLS).</i>'; return; }
  el.innerHTML = d.jogos.map(j => renderizarCard(j, d.grupos_ordem)).join('');
}

async function carregarJogosAmanha() {
  const d = await buscar('/api/jogos-do-dia?dia=amanha');
  const el = document.getElementById('jogos-amanha');
  if (!d.jogos.length) { el.innerHTML = '<i>Nenhum jogo real confirmado pra amanhã ainda.</i>'; return; }
  let html = '<table><tr><th>Liga</th><th>Jogo</th><th>Horário</th></tr>';
  for (const j of d.jogos) {
    const liga = j.modelado ? j.liga : `${j.liga} <span class="etiqueta-sem-edge">⚠️ sem modelo</span>`;
    html += `<tr><td>${liga}</td><td>${j.casa} x ${j.fora}</td><td>${j.commence_time}</td></tr>`;
  }
  el.innerHTML = html + '</table>';
}

async function carregarRegistros() {
  const liga = document.getElementById('filtro-liga').value;
  const familia = document.getElementById('filtro-familia').value;
  const params = new URLSearchParams();
  if (liga) params.set('liga', liga);
  if (familia) params.set('familia', familia);
  const d = await buscar('/api/registros?' + params.toString());
  const el = document.getElementById('registros');
  if (!d.registros.length) { el.innerHTML = '<i>Nenhum registro ainda.</i>'; return; }
  let html = '<table><tr><th>Data</th><th>Liga</th><th>Jogo</th><th>Mercado/Seleção</th><th>Odd</th><th>EV</th><th>Status</th><th>CLV</th><th>Resultado</th><th></th></tr>';
  for (const r of d.registros) {
    const clvClasse = r.clv > 0 ? 'positivo' : (r.clv < 0 ? 'negativo' : '');
    const clvTxt = r.clv != null ? fmtPct(r.clv) : '-';
    const acao = (r.status === 'aberto')
      ? `<button onclick="marcarResultado(${r.id}, 'ganhou')">ganhou</button> <button onclick="marcarResultado(${r.id}, 'perdeu')">perdeu</button>`
      : '';
    html += `<tr><td>${r.data_jogo||'-'}</td><td>${r.liga}</td><td>${r.time_casa} x ${r.time_fora}</td>`
      + `<td>${r.mercado}/${r.selecao}</td><td>${fmtOdd(r.odd_registrada)}</td><td>${fmtPct(r.ev)}</td>`
      + `<td>${r.status}</td><td class="${clvClasse}">${clvTxt}</td><td>${r.resultado||'-'}</td><td>${acao}</td></tr>`;
  }
  el.innerHTML = html + '</table>';
}

async function marcarResultado(id, resultado) {
  await fetch(`/api/registros/${id}/resultado?resultado=${resultado}`, { method: 'POST' });
  await carregarRegistros();
}

async function carregarGrafico() {
  const d = await buscar('/api/clv-serie');
  const el = document.getElementById('grafico');
  if (!d.pontos.length) { el.innerHTML = '<i>Sem registros fechados ainda para plotar.</i>'; return; }

  const w = 760, h = 260, pad = 40;
  const xs = d.pontos.map(p => p.indice);
  const todosValores = d.pontos.flatMap(p => [p.clv_ic_inferior, p.clv_ic_superior, p.roi_ic_inferior, p.roi_ic_superior]);
  const yMin = Math.min(...todosValores), yMax = Math.max(...todosValores);
  const xScale = x => pad + (x - 1) / Math.max(1, (xs.length - 1)) * (w - 2*pad);
  const yScale = y => h - pad - (y - yMin) / (yMax - yMin || 1) * (h - 2*pad);

  function linha(chave) {
    return d.pontos.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.indice)} ${yScale(p[chave])}`).join(' ');
  }
  function banda(chaveInf, chaveSup) {
    const ida = d.pontos.map(p => `${xScale(p.indice)},${yScale(p[chaveSup])}`).join(' ');
    const volta = [...d.pontos].reverse().map(p => `${xScale(p.indice)},${yScale(p[chaveInf])}`).join(' ');
    return `${ida} ${volta}`;
  }

  const svg = `
  <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <line x1="${pad}" y1="${yScale(0)}" x2="${w-pad}" y2="${yScale(0)}" stroke="#666" stroke-dasharray="4 4"/>
    <polygon points="${banda('clv_ic_inferior','clv_ic_superior')}" fill="#3fb950" opacity="0.15"/>
    <path d="${linha('clv_medio')}" fill="none" stroke="#3fb950" stroke-width="2"/>
    <polygon points="${banda('roi_ic_inferior','roi_ic_superior')}" fill="#58a6ff" opacity="0.12"/>
    <path d="${linha('roi_medio')}" fill="none" stroke="#58a6ff" stroke-width="2"/>
    <text x="${pad}" y="16" fill="#3fb950" font-size="12">— CLV médio acumulado</text>
    <text x="${pad+180}" y="16" fill="#58a6ff" font-size="12">— ROI de papel acumulado</text>
  </svg>`;
  el.innerHTML = svg + '<div style="font-size:0.75rem;opacity:0.7;margin-top:0.3rem">Faixa sombreada = intervalo de confiança de 95%. Enquanto a faixa cruzar o zero, a diferença de CLV/ROI não é estatisticamente distinguível de ruído.</div>';
}

async function carregarStatusOddspapi() {
  const s = await buscar('/api/oddspapi/status');
  document.getElementById('oddspapi-status').innerHTML =
    `Cota: ${s.gasto}/${s.limite} usos (restam ${s.restante})`;
}

async function buscarOddspapi() {
  const el = document.getElementById('oddspapi-resultado');
  el.innerHTML = 'Buscando na OddsPapi e cruzando com o modelo (o ajuste do modelo demora alguns segundos por jogo)…';
  const d = await buscar('/api/oddspapi/buscar', { method: 'POST' });
  await carregarStatusOddspapi();
  if (!d.sucesso) { el.innerHTML = `⚠️ ${d.erro}`; return; }
  if (!d.jogos.length) { el.innerHTML = '<i>Nenhum jogo com odds Pinnacle no momento pra essas ligas.</i>'; return; }
  let html = '';
  for (const j of d.jogos) {
    const rotuloModelo = j.tem_modelo ? '' : ' — <span class="etiqueta-sem-edge">⚠️ sem modelo (Série B) — só a odd</span>';
    const mercados = [...j.mercados].sort((a, b) => (b.ev ?? -99) - (a.ev ?? -99));
    html += `<div class="jogo-card"><b>${j.casa} x ${j.fora}</b> (${j.liga}) — ${j.commence_time}${rotuloModelo}`
      + '<table><tr><th>Mercado</th><th>Seleção</th><th>Odd</th><th>Prob. modelo</th><th>EV</th></tr>';
    for (const m of mercados) {
      const evClasse = m.ev > 0 ? 'positivo' : (m.ev < 0 ? 'negativo' : '');
      const evTxto = m.suspeito ? `${fmtPct(m.ev)} ⚠️ suspeito` : fmtPct(m.ev);
      html += `<tr><td>${m.mercado}</td><td>${m.selecao}</td><td>${fmtOdd(m.odd)}</td>`
        + `<td>${fmtPct(m.prob_modelo)}</td><td class="${evClasse}">${evTxto}</td></tr>`;
    }
    html += '</table>';
    if (j.melhor_aposta) {
      const a = j.melhor_aposta;
      html += `<div class="destaque-aposta">🎯 <b>Melhor aposta:</b> ${a.mercado}/${a.selecao} — odd ${fmtOdd(a.odd)} | prob. ${fmtPct(a.prob_modelo)} | EV ${fmtPct(a.ev)}</div>`;
    }
    if (j.melhor_bilhete) {
      const b = j.melhor_bilhete;
      const pernasTxt = b.pernas.map(p => `${p.mercado}/${p.selecao} (odd ${fmtOdd(p.odd)})`).join(' + ');
      html += `<div class="destaque-bilhete">🎟️ <b>Melhor bilhete:</b> ${pernasTxt} — odd combinada ${fmtOdd(b.odd_combinada)} | prob. conjunta ${fmtPct(b.prob_conjunta)} | EV ${fmtPct(b.ev_combinado)}</div>`;
    }
    html += '</div>';
  }
  el.innerHTML = html;
}

async function atualizarTudo(forcar) {
  await Promise.all([
    carregarStatus(), carregarStatusSistema(), carregarMaioresProbabilidades(), carregarApostarias(),
    carregarJogosHoje(forcar), carregarJogosAmanha(), carregarRegistros(), carregarGrafico(),
    carregarStatusOddspapi(),
  ]);
}
atualizarTudo(false);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def pagina_inicial():
    return PAGINA_HTML


def main():
    import uvicorn
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8000"))
    logger.info(f"Painel disponível em http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
