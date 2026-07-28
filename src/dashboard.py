# -*- coding: utf-8 -*-
"""
Painel web do futprob (FastAPI). Só leitura, exceto marcar manualmente o
resultado de um jogo já ocorrido — nenhuma ação do painel apaga ou corrompe
dados. Lê exclusivamente de db/previsoes.sqlite.

Host/porta configuráveis via .env (padrão: localhost, sem exposição
externa). Rodar com: python src/dashboard.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
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

load_dotenv(RAIZ / ".env")

from painel_db import inicializar_db_painel, marcar_resultado_manual  # noqa: E402
from predict import inicializar_db as inicializar_db_previsoes  # noqa: E402

app = FastAPI(title="futprob — painel")


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


@app.get("/api/jogos-do-dia")
def jogos_do_dia():
    """Últimas previsões geradas (predict.py) por confronto, com o EV de
    cada mercado quando houver odds coletadas casando com o mesmo
    liga+times+data em `registros` (fica vazio até o coletor do Bloco 4
    estar rodando de verdade)."""
    with _conectar() as conn:
        previsoes = pd.read_sql_query(
            """SELECT p.id, p.criado_em, p.liga, p.time_casa, p.time_fora, p.data_corte_modelo, p.alpha_xg
               FROM previsoes p
               WHERE p.id IN (
                   SELECT MAX(id) FROM previsoes GROUP BY liga, time_casa, time_fora
               )
               ORDER BY p.criado_em DESC LIMIT 30""",
            conn,
        )
        if previsoes.empty:
            return {"jogos": []}

        mercados = pd.read_sql_query(
            "SELECT previsao_id, mercado, selecao, probabilidade FROM previsoes_mercados "
            f"WHERE previsao_id IN ({','.join(map(str, previsoes['id']))})",
            conn,
        )
        registros = pd.read_sql_query(
            "SELECT liga, data_jogo, time_casa, time_fora, mercado, selecao, ev, status "
            "FROM registros WHERE status='aberto'",
            conn,
        )

    jogos = []
    for _, p in previsoes.iterrows():
        m = mercados[mercados["previsao_id"] == p["id"]]
        ev_do_jogo = registros[
            (registros["liga"] == p["liga"]) & (registros["time_casa"] == p["time_casa"]) & (registros["time_fora"] == p["time_fora"])
        ]
        jogos.append({
            "liga": p["liga"], "time_casa": p["time_casa"], "time_fora": p["time_fora"],
            "criado_em": p["criado_em"], "alpha_xg": p["alpha_xg"],
            "mercados_1x2": _sem_nan(m[m["mercado"] == "1X2"][["selecao", "probabilidade"]].to_dict("records")),
            "evs": _sem_nan(ev_do_jogo[["mercado", "selecao", "ev"]].to_dict("records")),
            "apostaria": bool((ev_do_jogo["ev"] > 0.05).any()) if not ev_do_jogo.empty else False,
        })
    return {"jogos": jogos}


@app.get("/api/registros")
def registros(status: str | None = None):
    with _conectar() as conn:
        query = "SELECT * FROM registros"
        params = []
        if status in ("aberto", "fechado"):
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY criado_em DESC LIMIT 500"
        df = pd.read_sql_query(query, conn, params=params)
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
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #333; }
  .badge { padding: 0.1rem 0.5rem; border-radius: 0.3rem; font-size: 0.75rem; }
  .apostaria { background: #2ea043; color: white; }
  .positivo { color: #3fb950; }
  .negativo { color: #f85149; }
  button { background: #238636; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.4rem; cursor: pointer; }
  #status-linha { font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem; }
  svg { background: #161b22; border-radius: 0.4rem; }
  .card { background: #161b22; border-radius: 0.5rem; padding: 1rem; margin-top: 0.5rem; }
</style>
</head>
<body>
<h1>futprob — painel</h1>
<div id="status-linha">carregando status…</div>
<button onclick="atualizarTudo()">Atualizar</button>

<h2>Jogos do dia (últimas previsões)</h2>
<div id="jogos" class="card">carregando…</div>

<h2>Registros — abertos e fechados</h2>
<div id="registros" class="card">carregando…</div>

<h2>Evolução do CLV médio e ROI de papel (com IC 95%)</h2>
<div id="grafico" class="card">carregando…</div>

<script>
async function buscar(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function carregarStatus() {
  const s = await buscar('/api/status-coleta');
  const uc = s.ultima_coleta;
  const linhaColeta = uc
    ? `Última coleta: ${uc.fonte} (${uc.tipo || '-'}) em ${uc.executado_em} — ${uc.sucesso ? 'sucesso' : 'FALHA'} — ${uc.mensagem || ''}`
    : 'Nenhuma coleta registrada ainda.';
  document.getElementById('status-linha').innerHTML =
    `${linhaColeta} · OddsPapi: ${s.oddspapi_gasto}/${s.oddspapi_limite} usos (restam ${s.oddspapi_restante})`;
}

async function carregarJogos() {
  const d = await buscar('/api/jogos-do-dia');
  const el = document.getElementById('jogos');
  if (!d.jogos.length) { el.innerHTML = '<i>Nenhuma previsão registrada ainda. Rode src/predict.py.</i>'; return; }
  let html = '<table><tr><th>Liga</th><th>Jogo</th><th>1X2 (modelo)</th><th>EV coletado</th><th></th></tr>';
  for (const j of d.jogos) {
    const probs = j.mercados_1x2.map(m => `${m.selecao}: ${(m.probabilidade*100).toFixed(1)}%`).join(' / ');
    const evs = j.evs.length ? j.evs.map(e => `${e.mercado}/${e.selecao}: ${(e.ev*100).toFixed(1)}%`).join('<br>') : '<i>sem odds coletadas</i>';
    const destaque = j.apostaria ? '<span class="badge apostaria">apostaria</span>' : '';
    html += `<tr><td>${j.liga}</td><td>${j.time_casa} x ${j.time_fora}</td><td>${probs}</td><td>${evs}</td><td>${destaque}</td></tr>`;
  }
  el.innerHTML = html + '</table>';
}

async function carregarRegistros() {
  const d = await buscar('/api/registros');
  const el = document.getElementById('registros');
  if (!d.registros.length) { el.innerHTML = '<i>Nenhum registro ainda.</i>'; return; }
  let html = '<table><tr><th>Data</th><th>Liga</th><th>Jogo</th><th>Mercado/Seleção</th><th>Odd</th><th>EV</th><th>Status</th><th>CLV</th><th>Resultado</th><th></th></tr>';
  for (const r of d.registros) {
    const clvClasse = r.clv > 0 ? 'positivo' : (r.clv < 0 ? 'negativo' : '');
    const clvTxt = r.clv != null ? (r.clv*100).toFixed(1) + '%' : '-';
    const acao = (r.status === 'aberto')
      ? `<button onclick="marcarResultado(${r.id}, 'ganhou')">ganhou</button> <button onclick="marcarResultado(${r.id}, 'perdeu')">perdeu</button>`
      : '';
    html += `<tr><td>${r.data_jogo||'-'}</td><td>${r.liga}</td><td>${r.time_casa} x ${r.time_fora}</td>`
      + `<td>${r.mercado}/${r.selecao}</td><td>${r.odd_registrada?.toFixed(2)}</td><td>${(r.ev*100).toFixed(1)}%</td>`
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

async function atualizarTudo() {
  await Promise.all([carregarStatus(), carregarJogos(), carregarRegistros(), carregarGrafico()]);
}
atualizarTudo();
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
    print(f"Painel disponível em http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
