# -*- coding: utf-8 -*-
"""
Coletor de odds (Playwright) para Betano BR — adaptado do projeto
surebet-bot (C:\\Users\\Arthur\\surebet-bot\\collectors\\playwright_br.py).

Escopo corrigido: SOMENTE Betano (Superbet foi removida do código e do
agendamento — só trazia h2h de qualquer forma nos testes reais). A coluna
`casa_apostas` em odds_coletadas continua genérica (texto livre), então o
banco já está pronto para receber "superbet" de novo no futuro sem
migração, caso o coletor volte a incluí-la.

Adaptações em relação ao original:
- SOMENTE futebol (removidas as URLs de basquete/tênis/vôlei/handebol).
- SOMENTE pré-jogo (eventos com commence_time no passado são descartados).
- SOMENTE os mercados que o futprob modela: 1X2, over/under de gols, ambas
  marcam, dupla chance, escanteios (total e por time) e cartões — qualquer
  outro mercado capturado (chutes, escanteios de 1º/2º tempo, handicap
  asiático, faltas etc.) é descartado no final (ver _mercado_permitido).
- Adicionada detecção explícita de "ambas marcam" (btts) e "dupla chance"
  (double_chance) em _infer_market_key — o coletor original não tinha essas
  duas categorias (caíam no fallback genérico).
- Sem dependência do resto do projeto antigo: os dataclasses EventOdds/
  MarketOdds/OutcomeOdd são redefinidos aqui, self-contained.

Nunca roda em loop contínuo: cada chamada de coletar() é UMA foto (ver
Bloco 4 do pedido — duas fotos por rodada, manhã e pré-jogo, nunca coleta
contínua). Quem decide a frequência é o chamador (bot/agendador).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Response, Playwright

try:
    from playwright_stealth import Stealth as _Stealth
    _STEALTH = _Stealth()
    _HAS_STEALTH = True
except ImportError:
    _STEALTH = None
    _HAS_STEALTH = False

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("futprob.collector")

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB_PADRAO = RAIZ / "db" / "previsoes.sqlite"

# só os mercados que o futprob modela (ver src/markets.py, model_corners.py,
# model_cards.py) — prefixo/stat usado como filtro após a inferência da chave
STATS_PERMITIDOS = {"goals", "corners", "cards"}


@dataclass
class OutcomeOdd:
    name: str
    price: float
    bookmaker: str


@dataclass
class MarketOdds:
    market_key: str
    outcomes: list[OutcomeOdd] = field(default_factory=list)


@dataclass
class EventOdds:
    external_id: str
    home_team: str
    away_team: str
    commence_time: datetime | None
    bookmaker: str
    markets: list[MarketOdds] = field(default_factory=list)


# ── Helpers (portados do coletor original) ──────────────────────────────────

_STRIP_PARENS = re.compile(r'\s*\([^)]*\)')
_NOISE = frozenset(["club", "clube", "futebol", "esporte", "sport", "football"])
_OU_RE = re.compile(r'(\d+(?:\.\d+)?)')
_OVER_WORDS = frozenset(["acima", "over", "mais"])
_UNDER_WORDS = frozenset(["abaixo", "under", "menos"])


def _norm(name: str) -> str:
    name = _STRIP_PARENS.sub('', name).strip()
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    words = re.findall(r'[a-z0-9]+', ascii_str.lower())
    filtered = [w for w in words if len(w) > 2 and w not in _NOISE]
    return "".join(filtered) if filtered else "".join(words)


def _safe_float(v) -> float:
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0


def _parse_time(raw) -> datetime | None:
    if not raw:
        return None
    try:
        if isinstance(raw, (int, float)):
            ts = float(raw)
            if ts > 1e10:
                ts /= 1000
            return datetime.utcfromtimestamp(ts)
        if isinstance(raw, str):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    return None


def _ext_id(home: str, away: str) -> str:
    return f"soccer_{_norm(home)}_vs_{_norm(away)}"


_H2H_MARKET_WORDS = frozenset(["resultado", "1x2", "match result", "moneyline", "vencedor da partida", "winner"])
_H2H_EXACT_NAMES = frozenset([
    "vencedor", "vencedor da partida", "resultado final", "resultado",
    "1x2", "full time result", "match result", "moneyline", "match winner",
])
# NOVO em relação ao original: dupla chance e ambas marcam
# "empate anula a aposta"/"draw no bet" é um mercado DIFERENTE (2 seleções,
# casa-ou-reembolso / fora-ou-reembolso) de "dupla chance" (3 seleções,
# 1X/12/X2) — batia na mesma categoria antes e todo double_chance capturado
# na prática era Draw No Bet (2 seleções nomeadas com o nome do time), o
# que cruzava errado com o modelo de "Dupla chance" (soma de 1X2). Por isso
# viram chaves separadas; draw_no_bet não tem modelo no futprob e é
# descartado em _mercado_permitido — melhor não coletar do que coletar
# errado.
_DUPLA_CHANCE_WORDS = frozenset(["dupla chance", "double chance"])
_EMPATE_ANULA_WORDS = frozenset(["empate anula", "draw no bet"])
_BTTS_WORDS = frozenset(["ambas equipes marcam", "ambas marcam", "both teams to score", "btts"])


def _infer_market_key(market_name: str, outcome_names: list[str], home_name: str = "", away_name: str = "") -> str:
    """Normaliza market_name + outcomes -> chave canônica. Portado do
    coletor original, com duas categorias adicionadas (dupla_chance, btts)
    que o original não distinguia."""
    combined = (market_name + " " + " ".join(outcome_names)).lower()
    outcome_set = {n.lower().strip() for n in outcome_names}

    if any(kw in combined for kw in _DUPLA_CHANCE_WORDS):
        return "double_chance"
    if any(kw in combined for kw in _EMPATE_ANULA_WORDS):
        return "draw_no_bet"
    if any(kw in combined for kw in _BTTS_WORDS):
        return "btts"

    has_over = any(kw in combined for kw in _OVER_WORDS)
    has_under = any(kw in combined for kw in _UNDER_WORDS)

    if has_over or has_under:
        qualifier = ""
        if re.search(r'1[°º]\s*tempo|1st\s*half|primeiro\s*tempo', combined):
            qualifier = "_ht"
        elif re.search(r'2[°º]\s*tempo|2nd\s*half|segundo\s*tempo', combined):
            qualifier = "_2h"
        elif re.search(r'1[°º]\s*time|time\s+da\s+casa|home\s+team', combined):
            qualifier = "_team1"
        elif re.search(r'2[°º]\s*time|time\s+visit|away\s+team', combined):
            qualifier = "_team2"

        combined_clean = re.sub(r'\d+\s*[°º]\s*\w+', '', combined)
        matches = _OU_RE.findall(combined_clean)
        if not matches:
            pass
        else:
            line = matches[-1]
            if "chut" in combined and ("alvo" in combined or "gol" in combined):
                stat = "shots_ot"
            elif "chut" in combined:
                stat = "shots"
            elif "gol" in combined or "goal" in combined:
                stat = "goals"
            elif "cart" in combined or "card" in combined:
                stat = "redcards" if ("verm" in combined or "red" in combined) else "cards"
            elif "escan" in combined or "corner" in combined:
                stat = "corners"
            elif "falt" in combined:
                stat = "fouls"
            elif "ofens" in combined or "imped" in combined or "offside" in combined:
                stat = "offside"
            else:
                words = re.findall(r'[a-z]+', _norm(market_name))
                stat = "_".join(w for w in words if len(w) > 2)[:20] or "total"

            if not qualifier and (home_name or away_name):
                mn_norm = _norm(market_name)
                if home_name and _norm(home_name) in mn_norm:
                    qualifier = "_team1"
                elif away_name and _norm(away_name) in mn_norm:
                    qualifier = "_team2"

            return f"ou_{line}_{stat}{qualifier}"

    mn_lower = market_name.lower()
    if (mn_lower in _H2H_EXACT_NAMES
            or outcome_set & {"1", "x", "2", "empate", "draw"}
            or any(w in mn_lower for w in _H2H_MARKET_WORDS)):
        return "h2h"

    key = _norm(market_name)[:25]
    return key or "misc"


def _mercado_permitido(market_key: str) -> bool:
    """Filtro final: só os mercados que o futprob modela. Descarta chutes,
    escanteios/gols de 1º-2º tempo, handicap, faltas, cartões vermelhos etc."""
    if market_key in ("h2h", "double_chance", "btts"):
        return True
    if market_key.startswith("ou_") and not market_key.endswith(("_ht", "_2h")):
        partes = market_key.split("_")
        stat = partes[2] if len(partes) > 2 else ""
        return stat in STATS_PERMITIDOS
    return False


def _is_odds_json(data: Any) -> bool:
    if not isinstance(data, (dict, list)):
        return False
    text = json.dumps(data, ensure_ascii=False)
    if len(text) < 300:
        return False
    indicadores = [
        "matchName", "marketIdList", "selectionIdList", "participants", "selections",
        "betOffers", "outcomes", "runners", "markets", "odds", "fixture", "match",
    ]
    return sum(1 for i in indicadores if i in text) >= 2


# ── Parsers (Betano / Superbet) ──────────────────────────────────────────────

def _parse_betano(data: Any, bookmaker: str) -> list[EventOdds]:
    resultados: list[EventOdds] = []
    vistos: set = set()
    candidatos: list[dict] = []

    def _achar(d, prof=0):
        if prof > 12 or not isinstance(d, dict):
            return
        if "events" in d and "markets" in d and "selections" in d:
            candidatos.append(d)
            return
        for v in d.values():
            if isinstance(v, dict):
                _achar(v, prof + 1)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _achar(item, prof + 1)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _achar(item)
    else:
        _achar(data)

    for bloco in candidatos:
        raw_eventos = bloco.get("events", {})
        raw_mercados = bloco.get("markets", {})
        raw_sels = bloco.get("selections", {})
        if not isinstance(raw_eventos, dict):
            continue

        for ev_id, ev in raw_eventos.items():
            if not isinstance(ev, dict):
                continue
            # a listagem geral embute eventos de VÁRIOS esportes (mesmo numa
            # página só de futebol) — sportId='FOOT' é o filtro real; sem
            # isso, jogos de beisebol/basquete etc. vazavam pro futprob
            sport_id = ev.get("sportId")
            if sport_id is not None and sport_id != "FOOT":
                continue
            parts = ev.get("participants", [])
            if not isinstance(parts, list) or len(parts) < 2:
                continue
            home = (parts[0].get("name") or parts[0].get("shortName", "")).strip()
            away = (parts[1].get("name") or parts[1].get("shortName", "")).strip()
            if len(home) < 2 or len(away) < 2:
                continue

            ev_url = ev.get("url") or ev.get("eventUrl") or ev.get("path")
            if ev_url and ev_url.startswith("/virtuals/"):
                continue
            combinado = (home + away).lower()
            if "esport" in combinado or "(sim)" in combinado:
                continue

            eid = _ext_id(home, away)
            if eid in vistos:
                continue
            start_raw = (ev.get("startDate") or ev.get("startTime") or ev.get("startsAt")
                         or ev.get("startAt") or ev.get("startDateUtc") or "")
            evento = EventOdds(external_id=eid, home_team=home, away_team=away,
                                commence_time=_parse_time(start_raw), bookmaker=bookmaker)

            chaves_vistas: set = set()
            for mkt_id in ev.get("marketIdList", []):
                mkt = raw_mercados.get(str(mkt_id)) or raw_mercados.get(mkt_id)
                if not isinstance(mkt, dict):
                    continue
                mkt_nome = mkt.get("name") or mkt.get("type") or mkt.get("marketType") or mkt.get("typeName") or ""

                outcomes: list[tuple[str, float]] = []
                for sel_id in mkt.get("selectionIdList", []):
                    sel = raw_sels.get(str(sel_id)) or raw_sels.get(sel_id)
                    if not isinstance(sel, dict):
                        continue
                    preco = _safe_float(sel.get("price", 0))
                    nome = (sel.get("name") or sel.get("shortName", "")).strip()
                    if preco > 1.0 and nome:
                        outcomes.append((nome, preco))

                if len(outcomes) < 2:
                    continue
                mk = _infer_market_key(mkt_nome, [o[0] for o in outcomes], home, away)
                if not _mercado_permitido(mk) or mk in chaves_vistas:
                    continue
                chaves_vistas.add(mk)

                mkt_obj = MarketOdds(market_key=mk)
                for nome, preco in outcomes:
                    mkt_obj.outcomes.append(OutcomeOdd(name=nome, price=preco, bookmaker=bookmaker))
                evento.markets.append(mkt_obj)

            if evento.markets:
                resultados.append(evento)
                vistos.add(eid)

    return resultados



def _parse_betano_odds_bt(data: Any, bookmaker: str) -> EventOdds | None:
    """Parser do endpoint bt=12 (odds de UM jogo específico da Betano) — é
    esse endpoint, não a listagem geral, que traz os mercados completos
    (cartões, escanteios etc). Portado do coletor original."""
    if not isinstance(data, dict):
        return None
    inner = data.get("data") or data
    if not isinstance(inner, dict):
        return None
    ev = inner.get("event")
    if not isinstance(ev, dict):
        return None
    parts = ev.get("participants", [])
    if not isinstance(parts, list) or len(parts) < 2:
        return None
    home = (parts[0].get("name") or "").strip()
    away = (parts[1].get("name") or "").strip()
    if len(home) < 2 or len(away) < 2:
        return None

    ev_url = ev.get("url")
    if ev_url and ev_url.startswith("/virtuals/"):
        return None
    if "esport" in (home + away).lower():
        return None

    eid = _ext_id(home, away)
    commence = _parse_time(ev.get("startTime")) if ev.get("startTime") else None
    evento = EventOdds(external_id=eid, home_team=home, away_team=away, commence_time=commence, bookmaker=bookmaker)

    markets_raw = ev.get("markets", [])
    if not isinstance(markets_raw, list):
        return None

    chaves_vistas: set = set()
    for mkt in markets_raw:
        if not isinstance(mkt, dict):
            continue
        mkt_nome = (mkt.get("name") or mkt.get("type") or "").strip()
        if not mkt_nome:
            continue
        sels_raw = mkt.get("selections", [])
        if not isinstance(sels_raw, list):
            continue

        por_handicap: dict[float, list[tuple[str, float]]] = {}
        for sel in sels_raw:
            if not isinstance(sel, dict):
                continue
            preco = _safe_float(sel.get("price", 0))
            nome_curto = (sel.get("name") or "").strip()
            hcap = _safe_float(sel.get("handicap") or mkt.get("handicap") or 0.0)
            if preco > 1.0 and nome_curto:
                por_handicap.setdefault(hcap, []).append((nome_curto, preco))

        for _hcap, grupo in por_handicap.items():
            if len(grupo) < 2:
                continue
            mk = _infer_market_key(mkt_nome, [g[0] for g in grupo], home, away)
            if not _mercado_permitido(mk) or mk in chaves_vistas:
                continue
            chaves_vistas.add(mk)
            mkt_obj = MarketOdds(market_key=mk)
            for nome_curto, preco in grupo:
                mkt_obj.outcomes.append(OutcomeOdd(name=nome_curto, price=preco, bookmaker=bookmaker))
            evento.markets.append(mkt_obj)

    return evento if evento.markets else None


_JS_EXTRAIR_BT12 = """(limit) => {
    const seen = new Set();
    const urls = [];
    for (const a of document.querySelectorAll('a[href*="/sport/futebol/"]')) {
        try {
            const path = new URL(a.href).pathname;
            if (path.includes('/virtuals/')) continue;
            const parts = path.split('/').filter(Boolean);
            for (let i = parts.length - 1; i >= 1; i--) {
                if (/^\\d{7,10}$/.test(parts[i])) {
                    const slug = parts[i - 1];
                    const u = '/api/odds/' + slug + '/' + parts[i] + '/?bt=12&req=s,stnf,c';
                    if (!seen.has(u)) { seen.add(u); urls.push(u); }
                    break;
                }
            }
        } catch(e) {}
    }
    const batch = urls.slice(0, limit);
    batch.forEach(u => fetch(u, {credentials: 'include'}).catch(() => {}));
    return {count: urls.length, sample: batch.slice(0, 3)};
}"""

# páginas de competição para garantir cobertura de campeonatos brasileiros
# (a página geral de futebol às vezes não lista todos os jogos do dia)
_COMPETICOES_BETANO = [
    "https://www.betano.bet.br/sport/futebol/brasil/brasileirao-serie-a-betano/10016/",
    "https://www.betano.bet.br/sport/futebol/proximos-jogos/",
]


async def _disparar_bt12_betano(page: Page) -> int:
    """Clica 'próximos jogos' e dispara as chamadas bt=12 (odds completas)
    de cada jogo visível na página atual. Retorna quantas URLs disparou."""
    total = 0
    try:
        clicado = await page.evaluate("""() => {
            const kws = ['próximos', 'proximos', 'pré-jogo', 'pre-jogo', 'upcoming'];
            for (const el of document.querySelectorAll('a, button, [role="tab"], li')) {
                const txt = (el.textContent || '').trim().toLowerCase();
                if (kws.some(k => txt === k || txt.startsWith(k))) { el.click(); return txt; }
            }
            return null;
        }""")
        if clicado:
            await asyncio.sleep(3)
    except Exception:
        pass

    try:
        res = await page.evaluate(_JS_EXTRAIR_BT12, 150)
        n = res.get("count", 0) if isinstance(res, dict) else 0
        if n:
            total += n
            await asyncio.sleep(6)
    except Exception as exc:
        logger.debug(f"[betano] bt12 principal falhou: {exc}")

    return total


# ── Coleta (Playwright) ──────────────────────────────────────────────────────

# Superbet removida do escopo (só trazia h2h nos testes reais, ver Fase
# ponta-a-ponta). Estrutura em dict mantida — basta adicionar de volta uma
# entrada aqui e um parser para reativar, sem mudança nenhuma no schema.
_BOOKMAKERS = {
    "betano": {"url": "https://www.betano.bet.br/sport/futebol/", "parser": _parse_betano},
}


async def _coletar_casa(browser: Browser, nome: str, config: dict) -> tuple[list[EventOdds], list[Any]]:
    """Retorna (eventos_parseados, payloads_json_brutos) — os brutos são
    devolvidos pra quem chamar poder guardar tudo que foi capturado, mesmo
    o que o mapeador atual ainda não sabe traduzir (ver salvar_bruto_coleta:
    dado coletado nunca mais se perde por dicionário de mercado incompleto,
    uma correção futura no mapeador pode reprocessar sem precisar coletar
    de novo)."""
    capturado: list[Any] = []
    context: BrowserContext = await browser.new_context(
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        viewport={"width": 1366, "height": 768}, locale="pt-BR", timezone_id="America/Sao_Paulo",
    )
    await context.route(
        "**/*.{png,jpg,jpeg,gif,svg,ico,webp,woff,woff2,ttf,mp4,webm}",
        lambda route: route.abort(),
    )
    page: Page = await context.new_page()

    if _HAS_STEALTH and nome != "betano":
        await _STEALTH.apply_stealth_async(page)

    async def on_response(response: Response):
        try:
            if response.status != 200:
                return
            ct = response.headers.get("content-type", "")
            if "json" not in ct and "text/plain" not in ct:
                return
            body = await response.json()
            if _is_odds_json(body):
                capturado.append(body)
        except Exception:
            pass

    page.on("response", on_response)

    try:
        wait_until = "networkidle" if nome == "betano" else "domcontentloaded"
        try:
            await page.goto(config["url"], wait_until=wait_until, timeout=60000)
        except Exception:
            logger.debug(f"[{nome}] goto timeout, continuando")

        if nome != "betano":
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                await asyncio.sleep(6)

        try:
            html = await page.content()
            nd_match = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>([^<]+)</script>', html)
            if nd_match:
                capturado.append(json.loads(nd_match.group(1)))
        except Exception:
            pass

        for scroll_y in range(0, 12000, 800):
            await page.evaluate(f"window.scrollTo(0, {scroll_y})")
            await asyncio.sleep(0.3)
        await asyncio.sleep(4)

        for js_var in ["window.__NEXT_DATA__", "window.__INITIAL_STATE__"]:
            try:
                val = await page.evaluate(f"() => JSON.stringify({js_var} || null)")
                if val and val != "null":
                    dado = json.loads(val)
                    if _is_odds_json(dado):
                        capturado.append(dado)
            except Exception:
                pass

        # Betano: os mercados completos (cartões, escanteios) só vêm do
        # endpoint por-jogo (bt=12) — a listagem geral só traz h2h
        if nome == "betano":
            await _disparar_bt12_betano(page)
            for comp_url in _COMPETICOES_BETANO:
                try:
                    await page.goto(comp_url, wait_until="domcontentloaded", timeout=15000)
                    await asyncio.sleep(3)
                    await _disparar_bt12_betano(page)
                except Exception as exc:
                    logger.debug(f"[betano] competição {comp_url} falhou: {exc}")
    except Exception as exc:
        logger.warning(f"[{nome}] falha ao navegar: {exc}")
    finally:
        await context.close()

    parser_fn = config["parser"]
    mesclados: dict[str, EventOdds] = {}

    for dado in capturado:
        # respostas bt=12 (odds completas de 1 jogo) usam parser dedicado
        if nome == "betano" and isinstance(dado, dict):
            inner = dado.get("data", {})
            ev_bt = inner.get("event", {}) if isinstance(inner, dict) else {}
            if (isinstance(ev_bt, dict) and ev_bt.get("sportId")
                    and isinstance(ev_bt.get("markets"), list) and len(ev_bt.get("markets", [])) > 3):
                evento = _parse_betano_odds_bt(dado, nome)
                eventos = [evento] if evento else []
            else:
                eventos = parser_fn(dado, nome)
        else:
            eventos = parser_fn(dado, nome)

        for ev in eventos:
            if ev.external_id not in mesclados:
                mesclados[ev.external_id] = ev
            else:
                existente = mesclados[ev.external_id]
                ja_tem = {m.market_key for m in existente.markets}
                for mkt in ev.markets:
                    if mkt.market_key not in ja_tem:
                        existente.markets.append(mkt)
                        ja_tem.add(mkt.market_key)

    resultados = list(mesclados.values())

    # SOMENTE pré-jogo: descarta eventos já iniciados
    agora = datetime.utcnow()
    resultados = [ev for ev in resultados if not ev.commence_time or ev.commence_time > agora]

    return resultados, capturado


async def coletar(casas: list[str] | None = None, timeout_total_s: int = 120,
                   ) -> tuple[dict[str, list[EventOdds]], dict[str, list[Any]]]:
    """Uma foto de coleta (pré-jogo, só futebol, só os mercados do futprob).
    Retorna (eventos_por_casa, payloads_brutos_por_casa) — o segundo é pra
    persistir com salvar_bruto_coleta e nunca perder dado já coletado.
    Nunca roda em loop contínuo."""
    casas = casas or list(_BOOKMAKERS.keys())
    resultado: dict[str, list[EventOdds]] = {}
    bruto: dict[str, list[Any]] = {}

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True, channel="chrome",
                                                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        except Exception:
            browser = await pw.chromium.launch(headless=True,
                                                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        try:
            tarefas = {nome: _coletar_casa(browser, nome, _BOOKMAKERS[nome]) for nome in casas if nome in _BOOKMAKERS}
            feitas = await asyncio.wait_for(asyncio.gather(*tarefas.values(), return_exceptions=True), timeout=timeout_total_s)
            for nome, res in zip(tarefas.keys(), feitas):
                if isinstance(res, Exception):
                    logger.warning(f"[{nome}] erro: {res}")
                    resultado[nome], bruto[nome] = [], []
                else:
                    resultado[nome], bruto[nome] = res
        finally:
            await browser.close()

    return resultado, bruto


# ── Persistência (snapshot no SQLite) ────────────────────────────────────────

SQL_CRIAR_TABELA_ODDS = """
CREATE TABLE IF NOT EXISTS odds_coletadas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coletado_em TEXT NOT NULL,
    tipo_foto TEXT NOT NULL,
    casa_apostas TEXT NOT NULL,
    time_casa_coletado TEXT NOT NULL,
    time_fora_coletado TEXT NOT NULL,
    commence_time TEXT,
    mercado TEXT NOT NULL,
    selecao TEXT NOT NULL,
    odd REAL NOT NULL
);
"""


def salvar_snapshot(resultado: dict[str, list[EventOdds]], tipo_foto: str, caminho_db: Path = CAMINHO_DB_PADRAO) -> int:
    """Grava a foto coletada no SQLite com timestamp e casa de apostas.
    Nunca apaga fotos anteriores — cada chamada só insere linhas novas."""
    caminho_db.parent.mkdir(parents=True, exist_ok=True)
    coletado_em = datetime.now().isoformat(timespec="seconds")
    linhas = []
    for casa, eventos in resultado.items():
        for ev in eventos:
            for mkt in ev.markets:
                for outcome in mkt.outcomes:
                    linhas.append((
                        coletado_em, tipo_foto, casa, ev.home_team, ev.away_team,
                        ev.commence_time.isoformat() if ev.commence_time else None,
                        mkt.market_key, outcome.name, outcome.price,
                    ))

    with sqlite3.connect(caminho_db) as conn:
        conn.executescript(SQL_CRIAR_TABELA_ODDS)
        conn.executemany(
            """INSERT INTO odds_coletadas
               (coletado_em, tipo_foto, casa_apostas, time_casa_coletado, time_fora_coletado,
                commence_time, mercado, selecao, odd)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            linhas,
        )
        conn.commit()
    return len(linhas)


SQL_CRIAR_TABELA_BRUTO = """
CREATE TABLE IF NOT EXISTS coletas_brutas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coletado_em TEXT NOT NULL,
    tipo_foto TEXT NOT NULL,
    casa_apostas TEXT NOT NULL,
    sequencia INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def salvar_bruto_coleta(bruto: dict[str, list[Any]], tipo_foto: str, caminho_db: Path = CAMINHO_DB_PADRAO) -> int:
    """Guarda CADA payload JSON bruto capturado na coleta, com data — pra
    que uma correção futura no mapeador (_infer_market_key, _mercados_para_
    jogo etc.) possa reprocessar tudo que já foi coletado sem precisar de
    uma nova coleta ao vivo. Nunca apaga coletas brutas anteriores."""
    caminho_db.parent.mkdir(parents=True, exist_ok=True)
    coletado_em = datetime.now().isoformat(timespec="seconds")
    linhas = []
    for casa, payloads in bruto.items():
        for i, payload in enumerate(payloads):
            try:
                payload_json = json.dumps(payload, ensure_ascii=False)
            except Exception:
                continue
            linhas.append((coletado_em, tipo_foto, casa, i, payload_json))

    with sqlite3.connect(caminho_db) as conn:
        conn.executescript(SQL_CRIAR_TABELA_BRUTO)
        conn.executemany(
            "INSERT INTO coletas_brutas (coletado_em, tipo_foto, casa_apostas, sequencia, payload_json) "
            "VALUES (?,?,?,?,?)",
            linhas,
        )
        conn.commit()
    return len(linhas)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Coleta de odds pré-jogo (Betano/Superbet)")
    parser.add_argument("--tipo", choices=["manha", "prejogo"], default="manha")
    parser.add_argument("--casas", nargs="+", default=None)
    args = parser.parse_args()

    from painel_db import registrar_coleta

    resultado, bruto = asyncio.run(coletar(args.casas))
    n_jogos = sum(len(evs) for evs in resultado.values())
    n_mercados = sum(len(mkt.outcomes) for evs in resultado.values() for ev in evs for mkt in ev.markets)

    for casa, eventos in resultado.items():
        print(f"[{casa}] {len(eventos)} jogos pré-jogo capturados")
        for ev in eventos[:5]:
            mercados = sorted({m.market_key for m in ev.markets})
            print(f"  {ev.home_team} x {ev.away_team} ({ev.commence_time}): {mercados}")

    n_linhas = salvar_snapshot(resultado, args.tipo)
    n_brutos = salvar_bruto_coleta(bruto, args.tipo)
    sucesso = n_jogos > 0
    registrar_coleta(
        CAMINHO_DB_PADRAO, "betano", sucesso=sucesso, tipo=args.tipo,
        mensagem="ok" if sucesso else "nenhum jogo capturado",
        n_jogos_capturados=n_jogos, n_mercados_capturados=n_mercados,
    )
    print(f"\n{n_linhas} linhas de odds salvas em {CAMINHO_DB_PADRAO} (tipo={args.tipo}); "
          f"{n_brutos} payloads brutos guardados em coletas_brutas.")


if __name__ == "__main__":
    main()
