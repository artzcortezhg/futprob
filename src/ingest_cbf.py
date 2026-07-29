# -*- coding: utf-8 -*-
"""
Resultados (Série B) + cartões/árbitro (Série A e B) direto do site oficial
da CBF — a súmula on-line é um documento público, sem login, e cobre pelo
menos 2018 em diante (testado). Ela NÃO tem escanteios/faltas (isso vem de
ge.globo, ver ingest_geglobo.py) — súmula brasileira sempre foi sobre
disciplina (gols, cartões, substituições), nunca estatística tática.

Duas responsabilidades bem separadas:
1. `listar_jogos_temporada`: enumera os confrontos de uma temporada (via a
   tabela de rodadas do site, um único carregamento de página por temporada
   — trocar de rodada é só um <select>, não precisa recarregar).
2. `obter_link_sumula` + `parsear_sumula`: por partida, acha o PDF da súmula
   e extrai árbitro + contagem de cartões amarelos/vermelhos por lado
   (casa/fora, usando o próprio cabeçalho "Jogo: X / UF Y" da súmula pra
   saber quem é quem — nunca assume pela ordem alfabética).
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pdfplumber
import requests
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parent.parent
PASTA_CACHE_SUMULAS = RAIZ / "data" / "raw" / "sumulas_cbf"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

SERIE_URL = {
    "brasileirao": "serie-a",
    "brasileirao_b": "serie-b",
}


_RE_BLOCO_RODADA = re.compile(
    r"([A-ZÀ-Ý]{3})\n(\d+)\nX\n([A-ZÀ-Ý]{3})\n(\d+)\nJogo \d+\n"
    r"(?:\d+ Altera.{1,4}o\n)?\n(\d{2}/\d{2}/\d{4}) - (\d{2}:\d{2})"
)


def listar_jogos_temporada(page, liga: str, temporada: int) -> list[dict]:
    """Devolve todos os confrontos da temporada: [{"rodada":, "url_jogo":,
    "time_casa_slug":, "time_fora_slug":, "id_jogo_cbf":, "data_iso":,
    "gols_casa":, "gols_fora":}]. Um load de página, depois só troca o
    <select> de rodada (rápido — não recarrega o DOM inteiro). Placar e
    data vêm do texto da própria listagem (mesma ordem dos links
    "documentos do jogo" na página, por isso o zip por índice) — pra
    Série B isso é o ÚNICO jeito de saber o resultado, já que não existe
    fonte alguma pra ela em data/processed/partidas.csv hoje."""
    serie = SERIE_URL[liga]
    url_base = f"https://www.cbf.com.br/futebol-brasileiro/tabelas/campeonato-brasileiro/{serie}/{temporada}"
    page.goto(url_base, timeout=30000, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)

    opcoes = page.eval_on_selector("select", "el => Array.from(el.options).map(o => o.value)")
    rodadas = [o for o in opcoes if o.isdigit()]

    jogos = []
    for rodada in rodadas:
        page.select_option("select", rodada)
        page.wait_for_timeout(1500)
        links = page.eval_on_selector_all(
            "a", "els => els.map(e => [e.textContent.trim(), e.href])"
        )
        hrefs_partida = []
        for texto, href in links:
            if "DOCUMENTOS" not in texto.upper():
                continue
            m = re.search(rf"/{serie}/{temporada}/([a-z0-9-]+)-x-([a-z0-9-]+)/(\d+)$", href)
            if m:
                hrefs_partida.append((href, m))

        blocos = _RE_BLOCO_RODADA.findall(page.inner_text("body"))
        if len(blocos) != len(hrefs_partida):
            # ordem/contagem não bateram -- não arrisca casar placar errado
            # com o jogo errado, melhor faltar data/placar que inventar
            blocos = [None] * len(hrefs_partida)

        for (href, m), bloco in zip(hrefs_partida, blocos):
            item = {
                "rodada": int(rodada), "url_jogo": href,
                "time_casa_slug": m.group(1), "time_fora_slug": m.group(2),
                "id_jogo_cbf": m.group(3),
                "data_iso": None, "gols_casa": None, "gols_fora": None,
            }
            if bloco is not None:
                _, gols_casa, _, gols_fora, data_br, _ = bloco
                dia, mes, ano = data_br.split("/")
                item["data_iso"] = f"{ano}-{mes}-{dia}"
                item["gols_casa"] = int(gols_casa)
                item["gols_fora"] = int(gols_fora)
            jogos.append(item)
    return jogos


def obter_link_sumula(page, url_jogo: str) -> str | None:
    page.goto(f"{url_jogo}?view=documentos", timeout=25000, wait_until="domcontentloaded")
    page.wait_for_timeout(2500)
    links = page.eval_on_selector_all("a", "els => els.map(e => [e.textContent.trim(), e.href])")
    for texto, href in links:
        if texto.strip().lower() == "súmula" or "sumulas" in href and href.endswith("se.pdf"):
            return href
    return None


_RE_CABECALHO = re.compile(r"Jogo:\s*(.+?)\s*/\s*[A-Z]{2}\s*X\s*(.+?)\s*/\s*[A-Z]{2}")
_RE_ARBITRO = re.compile(r"Arbitro:\s*(.+?)\s*\(")
_RE_CARTOES_AMARELOS = re.compile(r"Cart.es Amarelos(.*?)(?:Cart.es Vermelhos|$)", re.DOTALL)
_RE_CARTOES_VERMELHOS = re.compile(r"Cart.es Vermelhos(.*?)$", re.DOTALL)
# Cada linha de cartão: "<tempo> <1T/2T> <nº ou AT/TC> <nome...> <Time/UF>".
# O nº separado (grupo 2) é o que distingue jogador (numérico) de comissão
# técnica (AT=assistente, TC=técnico) — súmulas de futebol brasileiro
# incluem cartão de comissão técnica, mas o mercado de "cartões" de aposta
# é sempre sobre jogadores em campo, então comissão técnica NUNCA conta.
_RE_LINHA_CARTAO = re.compile(
    r"^\+?\d{2}:\d{2}\s+\dT\s+(\S+)\s+.+?\s+([A-Za-zÀ-ÿ .\-]+/[A-Z]{2})$",
    re.MULTILINE,
)


def parsear_sumula(pdf_bytes: bytes) -> dict:
    """Extrai o texto do PDF e delega pra _parsear_texto_sumula (mantido
    separado pra poder testar o parsing com um texto sintético, sem
    precisar gerar um PDF de verdade)."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        texto = "\n".join(p.extract_text() or "" for p in pdf.pages)
    return _parsear_texto_sumula(texto)


def _parsear_texto_sumula(texto: str) -> dict:
    """Extrai árbitro + contagem de cartões por lado (casa/fora), usando o
    cabeçalho da própria súmula pra saber os nomes oficiais dos dois times
    (nunca confia na ordem alfabética nem no slug da URL, que pode abreviar
    diferente — ex.: 'Corinthians / SP' no cabeçalho vs 'corinthians' no
    slug)."""
    m_cab = _RE_CABECALHO.search(texto)
    if m_cab is None:
        raise ValueError("súmula sem cabeçalho reconhecível (Jogo: X / UF Y X Z / UF W)")
    time_casa, time_fora = m_cab.group(1).strip(), m_cab.group(2).strip()

    m_arb = _RE_ARBITRO.search(texto)
    arbitro = m_arb.group(1).strip() if m_arb else None

    def _contar_por_time(bloco: str) -> tuple[int, int]:
        casa = fora = 0
        for numero, time_da_linha in _RE_LINHA_CARTAO.findall(bloco):
            if not numero.isdigit():
                continue  # comissão técnica (AT/TC) -- não conta pro mercado de cartões
            if time_casa.lower() in time_da_linha.lower():
                casa += 1
            elif time_fora.lower() in time_da_linha.lower():
                fora += 1
        return casa, fora

    m_am = _RE_CARTOES_AMARELOS.search(texto)
    hy, ay = _contar_por_time(m_am.group(1)) if m_am else (0, 0)
    m_verm = _RE_CARTOES_VERMELHOS.search(texto)
    hr, ar = _contar_por_time(m_verm.group(1)) if m_verm else (0, 0)

    return {"time_casa": time_casa, "time_fora": time_fora, "arbitro": arbitro,
            "HY": hy, "AY": ay, "HR": hr, "AR": ar}


def caminho_cache_sumula(liga: str, temporada: int, id_jogo_cbf: str) -> Path:
    return PASTA_CACHE_SUMULAS / liga / str(temporada) / f"{id_jogo_cbf}.pdf"


def obter_sumula_com_cache(page, liga: str, temporada: int, jogo: dict) -> dict | None:
    caminho = caminho_cache_sumula(liga, temporada, jogo["id_jogo_cbf"])
    if caminho.exists():
        conteudo = caminho.read_bytes()
        if not conteudo:
            return None
    else:
        link = obter_link_sumula(page, jogo["url_jogo"])
        caminho.parent.mkdir(parents=True, exist_ok=True)
        if link is None:
            caminho.write_bytes(b"")
            return None
        resp = None
        for tentativa in range(3):
            try:
                resp = requests.get(link, headers={"User-Agent": USER_AGENT}, timeout=30)
                break
            except requests.exceptions.RequestException:
                # falha transitória de rede (DNS, timeout) -- não pode
                # derrubar uma coleta de horas por causa de um blip;
                # tenta de novo antes de desistir dessa súmula específica
                if tentativa == 2:
                    raise
                time.sleep(5)
        if resp.status_code != 200:
            caminho.write_bytes(b"")
            return None
        conteudo = resp.content
        caminho.write_bytes(conteudo)

    try:
        return parsear_sumula(conteudo)
    except Exception:
        return None


def coletar_temporada(liga: str, temporada: int, atraso: float = 0.5) -> list[dict]:
    """Enumera + baixa tudo de uma temporada (usa cache em disco -- reruns
    só buscam o que falta). Devolve uma linha por jogo com rodada, times
    (slug da CBF), árbitro e cartões -- SEM escanteios/faltas (outra fonte)."""
    resultado = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        jogos = listar_jogos_temporada(page, liga, temporada)
        for jogo in jogos:
            sumula = obter_sumula_com_cache(page, liga, temporada, jogo)
            resultado.append({**jogo, "temporada": temporada, "liga": liga, "sumula": sumula})
            time.sleep(atraso)
        browser.close()
    return resultado
