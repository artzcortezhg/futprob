# -*- coding: utf-8 -*-
"""
Escanteios e faltas por partida do Brasileirão (Série A/B), via ge.globo.com
(Globo Esporte) — a única fonte gratuita e ESTÁVEL que encontramos com esses
dois mercados pro Brasileirão (football-data.co.uk não tem; FBref bloqueia
com Cloudflare; Flashscore/Sofascore têm proteção ativa contra automação em
escala). Testado manualmente em partidas de 2019 e 2026, sem sinal de
bloqueio — mas o widget de estatística às vezes demora a renderizar, por
isso os retries com espera adicional antes de desistir.

Cada página de jogo tem a URL:
    https://ge.globo.com/{estado}/futebol/{competicao}/jogo/{DD-MM-AAAA}/{time-casa}-{time-fora}.ghtml

`estado` é a UF do time mandante (ver ESTADO_TIME) — sem ela a página dá 404
mesmo que o resto da URL esteja certo (não existe redirecionamento). Times
com nome de slug diferente do normalizado (acentos/hífen) têm override em
SLUG_GLOBO.
"""
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parent.parent
PASTA_CACHE = RAIZ / "data" / "raw" / "geglobo"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

COMPETICAO = {
    "brasileirao": "brasileirao-serie-a",
    "brasileirao_b": "brasileirao-serie-b",
}

# Chave = slug da CBF (da URL "jogos/.../{slug}-x-{slug}/id") depois de
# tirar sufixo de reorganização societária (-saf) — ver _normalizar_slug_cbf.
# Cada entrada dá o estado do mandante (obrigatório na URL do ge.globo, sem
# ele é 404 direto — CBF e ge.globo não redirecionam um pro outro) e o slug
# que o PRÓPRIO ge.globo usa, quando é diferente do slug normalizado da CBF
# (ex.: CBF usa o nome completo "athletico-paranaense", ge.globo usa o
# apelido "atletico-pr" — confirmado comparando URLs reais do próprio
# ge.globo, nunca advinhado às cegas).
TIME_CBF_PARA_GLOBO: dict[str, dict[str, str]] = {
    "america": {"estado": "mg", "slug": "america-mg"},
    "athletico-paranaense": {"estado": "pr", "slug": "atletico-pr"},
    "atletico-goianiense": {"estado": "go", "slug": "atletico-go"},
    "atletico-mineiro": {"estado": "mg", "slug": "atletico-mg"},
    "avai": {"estado": "sc", "slug": "avai"},
    "bahia": {"estado": "ba", "slug": "bahia"},
    "botafogo": {"estado": "rj", "slug": "botafogo"},  # Botafogo/RJ (Série A)
    "red-bull-bragantino": {"estado": "sp", "slug": "bragantino"},
    "csa": {"estado": "al", "slug": "csa"},
    "ceara": {"estado": "ce", "slug": "ceara"},
    "chapecoense": {"estado": "sc", "slug": "chapecoense"},
    "corinthians": {"estado": "sp", "slug": "corinthians"},
    "coritiba": {"estado": "pr", "slug": "coritiba"},
    "criciuma": {"estado": "sc", "slug": "criciuma"},
    "cruzeiro": {"estado": "mg", "slug": "cruzeiro"},
    "cuiaba": {"estado": "mt", "slug": "cuiaba"},
    "fluminense": {"estado": "rj", "slug": "fluminense"},
    "flamengo": {"estado": "rj", "slug": "flamengo"},
    "fortaleza": {"estado": "ce", "slug": "fortaleza"},
    "goias": {"estado": "go", "slug": "goias"},
    "gremio": {"estado": "rs", "slug": "gremio"},
    "internacional": {"estado": "rs", "slug": "internacional"},
    "juventude": {"estado": "rs", "slug": "juventude"},
    "mirassol": {"estado": "sp", "slug": "mirassol"},
    "palmeiras": {"estado": "sp", "slug": "palmeiras"},
    "parana": {"estado": "pr", "slug": "parana"},
    "ponte-preta": {"estado": "sp", "slug": "ponte-preta"},
    "remo": {"estado": "pa", "slug": "remo"},
    "santos-fc": {"estado": "sp", "slug": "santos"},
    "santos": {"estado": "sp", "slug": "santos"},
    "sao-paulo": {"estado": "sp", "slug": "sao-paulo"},
    "sport-recife": {"estado": "pe", "slug": "sport"},
    "sport": {"estado": "pe", "slug": "sport"},
    "vasco-da-gama": {"estado": "rj", "slug": "vasco"},
    "vitoria": {"estado": "ba", "slug": "vitoria"},
    # Série B (roster estático, ver resolucao_times.TIMES_SERIE_B_2026) --
    # times que não aparecem também na A
    "athletic": {"estado": "mg", "slug": "athletic-mg"},
    "botafogo-sp": {"estado": "sp", "slug": "botafogo-sp", "regiao": "ribeirao-preto-e-regiao"},
    "crb": {"estado": "al", "slug": "crb"},
    "londrina": {"estado": "pr", "slug": "londrina"},
    "nautico": {"estado": "pe", "slug": "nautico"},
    "gremio-novorizontino": {"estado": "sp", "slug": "novorizontino"},
    "operario": {"estado": "pr", "slug": "operario-pr"},
    "sao-bernardo": {"estado": "sp", "slug": "sao-bernardo"},
    "vila-nova": {"estado": "go", "slug": "vila-nova"},
}

# Sufixos de reorganização societária que a CBF adiciona ao slug em algumas
# temporadas (o clube é o mesmo, o nome societário mudou) — removidos antes
# de procurar em TIME_CBF_PARA_GLOBO.
_SUFIXOS_SAF = ("-saf",)

# A CBF usa o MESMO slug "botafogo" pro Botafogo/RJ (Série A) e pro
# Botafogo-SP (Série B, sede em Ribeirão Preto) -- só dá pra desambiguar
# pela liga da partida, por isso um override específico (liga, slug) antes
# de cair no dicionário geral.
_OVERRIDE_POR_LIGA: dict[tuple[str, str], dict[str, str]] = {
    ("brasileirao_b", "botafogo"): {"estado": "sp", "slug": "botafogo-sp", "regiao": "ribeirao-preto-e-regiao"},
}


def _normalizar_slug_cbf(slug: str) -> str:
    for sufixo in _SUFIXOS_SAF:
        if slug.endswith(sufixo):
            return slug[: -len(sufixo)]
    return slug


def _slug(nome: str) -> str:
    nfkd = unicodedata.normalize("NFKD", nome)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", sem_acento.lower()).strip("-")


def _info_time(liga: str, slug_cbf: str) -> dict | None:
    slug_norm = _normalizar_slug_cbf(slug_cbf)
    return _OVERRIDE_POR_LIGA.get((liga, slug_norm)) or TIME_CBF_PARA_GLOBO.get(slug_norm)


def construir_urls_candidatas(liga: str, data_iso: str, time_casa_slug_cbf: str, time_fora_slug_cbf: str) -> list[str]:
    """data_iso no formato AAAA-MM-DD; os dois times vêm como slug da CBF
    (ex.: "atletico-mineiro", "coritiba-saf") — a mesma string que já sai de
    ingest_cbf.listar_jogos_temporada, sem precisar de outra normalização.
    Devolve uma lista com no máximo 1 URL (a documentada em
    TIME_CBF_PARA_GLOBO) — time desconhecido = lista vazia, nunca um
    palpite não verificado."""
    casa = _info_time(liga, time_casa_slug_cbf)
    fora = _info_time(liga, time_fora_slug_cbf)
    if casa is None or fora is None:
        return []
    competicao = COMPETICAO[liga]
    ano, mes, dia = data_iso.split("-")
    data_url = f"{dia}-{mes}-{ano}"
    regiao = casa.get("regiao")
    prefixo = f"{casa['estado']}/{regiao}" if regiao else casa["estado"]
    return [f"https://ge.globo.com/{prefixo}/futebol/{competicao}/jogo/{data_url}/{casa['slug']}-{fora['slug']}.ghtml"]


_PADRAO_ESCANTEIOS = re.compile(r"(\d+)\s*\n+Escanteios\s*\n+(\d+)")
_PADRAO_FALTAS = re.compile(r"(\d+)\s*\n+Faltas cometidas\s*\n+(\d+)")


def _extrair_estatisticas(texto: str) -> dict | None:
    m_esc = _PADRAO_ESCANTEIOS.search(texto)
    m_falt = _PADRAO_FALTAS.search(texto)
    if m_esc is None and m_falt is None:
        return None
    return {
        "escanteios_casa": int(m_esc.group(1)) if m_esc else None,
        "escanteios_fora": int(m_esc.group(2)) if m_esc else None,
        "faltas_casa": int(m_falt.group(1)) if m_falt else None,
        "faltas_fora": int(m_falt.group(2)) if m_falt else None,
    }


def buscar_estatisticas_jogo(page, liga: str, data_iso: str, time_casa: str, time_fora: str,
                              tentativas_espera: tuple[int, ...] = (3000, 5000)) -> dict | None:
    """Tenta cada URL candidata (nome/estado do mandante); em cada uma,
    espera um pouco mais na segunda tentativa (o widget de estatística às
    vezes carrega devagar). Retorna None só depois de esgotar todas as
    combinações plausíveis — nunca inventa números."""
    urls = construir_urls_candidatas(liga, data_iso, time_casa, time_fora)
    for url in urls:
        try:
            resp = page.goto(url, timeout=20000, wait_until="domcontentloaded")
        except Exception:
            continue
        if resp is None or resp.status == 404:
            continue
        for espera in tentativas_espera:
            page.wait_for_timeout(espera)
            texto = page.inner_text("body")
            stats = _extrair_estatisticas(texto)
            if stats is not None:
                return stats
        # página existe (200) mas o widget não carregou -> não insiste mais nessa URL
    return None


def caminho_cache(liga: str, data_iso: str, time_casa: str, time_fora: str) -> Path:
    nome = f"{data_iso}_{_slug(time_casa)}_{_slug(time_fora)}.json"
    return PASTA_CACHE / liga / nome


def obter_estatisticas_com_cache(page, liga: str, data_iso: str, time_casa: str, time_fora: str) -> dict | None:
    """Cacheia em disco (partida histórica nunca muda) — reruns não
    refazem requisições já bem-sucedidas OU já confirmadas ausentes."""
    import json

    caminho = caminho_cache(liga, data_iso, time_casa, time_fora)
    if caminho.exists():
        conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        return conteudo if conteudo else None

    stats = buscar_estatisticas_jogo(page, liga, data_iso, time_casa, time_fora)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(stats or {}), encoding="utf-8")
    return stats


def coletar_lote(jogos: list[dict], liga: str, atraso_entre_requisicoes: float = 1.0) -> dict[tuple, dict | None]:
    """jogos: lista de {"data_iso":, "time_casa":, "time_fora":}. Roda um
    único navegador pra todo o lote (bem mais rápido que abrir um por
    jogo). Devolve dict indexado por (data_iso, time_casa, time_fora)."""
    resultado = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        for jogo in jogos:
            chave = (jogo["data_iso"], jogo["time_casa"], jogo["time_fora"])
            resultado[chave] = obter_estatisticas_com_cache(page, liga, *chave)
            time.sleep(atraso_entre_requisicoes)
        browser.close()
    return resultado
