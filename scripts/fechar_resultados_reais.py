# -*- coding: utf-8 -*-
"""
Fecha registros abertos usando o PLACAR REAL da partida (pesquisado — hoje
o único jeito confiável é buscar na internet, ver resolucao_resultados.py
pra por que isso não dá pra automatizar 100% sem humano/LLM no loop, ou uma
API de resultados esportivos de verdade).

Fluxo:
1. `python scripts/fechar_resultados_reais.py --listar` mostra os jogos com
   registros abertos cujo data_jogo já passou -- é essa lista que precisa
   ser pesquisada (WebSearch) pra saber o placar real de cada um.
2. Depois de pesquisar, preenche data/resultados_reais_pendentes.json com
   {"liga|casa|fora|data_jogo": [gols_casa, gols_fora]} pra cada jogo.
3. `python scripts/fechar_resultados_reais.py --aplicar` lê esse arquivo e
   fecha (status='fechado' + resultado) todo registro aberto desses jogos
   que dá pra resolver só com o placar (1X2, dupla chance, ambas marcam,
   over/under gols) -- escanteios/cartões/faltas continuam abertos (não dá
   pra saber só com o placar final, nunca inventa).

Sem isso rodando, o Brasileirão/Série B nunca acumula registro fechado o
suficiente pra rodar o mesmo diagnóstico de "é ruído ou vantagem real" que
já rodamos nas ligas europeias (ver diagnosticar_vantagem_real.py).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from resolucao_resultados import resultado_bate_selecao  # noqa: E402
from painel_db import fechar_registro_com_resultado_real, CAMINHO_DB_PADRAO  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PENDENTES = RAIZ / "data" / "resultados_reais_pendentes.json"


def listar_jogos_pendentes(caminho_db: Path = CAMINHO_DB_PADRAO) -> list[dict]:
    with sqlite3.connect(caminho_db) as conn:
        conn.row_factory = sqlite3.Row
        linhas = conn.execute(
            """SELECT DISTINCT liga, time_casa, time_fora, data_jogo FROM registros
               WHERE status='aberto' AND data_jogo < date('now', 'localtime')
               ORDER BY data_jogo"""
        ).fetchall()
    return [dict(linha) for linha in linhas]


def _chave(liga: str, casa: str, fora: str, data_jogo: str) -> str:
    return f"{liga}|{casa}|{fora}|{data_jogo}"


def aplicar_resultados(caminho_db: Path = CAMINHO_DB_PADRAO, caminho_pendentes: Path = CAMINHO_PENDENTES) -> dict:
    if not caminho_pendentes.exists():
        raise FileNotFoundError(f"{caminho_pendentes} não existe -- rode --listar primeiro e preencha os placares.")
    placares = json.loads(caminho_pendentes.read_text(encoding="utf-8"))

    fechados, pulados_sem_placar, sem_resultado_disponivel = 0, 0, 0
    with sqlite3.connect(caminho_db) as conn:
        conn.row_factory = sqlite3.Row
        abertos = conn.execute(
            """SELECT id, liga, time_casa, time_fora, data_jogo, mercado, selecao FROM registros
               WHERE status='aberto' AND data_jogo < date('now', 'localtime')"""
        ).fetchall()

    for linha in abertos:
        chave = _chave(linha["liga"], linha["time_casa"], linha["time_fora"], linha["data_jogo"])
        if chave not in placares:
            sem_resultado_disponivel += 1
            continue
        gols_casa, gols_fora = placares[chave]
        bateu = resultado_bate_selecao(linha["mercado"], linha["selecao"], gols_casa, gols_fora)
        if bateu is None:
            pulados_sem_placar += 1
            continue
        fechar_registro_com_resultado_real(caminho_db, linha["id"], "ganhou" if bateu else "perdeu")
        fechados += 1

    return {
        "fechados": fechados,
        "pulados_mercado_sem_placar_suficiente": pulados_sem_placar,
        "sem_resultado_pesquisado_ainda": sem_resultado_disponivel,
    }


def main():
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--listar", action="store_true", help="Lista jogos com registros abertos que precisam de placar")
    grupo.add_argument("--aplicar", action="store_true", help="Aplica os placares de data/resultados_reais_pendentes.json")
    args = parser.parse_args()

    if args.listar:
        jogos = listar_jogos_pendentes()
        print(json.dumps(jogos, indent=2, ensure_ascii=False))
        print(f"\n{len(jogos)} jogo(s) precisam de placar real em {CAMINHO_PENDENTES}")
    else:
        resultado = aplicar_resultados()
        print(resultado)


if __name__ == "__main__":
    main()
