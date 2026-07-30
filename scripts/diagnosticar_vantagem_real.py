# -*- coding: utf-8 -*-
"""
Diagnóstico permanente: existe uma vantagem REAL e explorável contra o
mercado, ou o EV que o sistema calcula é sobretudo ruído de estimativa
("maldição do vencedor" -- ver backtest_clv.calibracao_apostas_selecionadas
e varredura_limiar_ev)?

Origem: usuário testou 4 apostas reais de um dia (1 acertou), perguntou se
o sistema é utilizável. Investigação encontrou que a probabilidade do
MODELO é bem calibrada quando olhada em TODAS as previsões (evaluate.py),
mas a SELEÇÃO de maior EV por jogo mostra superconfiança sistemática --
replicado de forma independente aplicando a mesma seleção na base já
comprovada calibrada, com resultado quase idêntico ao backtest real. Um
sinal de vantagem genuína mostraria retorno estável/crescente conforme o
limiar de EV sobe; retorno PIORANDO é sinal de que "EV alto" é ruído, não
vantagem -- confirmado nas 3 ligas com dado suficiente (Premier League,
La Liga, Championship).

Roda de novo conforme mais dado real for se acumulando (esp. Brasileirão/
Série B, que ainda não tem avaliacao_*.csv walk-forward) -- é a única
forma honesta de saber QUANDO (se algum dia) uma vantagem real aparecer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from backtest_clv import simular_backtest_clv, calibracao_apostas_selecionadas, varredura_limiar_ev  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
LIGAS_ARQUIVO = {
    "premier_league": "Premier League",
    "la_liga": "La Liga",
    "championship": "Championship",
}


def diagnosticar(nome_arquivo: str, nome_liga: str) -> None:
    caminho = RAIZ / "data" / "processed" / f"avaliacao_{nome_arquivo}.csv"
    if not caminho.exists():
        print(f"[{nome_liga}] sem avaliacao_{nome_arquivo}.csv ainda -- pule.")
        return

    df = pd.read_csv(caminho, parse_dates=["Date"])
    print(f"\n=== {nome_liga} ({len(df)} jogos avaliados) ===")

    apostas = simular_backtest_clv(df)
    if apostas.empty:
        print("  nenhuma aposta passou do limiar padrão (EV>5%).")
        return

    print(f"  apostas selecionadas (EV>5%): {len(apostas)}")
    print(f"  prob. média prevista: {apostas['prob_modelo'].mean():.3f} | taxa real de acerto: {apostas['ganhou'].mean():.3f}")
    print(f"  ROI de papel real: {apostas['retorno_papel'].mean():+.3f}")

    print("\n  --- calibração das apostas selecionadas (prevista vs real) ---")
    tabela = calibracao_apostas_selecionadas(apostas)
    for _, linha in tabela.iterrows():
        print(f"    {linha['faixa']}: n={int(linha['n']):>4}  previsto={linha['prob_media_prevista']:.3f}  real={linha['taxa_real_acerto']:.3f}")

    print("\n  --- retorno real por limiar de EV exigido ---")
    varredura = varredura_limiar_ev(df)
    for _, linha in varredura.iterrows():
        roi = f"{linha['roi_papel']:+.3f}" if linha["roi_papel"] is not None else "sem apostas"
        print(f"    EV>{linha['limiar_ev']:.0%}: n={linha['n_apostas']:>4}  ROI real={roi}")

    piora_com_limiar_mais_alto = (
        varredura["roi_papel"].notna().sum() >= 2
        and varredura.dropna(subset=["roi_papel"])["roi_papel"].iloc[-1] < varredura.dropna(subset=["roi_papel"])["roi_papel"].iloc[0]
    )
    if piora_com_limiar_mais_alto:
        print("  >> VEREDITO: retorno PIORA com limiar mais alto -- sinal de ruído (maldição do vencedor), não vantagem real.")
    else:
        print("  >> VEREDITO: retorno não piora com limiar mais alto -- vale investigar mais, pode haver sinal real.")


def main():
    for arquivo, liga in LIGAS_ARQUIVO.items():
        diagnosticar(arquivo, liga)


if __name__ == "__main__":
    main()
