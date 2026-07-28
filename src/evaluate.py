# -*- coding: utf-8 -*-
"""
Validação walk-forward do modelo Dixon-Coles.

Ideia: percorre os jogos de uma liga em ordem cronológica; a cada
`refit_dias`, reajusta o modelo usando SOMENTE jogos anteriores à data do
próximo jogo a avaliar (nunca dados posteriores). Entre reajustes, o mesmo
modelo é usado para prever os jogos seguintes — isso ainda respeita a regra
de nunca usar dados futuros, pois a data de corte de cada ajuste é sempre
<= à data de todo jogo avaliado com aquele modelo.

Reporta log loss, Brier score e tabela de calibração do modelo, comparando
com as probabilidades implícitas nas odds de fechamento (PSCH/PSCD/PSCA),
com a margem da casa de apostas removida (normalização 1/odd).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from markets import mercado_1x2
from model_goals import XI_PADRAO, MAX_GOLS_PADRAO, ajustar_modelo, matriz_placares

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS_PADRAO = RAIZ / "data" / "processed" / "partidas.csv"

CLASSES = ["casa", "empate", "fora"]  # ordem fixa usada em todos os vetores/matrizes de prob.


def remover_margem_odds(odd_casa: float, odd_empate: float, odd_fora: float) -> tuple[float, float, float]:
    """Converte odds decimais de fechamento em probabilidades implícitas,
    removendo a margem da casa (overround) via normalização de 1/odd."""
    if any(pd.isna(o) or o <= 1 for o in (odd_casa, odd_empate, odd_fora)):
        return (np.nan, np.nan, np.nan)
    inv = np.array([1.0 / odd_casa, 1.0 / odd_empate, 1.0 / odd_fora])
    return tuple(inv / inv.sum())


def _resultado_para_indice(ftr: str) -> int:
    return {"H": 0, "D": 1, "A": 2}[ftr]


def avaliar_walkforward(
    df: pd.DataFrame,
    liga: str,
    data_inicio_avaliacao,
    data_fim_avaliacao=None,
    refit_dias: int = 30,
    xi: float = XI_PADRAO,
    max_gols: int = MAX_GOLS_PADRAO,
    verboso: bool = True,
) -> pd.DataFrame:
    """Executa a validação walk-forward para uma liga e retorna um DataFrame
    com uma linha por jogo avaliado (probabilidades do modelo e das odds)."""
    data_inicio_avaliacao = pd.Timestamp(data_inicio_avaliacao)
    df_liga = df[df["liga"] == liga].sort_values("Date").reset_index(drop=True)

    data_fim_avaliacao = pd.Timestamp(data_fim_avaliacao) if data_fim_avaliacao else df_liga["Date"].max() + pd.Timedelta(days=1)
    jogos = df_liga[(df_liga["Date"] >= data_inicio_avaliacao) & (df_liga["Date"] < data_fim_avaliacao)].copy()
    jogos = jogos.dropna(subset=["FTR"])

    if jogos.empty:
        raise ValueError("Nenhum jogo no intervalo de avaliação informado.")

    registros = []
    modelo_atual = None
    proximo_refit = None
    n_refits = 0

    for _, jogo in jogos.iterrows():
        data_jogo = jogo["Date"]

        if modelo_atual is None or data_jogo >= proximo_refit:
            modelo_atual = ajustar_modelo(df, liga, data_corte=data_jogo, xi=xi)
            proximo_refit = data_jogo + pd.Timedelta(days=refit_dias)
            n_refits += 1
            if verboso:
                print(f"  [refit #{n_refits}] corte={data_jogo.date()} n_jogos_treino={modelo_atual.n_jogos_usados}")

        casa, fora = jogo["HomeTeam"], jogo["AwayTeam"]
        if casa not in modelo_atual.ataque or fora not in modelo_atual.ataque:
            # time sem histórico suficiente antes deste corte (ex.: recém-promovido)
            continue

        matriz = matriz_placares(modelo_atual, casa, fora, max_gols=max_gols)
        probs_mod = mercado_1x2(matriz)
        p_casa_odds, p_empate_odds, p_fora_odds = remover_margem_odds(jogo["PSCH"], jogo["PSCD"], jogo["PSCA"])

        registros.append({
            "Date": data_jogo,
            "liga": liga,
            "HomeTeam": casa,
            "AwayTeam": fora,
            "FTR": jogo["FTR"],
            "prob_casa_modelo": probs_mod["casa"],
            "prob_empate_modelo": probs_mod["empate"],
            "prob_fora_modelo": probs_mod["fora"],
            "prob_casa_odds": p_casa_odds,
            "prob_empate_odds": p_empate_odds,
            "prob_fora_odds": p_fora_odds,
            "data_corte_modelo": modelo_atual.data_corte,
        })

    if verboso:
        print(f"  total de reajustes: {n_refits} | jogos avaliados: {len(registros)}")

    return pd.DataFrame(registros)


def log_loss_multiclasse(y_idx: np.ndarray, probs: np.ndarray) -> float:
    """probs: array (n,3) na ordem CLASSES. y_idx: array (n,) com 0/1/2."""
    p_verdadeira = probs[np.arange(len(y_idx)), y_idx]
    p_verdadeira = np.clip(p_verdadeira, 1e-15, 1.0)
    return float(-np.mean(np.log(p_verdadeira)))


def brier_score_multiclasse(y_idx: np.ndarray, probs: np.ndarray) -> float:
    """Brier score multiclasse (quadratic score): media de sum_k (p_k - y_k)^2."""
    n = len(y_idx)
    y_onehot = np.zeros((n, 3))
    y_onehot[np.arange(n), y_idx] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def tabela_calibracao(y_idx: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Achata as probabilidades das 3 classes em um único vetor (com seus
    indicadores de acerto) e monta a tabela de calibração por faixa de
    probabilidade prevista."""
    n = len(y_idx)
    y_onehot = np.zeros((n, 3))
    y_onehot[np.arange(n), y_idx] = 1.0

    probs_flat = probs.flatten()
    y_flat = y_onehot.flatten()

    bordas = np.linspace(0, 1, n_bins + 1)
    faixas = pd.cut(probs_flat, bordas, include_lowest=True)

    df_flat = pd.DataFrame({"faixa": faixas, "prob_prevista": probs_flat, "ocorreu": y_flat})
    tabela = df_flat.groupby("faixa", observed=True).agg(
        n=("ocorreu", "size"),
        prob_media_prevista=("prob_prevista", "mean"),
        freq_observada=("ocorreu", "mean"),
    ).reset_index()
    return tabela


def relatorio_metricas(df_avaliacao: pd.DataFrame) -> dict:
    """Calcula log loss, Brier score e tabela de calibração do modelo, e
    compara com as odds de fechamento (apenas nos jogos em que há odds)."""
    y_idx = df_avaliacao["FTR"].map(_resultado_para_indice).to_numpy()
    probs_modelo = df_avaliacao[["prob_casa_modelo", "prob_empate_modelo", "prob_fora_modelo"]].to_numpy()

    resultado = {
        "n_jogos": len(df_avaliacao),
        "modelo": {
            "log_loss": log_loss_multiclasse(y_idx, probs_modelo),
            "brier_score": brier_score_multiclasse(y_idx, probs_modelo),
            "calibracao": tabela_calibracao(y_idx, probs_modelo),
        },
    }

    com_odds = df_avaliacao.dropna(subset=["prob_casa_odds", "prob_empate_odds", "prob_fora_odds"])
    if not com_odds.empty:
        y_idx_odds = com_odds["FTR"].map(_resultado_para_indice).to_numpy()
        probs_odds = com_odds[["prob_casa_odds", "prob_empate_odds", "prob_fora_odds"]].to_numpy()
        probs_modelo_mesmo_subset = com_odds[["prob_casa_modelo", "prob_empate_modelo", "prob_fora_modelo"]].to_numpy()

        resultado["odds"] = {
            "n_jogos": len(com_odds),
            "log_loss": log_loss_multiclasse(y_idx_odds, probs_odds),
            "brier_score": brier_score_multiclasse(y_idx_odds, probs_odds),
        }
        resultado["modelo_mesmo_subset_odds"] = {
            "n_jogos": len(com_odds),
            "log_loss": log_loss_multiclasse(y_idx_odds, probs_modelo_mesmo_subset),
            "brier_score": brier_score_multiclasse(y_idx_odds, probs_modelo_mesmo_subset),
        }

    return resultado


def main():
    parser = argparse.ArgumentParser(description="Validação walk-forward do modelo Dixon-Coles")
    parser.add_argument("--liga", required=True)
    parser.add_argument("--dados", default=str(CAMINHO_DADOS_PADRAO))
    parser.add_argument("--data-inicio", default=None, help="YYYY-MM-DD (padrão: 3 anos antes do último jogo)")
    parser.add_argument("--data-fim", default=None, help="YYYY-MM-DD (padrão: até o último jogo disponível)")
    parser.add_argument("--refit-dias", type=int, default=30)
    parser.add_argument("--xi", type=float, default=XI_PADRAO)
    parser.add_argument("--max-gols", type=int, default=MAX_GOLS_PADRAO)
    parser.add_argument("--n-bins-calibracao", type=int, default=10)
    parser.add_argument("--saida", default=None, help="Caminho para salvar as previsões jogo a jogo em CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.dados, parse_dates=["Date"])
    data_max = df[df["liga"] == args.liga]["Date"].max()
    data_inicio = pd.Timestamp(args.data_inicio) if args.data_inicio else data_max - pd.Timedelta(days=365 * 3)

    print(f"Validação walk-forward — {args.liga}")
    print(f"Período avaliado: {data_inicio.date()} até {args.data_fim or data_max.date()} | refit a cada {args.refit_dias} dias\n")

    df_aval = avaliar_walkforward(
        df, args.liga, data_inicio,
        data_fim_avaliacao=args.data_fim, refit_dias=args.refit_dias, xi=args.xi, max_gols=args.max_gols,
    )

    if args.saida:
        df_aval.to_csv(args.saida, index=False)
        print(f"\nPrevisões jogo a jogo salvas em {args.saida}")

    metricas = relatorio_metricas(df_aval)

    print(f"\n=== Métricas (n={metricas['n_jogos']} jogos) ===")
    print(f"Modelo Dixon-Coles : log loss = {metricas['modelo']['log_loss']:.4f} | Brier = {metricas['modelo']['brier_score']:.4f}")
    if "odds" in metricas:
        n_odds = metricas["odds"]["n_jogos"]
        print(f"\n--- Comparação restrita aos {n_odds} jogos com odds de fechamento disponíveis ---")
        print(f"Modelo Dixon-Coles       : log loss = {metricas['modelo_mesmo_subset_odds']['log_loss']:.4f} | Brier = {metricas['modelo_mesmo_subset_odds']['brier_score']:.4f}")
        print(f"Odds de fechamento (sem margem): log loss = {metricas['odds']['log_loss']:.4f} | Brier = {metricas['odds']['brier_score']:.4f}")
    else:
        print("\n[aviso] Nenhum jogo avaliado tinha odds de fechamento (PSCH/PSCD/PSCA) disponíveis.")

    print("\n=== Tabela de calibração (modelo) ===")
    print(metricas["modelo"]["calibracao"].to_string(index=False))


if __name__ == "__main__":
    main()
