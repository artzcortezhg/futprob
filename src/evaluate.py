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
from model_corners import ajustar_modelo_escanteios, matriz_escanteios, distribuicao_total
from model_cards import ajustar_modelo_cartoes, matriz_contagem

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
    alpha_xg: float = 0.0,
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
            modelo_atual = ajustar_modelo(df, liga, data_corte=data_jogo, xi=xi, alpha_xg=alpha_xg)
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
        # fechamento (PSC*): SÓ para validação/CLV, nunca insumo de decisão
        p_casa_odds, p_empate_odds, p_fora_odds = remover_margem_odds(jogo["PSCH"], jogo["PSCD"], jogo["PSCA"])
        # pré-jogo (PS*, sem C): insumo real de EV/mistura modelo+mercado
        p_casa_prejogo, p_empate_prejogo, p_fora_prejogo = remover_margem_odds(jogo["PSH"], jogo["PSD"], jogo["PSA"])

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
            "prob_casa_prejogo": p_casa_prejogo,
            "prob_empate_prejogo": p_empate_prejogo,
            "prob_fora_prejogo": p_fora_prejogo,
            "PSH": jogo["PSH"], "PSD": jogo["PSD"], "PSA": jogo["PSA"],
            "PSCH": jogo["PSCH"], "PSCD": jogo["PSCD"], "PSCA": jogo["PSCA"],
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
    """Brier score multiclasse (quadratic score): media de sum_k (p_k - y_k)^2.
    Genérico no número de classes (3 para 1x2, 2 para over/under)."""
    n, n_classes = probs.shape
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y_idx] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def tabela_calibracao(y_idx: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Achata as probabilidades das classes em um único vetor (com seus
    indicadores de acerto) e monta a tabela de calibração por faixa de
    probabilidade prevista. Genérico no número de classes."""
    n, n_classes = probs.shape
    y_onehot = np.zeros((n, n_classes))
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


# linhas principais avaliadas por padrão em cada mercado auxiliar
LINHAS_PADRAO_AUXILIAR = {"escanteios": [9.5, 10.5], "cartoes": [3.5]}


def avaliar_walkforward_auxiliar(
    df: pd.DataFrame,
    liga: str,
    tipo: str,
    data_inicio_avaliacao,
    data_fim_avaliacao=None,
    refit_dias: int = 30,
    xi: float = XI_PADRAO,
    linhas: list[float] | None = None,
    verboso: bool = True,
) -> pd.DataFrame:
    """Walk-forward dos mercados de over/under de escanteios ou cartões
    (total do jogo). Não há odds de fechamento para esses mercados na
    fonte, então a validação aqui é só de calibração própria do modelo —
    sem comparação com mercado. Mesmo esquema de reajuste periódico e
    mesma garantia de não usar dados futuros que avaliar_walkforward."""
    if tipo not in ("escanteios", "cartoes"):
        raise ValueError("tipo deve ser 'escanteios' ou 'cartoes'.")
    linhas = linhas or LINHAS_PADRAO_AUXILIAR[tipo]
    col_casa, col_fora = ("HC", "AC") if tipo == "escanteios" else ("HY", "AY")

    data_inicio_avaliacao = pd.Timestamp(data_inicio_avaliacao)
    df_liga = df[df["liga"] == liga].sort_values("Date").reset_index(drop=True)

    data_fim_avaliacao = pd.Timestamp(data_fim_avaliacao) if data_fim_avaliacao else df_liga["Date"].max() + pd.Timedelta(days=1)
    jogos = df_liga[(df_liga["Date"] >= data_inicio_avaliacao) & (df_liga["Date"] < data_fim_avaliacao)].copy()
    jogos = jogos.dropna(subset=[col_casa, col_fora])

    if jogos.empty:
        raise ValueError("Nenhum jogo no intervalo de avaliação informado.")

    registros = []
    modelo_atual = None
    proximo_refit = None
    n_refits = 0

    for _, jogo in jogos.iterrows():
        data_jogo = jogo["Date"]

        if modelo_atual is None or data_jogo >= proximo_refit:
            if tipo == "escanteios":
                modelo_atual = ajustar_modelo_escanteios(df, liga, data_corte=data_jogo, xi=xi)
            else:
                modelo_atual = ajustar_modelo_cartoes(df, liga, data_corte=data_jogo, xi=xi)
            proximo_refit = data_jogo + pd.Timedelta(days=refit_dias)
            n_refits += 1
            if verboso:
                print(f"  [refit #{n_refits}] corte={data_jogo.date()} n_jogos_treino={modelo_atual.n_jogos_usados}")

        casa, fora = jogo["HomeTeam"], jogo["AwayTeam"]
        if casa not in modelo_atual.ataque or fora not in modelo_atual.ataque:
            continue

        if tipo == "escanteios":
            matriz = matriz_escanteios(modelo_atual, casa, fora)
        else:
            arbitro = jogo["Referee"] if modelo_atual.usar_arbitro else None
            matriz = matriz_contagem(modelo_atual, casa, fora, arbitro=arbitro, max_valor=15)

        dist_total = distribuicao_total(matriz)
        valores = np.arange(len(dist_total))
        total_real = jogo[col_casa] + jogo[col_fora]

        registro = {
            "Date": data_jogo, "liga": liga, "HomeTeam": casa, "AwayTeam": fora,
            "total_real": total_real, "data_corte_modelo": modelo_atual.data_corte,
        }
        for linha in linhas:
            registro[f"prob_over_{linha}"] = float(dist_total[valores > linha].sum())
            registro[f"over_real_{linha}"] = 1.0 if total_real > linha else 0.0
        registros.append(registro)

    if verboso:
        print(f"  total de reajustes: {n_refits} | jogos avaliados: {len(registros)}")

    return pd.DataFrame(registros)


def relatorio_metricas_auxiliar(df_avaliacao: pd.DataFrame, linhas: list[float]) -> dict:
    """Log loss, Brier score e tabela de calibração por linha de over/under,
    tratando over/under como duas classes (reaproveita as mesmas funções
    genéricas usadas no 1x2 de gols, aqui com k=2 em vez de k=3)."""
    resultado = {"n_jogos": len(df_avaliacao), "linhas": {}}
    for linha in linhas:
        y_idx = np.where(df_avaliacao[f"over_real_{linha}"].to_numpy() == 1.0, 0, 1)
        p_over = df_avaliacao[f"prob_over_{linha}"].to_numpy()
        probs = np.column_stack([p_over, 1.0 - p_over])  # colunas: [over, under]
        resultado["linhas"][linha] = {
            "log_loss": log_loss_multiclasse(y_idx, probs),
            "brier_score": brier_score_multiclasse(y_idx, probs),
            "calibracao": tabela_calibracao(y_idx, probs, n_bins=10),
        }
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Validação walk-forward dos modelos do futprob")
    parser.add_argument("--liga", required=True)
    parser.add_argument("--tipo", choices=["gols", "escanteios", "cartoes"], default="gols",
                         help="Qual modelo validar (gols compara com odds; escanteios/cartões só calibração)")
    parser.add_argument("--dados", default=None, help="Padrão: partidas.csv (ou partidas_xg.csv se --alpha-xg > 0)")
    parser.add_argument("--data-inicio", default=None, help="YYYY-MM-DD (padrão: 3 anos antes do último jogo)")
    parser.add_argument("--data-fim", default=None, help="YYYY-MM-DD (padrão: até o último jogo disponível)")
    parser.add_argument("--refit-dias", type=int, default=30)
    parser.add_argument("--xi", type=float, default=XI_PADRAO)
    parser.add_argument("--max-gols", type=int, default=MAX_GOLS_PADRAO)
    parser.add_argument("--alpha-xg", type=float, default=0.0, help="Peso do xG no alvo de treino, 0-1 (só --tipo gols)")
    parser.add_argument("--linhas", default=None, help="Linhas de over/under separadas por vírgula (só --tipo escanteios/cartoes)")
    parser.add_argument("--saida", default=None, help="Caminho para salvar as previsões jogo a jogo em CSV")
    args = parser.parse_args()

    caminho_dados = args.dados or str(RAIZ / "data" / "processed" / ("partidas_xg.csv" if args.alpha_xg > 0 else "partidas.csv"))
    df = pd.read_csv(caminho_dados, parse_dates=["Date"])
    data_max = df[df["liga"] == args.liga]["Date"].max()
    data_inicio = pd.Timestamp(args.data_inicio) if args.data_inicio else data_max - pd.Timedelta(days=365 * 3)

    if args.tipo == "gols":
        print(f"Validação walk-forward — {args.liga} (alpha_xg={args.alpha_xg})")
        print(f"Período avaliado: {data_inicio.date()} até {args.data_fim or data_max.date()} | refit a cada {args.refit_dias} dias\n")

        df_aval = avaliar_walkforward(
            df, args.liga, data_inicio,
            data_fim_avaliacao=args.data_fim, refit_dias=args.refit_dias, xi=args.xi, max_gols=args.max_gols,
            alpha_xg=args.alpha_xg,
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

    else:
        linhas = [float(x) for x in args.linhas.split(",")] if args.linhas else LINHAS_PADRAO_AUXILIAR[args.tipo]
        print(f"Validação walk-forward — {args.liga} ({args.tipo}, linhas={linhas})")
        print(f"Período avaliado: {data_inicio.date()} até {args.data_fim or data_max.date()} | refit a cada {args.refit_dias} dias\n")

        df_aval = avaliar_walkforward_auxiliar(
            df, args.liga, args.tipo, data_inicio,
            data_fim_avaliacao=args.data_fim, refit_dias=args.refit_dias, xi=args.xi, linhas=linhas,
        )

        if args.saida:
            df_aval.to_csv(args.saida, index=False)
            print(f"\nPrevisões jogo a jogo salvas em {args.saida}")

        metricas = relatorio_metricas_auxiliar(df_aval, linhas)
        print(f"\n=== Métricas por linha (n={metricas['n_jogos']} jogos, sem comparação com mercado) ===")
        for linha in linhas:
            m = metricas["linhas"][linha]
            print(f"\n--- Over/Under {linha} ---")
            print(f"log loss = {m['log_loss']:.4f} | Brier = {m['brier_score']:.4f}")
            print(m["calibracao"].to_string(index=False))


if __name__ == "__main__":
    main()
