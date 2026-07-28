# -*- coding: utf-8 -*-
"""
Modelo de escanteios por time (a favor e contra), com a mesma estrutura de
ataque/defesa/vantagem de mando/decaimento temporal do Dixon-Coles, mas com
distribuição binomial negativa (BN) no lugar de Poisson: escanteios têm
variância maior que a média (superdispersão), o que a Poisson não captura.

Parametrização BN2: var(Y) = mu + alpha * mu^2, com mu a média esperada e
alpha o parâmetro de dispersão (alpha -> 0 recupera o caso Poisson). alpha é
estimado nos dados junto com os demais parâmetros.

    mu_casa = exp(ataque_casa - defesa_fora + home_adv)   # escanteios esperados do mandante
    mu_fora = exp(ataque_fora - defesa_casa)              # escanteios esperados do visitante

Ajuste por máxima verossimilhança ponderada no tempo (peso =
exp(-xi*dias_atras)), restrição média(ataque)=0, um modelo por liga (nunca
misturando times de ligas diferentes). Nenhuma função de ajuste usa dados
posteriores à data de corte informada explicitamente pelo chamador.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "data" / "processed" / "modelos"

XI_PADRAO = 0.0018
MAX_ESCANTEIOS_PADRAO = 20  # por time; o total (casa+fora) vai até 2x isso


@dataclass
class ModeloEscanteios:
    """Parâmetros ajustados do modelo de escanteios (binomial negativa) de uma liga."""

    liga: str
    data_corte: str
    xi: float
    times: list[str]
    ataque: dict[str, float]   # tendência do time de conquistar escanteios
    defesa: dict[str, float]   # tendência do time de conceder escanteios
    home_adv: float
    alpha: float               # dispersão da binomial negativa (var = mu + alpha*mu^2)
    n_jogos_usados: int = 0

    def to_dict(self) -> dict:
        return {
            "liga": self.liga, "data_corte": self.data_corte, "xi": self.xi,
            "times": self.times, "ataque": self.ataque, "defesa": self.defesa,
            "home_adv": self.home_adv, "alpha": self.alpha, "n_jogos_usados": self.n_jogos_usados,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModeloEscanteios":
        return cls(**d)


def _nb_n_p(mu: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Converte a parametrização (media mu, dispersão alpha) para os
    parâmetros (n, p) exigidos por scipy.stats.nbinom."""
    n = 1.0 / alpha
    p = n / (n + mu)
    return n, p


def calcular_pesos_temporais(datas: pd.Series, data_corte, xi: float = XI_PADRAO) -> np.ndarray:
    data_corte = pd.Timestamp(data_corte)
    dias_atras = (data_corte - datas).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * dias_atras)


def _preparar_indices(df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    times = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    indice = {t: i for i, t in enumerate(times)}
    idx_casa = df["HomeTeam"].map(indice).to_numpy()
    idx_fora = df["AwayTeam"].map(indice).to_numpy()
    return times, idx_casa, idx_fora


def _desempacotar_parametros(x: np.ndarray, n_times: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Desempacota em (ataque com média zero, defesa, home_adv, alpha).
    O ataque do último time é -soma dos demais (reparametrização exata da
    restrição de identificabilidade, igual ao Dixon-Coles de gols)."""
    ataque_livre = x[: n_times - 1]
    ultimo_ataque = -np.sum(ataque_livre)
    ataque = np.append(ataque_livre, ultimo_ataque)
    defesa = x[n_times - 1: 2 * n_times - 1]
    home_adv = x[-2]
    alpha = np.exp(x[-1])  # log_alpha -> alpha, garante alpha > 0
    return ataque, defesa, home_adv, alpha


def _neg_log_verossimilhanca(x, idx_casa, idx_fora, esc_casa, esc_fora, pesos, n_times) -> float:
    ataque, defesa, home_adv, alpha = _desempacotar_parametros(x, n_times)

    # clip defensivo: durante a otimização (busca de linha do L-BFGS-B) o
    # vetor de parâmetros pode passar por pontos extremos antes de convergir;
    # sem isso, mu ou p podem estourar para 0/inf, a log-verossimilhança vira
    # -inf, e a diferença numérica do gradiente na iteração seguinte vira NaN
    # (o otimizador então aborta achando que já convergiu em x0).
    mu_casa = np.exp(np.clip(ataque[idx_casa] - defesa[idx_fora] + home_adv, -20, 20))
    mu_fora = np.exp(np.clip(ataque[idx_fora] - defesa[idx_casa], -20, 20))

    n_casa, p_casa = _nb_n_p(mu_casa, alpha)
    n_fora, p_fora = _nb_n_p(mu_fora, alpha)
    p_casa = np.clip(p_casa, 1e-10, 1 - 1e-10)
    p_fora = np.clip(p_fora, 1e-10, 1 - 1e-10)

    log_lik = nbinom.logpmf(esc_casa, n_casa, p_casa) + nbinom.logpmf(esc_fora, n_fora, p_fora)
    return -np.sum(pesos * log_lik)


def ajustar_modelo_escanteios(
    df: pd.DataFrame,
    liga: str,
    data_corte,
    xi: float = XI_PADRAO,
) -> ModeloEscanteios:
    """Ajusta o modelo de escanteios para UMA liga usando apenas partidas
    com Date < data_corte. `df` deve conter liga, Date, HomeTeam, AwayTeam,
    HC (escanteios do mandante), AC (escanteios do visitante)."""
    data_corte = pd.Timestamp(data_corte)

    df_liga = df[df["liga"] == liga].copy()
    df_liga = df_liga[df_liga["Date"] < data_corte]
    df_liga = df_liga.dropna(subset=["HC", "AC", "HomeTeam", "AwayTeam"])

    if df_liga.empty:
        raise ValueError(f"Sem partidas de '{liga}' anteriores a {data_corte.date()} para ajustar o modelo de escanteios.")

    times, idx_casa, idx_fora = _preparar_indices(df_liga)
    n_times = len(times)
    esc_casa = df_liga["HC"].to_numpy(dtype=float)
    esc_fora = df_liga["AC"].to_numpy(dtype=float)
    pesos = calcular_pesos_temporais(df_liga["Date"], data_corte, xi)

    n_params = (n_times - 1) + n_times + 1 + 1
    x0 = np.zeros(n_params)
    # com ataque/defesa=0, mu_casa = exp(home_adv): inicializa no log da média
    # observada para não começar longe demais da escala real (escanteios têm
    # média bem mais alta que gols) — chute ruim aqui gera gradientes enormes
    # e quebra a diferenciação numérica do L-BFGS-B.
    x0[-2] = np.log(max(esc_casa.mean(), 0.5))
    x0[-1] = 0.0   # log_alpha inicial (alpha=1)

    bounds = [(-4, 4)] * (n_times - 1) + [(-4, 4)] * n_times + [(-2, 4)] + [(-3, 3)]

    resultado = minimize(
        _neg_log_verossimilhanca, x0,
        args=(idx_casa, idx_fora, esc_casa, esc_fora, pesos, n_times),
        method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-10},
    )

    ataque, defesa, home_adv, alpha = _desempacotar_parametros(resultado.x, n_times)

    return ModeloEscanteios(
        liga=liga, data_corte=data_corte.date().isoformat(), xi=xi, times=times,
        ataque={t: float(a) for t, a in zip(times, ataque)},
        defesa={t: float(d) for t, d in zip(times, defesa)},
        home_adv=float(home_adv), alpha=float(alpha), n_jogos_usados=len(df_liga),
    )


def matriz_escanteios(
    modelo: ModeloEscanteios,
    time_casa: str,
    time_fora: str,
    max_escanteios: int = MAX_ESCANTEIOS_PADRAO,
) -> np.ndarray:
    """Matriz (max_escanteios+1) x (max_escanteios+1) de probabilidades
    conjuntas de escanteios (linha=casa, coluna=fora), assumindo
    independência condicional entre os dois totais (produto das
    marginais BN, sem correção de correlação — diferente do Dixon-Coles
    de gols, que corrige placares baixos)."""
    for t in (time_casa, time_fora):
        if t not in modelo.ataque:
            raise ValueError(f"Time '{t}' não encontrado no modelo de escanteios da liga '{modelo.liga}'.")

    mu_casa = np.exp(modelo.ataque[time_casa] - modelo.defesa[time_fora] + modelo.home_adv)
    mu_fora = np.exp(modelo.ataque[time_fora] - modelo.defesa[time_casa])

    n_casa, p_casa = _nb_n_p(mu_casa, modelo.alpha)
    n_fora, p_fora = _nb_n_p(mu_fora, modelo.alpha)

    valores = np.arange(0, max_escanteios + 1)
    pmf_casa = nbinom.pmf(valores, n_casa, p_casa)
    pmf_fora = nbinom.pmf(valores, n_fora, p_fora)

    matriz = np.outer(pmf_casa, pmf_fora)
    matriz = matriz / matriz.sum()
    return matriz


def distribuicao_total(matriz: np.ndarray) -> np.ndarray:
    """Distribuição do total de escanteios do jogo (casa+fora), obtida pela
    convolução das marginais da matriz conjunta."""
    marginal_casa = matriz.sum(axis=1)
    marginal_fora = matriz.sum(axis=0)
    total = np.convolve(marginal_casa, marginal_fora)
    return total / total.sum()


def _nome_arquivo_modelo(liga: str) -> str:
    slug = liga.lower().replace(" ", "_")
    return f"escanteios_{slug}.json"


def salvar_modelo(modelo: ModeloEscanteios, caminho: Path | None = None) -> Path:
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    if caminho is None:
        caminho = PASTA_MODELOS / _nome_arquivo_modelo(modelo.liga)
    caminho.write_text(json.dumps(modelo.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


def carregar_modelo(liga: str) -> ModeloEscanteios:
    caminho = PASTA_MODELOS / _nome_arquivo_modelo(liga)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo de escanteios não encontrado para a liga '{liga}' em {caminho}.")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    return ModeloEscanteios.from_dict(d)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ajusta modelos de escanteios (BN) por liga")
    parser.add_argument("--dados", default=str(RAIZ / "data" / "processed" / "partidas.csv"))
    parser.add_argument("--data-corte", default=None, help="YYYY-MM-DD (padrão: hoje)")
    parser.add_argument("--xi", type=float, default=XI_PADRAO)
    args = parser.parse_args()

    df = pd.read_csv(args.dados, parse_dates=["Date"])
    data_corte = args.data_corte or date.today().isoformat()

    for liga in sorted(df["liga"].unique()):
        modelo = ajustar_modelo_escanteios(df, liga, data_corte, xi=args.xi)
        caminho = salvar_modelo(modelo)
        print(f"[{liga}] escanteios ajustado com {modelo.n_jogos_usados} jogos | "
              f"home_adv={modelo.home_adv:.3f} alpha={modelo.alpha:.3f} -> {caminho}")


if __name__ == "__main__":
    main()
