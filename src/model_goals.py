# -*- coding: utf-8 -*-
"""
Modelo Dixon-Coles de gols por partida.

Cada time tem um parâmetro de ataque e um de defesa. Além disso, o modelo
tem uma vantagem de mando (home_adv) e um parâmetro rho que corrige a
correlação entre os gols em placares baixos (0-0, 1-0, 0-1, 1-1), que o
modelo de Poisson independente por natureza subestima/superestima.

    lambda = exp(ataque_casa - defesa_fora + home_adv)   # gols esperados do mandante
    mu     = exp(ataque_fora - defesa_casa)              # gols esperados do visitante

O ajuste é feito por máxima verossimilhança ponderada no tempo: partidas
mais antigas pesam menos, com peso = exp(-xi * dias_atras).

IMPORTANTE: nenhuma função de ajuste aqui usa dados posteriores à data de
corte informada explicitamente pelo chamador. Um modelo é ajustado por vez,
para uma única liga (nunca misturando times de ligas diferentes).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "data" / "processed" / "modelos"

XI_PADRAO = 0.0018
MAX_GOLS_PADRAO = 10


@dataclass
class ModeloDixonColes:
    """Parâmetros ajustados de um modelo Dixon-Coles para uma liga."""

    liga: str
    data_corte: str  # ISO (YYYY-MM-DD) - nenhum jogo em/após essa data foi usado
    xi: float
    times: list[str]
    ataque: dict[str, float]
    defesa: dict[str, float]
    home_adv: float
    rho: float
    n_jogos_usados: int = 0
    alpha_xg: float = 0.0  # peso do xG no alvo de treino: alvo = alpha_xg*xG + (1-alpha_xg)*gols

    def to_dict(self) -> dict:
        return {
            "liga": self.liga,
            "data_corte": self.data_corte,
            "xi": self.xi,
            "times": self.times,
            "ataque": self.ataque,
            "defesa": self.defesa,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "n_jogos_usados": self.n_jogos_usados,
            "alpha_xg": self.alpha_xg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModeloDixonColes":
        d = dict(d)
        if "alpha_xg" not in d:
            # compatibilidade com modelos salvos antes desta correção (campo
            # antigo "fonte": "gols"/"xg" com arredondamento)
            d["alpha_xg"] = 1.0 if d.pop("fonte", "gols") == "xg" else 0.0
        else:
            d.pop("fonte", None)
        return cls(**d)


def _log_poisson_continuo(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """log-densidade de Poisson generalizada para contagens não-inteiras:
    logpmf(k, lam) = k*log(lam) - lam - lgamma(k+1).

    scipy.stats.poisson.logpmf exige suporte discreto (retorna -inf para k
    fracionário), o que impede usar xG contínuo como alvo de treino. Esta
    extensão via log-gama (scipy.special.gammaln) é lisa em k >= 0 e reduz
    exatamente à log-pmf de Poisson quando k é inteiro (gammaln(n+1) =
    log(n!)), então usá-la sempre (mesmo com k inteiro) não muda nada em
    relação à Poisson padrão.
    """
    return k * np.log(lam) - lam - gammaln(k + 1)


def _correcao_dixon_coles(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Fator de correção tau(x,y) aplicado nas quatro células de placar baixo."""
    tau = np.ones_like(lam, dtype=float)
    tau = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1.0 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1.0 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1.0 - rho, tau)
    return tau


def calcular_pesos_temporais(datas: pd.Series, data_corte, xi: float = XI_PADRAO) -> np.ndarray:
    """Peso = exp(-xi * dias_atras), onde dias_atras é a distância (em dias)
    entre a data do jogo e a data de corte."""
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
    """Desempacota o vetor de otimização em (ataque, defesa, home_adv, rho).

    O ataque do último time é definido como -soma dos demais, garantindo
    exatamente a restrição de identificabilidade média(ataque) = 0 durante
    toda a otimização (reparametrização, não penalidade).
    """
    ataque_livre = x[: n_times - 1]
    ultimo_ataque = -np.sum(ataque_livre)
    ataque = np.append(ataque_livre, ultimo_ataque)
    defesa = x[n_times - 1: n_times - 1 + n_times]
    home_adv = x[-2]
    rho = x[-1]
    return ataque, defesa, home_adv, rho


def _neg_log_verossimilhanca(
    x: np.ndarray, idx_casa, idx_fora,
    alvo_casa, alvo_fora, gols_casa_obs, gols_fora_obs,
    pesos, n_times,
) -> float:
    """alvo_casa/alvo_fora alimentam o termo de Poisson (podem ser gols
    observados, xG contínuo, ou uma mistura dos dois). gols_casa_obs/
    gols_fora_obs são sempre os placares observados (inteiros) e alimentam
    APENAS a correção rho de Dixon-Coles, que corrige a correlação em
    placares baixos reais — não faz sentido aplicá-la sobre xG."""
    ataque, defesa, home_adv, rho = _desempacotar_parametros(x, n_times)

    lam = np.exp(ataque[idx_casa] - defesa[idx_fora] + home_adv)
    mu = np.exp(ataque[idx_fora] - defesa[idx_casa])

    tau = _correcao_dixon_coles(gols_casa_obs, gols_fora_obs, lam, mu, rho)
    tau = np.clip(tau, 1e-10, None)  # evita log(0) em regiões inválidas de rho

    log_lik = (
        np.log(tau)
        + _log_poisson_continuo(alvo_casa, lam)
        + _log_poisson_continuo(alvo_fora, mu)
    )
    return -np.sum(pesos * log_lik)


def ajustar_modelo(
    df: pd.DataFrame,
    liga: str,
    data_corte,
    xi: float = XI_PADRAO,
    alpha_xg: float = 0.0,
) -> ModeloDixonColes:
    """Ajusta um modelo Dixon-Coles para UMA liga usando apenas partidas com
    Date < data_corte. `df` deve conter (ao menos) as colunas: liga, Date,
    HomeTeam, AwayTeam, FTHG, FTAG (e xG_casa, xG_fora se alpha_xg > 0).

    alpha_xg em [0, 1] controla o alvo de treino do termo de Poisson:
        alvo = alpha_xg * xG + (1 - alpha_xg) * gols_observados
    alpha_xg=0 (padrão): Dixon-Coles clássico, treina só com gols observados.
    alpha_xg=1: treina inteiramente com xG contínuo (sem arredondar — usa
    _log_poisson_continuo, a extensão da log-pmf de Poisson via log-gama,
    que aceita contagens fracionárias).
    Valores intermediários misturam os dois insumos.

    Independentemente de alpha_xg, a correção rho de Dixon-Coles é sempre
    avaliada nos placares REALMENTE observados (FTHG/FTAG), nunca no xG —
    ela corrige a correlação em placares baixos de fato ocorridos.
    """
    if not (0.0 <= alpha_xg <= 1.0):
        raise ValueError("alpha_xg deve estar entre 0 e 1.")

    data_corte = pd.Timestamp(data_corte)

    df_liga = df[df["liga"] == liga].copy()
    df_liga = df_liga[df_liga["Date"] < data_corte]

    colunas_necessarias = ["FTHG", "FTAG", "HomeTeam", "AwayTeam"]
    if alpha_xg > 0:
        if not {"xG_casa", "xG_fora"}.issubset(df_liga.columns):
            raise ValueError("alpha_xg > 0 exige as colunas xG_casa/xG_fora no DataFrame (ver src/ingest_xg.py).")
        colunas_necessarias += ["xG_casa", "xG_fora"]
    df_liga = df_liga.dropna(subset=colunas_necessarias)

    if df_liga.empty:
        raise ValueError(f"Sem partidas de '{liga}' anteriores a {data_corte.date()} para ajustar o modelo (alpha_xg={alpha_xg}).")

    times, idx_casa, idx_fora = _preparar_indices(df_liga)
    n_times = len(times)

    gols_casa_obs = df_liga["FTHG"].to_numpy(dtype=float)
    gols_fora_obs = df_liga["FTAG"].to_numpy(dtype=float)

    if alpha_xg > 0:
        xg_casa = df_liga["xG_casa"].to_numpy(dtype=float)
        xg_fora = df_liga["xG_fora"].to_numpy(dtype=float)
        alvo_casa = alpha_xg * xg_casa + (1 - alpha_xg) * gols_casa_obs
        alvo_fora = alpha_xg * xg_fora + (1 - alpha_xg) * gols_fora_obs
    else:
        alvo_casa = gols_casa_obs
        alvo_fora = gols_fora_obs

    pesos = calcular_pesos_temporais(df_liga["Date"], data_corte, xi)

    # x0: ataque/defesa livres = 0, home_adv modesto, rho pequeno negativo
    n_params = (n_times - 1) + n_times + 1 + 1
    x0 = np.zeros(n_params)
    x0[-2] = 0.2   # home_adv inicial
    x0[-1] = -0.05  # rho inicial

    bounds = [(-5, 5)] * (n_times - 1) + [(-5, 5)] * n_times + [(-2, 2)] + [(-0.9, 0.9)]

    resultado = minimize(
        _neg_log_verossimilhanca,
        x0,
        args=(idx_casa, idx_fora, alvo_casa, alvo_fora, gols_casa_obs, gols_fora_obs, pesos, n_times),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    ataque, defesa, home_adv, rho = _desempacotar_parametros(resultado.x, n_times)

    return ModeloDixonColes(
        liga=liga,
        data_corte=data_corte.date().isoformat(),
        xi=xi,
        times=times,
        ataque={t: float(a) for t, a in zip(times, ataque)},
        defesa={t: float(d) for t, d in zip(times, defesa)},
        home_adv=float(home_adv),
        rho=float(rho),
        n_jogos_usados=len(df_liga),
        alpha_xg=alpha_xg,
    )


def contar_jogos_por_time(df: pd.DataFrame, liga: str, data_corte) -> dict[str, int]:
    """Quantos jogos (mandante ou visitante) cada time da liga tem na base
    ANTES de data_corte — mesmo filtro usado por ajustar_modelo() pra treinar.
    Usado pra sinalizar times com histórico curto demais (ex.: recém
    promovidos): sem esse aviso, um ataque/defesa ajustado com poucos jogos
    pode gerar probabilidades instáveis que parecem (mas não são) uma
    vantagem real contra o mercado."""
    data_corte = pd.Timestamp(data_corte)
    df_liga = df[(df["liga"] == liga) & (df["Date"] < data_corte)]
    contagem = df_liga["HomeTeam"].value_counts().add(df_liga["AwayTeam"].value_counts(), fill_value=0)
    return contagem.astype(int).to_dict()


def matriz_placares(
    modelo: ModeloDixonColes,
    time_casa: str,
    time_fora: str,
    max_gols: int = MAX_GOLS_PADRAO,
) -> np.ndarray:
    """Retorna a matriz de probabilidades de placar (max_gols+1) x (max_gols+1),
    normalizada, com a correção Dixon-Coles aplicada nas 4 células de placar
    baixo. Linha = gols do mandante, coluna = gols do visitante.
    """
    for t in (time_casa, time_fora):
        if t not in modelo.ataque:
            raise ValueError(f"Time '{t}' não encontrado no modelo da liga '{modelo.liga}'.")

    lam = np.exp(modelo.ataque[time_casa] - modelo.defesa[time_fora] + modelo.home_adv)
    mu = np.exp(modelo.ataque[time_fora] - modelo.defesa[time_casa])

    gols = np.arange(0, max_gols + 1)
    prob_casa = poisson.pmf(gols, lam)
    prob_fora = poisson.pmf(gols, mu)
    matriz = np.outer(prob_casa, prob_fora)

    x_grid, y_grid = np.meshgrid(gols, gols, indexing="ij")
    lam_grid = np.full_like(x_grid, lam, dtype=float)
    mu_grid = np.full_like(y_grid, mu, dtype=float)
    tau = _correcao_dixon_coles(x_grid, y_grid, lam_grid, mu_grid, modelo.rho)
    matriz = matriz * tau
    matriz = np.clip(matriz, 0, None)

    matriz = matriz / matriz.sum()
    return matriz


def _nome_arquivo_modelo(liga: str, alpha_xg: float) -> str:
    slug = liga.lower().replace(" ", "_")
    sufixo = "" if alpha_xg == 0.0 else f"_xg_a{round(alpha_xg * 100):03d}"
    return f"{slug}{sufixo}.json"


def salvar_modelo(modelo: ModeloDixonColes, caminho: Path | None = None) -> Path:
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    if caminho is None:
        caminho = PASTA_MODELOS / _nome_arquivo_modelo(modelo.liga, modelo.alpha_xg)
    caminho.write_text(json.dumps(modelo.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


def carregar_modelo(liga: str, alpha_xg: float = 0.0) -> ModeloDixonColes:
    caminho = PASTA_MODELOS / _nome_arquivo_modelo(liga, alpha_xg)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo (alpha_xg={alpha_xg}) não encontrado para a liga '{liga}' em {caminho}.")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    return ModeloDixonColes.from_dict(d)


def main():
    """Ajusta e salva um modelo por liga, usando a data de hoje como corte
    (ou seja, usa todo o histórico disponível)."""
    import argparse

    parser = argparse.ArgumentParser(description="Ajusta modelos Dixon-Coles por liga")
    parser.add_argument("--dados", default=None, help="Padrão: partidas.csv (ou partidas_xg.csv se --alpha-xg > 0)")
    parser.add_argument("--data-corte", default=None, help="YYYY-MM-DD (padrão: hoje)")
    parser.add_argument("--xi", type=float, default=XI_PADRAO)
    parser.add_argument("--alpha-xg", type=float, default=0.0, help="Peso do xG no alvo de treino (0=gols, 1=xG puro)")
    args = parser.parse_args()

    caminho_dados = args.dados or str(RAIZ / "data" / "processed" / ("partidas_xg.csv" if args.alpha_xg > 0 else "partidas.csv"))
    df = pd.read_csv(caminho_dados, parse_dates=["Date"])
    data_corte = args.data_corte or date.today().isoformat()

    for liga in sorted(df["liga"].unique()):
        modelo = ajustar_modelo(df, liga, data_corte, xi=args.xi, alpha_xg=args.alpha_xg)
        caminho = salvar_modelo(modelo)
        print(f"[{liga}] ajustado (alpha_xg={args.alpha_xg}) com {modelo.n_jogos_usados} jogos | "
              f"home_adv={modelo.home_adv:.3f} rho={modelo.rho:.3f} -> {caminho}")


if __name__ == "__main__":
    main()
