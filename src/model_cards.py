# -*- coding: utf-8 -*-
"""
Modelo de disciplina por jogo (cartões amarelos e faltas), com a mesma
estrutura ataque/defesa/vantagem de mando/decaimento temporal/binomial
negativa do modelo de escanteios (ver model_corners.py), mais um efeito
aditivo (no log da média) do árbitro da partida.

Interpretação dos parâmetros por time (o nome "ataque"/"defesa" é mantido
por analogia ao Dixon-Coles, mas aqui o sentido futebolístico é outro):
- ataque: propensão do próprio time a cometer faltas/receber cartões.
- defesa: o quanto o ESTILO DE JOGO do adversário induz faltas/cartões
  nesse time (times mais tecnicos/de posse tendem a sofrer mais faltas).

    mu_casa = exp(ataque_casa - defesa_fora + home_adv + efeito_arbitro)
    mu_fora = exp(ataque_fora - defesa_casa + efeito_arbitro)

home_adv aqui pode sair NEGATIVO — e isso é esperado: é comum mandantes
receberem menos cartões que visitantes (viés de arbitragem a favor da
torcida da casa), o oposto do que ocorre em gols/escanteios.

Efeito de árbitro: incluído para Premier League e Championship (têm a
coluna Referee); La Liga não tem essa coluna na fonte, então o ajuste roda
sem esse termo. Árbitros com menos de `min_jogos_arbitro` partidas no
histórico (antes da data de corte) não recebem parâmetro próprio — usam o
efeito 0 (média da liga), evitando estimar um efeito de árbitro com poucos
dados. Restrição de identificabilidade: média(efeito_arbitro) = 0 entre os
árbitros com parâmetro próprio (mesma reparametrização usada em ataque).

Um modelo por liga; nenhuma função de ajuste usa dados posteriores à data
de corte informada explicitamente pelo chamador.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

from model_corners import distribuicao_total  # reaproveita a convolução genérica

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "data" / "processed" / "modelos"

XI_PADRAO = 0.0018
MIN_JOGOS_ARBITRO_PADRAO = 10
MAX_CARTOES_PADRAO = 10
MAX_FALTAS_PADRAO = 30

# ligas em que a coluna Referee está disponível na fonte (ver src/ingest.py)
LIGAS_COM_ARBITRO = {"Premier League", "Championship"}


@dataclass
class ModeloDisciplina:
    """Parâmetros ajustados de um modelo de disciplina (cartões ou faltas)."""

    liga: str
    tipo: str  # "cartoes" ou "faltas"
    data_corte: str
    xi: float
    times: list[str]
    ataque: dict[str, float]
    defesa: dict[str, float]
    home_adv: float
    alpha: float
    usar_arbitro: bool
    efeito_arbitro: dict[str, float] = field(default_factory=dict)
    min_jogos_arbitro: int = MIN_JOGOS_ARBITRO_PADRAO
    n_jogos_usados: int = 0

    def to_dict(self) -> dict:
        return {
            "liga": self.liga, "tipo": self.tipo, "data_corte": self.data_corte, "xi": self.xi,
            "times": self.times, "ataque": self.ataque, "defesa": self.defesa,
            "home_adv": self.home_adv, "alpha": self.alpha, "usar_arbitro": self.usar_arbitro,
            "efeito_arbitro": self.efeito_arbitro, "min_jogos_arbitro": self.min_jogos_arbitro,
            "n_jogos_usados": self.n_jogos_usados,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModeloDisciplina":
        return cls(**d)


def _nb_n_p(mu: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
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


def _preparar_indices_arbitro(df: pd.DataFrame, min_jogos: int) -> tuple[list[str], np.ndarray]:
    """Retorna a lista de árbitros com >= min_jogos partidas no período de
    treino (só esses recebem parâmetro próprio) e o índice de cada jogo
    nessa lista (-1 = árbitro sem histórico suficiente, usa efeito 0)."""
    contagem = df["Referee"].value_counts()
    arbitros_validos = sorted(contagem[contagem >= min_jogos].index.tolist())
    indice = {a: i for i, a in enumerate(arbitros_validos)}
    idx_arbitro = df["Referee"].map(indice)
    idx_arbitro = idx_arbitro.where(idx_arbitro.notna(), -1).astype(int).to_numpy()
    return arbitros_validos, idx_arbitro


def _desempacotar(x: np.ndarray, n_times: int, n_arbitros: int) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Desempacota em (ataque média-zero, defesa, home_adv, alpha,
    efeito_arbitro média-zero [array vazio se n_arbitros=0])."""
    i = 0
    ataque_livre = x[i:i + n_times - 1]; i += n_times - 1
    ataque = np.append(ataque_livre, -np.sum(ataque_livre))
    defesa = x[i:i + n_times]; i += n_times
    home_adv = x[i]; i += 1
    alpha = np.exp(x[i]); i += 1
    if n_arbitros > 0:
        efeito_livre = x[i:i + n_arbitros - 1]
        efeito_arbitro = np.append(efeito_livre, -np.sum(efeito_livre))
    else:
        efeito_arbitro = np.zeros(0)
    return ataque, defesa, home_adv, alpha, efeito_arbitro


def _neg_log_verossimilhanca(x, idx_casa, idx_fora, idx_arb, y_casa, y_fora, pesos, n_times, n_arbitros) -> float:
    ataque, defesa, home_adv, alpha, efeito_arbitro = _desempacotar(x, n_times, n_arbitros)

    efeito_por_jogo = np.zeros(len(idx_casa))
    if n_arbitros > 0:
        valido = idx_arb >= 0
        efeito_por_jogo[valido] = efeito_arbitro[idx_arb[valido]]

    mu_casa = np.exp(np.clip(ataque[idx_casa] - defesa[idx_fora] + home_adv + efeito_por_jogo, -20, 20))
    mu_fora = np.exp(np.clip(ataque[idx_fora] - defesa[idx_casa] + efeito_por_jogo, -20, 20))

    n_casa, p_casa = _nb_n_p(mu_casa, alpha)
    n_fora, p_fora = _nb_n_p(mu_fora, alpha)
    p_casa = np.clip(p_casa, 1e-10, 1 - 1e-10)
    p_fora = np.clip(p_fora, 1e-10, 1 - 1e-10)

    log_lik = nbinom.logpmf(y_casa, n_casa, p_casa) + nbinom.logpmf(y_fora, n_fora, p_fora)
    return -np.sum(pesos * log_lik)


def _ajustar_generico(
    df: pd.DataFrame, liga: str, data_corte, col_casa: str, col_fora: str, tipo: str,
    xi: float, usar_arbitro: bool, min_jogos_arbitro: int,
) -> ModeloDisciplina:
    data_corte = pd.Timestamp(data_corte)

    df_liga = df[df["liga"] == liga].copy()
    df_liga = df_liga[df_liga["Date"] < data_corte]
    colunas_obrigatorias = [col_casa, col_fora, "HomeTeam", "AwayTeam"]
    if usar_arbitro:
        colunas_obrigatorias.append("Referee")
    df_liga = df_liga.dropna(subset=colunas_obrigatorias)

    if df_liga.empty:
        raise ValueError(f"Sem partidas de '{liga}' anteriores a {data_corte.date()} para ajustar o modelo de {tipo}.")

    times, idx_casa, idx_fora = _preparar_indices(df_liga)
    n_times = len(times)
    y_casa = df_liga[col_casa].to_numpy(dtype=float)
    y_fora = df_liga[col_fora].to_numpy(dtype=float)
    pesos = calcular_pesos_temporais(df_liga["Date"], data_corte, xi)

    if usar_arbitro:
        arbitros_validos, idx_arb = _preparar_indices_arbitro(df_liga, min_jogos_arbitro)
    else:
        arbitros_validos, idx_arb = [], np.full(len(df_liga), -1)
    n_arbitros = len(arbitros_validos)

    n_params = (n_times - 1) + n_times + 1 + 1 + max(n_arbitros - 1, 0)
    x0 = np.zeros(n_params)
    x0[2 * n_times - 1] = np.log(max(y_casa.mean(), 0.5))  # home_adv inicial na escala certa
    # log_alpha (índice 2*n_times) fica em 0 (alpha=1)

    # log_alpha com piso em -5 (alpha ~ 0.0067): cartões, em especial, tendem
    # a ficar perto do limite de Poisson (alpha->0) neste dataset — ver nota
    # no ajuste de main(). O piso evita apenas a degenerescência numérica de
    # alpha exatamente 0 (n=1/alpha -> infinito).
    bounds = (
        [(-4, 4)] * (n_times - 1) + [(-4, 4)] * n_times + [(-3, 4)] + [(-5, 4)]
        + [(-3, 3)] * max(n_arbitros - 1, 0)
    )

    resultado = minimize(
        _neg_log_verossimilhanca, x0,
        args=(idx_casa, idx_fora, idx_arb, y_casa, y_fora, pesos, n_times, n_arbitros),
        method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-10},
    )

    ataque, defesa, home_adv, alpha, efeito_arbitro = _desempacotar(resultado.x, n_times, n_arbitros)

    return ModeloDisciplina(
        liga=liga, tipo=tipo, data_corte=data_corte.date().isoformat(), xi=xi, times=times,
        ataque={t: float(a) for t, a in zip(times, ataque)},
        defesa={t: float(d) for t, d in zip(times, defesa)},
        home_adv=float(home_adv), alpha=float(alpha), usar_arbitro=usar_arbitro,
        efeito_arbitro={a: float(e) for a, e in zip(arbitros_validos, efeito_arbitro)},
        min_jogos_arbitro=min_jogos_arbitro, n_jogos_usados=len(df_liga),
    )


def ajustar_modelo_cartoes(df, liga, data_corte, xi: float = XI_PADRAO,
                            min_jogos_arbitro: int = MIN_JOGOS_ARBITRO_PADRAO) -> ModeloDisciplina:
    """Ajusta o modelo de cartões amarelos (HY/AY) para UMA liga. Usa efeito
    de árbitro automaticamente para Premier League e Championship; La Liga
    não tem a coluna Referee na fonte e é ajustada sem esse termo."""
    usar_arbitro = liga in LIGAS_COM_ARBITRO
    return _ajustar_generico(df, liga, data_corte, "HY", "AY", "cartoes", xi, usar_arbitro, min_jogos_arbitro)


def ajustar_modelo_faltas(df, liga, data_corte, xi: float = XI_PADRAO,
                           min_jogos_arbitro: int = MIN_JOGOS_ARBITRO_PADRAO) -> ModeloDisciplina:
    """Ajusta o modelo de faltas (HF/AF) para UMA liga, mesma estrutura do
    modelo de cartões (inclusive efeito de árbitro nas mesmas ligas)."""
    usar_arbitro = liga in LIGAS_COM_ARBITRO
    return _ajustar_generico(df, liga, data_corte, "HF", "AF", "faltas", xi, usar_arbitro, min_jogos_arbitro)


def matriz_contagem(
    modelo: ModeloDisciplina, time_casa: str, time_fora: str,
    arbitro: str | None = None, max_valor: int = MAX_CARTOES_PADRAO,
) -> np.ndarray:
    """Matriz (max_valor+1) x (max_valor+1) de probabilidades conjuntas
    (linha=casa, coluna=fora), produto das marginais BN (sem correção de
    correlação). Se `arbitro` for None ou não tiver parâmetro próprio
    (história insuficiente), usa o efeito 0 (média da liga)."""
    for t in (time_casa, time_fora):
        if t not in modelo.ataque:
            raise ValueError(f"Time '{t}' não encontrado no modelo de {modelo.tipo} da liga '{modelo.liga}'.")

    efeito_arb = modelo.efeito_arbitro.get(arbitro, 0.0) if (modelo.usar_arbitro and arbitro) else 0.0

    mu_casa = np.exp(modelo.ataque[time_casa] - modelo.defesa[time_fora] + modelo.home_adv + efeito_arb)
    mu_fora = np.exp(modelo.ataque[time_fora] - modelo.defesa[time_casa] + efeito_arb)

    n_casa, p_casa = _nb_n_p(mu_casa, modelo.alpha)
    n_fora, p_fora = _nb_n_p(mu_fora, modelo.alpha)

    valores = np.arange(0, max_valor + 1)
    pmf_casa = nbinom.pmf(valores, n_casa, p_casa)
    pmf_fora = nbinom.pmf(valores, n_fora, p_fora)

    matriz = np.outer(pmf_casa, pmf_fora)
    return matriz / matriz.sum()


def _nome_arquivo_modelo(liga: str, tipo: str) -> str:
    slug = liga.lower().replace(" ", "_")
    return f"{tipo}_{slug}.json"


def salvar_modelo(modelo: ModeloDisciplina, caminho: Path | None = None) -> Path:
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    if caminho is None:
        caminho = PASTA_MODELOS / _nome_arquivo_modelo(modelo.liga, modelo.tipo)
    caminho.write_text(json.dumps(modelo.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


def carregar_modelo(liga: str, tipo: str) -> ModeloDisciplina:
    caminho = PASTA_MODELOS / _nome_arquivo_modelo(liga, tipo)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo de {tipo} não encontrado para a liga '{liga}' em {caminho}.")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    return ModeloDisciplina.from_dict(d)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ajusta modelos de cartões e faltas (BN) por liga")
    parser.add_argument("--dados", default=str(RAIZ / "data" / "processed" / "partidas.csv"))
    parser.add_argument("--data-corte", default=None, help="YYYY-MM-DD (padrão: hoje)")
    parser.add_argument("--xi", type=float, default=XI_PADRAO)
    parser.add_argument("--min-jogos-arbitro", type=int, default=MIN_JOGOS_ARBITRO_PADRAO)
    args = parser.parse_args()

    df = pd.read_csv(args.dados, parse_dates=["Date"])
    data_corte = args.data_corte or date.today().isoformat()

    for liga in sorted(df["liga"].unique()):
        m_cartoes = ajustar_modelo_cartoes(df, liga, data_corte, xi=args.xi, min_jogos_arbitro=args.min_jogos_arbitro)
        c1 = salvar_modelo(m_cartoes)
        print(f"[{liga}] cartoes: {m_cartoes.n_jogos_usados} jogos | home_adv={m_cartoes.home_adv:.3f} "
              f"alpha={m_cartoes.alpha:.3f} arbitro={m_cartoes.usar_arbitro} "
              f"n_arbitros_com_parametro={len(m_cartoes.efeito_arbitro)} -> {c1}")

        m_faltas = ajustar_modelo_faltas(df, liga, data_corte, xi=args.xi, min_jogos_arbitro=args.min_jogos_arbitro)
        c2 = salvar_modelo(m_faltas)
        print(f"[{liga}] faltas : {m_faltas.n_jogos_usados} jogos | home_adv={m_faltas.home_adv:.3f} "
              f"alpha={m_faltas.alpha:.3f} arbitro={m_faltas.usar_arbitro} "
              f"n_arbitros_com_parametro={len(m_faltas.efeito_arbitro)} -> {c2}")


if __name__ == "__main__":
    main()
