# -*- coding: utf-8 -*-
"""Script auxiliar (Bloco 1): roda o walk-forward de gols de uma liga e
seleciona o w otimo da mistura modelo+mercado via grade 0.0-1.0."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import avaliar_walkforward
from blend import avaliar_grade_w, escolher_w_otimo, salvar_peso

liga = sys.argv[1]
RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "data" / "processed" / "partidas.csv", parse_dates=["Date"])
data_max = df[df["liga"] == liga]["Date"].max()
data_inicio = data_max - pd.Timedelta(days=365 * 3)

print(f"Walk-forward — {liga}")
df_aval = avaliar_walkforward(df, liga, data_inicio, refit_dias=30, verboso=True)
df_aval.to_csv(RAIZ / "data" / "processed" / f"avaliacao_blend_{liga.lower().replace(' ', '_')}.csv", index=False)

tabela = avaliar_grade_w(df_aval)
print(tabela.to_string(index=False))
w_otimo = escolher_w_otimo(tabela)
print(f"w otimo para {liga}: {w_otimo}")
salvar_peso(liga, w_otimo)
