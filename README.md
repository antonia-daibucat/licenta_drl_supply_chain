# licenta_drl_supply_chain
Analiza comportamentului unui agent DRL ierarhic în Supply Chain

Acest repository conține contribuțiile originale ale lucrării de licență,
realizate pe baza unui simulator DRL open-source (Stefan Lopez, 2020).

## Conținut

- `Licenta.ipynb` — Analiză statistică oentru 4 ipoteze (H1-H4) realizată în Google Colab
- `lead_time_sensitivity.py` — Script pentru testarea sensibilității agentului 
  la variații ale timpilor de livrare
- `sensitivity_results.csv` — Rezultatele celor 12 scenarii × 200 episoade
-  `grafic.py`- script pentru generarea unui grafic care demonstrează că agentul învață

Cele patru analize experimentale acoperă:
- **H1** — Testarea normalității cererii simulate (Shapiro-Wilk)
- **H2** — Efectul Bullwhip în arhitectura ierarhică (Coeficient de Variație)
- **H3** — Asimetria sistematică a comenzilor agentului (Skewness + test t)
- **H4** — Sensibilitatea agentului la variații ale timpilor de livrare
