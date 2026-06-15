# licenta_drl_supply_chain
Analiza comportamentului unui agent DRL ierarhic în Supply Chain

Acest repository conține contribuțiile originale ale lucrării de licență,
realizate pe baza unui simulator DRL open-source (Stefan Lopez, 2020).

## Conținut

- `Licenta.ipynb` — Analiză statistică oentru 4 ipoteze (H1-H4) realizată în Google Colab
- `lead_time_sensitivity.py` — Script pentru testarea sensibilității agentului 
  la variații ale timpilor de livrare
- `sensitivity_results.csv` — Rezultatele celor 12 scenarii × 200 episoade
-  `grafic.py`- script pentru generarea graficului care evidențiază convergența algoritmului A2C pe parcursul antrenării

Cele patru analize experimentale acoperă:
- **H1** — Testarea normalității cererii simulate (Shapiro-Wilk)
- **H2** — Efectul Bullwhip în arhitectura ierarhică (Coeficient de Variație)
- **H3** — Asimetria sistematică a comenzilor agentului (Skewness + test t)
- **H4** — Sensibilitatea agentului la variații ale timpilor de livrare


## Ghid de execuție

### H1, H2, H3 — Google Colab
1. Se deschide `Licenta.ipynb` în Google Colab
2. Se încarcă în Google Drive fișierele de date din proiectul original:
   - `incoming_orders.csv`
   - `dc_order_list.csv`
3. Se actualizează căile din celula de Setup cu locația din Drive
4. Se rulează celulele în ordine

### H4 — Local
1. Se clonează repository-ul proiectului original:
   `git clone https://github.com/stefan-lopez/supply_chain_reinforcement_learning`
2. Se copiază `lead_time_sensitivity.py` în folderul rădăcină al proiectului
3. Ne asigurăm că modelul antrenat există la calea:
   `saves/best/best_+1977.900_3692000.dat`
4. Se rulează:
   `python lead_time_sensitivity.py`
6. Rezultatele sunt salvate automat în `sensitivity_results.csv`

### Generarea graficului
1. Se copiază `grafic.py` în folderul rădăcină al proiectului
2. Se rulează:
   `python grafic.py`

## Dependențe
- Python 3.6+
- PyTorch, numpy, scipy, pandas, matplotlib
- Repository original: https://github.com/stefan-lopez/supply_chain_reinforcement_learning
