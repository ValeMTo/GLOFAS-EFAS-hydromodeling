# Sensibilità del salto idraulico sulla produzione run-of-river
### Riassunto tecnico — risposta al Reviewer #4

---

## 1. Contesto e richiesta del reviewer

Il Reviewer #4 osserva che i modelli macro-energetici rappresentano la conversione idraulico-elettrica con una relazione lineare a tratti (3 segmenti) tra portata relativa e potenza normalizzata, valida per turbine a flusso incrociato (Banki/Michell/Ossberger) usate su piccoli impianti (< 10 MW). Le tecnologie prevalenti su larga scala (Francis, Kaplan, Pelton) sono invece rappresentate in letteratura con approssimazioni a 3-5 segmenti, adeguate per impianti a bacino con **salto costante**, ma **non valide per impianti run-of-river (RoR)**, dove la variazione di salto (oltre alla portata) incide significativamente sulla potenza prodotta.

Il reviewer richiede quindi:

1. uno studio di sensitività che quantifichi il limite dell'approssimazione a 3 segmenti per gli impianti RoR;
2. che tenga conto della variazione relativa del salto **in aggiunta** alla variazione di portata;
3. contestualizzato rispetto alla quota di capacità RoR sul totale idroelettrico, poiché questi impianti sono tipicamente scelti per l'integrazione delle rinnovabili variabili (VRES).

---

## 2. Obiettivo e perimetro dell'analisi

Si isola **esclusivamente l'effetto del salto variabile**, confrontando nove formulazioni applicate alla stessa serie di afflusso, tutte derivate dalla stessa rampa V0 e dalla stessa geometria di canale reale — nessuna curva di efficienza turbina è coinvolta (v. §4).

**Famiglia a un lato** (assume il livello di monte fisso, solo il livello di valle varia con la portata):
- **V0** — la formula pubblicata nel paper (rampa affine a 3 segmenti, salto costante);
- **V_head** — la stessa rampa, moltiplicata per un fattore di salto reale (guadagno a carico parziale + perdita in piena);
- **V_head_adverse** — solo la perdita in piena (nessun credito a carico parziale) — il caso limite più severo;
- **V_head_gain** — mirror: solo il guadagno a carico parziale, nessuna perdita in piena.

**Famiglia a due lati** (lascia variare con la portata sia il livello di monte sia quello di valle, ciascuno con la propria geometria di canale reale — supera l'assunzione "monte fisso" contestata durante la revisione interna):
- **V_head_cellcell** — cella abbinata (valle) e cella immediatamente a monte nella rete di drenaggio (~5 km di distanza);
- **V_head_cellcell_downstream** — mirror: cella abbinata (monte) e cella immediatamente a valle nella rete di drenaggio (~5 km, copertura maggiore);
- **V_head_local100** — come V_head_cellcell, ma monte e valle sono la posizione vera dell'impianto proiettata sul segmento e un punto 100 m più a valle (scala reale intake→restituzione, non la scala della cella GloFAS);
- **V_head_cellcell_gain**, **V_head_local100_gain**, **V_head_cellcell_downstream_gain** — mirror "solo guadagno" delle tre varianti a due lati sopra.

---

## 3. Metodologia

### 3.1 Formule di base

Tutte le formule partono dalla portata relativa:

```
q = Q / Qmax
```

dove `Qmax` è la portata di progetto della turbina:

```
Qmax = P_nom / (9.81 · head_m · 1e-3 · η)        η = 0.85 (assunto)
```

**V0 — formula pubblicata (rampa affine, salto costante)**

```
p(q) = 0                          se q < qmin
p(q) = (q - qmin) / (1 - qmin)    se qmin ≤ q ≤ 1
p(q) = 1                          se q > 1
```

con `qmin = 0.20` (assunto).

**V_head — rampa V0 modulata dal salto reale (un lato)**

```
p(q) = clip( p_V0(q) · h(q),  0, 1 )
h(q) = 1 - (y(q) - y(q=1)) / head_m
```

dove `y(Q)` è il tirante idrico reale del fiume nella cella abbinata (non della turbina). `h(q) > 1` a carico parziale (guadagno: il livello a valle è più basso), `h(q) < 1` in piena (perdita: il livello a valle sale). **Nessun parametro stimato**: `y(Q)` proviene da geometria di canale reale (§3.2).

**V_head_adverse / V_head_gain — varianti limite**

```
h_adverse(q) = min(1, h(q))                              solo perdita
h_gain(q)    = h(q) per q<1,  = 1 per q≥1                 solo guadagno
```

Nessun parametro aggiuntivo: sono troncamenti della stessa curva reale già calcolata per V_head.

**Famiglia a due lati — supera l'assunzione "monte fisso"**

```
h(q) = 1 - [Δy_valle(q) - Δy_monte(q)] / head_m
Δy_x(q) = y_x(q) - y_x(q=1)
```

dove `y_monte(Q)` e `y_valle(Q)` sono risolti indipendentemente da Manning (§3.2), ciascuno con la propria geometria di canale. Se monte e valle salgono/scendono insieme, i due termini si compensano parzialmente — l'effetto atteso è quindi **più piccolo** di quello a un lato. Le varianti `_gain` applicano lo stesso troncamento (`h(q)=1` per q≥1) a questa formula a due lati.

Identificazione di monte/valle:
- **V_head_cellcell**: monte = nodo immediatamente a monte nella catena di drenaggio (`upstream_model_ids` nella tabella `glofas_drain_lv3_europe_full.parquet`; alle confluenze si sceglie il ramo con `drainage_area` maggiore); valle = cella abbinata (quella già usata per estrarre la portata dell'impianto).
- **V_head_cellcell_downstream**: mirror — cella abbinata come monte, `downstream_model_id` (valore singolo, nessuna ambiguità di confluenza) come valle. Copertura maggiore perché ogni nodo ha un vicino di valle tranne i veri sbocchi di bacino.
- **V_head_local100**: le coordinate vere dell'impianto vengono proiettate sul segmento monte→valle (proiezione ortogonale in un piano metrico locale); il punto proiettato è "monte", un punto 100 m più a valle sullo stesso segmento è "valle". La geometria nei due punti è un'interpolazione lineare tra i valori già campionati alle due celle estreme (nessuna nuova lettura raster). Con 100 m ≪ 5 km di distanza tra celle, geometria e tiranti ai due punti sono quasi identici → cancellazione quasi totale attesa (confermata dai risultati, §5).

### 3.2 Calcolo di y(Q): equazione di Manning su sezione composita

```
Q = (1/n) · A(y)^(5/3) / P(y)^(2/3) · √S₀
```

- **Sotto il livello di piena banchina** (y ≤ chanbnkf) — sezione trapezoidale:
  ```
  A(y) = (chanbw + chans·y) · y
  P(y) = chanbw + 2y · √(1 + chans²)
  ```
- **Sopra il livello di piena banchina** (y > chanbnkf) — si aggiunge la golena (floodplain) di larghezza `chanflpn`, con pareti laterali approssimate verticali (semplificazione standard per canali compound):
  ```
  A(y) = A(chanbnkf) + (y - chanbnkf) · chanflpn
  P(y) = P(chanbnkf) + 2·(y - chanbnkf) + (chanflpn - chanbw)
  ```
- `S₀ = changrad` (pendenza longitudinale), `n = chanman` (scabrezza di Manning).

Per ogni portata `Q` si risolve `y` per bisezione (nessuna forma chiusa per la sezione composita); si tabula `y(q)` su una griglia di `q` (0.001–20) per ciascun punto (cella abbinata, cella a monte, cella a valle, o punto interpolato), poi si interpola sulla serie temporale reale dell'impianto.

### 3.3 Dati usati (nessuna stima dove il dato esiste)

| Parametro | Fonte | Note |
|---|---|---|
| `chanbw`, `chans`, `changrad`, `chanman`, `chanbnkf`, `chanflpn` | CEMS_SurfaceFields_2022, static maps LISFLOOD per GloFAS v1.1.1 (JRC Data Catalogue), griglia globale 0.05°/3600×7200 | Usati operativamente per GloFASv4/EFASv5 |
| Cella abbinata (valle di riferimento) | `final_hydro_plant_glofas_points.parquet` — la **stessa cella** già usata dal pipeline per estrarre la serie di portata di ciascun impianto | Evita di associare la portata di una cella alla geometria di un'altra |
| Cella a monte / a valle (famiglia a due lati) | `glofas_drain_lv3_europe_full.parquet` — colonne `upstream_model_ids` (monte, con selezione del ramo a `drainage_area` maggiore alle confluenze) e `downstream_model_id` (valle, valore singolo) | Stessa rete di drenaggio già costruita dal pipeline per il matching |
| Coordinate vere dell'impianto (solo V_head_local100) | `GloHydroRes_vs1.csv`, colonne `plant_lon`/`plant_lat` | Usate solo per la proiezione sul segmento monte-valle |
| `head_m`, `p_nom` | `GloHydroRes_vs1.csv`, imputazione dei mancanti per mediana paese/tecnologia (logica già esistente nel pipeline) | Non modificata da questa analisi |

---

## 4. Assunzioni esplicite

- `η = 0.85`, `qmin = 0.20` — costanti globali, condivise da tutte le formule confrontate (non influenzano il confronto ΔE).
- Pareti laterali verticali per la golena oltre il bankfull — approssimazione standard per canali compound, dichiarata come tale.
- Floor numerici applicati solo in caso di geometria mancante/degenere: `changrad ≥ 1e-4`, `chanbnkf ≥ 0.10 m`, `chanman` di default 0.035 se mancante, `chans` di default 1.0, `chanflpn` di default 3×`chanbw` (stessa formula empirica usata da LISFLOOD).
- Alle confluenze (più candidati in `upstream_model_ids`), si sceglie il ramo con area di drenaggio maggiore (ramo principale) — una convenzione, non una misura.
- `V_head_local100` usa una distanza fissa di 100 m (assunta come lunghezza tipica derivazione intake→restituzione per un impianto RoR) — parametro configurabile (`DOWNSTREAM_OFFSET_M`), non stimato dai dati.
- Curva di efficienza turbina: esclusa di proposito da questa analisi (vedi §2).

---

## 5. Risultati

**Copertura**: 1553 impianti RoR, 45.24 GW, anni 2015–2018 (35040 ore).

- Famiglia a un lato (V_head/V_head_adverse/V_head_gain): **1553/1553 (100%)** con geometria abbinata valida.
- V_head_cellcell / V_head_local100: **1147/1553 (74%)** con cella a monte risolta nella catena di drenaggio (406 impianti su sorgenti o senza match valido).
- V_head_cellcell_downstream: **1320/1553 (85%)** con cella a valle risolta (233 impianti su sbocchi di bacino o senza match valido) — copertura maggiore come atteso, perché `downstream_model_id` non ha l'ambiguità delle confluenze.
- Per il 46% degli impianti con monte risolto (531/1147), i 100 m di `V_head_local100` superano l'intero segmento monte-valle (celle molto ravvicinate nella rete) — per questi il punto valle coincide con la cella abbinata.

**Baseline**: V0 = 125.63 TWh/anno. Osservato (ENTSO-E): RoR = 176.84 TWh/anno, RoR+Reservoir (esclude il pompaggio, non modellato) = 375.10 TWh/anno.

### 5.1 Quota RoR sulla produzione idroelettrica (esclusi impianti di pompaggio, PHS)

| Tecnologia | Capacità | Quota (RoR + Reservoir) |
|---|---|---|
| Run-of-River | 42.40 GW | **31.7 %** |
| Reservoir | 91.19 GW | 68.3 % |
| *(Pumped Storage, escluso)* | *(54.58 GW)* | *(non conteggiato)* |

### 5.2 ΔE = formula con salto − V0 (% dell'energia V0 dello stesso sottoinsieme)

**Famiglia a un lato** (monte assunto fisso):

| Sottoinsieme | V_head (simmetrico) | V_head_adverse (solo perdita) | V_head_gain (solo guadagno) |
|---|---|---|---|
| Flotta intera | +0.03 % | **-1.12 %** | **+1.15 %** |
| Basso salto (< 15 m) | +1.22 % | **-2.34 %** | **+3.56 %** |
| Alto salto (≥ 15 m) | -0.30 % | -0.79 % | +0.49 % |
| Ben assegnati (Qmax/Q̄ ∈ [0.5,4]) | +0.99 % | -0.70 % | +1.70 % |

**Famiglia a due lati** (monte e valle entrambi variabili — copertura parziale, v. sopra):

| Sottoinsieme | V_head_cellcell (monte trovato, ~5km) | V_head_cellcell_downstream (valle trovata, ~5km) | V_head_local100 (100m reali) |
|---|---|---|---|
| Flotta intera | -0.31 % | -0.40 % | **-0.01 %** |
| Basso salto (< 15 m) | -0.77 % | -0.84 % | -0.02 % |
| Alto salto (≥ 15 m) | -0.19 % | -0.28 % | -0.01 % |
| Ben assegnati | -0.15 % | -0.26 % | -0.01 % |

**Mirror "solo guadagno" della famiglia a due lati:**

| Sottoinsieme | V_head_cellcell_gain | V_head_cellcell_downstream_gain | V_head_local100_gain |
|---|---|---|---|
| Flotta intera | +0.06 % | -0.05 % | +0.03 % |
| Basso salto (< 15 m) | **-0.18 %** | **-0.12 %** | -0.02 % |
| Alto salto (≥ 15 m) | +0.13 % | -0.03 % | +0.05 % |

---

## 6. Interpretazione

1. **Il modello a un lato (monte fisso) sovrastima l'effetto.** Confrontando V_head_adverse (-2.34% a basso salto) con la famiglia a due lati (-0.77% / -0.84% a ~5 km, -0.02% a 100 m), è evidente che gran parte dell'effetto stimato con il monte fisso deriva proprio da quell'assunzione, non dalla fisica del salto variabile in sé.

2. **La scala conta più della fisica di base.** Alla scala delle celle GloFAS (~5 km) l'effetto a due lati si stabilizza intorno a **-0.3/-0.4%** in flotta e **-0.8%** a basso salto, confermato da due percorsi indipendenti (cella a monte trovata vs cella a valle trovata — risultati entro 0.1 punti percentuali l'uno dall'altro). Alla scala reale dell'impianto (100 m) l'effetto crolla a **quasi zero**, perché su una derivazione corta i livelli di monte e valle si muovono quasi all'unisono.

3. **Il guadagno a carico parziale non è robusto quando si modella correttamente il monte.** Nel modello a un lato il guadagno a basso salto è marcato (+3.56%); nella famiglia a due lati **si annulla e in due casi su tre si inverte di segno** (-0.18% e -0.12% per V_head_cellcell_gain e V_head_cellcell_downstream_gain a basso salto). Questo perché a carico parziale non solo il valle scende, scende anche il monte — e se scende di più, il salto netto cala anziché aumentare, ribaltando l'assunzione del modello a un lato.

4. **Il messaggio complessivo per il reviewer**: l'ampiezza (e in parte il segno) dell'effetto di salto variabile dipende criticamente da due assunzioni esplicite — se il livello di monte è fisso o no, e su quale scala spaziale si misura la variazione. Il range onesto da riportare va da **circa -2.3%** (caso limite, monte fisso, solo basso salto) a **quasi zero** (derivazione corta, 100 m, due lati) passando per **-0.3/-0.8%** (scala di cella, due lati, il valore probabilmente più rappresentativo). Questa sensitività alla scala è essa stessa un risultato scientifico da comunicare, non un limite da nascondere.

5. Gli impianti RoR rappresentano il **31.7%** della capacità idroelettrica non-pompaggio nel dominio modellato — anche gli effetti più piccoli (~-0.3/-0.8%) restano potenzialmente rilevanti a livello di sistema data la dimensione della flotta interessata.

6. Il metodo elimina i parametri liberi che un approccio a letteratura richiederebbe (`dz`, esponente di forma, soglie percentili), sostituendoli con geometria di canale reale — l'unica scelta non misurata è la selezione del ramo principale alle confluenze e la distanza di 100 m in V_head_local100 (dichiarata come parametro, non stimata).

---

## 7. Limiti dichiarati e possibili prossimi passi

- **Curva di efficienza turbina**: non affrontata qui di proposito (v. §2). Il reviewer la cita come contesto ma la richiesta esplicita riguarda il salto; se necessario, un'analisi complementare andrebbe condotta separatamente.
- **Approccio "fisico" vs "mesh a 12-20 segmenti"**: il reviewer suggerisce un approccio empirico multidimensionale (superficie tabulata portata-salto-potenza). Questa analisi propone invece un modello fisico continuo (Manning + geometria di canale reale, monte e valle) che genera la relazione anziché tabularla — concettualmente più solido, ma va argomentato esplicitamente nella risposta al reviewer come alternativa equivalente o superiore, non come omissione del suggerimento.
- **Copertura parziale della famiglia a due lati**: 74-85% degli impianti (non 100%) hanno una cella a monte/valle risolvibile nella rete di drenaggio; il resto (sorgenti, sbocchi, match di fallback) non contribuisce alla stima a due lati.
- **Interpolazione lineare a 100 m**: la geometria interpolata tra due celle ~5 km apart cambia pochissimo su soli 100 m — il risultato quasi nullo di V_head_local100 è una conseguenza diretta di questa interpolazione lineare, non necessariamente della fisica reale su quella scala (il vero profilo del fiume tra le due celle non è una retta).
- **Semplificazione della golena**: pareti laterali verticali oltre il bankfull (§4) — approssimazione standard, ma dichiarata.
- **Selezione del ramo principale alle confluenze**: basata su `drainage_area` massima, una convenzione ragionevole ma non l'unica possibile.
