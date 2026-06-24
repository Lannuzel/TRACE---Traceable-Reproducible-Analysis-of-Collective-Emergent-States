# Évaluation de la performance par scénario

Pipeline de calcul des scores de performance pour les tâches collaboratives BIM (PC et VR).

## Structure du dossier

```
evaluation_performance_scenario/
├── score_groupes.py              # Score de performance (Score_perf_tsk)
│
├── S1/                           # Scénario 1 — Aménagement de salle (tables)
│   ├── performance_eval_unified.py   # Évaluation spatiale (PC et VR)
│   ├── PC/
│   │   ├── eval_consignes_align.py   # Évaluation des règles R1–R7
│   │   └── run_perf_s1_PC.ps1        # Script d'exécution batch
│   ├── VR/
│   │   ├── eval_consignes_align.py   # Évaluation des règles R1–R7 (VR)
│   │   └── run_perf_s1_vr.ps1        # Script d'exécution batch
│
├── S2/                           # Scénario 2 — Réservations BIM
│   ├── PC/
│   │   ├── performance_eval.py       # Évaluation spatiale S2 PC
│   │   ├── eval_consignes_s2.py      # Vérification host/shape/dimensions
│   │   └── run_perf_s2_PC.ps1
│   ├── VR/
│   │   ├── performance_eval.py       # Évaluation spatiale S2 VR
│   │   ├── eval_consignes_s2.py      # Vérification host/shape/dimensions
│   │   └── run_perf_s2_VR.ps1
│
├── corrections/                  # Fichiers solution (corrections attendues)
│   ├── PC_s1.csv                     # 16 positions de tables attendues (PC)
│   ├── VR_S1.xlsx                    # 16 positions de tables attendues (VR)
│   ├── PC_s2.csv                     # 16 réservations attendues (PC)
│   └── VR_S2.csv                     # 16 réservations attendues (VR)
│
├── script_revit/                 # Scripts Dynamo pour export Revit
│   ├── export.S1.dyn
│   └── export.S2.dyn
```

## Scénarios

### S1 — Aménagement d'une salle de cours

Les participants placent des **tables** dans une salle en respectant 7 règles de disposition :

| Règle | Description |
|-------|-------------|
| R1 | Disposition en rangées |
| R2 | Alignement rangée avant (Z max) |
| R3 | Espacement inter-rangées |
| R4 | Espacement intra-rangée |
| R5 | Alignement colonne droite (X max) |
| R6 | Même nombre de tables par rangée |
| R7 | Alignement colonne gauche (X min) |

### S2 — Réservations pour réseaux

Les participants placent des **réservations** (vides dans la structure) pour le passage de gaines et canalisations. Chaque réservation est évaluée sur :

| Critère | Description |
|---------|-------------|
| Détection (present_ok) | La réservation attendue a-t-elle été placée ? |
| Support (host_ok) | SOL ou MUR correct ? |
| Forme (shape_ok) | CIRCULAIRE ou RECTANGULAIRE correct ? |
| Dimensions (plan_dims_ok) | Dimensions en plan correctes ? |
| Épaisseur (thickness_ok) | Épaisseur correcte ? |

---

## Métriques de performance

### Score de performance — `Score_perf_tsk`

Score composite unique, identique pour S1 et S2.

```
Score_perf_tsk = 0.35 × M1 + 0.20 × M2 + 0.35 × M3 + 0.10 × M4
```

| Composante | Formule | Description |
|------------|---------|-------------|
| M1 (consignes) | `(règles passées / règles totales) × 100` | Respect des contraintes métier |
| M2 (nombre) | `F1-score entre objets placés et attendus × 100` | Comptage d'objets |
| M3 (précision) | `exp(-MAE / tolérance) × 100` | Précision spatiale |
| M4 (temps) | `max(0, 1 - durée/durée_max) × 100` | Bonus vitesse |

### Normalisation inter-scénarios (z-score)

S1 et S2 ayant des contextes différents (aménagement de salle vs réservations), un **z-score par scénario** permet la comparaison inter-scénarios :

```
Score_perf_tsk_z = (Score_perf_tsk - mean_scénario) / sd_scénario
```

- Centré sur 0, écart-type = 1 au sein de chaque scénario.
- Un z-score de +1.0 signifie « 1 écart-type au-dessus de la moyenne de son scénario ».
- Calculé automatiquement dans le rapport (`report_ci_group_pdf.py`), pas dans `score_groupes.py`.

---

## Pipeline d'exécution

### Étape 1 — Évaluation par groupe

Les scripts PowerShell (`run_perf_*.ps1`) parcourent les dossiers de données brutes et lancent pour chaque groupe :

1. **performance_eval** : Appariement spatial (algorithme hongrois) entre positions participant et solution. Produit `performance_per_object.csv` et `performance_summary.json`.

2. **eval_consignes** : Évaluation des contraintes métier.
   - S1 : `eval_consignes_align.py` vérifie les 7 règles de disposition (R1–R7).
   - S2 : `eval_consignes_s2.py` vérifie host/shape/dimensions/épaisseur item par item.

```powershell
# Exécuter S1 PC
cd D:\Analyse_donnee\Longitudinale\scripts\evaluation_performance_scenario
.\S1\PC\run_perf_s1_PC.ps1

# Exécuter S1 VR
.\S1\VR\run_perf_s1_vr.ps1

# Exécuter S2 PC
.\S2\PC\run_perf_s2_PC.ps1

# Exécuter S2 VR
.\S2\VR\run_perf_s2_VR.ps1
```

### Étape 2 — Agrégation des scores

```powershell
# Score de performance (Score_perf_tsk)
python score_groupes.py --roots D:\...\results\performance_task\performance_VR\S1 D:\...\results\performance_task\performance_PC\S1 `
                                D:\...\results\performance_task\performance_VR\S2 D:\...\results\performance_task\performance_PC\S2 `
                        --out D:\...\results\performance_task\recap_scores_all.csv
```

> Le z-score (`Score_perf_tsk_z`) est calculé automatiquement dans le rapport, par scénario.

### Sorties

| Fichier | Description |
|---------|-------------|
| `results/performance_task/recap_scores_all.csv` | Scores agrégés (tous groupes) : `Score_perf_tsk`, M1–M4 |
| `results/performance_task/performance_{PC,VR}/S{1,2}/bimXXX/` | Résultats bruts par groupe |

---

## Calibration (S1)

Les données VR nécessitent une calibration pour aligner l'espace participant avec l'espace solution.

- **Méthode** : Translation uniquement (pas de mise à l'échelle) — cohérent entre PC et VR.
- **Algorithme** : Regroupement par clusters (3 positions min parmi les solutions), calcul de l'offset médian sur le cluster le plus proche.
- **Tolérance adaptative** : `row_align_eff = max(args.row_align, 0.10 × step_z)` — la tolérance d'accrochage aux rangées s'adapte au pas inter-rangées.

---

## Groupes exclus

| Groupe | Raison |
|--------|--------|
| bim002 | Fichier participant non lisible / données corrompues |
| bim032 | CSV vide |
| bim065 | Pas de fichier performance |
| bim075 | Piste audio absente, aucun marker, pas de dossier performance |

---

## Fichiers solution (corrections/)

- **S1** : 16 positions de tables attendues (4 rangées × 4 colonnes).
- **S2** : 16 réservations attendues (circulaires et rectangulaires, dans murs et sols).

Les fichiers PC utilisent les coordonnées Revit (mètres). Les fichiers VR utilisent les coordonnées Unity (normalisées par les dimensions de la scène).

---

## Dépendances

- Python 3.10+
- pandas, numpy, scipy, matplotlib
- `common/constants.py` : `PERF_WEIGHTS`, `EXCLUDED_GROUPS`, `MAX_TASK_DURATION_S`
