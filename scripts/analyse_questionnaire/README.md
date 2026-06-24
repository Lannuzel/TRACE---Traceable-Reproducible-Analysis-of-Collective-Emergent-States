# Analyse des questionnaires TMS / Cohesion

Pipeline d'analyse des questionnaires (TMS, Cohesion sociale/tache, Communication) — version Python.

## Structure

```
analyse_questionnaire/
├── main.py                  # Point d'entree principal
├── py/                      # Package Python (modules)
│   ├── __init__.py
│   ├── config.py            # Constantes (dimensions, Likert, inversions, seuils)
│   ├── io_read.py           # Lecture du fichier source (xlsx/csv)
│   ├── transform.py         # Transformation wide → long + recodage + inversion
│   ├── reliability.py       # Alpha de Cronbach + stats item (r.drop, alpha_if_deleted)
│   ├── descriptives.py      # Descriptifs par dimension
│   ├── role_tests.py        # Tests inter-roles (ANOVA / Kruskal-Wallis)
│   ├── plots.py             # Exports PDF (distributions items, barplots roles)
│   ├── item_pruning.py      # Detection items faibles (r.drop) + epuration
│   ├── g3_context.py        # Profil participant (G3) + commentaires libres
│   └── scenario_modalite.py # Analyse Scenario x Modalite (scores, modeles, plots)
└── README.md                # Ce fichier
```

## Dependances Python

```bash
pip install numpy pandas scipy matplotlib openpyxl statsmodels
```

## Utilisation

```bash
cd Longitudinale/scripts/analyse_questionnaire

# Analyse globale (TMS/Cohesion) + scenario x modalite
python main.py --data ../../data_e2/results-survey.xlsx --out ../../results/questionnaire

# Analyse globale uniquement
python main.py --data ../../data_e2/results-survey.xlsx --out ../../results/questionnaire/global --mode global

# Analyse scenario x modalite uniquement
python main.py --data ../../data_e2/results-survey.xlsx --out ../../results/questionnaire/analyse --mode scenario
```

## Sorties

### Mode global (`--mode global`)
| Fichier | Contenu |
|---------|---------|
| `cronbach_alpha_tms.csv` | Alpha de Cronbach par dimension |
| `stats_items_tms.csv` | Stats item : n, mean, sd, r.drop, alpha_if_deleted |
| `desc_dim_tms.csv` | Descriptifs par dimension (n, mean, sd, median, min, max) |
| `stats_par_dimension_role_perf.csv` | Synthese par role + p-values inter-roles |
| `participant_profile_responses.csv` | Reponses G3 nettoyees (genre, age, familiarite VR, commentaires) |
| `participant_profile_summary.csv` | Resume global du profil participant |
| `participant_profile_category_counts.csv` | Comptages/pourcentages par categorie G3 |
| `free_comments_long.csv` | Commentaires libres non vides (tache / groupe) |
| `free_comments_theme_summary.csv` | Themes heuristiques detectes dans les commentaires libres |
| `questionnaire_reponses_numeriques_long.csv` | Reponses item-level G1/G2 avec valeur texte, score numerique brut, drapeau d'inversion et score final |
| `items_dropped.csv` | Items supprimes (r.drop < 0.20) |
| `analyse_perf.pdf` | Histogrammes items par dimension |
| `plots_dimensions_roles_perf.pdf` | Barplots moyennes +/- IC95 par role |

### Mode scenario (`--mode scenario`)
| Fichier | Contenu |
|---------|---------|
| `scores_dimension_par_participant.csv` | Score par participant x scenario x modalite |
| `descriptifs_scenario_modalite.csv` | Descriptifs par dimension x scenario x modalite |
| `modeles_scenario_modalite_par_dimension.csv` | P-values des modeles (lmer ou OLS) |
| `table_moyennes_pivot.csv` | Tableau pivot des moyennes |
| `plot_means_ic95_scenario_modalite.pdf` | Barplots scenario x modalite |

## Notes
- Le pipeline reproduit la logique des scripts R : extraction de dimension, recodage Likert 1-9, inversion des items marques, Cronbach alpha, r.drop, tests de normalite → ANOVA/Kruskal.
- L'alpha de Cronbach et le r.drop sont calcules en pur Python/numpy (meme formule que `psych::alpha`).
- Le fichier d'entree peut etre `.xlsx` (avec colonnes Session/Modalite/Scenario) ou `.csv` (format LimeSurvey, sep=`;`).
