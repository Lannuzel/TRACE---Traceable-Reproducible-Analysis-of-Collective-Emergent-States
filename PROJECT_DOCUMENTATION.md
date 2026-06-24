# Documentation du projet — Analyse longitudinale de la collaboration BIM en triade

> Dernière mise à jour : 2026-04-07  
> Auteur : T. Lannuzel

---

## Table des matières

1. [Présentation générale du projet](#1-présentation-générale-du-projet)
2. [Design expérimental](#2-design-expérimental)
3. [Données brutes — `data_e2/`](#3-données-brutes--data_e2)
4. [Structure du dossier `Longitudinale/`](#4-structure-du-dossier-longitudinale)
5. [Pipeline de traitement — vue d'ensemble](#5-pipeline-de-traitement--vue-densemble)
6. [Scripts — description détaillée](#6-scripts--description-détaillée)
7. [Résultats produits](#7-résultats-produits)
8. [Cadre théorique](#8-cadre-théorique)
9. [Variables clés et définitions](#9-variables-clés-et-définitions)
10. [Acquisition des données INV — paramètres techniques](#10-acquisition-des-données-inv--paramètres-techniques)
11. [Notes techniques et anomalies connues](#11-notes-techniques-et-anomalies-connues)
12. [Commandes d'exécution](#12-commandes-dexécution)

---

## 1. Présentation générale du projet

Ce projet étudie la **collaboration en triade** lors de tâches de conception architecturale en **BIM (Building Information Modeling)**, selon deux modalités :

- **PC** : collaboration à distance via ordinateur (interface 2D Revit)
- **VR** : collaboration en environnement de réalité virtuelle immersif

L'objectif est d'identifier les **indicateurs comportementaux non verbaux (INV)** — vocaux, faciaux, de regard — et les **indicateurs sociaux/cognitifs** (TCI, questionnaires, Riedl/Hackman) qui prédisent ou sont associés à la **performance collective**, dans une logique de **CI (Collective Intelligence)**.

Le design est **longitudinal à deux temps de mesure (T1/T2)** — deux sessions distinctes pour chaque groupe dans chaque modalité, permettant d'étudier l'évolution des dynamiques collectives sur deux scénarios de tâche.

---

## 2. Design expérimental

### Population et unité d'analyse

- **Unité d'analyse** : triades (groupes de 3 personnes)
- **Effectif** : ~25 triades identifiées dans le dataset fusionné (IDs `bim006` à `bim113`)
- **Cohortes** : Promo 2025 (sessions novembre 2025) + Promo 2026 (sessions janvier 2026)
- **Participants** : étudiants en BIM / architecture

### Rôles dans la triade

Chaque membre de la triade a un rôle attribué :

| Rôle | Description |
|---|---|
| `calculateur` | Responsable des calculs / vérifications techniques |
| `modelisateur` | Responsable de la modélisation 3D dans Revit |
| `lecteur` | Responsable de la lecture des plans et consignes |

### Modalités et scénarios

| Facteur | Niveaux |
|---|---|
| **Modalité** | PC, VR |
| **Scénario (session)** | S1 (BSI_A1 ou FISA_A3), S2 (FISA_A5) |
| **Timepoint** | T1, T2 |
| **Condition** | Combinaison modalité × timepoint (ex. `VR_T1`) |

> Note : tous les groupes n'ont pas réalisé les deux modalités ni les deux scénarios. Certains groupes ont une seule observation.

### Scénarios de tâche BIM

Les tâches BIM impliquent du **travail collaboratif dans un modèle Revit** :

- **S1 (BSI_A1)** : placement et validation d'éléments architecturaux dans un bâtiment résidentiel simple (AccessPanels, tableau blanc, projecteur). Correction de référence dans `correction_s1.csv` — liste des éléments attendus avec positions 3D (X, Y, Z en mètres) dans le modèle Revit.
- **S2 (FISA_A5)** : placement de réservations (passages dans murs et sols) dans un bâtiment industriel. Éléments : RESERVATION SOL CIRCULAIRE, RESERVATION MUR CIRCULAIRE, RESERVATION MUR RECTANGULAIRE, etc. Correction de référence dans `correction_s2.csv` avec positions et dimensions (SizeX/Y/Z en mètres).

### Mesures de performance — 4 sous-scores

| Métrique | Code | Description |
|---|---|---|
| Conformité aux consignes | `M1_consignes_%` | % de règles de placement respectées |
| Exhaustivité | `M2_nombre_%` | % d'éléments placés parmi les attendus |
| Précision spatiale | `M3_precision_%` | % d'éléments placés dans la tolérance (0.8 m) |
| Temps | `M4_temps_%` | Proportion du temps imparti utilisée (pénalité si dépassement) |
| **Score global** | `Score_perf_tsk` | Moyenne pondérée M1–M4 |

**Note analytique v2** :
- les scripts amont continuent de calculer M1–M4
- le rapport `v2` et les analyses de performance utilisent prioritairement `Score_perf_tsk` et `Score_perf_tsk_z`

**Distribution des scores (n=19 obs avec données complètes) :**
- Moyenne globale : 46.4 ± 17.5
- PC (n=7) : M=58.4 ± 4.9
- VR (n=12) : M=39.4 ± 18.6
- Effet significatif modalité : F(1,14)=6.12, p=0.027, η²p=0.30

---

## 3. Données brutes — `data_e2/`

### Structure de l'arborescence

```
data_e2/
├── correction_s1.csv              # Référence Revit S1 (18 éléments : positions 3D)
├── correction_s2.csv              # Référence Revit S2 (16+ réservations : positions + dimensions)
├── PC_duration.csv                # Durées des sessions PC par groupe
├── results-survey.xlsx / .csv     # Export LimeSurvey — questionnaire complet
├── README.txt                     # Notes sur les données
│
├── data_TCI/                      # Données TCI brutes par groupe (ZIP)
│   ├── bim002/ … bim113/          # Un dossier par groupe
│   │   ├── Event_Log_Session_*.zip      # Journal d'événements de la session TCI
│   │   ├── TaskScore_*.zip              # Scores par tâche TCI
│   │   └── TaskSnapshot_*.zip           # Snapshots d'état des tâches TCI
│   └── unok/                      # Sessions invalides / exclues
│
├── T1_BSI_A1/                     # Session T1, scénario BSI_A1
│   ├── PC/                        # Modalité PC
│   │   ├── S1/ … S2/              # Par scénario
│   │   │   └── bim067/            # Par groupe
│   │   │       ├── bim067.csv     # Export Revit (positions des éléments placés)
│   │   │       ├── bim067.rte     # Fichier Revit natif
│   │   │       ├── calculateur.mp4         # Enregistrement vidéo rôle calculateur
│   │   │       ├── lecteur.mp4             # Enregistrement vidéo rôle lecteur
│   │   │       ├── modelisateur_c.mp4      # Vidéo modelisateur (cam centrale)
│   │   │       ├── modelisateur_e.mp4      # Vidéo modelisateur (cam externe)
│   │   │       ├── processed_openface/     # Données OpenFace extraites
│   │   │       └── raw/                    # Données brutes (audio, gaze, etc.)
│   └── VR/                        # Modalité VR (même structure)
│
├── T1_FISA_A3/ T1_FISA_A5/        # Session T1, scénarios FISA_A3 et FISA_A5
├── T2_BSI_A1/ T2_FISA_A5/         # Session T2 (mêmes scénarios)
```

### Fichier `results-survey.xlsx` — Questionnaire LimeSurvey

**50 colonnes** incluant les métadonnées de session et 6 dimensions psychométriques :

| Dimension | Code | Nb items | α Cronbach | Source |
|---|---|---|---|---|
| Coordination | COR | 5 | 0.784 | Lewis (2003) — TMS |
| Credibilité | CRE | 5 | 0.686 (surveillée via pruning exploratoire et recalcul `v2`) | Lewis (2003) — TMS |
| Spécialisation | SPE | 5 | 0.743 | Lewis (2003) — TMS |
| Cohésion sociale | SOC | 10 | 0.931 | Sassier-Roublin et al. (2025) |
| Cohésion de tâche | TSK | 3 | 0.859 | Sassier-Roublin et al. (2025) |
| Communication | COM | 3 | 0.918 | Sassier-Roublin et al. (2025) |

**Variables de contexte** dans le questionnaire :
- `Session`, `Modalité`, `Scénario`, `Groupe` — identifiants
- `G3Q00001` : genre
- `G3Q00002` : âge
- `G3Q00003` : niveau de familiarité avec la VR
- `G3Q00004/05` : connaissance préalable des membres de l'équipe
- `G3Q00006/07` : commentaires libres (tâche + groupe)
- `G3Q00008` : rôle tenu pendant l'activité VR

Ces questions G3 sont désormais exploitées par le pipeline Python pour produire des sorties dédiées de profil participant et de commentaires libres, ensuite reprises au début du rapport `v2`.

**Échelle de réponse** : Likert 1-7 (accord)

**Pruning exploratoire CRE** :
- override analytique courant : seule la suppression de `CRE04` est conservée dans la trace exploratoire globale
- dans le rapport `v2`, les analyses questionnaire sont ensuite recalculées après exclusion des groupes à ICC2k `poor`
- selon le sous-échantillon retenu, `CRE` peut alors être conservée sans retrait supplémentaire dans les analyses finales

### Données TCI (`data_TCI/`)

**Groupes avec données TCI** (18 groupes) :
`bim002`, `bim006`, `bim007`, `bim010`, `bim011`, `bim015`, `bim024`, `bim025`, `bim035`, `bim036`, `bim044`, `bim053`, `bim056`, `bim057`, `bim060`, `bim066`, `bim067`, `bim068`, `bim073`, `bim075`, `bim081`, `bim113`

**Anomalies connues** (fichier `data_TCI/README.txt`) :
| Groupe | Problème |
|---|---|
| bim001 | 1/3 RMET seulement, peu de scores totaux |
| bim006 | Tâche en double (score = 0 pour le doublon) |
| bim009 | 1/3 RMET |
| bim018 | Data NOK (pas de score) + double session |
| bim026 | 1/3 RMET + pas de score |
| bim032 | 1/3 RMET + pas de score |
| bim036 | Tâche en double (score = 0) |
| bim056 | 2/3 RMET |
| bim062 | 1/3 RMET |

**Tâches cognitives TCI** :
| Tâche | Description |
|---|---|
| Brainstrorming_equation | Équations de brainstorming |
| Brainstrorming_word_P_N | Catégories de mots positifs/négatifs |
| MatrixSolvingN1_FR | Matrices de Raven (résolution) |
| MemoryGrid1/2/3_FR | Grilles mémoire (3 niveaux) |
| Sudoku_FR | Sudoku |
| TypingText_FR | Frappe de texte (vitesse/précision) |

**Format des ZIP** :
- `Event_Log_Session_*.zip` : journal horodaté de tous les événements (démarrages, réponses, scores intermédiaires)
- `TaskScore_*.zip` : score final par tâche
- `TaskSnapshot_*.zip` : état instantané de toutes les tâches à un instant t

### Fichier `PC_duration.csv`

Durées de sessions en modalité PC — utilisé pour normaliser les métriques audio (par ex. `backchannel_rate_per_min = n_backchannels / duration_min`).

---

## 4. Structure du dossier `Longitudinale/`

```
Longitudinale/
├── PROJECT_DOCUMENTATION.md    # Ce fichier
├── scripts/                    # Tous les scripts d'analyse
│   ├── run_inv.py              # Point d'entrée pipeline INV
│   ├── run_performance.py      # Point d'entrée pipeline performance
│   ├── README.md               # Guide d'utilisation des scripts
│   │
│   ├── analyse_inv/            # Pipeline INV (multimodal)
│   │   ├── analyze_inv_structure.py   # PCA + clustering + redondance + pruning INV
│   │   ├── speech/            # Analyse audio
│   │   │   ├── analyze_audio.py
│   │   │   ├── compute_audio_features.py
│   │   │   └── viewer.py
│   │   ├── gaze/              # Analyse regard
│   │   │   └── analyze_gaze.py
│   │   ├── face/              # Analyse faciale (OpenFace)
│   │   │   ├── analyze_aus_group.py
│   │   │   └── openface/      # Prétraitement vidéos OpenFace
│   │   │       ├── preprocess_pc_videos.py
│   │   │       └── run_openface_and_export_facs.py
│   │   └── hlf/               # High-Level Features multimodaux
│   │       └── compute_high_level_features.py
│   │
│   ├── analyse_performance/    # Modèles statistiques sur la performance
│   │   └── analyze_performance_effects.py
│   │
│   ├── analyse_questionnaire/  # Fiabilité + scoring du questionnaire
│   │   ├── main.py
│   │   └── README.md
│   │
│   ├── analyse_TCI/            # Analyse TCI (c_score, RME, Riedl)
│   │   ├── TCI.py
│   │   └── compute_team_indicators_rields.py
│   │
│   ├── evaluation_performance_scenario/  # Scoring des tâches BIM
│   │   ├── score_groupes.py
│   │   ├── README.md
│   │   ├── S1/                # Scénario 1 (tables)
│   │   │   ├── performance_eval_unified.py
│   │   │   ├── PC/eval_consignes_align.py
│   │   │   └── VR/eval_consignes_align.py
│   │   ├── S2/                # Scénario 2 (réservations)
│   │   │   ├── PC/eval_consignes_s2.py + performance_eval.py
│   │   │   └── VR/eval_consignes_s2.py + performance_eval.py
│   │   ├── corrections/       # Fichiers de référence (corrections)
│   │   │   ├── PC_s1.csv, PC_s2.csv
│   │   │   ├── VR_S1.xlsx, VR_S2.csv
│   │   └── script_revit/      # Scripts Dynamo pour export Revit
│   │       ├── export.S1.dyn
│   │       └── export.S2.dyn
│   │
│   ├── rapport/                # Génération rapports PDF multimodaux
│   │   ├── main.py             # Point d'entrée principal
│   │   └── py/                 # Sous-modules internes (data, corr, inv, network, etc.)
│   │
│   ├── common/                 # Fonctions partagées
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── io_utils.py
│   │   ├── metadata.py
│   │   ├── stats.py
│   │   └── temporal.py
│   │
│   ├── visualisation_sociale/  # Sociogramme multimodal (inspiré MIRAGE)
│   │   └── mirage_sociogram.py # Sociogramme offline : parole, regard, JVA, synchronie
│   │
│   └── utilitaires/            # Outils divers
│       ├── adapter_revit_to_pc.py
│       ├── cmd mkv to mp4.txt
│       ├── compute_lecteur_durations.py
│       └── merge_csv_by_suffix.py
│
└── results/                    # Tous les résultats produits
    ├── visualisation_sociale/      # Sociogrammes MIRAGE par groupe
    │   └── bimXXX/                 # Un dossier par run
    │       ├── mirage_snapshot.png
    │       ├── window_summary.csv
    │       ├── node_metrics.csv
    │       ├── edge_metrics.csv
    │       ├── run_metadata.json
    │       ├── frames/             # Frames PNG (optionnel)
    │       └── mirage_animation.gif # Animation GIF (optionnel)
    ├── merged_dataset/         # Exports fusionnés produits par le rapport
    │   ├── with_pruning/       # Export aligné sur --inv-analysis-mode pruning
    │   └── without_pruning/    # Export aligné sur --inv-analysis-mode no-pruning
    ├── performance_task/       # Sorties performance brutes + agrégées
    │   ├── recap_scores_all.csv          # Scores de performance agrégés
    │   ├── performance_PC/               # Scoring brut PC
    │   └── performance_VR/               # Scoring brut VR
    ├── INV/                    # Outputs INV bruts et HLF
    ├── results_inv_structure/  # PCA + redondance + pruning INV
    │   ├── inv_pruned_features.csv       # Résultat du hard pruning
    │   ├── inv_correlation_matrix_pruned.csv  # Matrice corr après pruning
    │   ├── inv_cluster_representatives.csv    # Features représentatives par cluster
    │   └── inv_final_selected_features.csv    # Liste finale des features sélectionnées
    ├── indices_collab/         # Indices de collaboration
    ├── analyse_performance/    # ANOVA, modèles, plots
    ├── questionnaire/          # Scores, fiabilité, plots
    ├── TCI/                    # C-scores, RME, corrélations
    ├── rapport_ci/             # Rapports PDF finaux
```

---

## 5. Pipeline de traitement — vue d'ensemble

```
data_e2/
    │
    ├──[1] score_groupes.py ──────────────────────────────► performance_task/recap_scores_all.csv
    │       (évaluation performance BIM : M1-M4, Score_perf_tsk)
    │
    ├──[2] run_inv.py ──────────────────────────────────────────────────────────┐
    │       ├── analyze_audio.py ──────────────► speech_metrics.csv            │
    │       ├── compute_audio_features.py ──────► audio_features.csv           │
    │       ├── analyze_gaze.py ────────────────► gaze/ (ALL_metrics_*.csv)    │
    │       ├── analyze_aus_group.py ────────────► face_emotion_metrics_all.csv │
    │       └── compute_high_level_features.py ──► high_level_features.csv     │
    │                                              high_level_features_audit.csv│
    │                                                                           │
    ├──[3] TCI.py ──────────────────────────────► c_scores.csv                 │
    │       + compute_team_indicators_rields.py ─► (effort, skill, strategy)   │
    │                                                                           │
    ├──[4] main.py (questionnaire) ─────────────► questionnaire_scores.csv     │
    │                                              cronbach_alpha.csv           │
    │                                                                           │
    └──[5] run_performance.py ──────────────────► merged_dataset_complete.csv ─┘
            (fusionne tous les résultats ci-dessus)
                │
                ▼
    [6] analyze_inv_structure.py ───────────────► PCA INV, clustering, redondance
    [7] analyze_performance_effects.py ─────────► ANOVA, modèles factoriels
    [8] rapport/v2/main.py ─────────────────────► rapports PDF publication (PC_VR / VR_only)
```

---

## 6. Scripts — description détaillée

### 6.1 `run_inv.py` — Point d'entrée pipeline INV

**Rôle** : orchestre l'ensemble du pipeline INV de façon modulaire.  
**Entrée** : `--data-dir data_e2/` + options (`--inv speech gaze face all`)  
**Sorties** : tous les CSV dans `results/INV/`  

**Logique** :
1. Lance `analyze_audio.py` → `speech_metrics.csv`
2. Lance `compute_audio_features.py` → `audio_features.csv` (superset enrichi)
3. Lance `analyze_gaze.py` → dossier `gaze/`
4. Lance `analyze_aus_group.py` → `face_emotion_metrics_all.csv`
5. Lance `compute_high_level_features.py` avec `audio_features.csv` (préféré à `speech_metrics.csv` car superset)

**Arguments clés** :
- `--hlf-only` : recalculer uniquement les HLF sans relancer les analyses modales
- `--speech-csv` : chemin vers un CSV audio externe
- `--inv speech gaze face all` : sélection des modalités à traiter

---

### 6.2 `analyse_inv/speech/analyze_audio.py`

**Rôle** : analyse des métriques de parole à partir des enregistrements audio ou transcriptions.  
**Entrée** : dossiers audio par groupe dans `data_e2/`  
**Sortie** : `speech_metrics.csv` (23 obs × 48 colonnes)

**Métriques produites** :
| Famille | Variables | Description |
|---|---|---|
| Global | `duration_s`, `overlap_s`, `overlap_ratio` | Durée totale, chevauchements |
| Global | `pause_time_s`, `pause_ratio`, `mean_pause_s` | Pauses (durée totale, ratio, moyenne) |
| Par rôle | `speech_{role}_s`, `speech_{role}_ratio` | Temps de parole par rôle |
| Par rôle | `turns_{role}_n`, `backchannels_{role}_n` | Tours de parole et backchannels |
| Par rôle | `mean_turn_{role}_s` | Durée moyenne des tours par rôle |
| Global | `mean_turn_s`, `total_turns` | Métriques globales |
| Interactions | `int_{role1}→{role2}` | Interruptions entre paires de rôles |
| Synchronie | `sync_{role1}_{role2}` | Synchronie vocale entre paires |

---

### 6.3 `analyse_inv/speech/compute_audio_features.py`

**Rôle** : calcule des métriques audio agrégées et enrichies depuis `speech_metrics.csv`.  
**Sortie** : `audio_features.csv` (23 obs × 79 colonnes après régénération avec le pipeline courant)

**Variables supplémentaires vs speech_metrics.csv** :
- `total_turns`, `max_speech_ratio`, `turn_balance_cv`, `speech_balance_cv`
- `participation_entropy` — entropie de Shannon sur la distribution de parole
- `interruptions_total`, `interruptions_rate_per_min`
- `n_attempted_interruptions`, `n_successful_interruptions`, `successful_interruption_ratio`
- `tms_cred_*`, `tms_coord_*`, `tms_spec_*` — indices TMS dérivés de l'audio
- `z_*` — versions z-score de toutes les métriques clés

---

### 6.4 `analyse_inv/gaze/analyze_gaze.py`

**Rôle** : analyse des données de regard (eye-tracking, VR uniquement).  
**Entrée** : fichiers de tracking gaze par session dans `data_e2/`  
**Sortie** : dossier `results/INV/gaze/` (par groupe) + `ALL_metrics_*.csv`

**Note importante** : les métriques gaze ne sont disponibles **que pour la modalité VR** (le dispositif de tracking de regard n'est pas disponible en PC). Environ 42% des observations globales (celles en PC) ont des NaN sur les métriques gaze.

**Métriques produites** :
| Variable | Description |
|---|---|
| `shared_obj_ratio` | Ratio de temps de regard simultané sur le même objet |
| `shared_obj_n_episodes` | Nombre d'épisodes de regard partagé |
| `mutual_gaze_ratio_mean_pairs` | Ratio de regard mutuel entre paires |
| `gaze_mutual_gaze_ratio` | Ratio global regard mutuel |
| `gaze_shared_visual_attention_ratio` | Attention visuelle partagée |
| `gaze_entropy` | Entropie de la distribution des points de regard |
| `gaze_entropy_mean_participants` | Entropie moyenne par participant |
| `gaze_to_speaker_ratio` | Ratio regard vers le locuteur actif |
| `gaze_focus_proxy` | Proxy de focalisation du groupe |

**Fichiers par groupe** :
- `ALL_metrics_overall.csv` : métriques agrégées par session
- `ALL_metrics_pairs.csv` : métriques par paire de participants
- `ALL_metrics_participants.csv` : métriques par participant

---

### 6.5 `analyse_inv/face/analyze_aus_group.py`

**Rôle** : analyse des Action Units (AU) faciales depuis les données OpenFace.  
**Entrée** : dossiers `processed_openface/` dans `data_e2/`  
**Sortie** : `face_emotion_metrics_all.csv`

**Métriques produites** — toutes issues du tracking facial (modalité « face ») :

| Sous-catégorie | Variables | Description |
|---|---|---|
| Joie | `joy_tri_rate_per_min`, `joy_tri_occupancy` | Taux et occupation des épisodes de joie en triade |
| Joie | `joy_sync_jaccard_mean`, `joy_sync_pearson_mean` | Synchronie joy entre membres |
| Joie | `joy_mean_mean`, `joy_active_pct_mean` | Intensité et activité joy |
| Tristesse | `sad_tri_rate_per_min`, `sad_tri_occupancy` | Taux et occupation épisodes tristesse |
| Tristesse | `sad_sync_jaccard_mean`, `sad_sync_pearson_mean` | Synchronie sad |
| Affect global | `affect_alignment_idx` | Index d'alignement affectif global |
| Affect global | `affect_balance_occ`, `affect_balance_rate` | Équilibre joie/tristesse |
| Affect global | `affect_sync_jaccard_contrast`, `affect_sync_pearson_contrast` | Contraste de synchronie |
| Affect global | `pos_neg_occ_ratio`, `pos_neg_rate_ratio` | Ratio positif/négatif |
| Face groupe | `face_smile_ratio` | Proportion de sourires (AU6+AU12 actifs) |
| Face groupe | `face_negative_affect_ratio` | Ratio affect négatif |
| Face groupe | `face_facial_synchrony` | Synchronie faciale globale (corrélation croisée) |
| Face groupe | `face_sync_pearson_global_idx` | Index de synchronie Pearson global |

---

### 6.6 `analyse_inv/hlf/compute_high_level_features.py`

**Rôle** : calcule les **features de haut niveau (HLF)** multimodales en intégrant toutes les sources INV.  
**Entrées** : `audio_features.csv`, fichiers gaze, `face_emotion_metrics_all.csv`  
**Sorties** :
- `high_level_features.csv` — dataset principal HLF
- `high_level_features_audit.csv` — version étendue avec colonnes `_source` (traçabilité)
- `high_level_features_missingness.csv` — rapport de données manquantes

**Colonnes canoniques produites** (préfixe `audio_`) :
| Colonne | Source d'origine | Description |
|---|---|---|
| `audio_total_speaking_turns` | `total_turns` (audio_features.csv) | Nombre total de tours de parole |
| `audio_avg_speaking_turn_duration_s` | `mean_turn_s` | Durée moyenne d'un tour |
| `audio_floor_exchange_pause_mean_s` | `mean_pause_s` (audio_features.csv) | Durée moyenne des pauses |
| `audio_overlap_speaking_ratio` | `overlap_ratio` | Ratio de chevauchement vocal |
| `audio_backchannel_rate_per_min` | Calculé depuis `backchannels_{role}_n` / `duration_s` | Taux de backchannels |
| `audio_successful_interruption_ratio` | `successful_interruption_ratio` (audio_features.csv) | Ratio `n_successful_interruptions / n_attempted_interruptions` calculé en amont sur les interruptions overlap-based |

**Mécanisme de fallback** : pour chaque colonne canonique, `first_valid_series()` essaie plusieurs noms de colonnes source par ordre de priorité. Si aucune n'est disponible, la colonne est NaN.

**Colonnes `_source`** (traçabilité) : indiquent quel champ source a été utilisé pour chaque colonne canonique.

---

### 6.7 `analyse_inv/analyze_inv_structure.py`

**Rôle** : analyse exploratoire de la structure factorielle des INV — PCA, clustering, redondance.  
**Entrée** : `high_level_features_audit.csv` (19 obs × 136 colonnes après les enrichissements diagnostics actuels)  
**Sortie** : dossier `results/results_inv_structure/`

#### Paramètres importants

```python
MAX_MISSING_RATIO = 0.20        # Seuil NaN pour inclusion d'une feature
REDUNDANCY_CORR_THRESHOLD = 0.80  # Seuil corrélation pairwise pour redondance
BOOTSTRAP_N_ITER = 300          # Iterations bootstrap pour stabilité
BOOTSTRAP_RANDOM_SEED = 42
```

#### Variables exclues par défaut (`ID_COLS`)

```python
{"group_id", "group_base_id", "condition", "scenario", "timepoint",
 "interaction_dur_s_ref", "interaction_dur_s",
 "n_missing_speech_core", "n_missing_gaze_core", "n_missing_face_core",
 "duration_s"}   # ← durée brute, pas un INV
```

#### Familles INV (`MODALITY_PREFIX`)

| Famille | Préfixes identifiants | Remarque |
|---|---|---|
| `audio` | `audio_`, `turn_`, `overlap_`, `pause_`, `speaking_`, `mean_turn`, `tms_`, `specialization_` | Comptage variable selon scope et filtrage |
| `face` | `face_`, `smile_`, `negative_affect_`, `joy_`, `sad_`, `affect_`, `pos_neg_` | Comptage variable selon scope et exclusions |
| `gaze` | `gaze_`, `mutual_gaze`, `shared_obj` | VR uniquement ; disponibilité dépendante du tracking |

**Note face** : la famille « face » regroupe les métriques AU brutes (smile_ratio, negative_affect_ratio, facial_synchrony) **et** les émotions dérivées (joy_*, sad_*, affect_*, pos_neg_*). Source commune : tracking facial OpenFace → Action Units → émotions.

**Note gaze** : les métriques gaze ont 42% de NaN sur l'ensemble PC+VR (threshold global 20%). Elles sont exclues de la PCA globale mais injectées séparément dans `pca_by_inv_modality()` avec filtrage des lignes all-NaN (groupes PC sans tracking).

#### Étapes du pipeline d'analyse

1. **Chargement + sélection de features** (seuil NaN ≤ 20%)
2. **Nettoyage** : suppression features constantes, exclusion `_source` et `z_`
3. **Matrice de corrélation** → `inv_correlation_matrix.csv`, `corr_matrix_inv.png`
3b. **Hard pruning** : suppression des features redondantes (|r| > 0.80) → `inv_pruned_features.csv`, `inv_correlation_matrix_pruned.csv`
4. **PCA globale** (features prunées)
5. **Extraction dimensions** : critère Kaiser (λ>1), variance cumulée ≥70%
6. **Clustering hiérarchique** des features (Ward, distance euclidienne)
6b. **Sélection de représentants** : 1 feature par cluster (corrélation avec performance) → `inv_cluster_representatives.csv`, `inv_final_selected_features.csv`
7. **Projection PCA** par condition (`pca_projection_groups.png`)
8. **PCA par modalité PC vs VR** (`pca_loadings_PC.csv`, `pca_loadings_VR.csv`)
9. **PCA par famille INV** (audio/face/gaze) avec redondance pairwise et bootstrap

#### Résumé PCA globale

Les comptages de features et les composantes retenues ne doivent plus être considérés comme figés dans cette documentation :
- le nombre exact de features utilisées dépend du **scope** (`all` vs `VR_only`), du **mode** (`with_pruning` vs `without_pruning`) et des **exclusions analytiques** actives
- pour un run donné, la référence canonique est :
  - `results/results_inv_structure/*/inv_features_used.csv`
  - `results/results_inv_structure/*/analysis_summary.csv`
  - et, pour le rapport `v2`, les fichiers `rapport_PCA_*.md`

**Exemple courant** :
- `rapport_v2`, `VR_only`, mode `pruning` : **25 features effectivement utilisées** en PCA
- ces 25 features prunées servent aussi de base à la régression stepwise `v2`

---

### 6.8 `evaluation_performance_scenario/score_groupes.py`

**Rôle** : calcule les 4 métriques de performance (M1–M4) et le score composite pour chaque groupe/scénario.  
**Entrée** : exports Revit (`.csv` par groupe) + `correction_s1/s2.csv` (références)  
**Sortie** : `performance_task/recap_scores_all.csv` (21 obs × 18 colonnes)

**Algorithme de scoring M3 (précision)** :
- Pour chaque élément placé, calculer la distance euclidienne 3D à l'élément de référence le plus proche
- Tolérance : 0.8 m
- `M3 = (n_éléments dans tolérance / n_éléments attendus) × 100`

**Algorithme de scoring M2 (exhaustivité)** :
- `M2 = min(n_placés / n_attendus, 1.0) × 100`
- Pénalité si n_placés > n_attendus (éléments surnuméraires)

---

### 6.9 `run_performance.py`

**Rôle** : calcule les scores de performance (M1–M4 + score global), sans fusion multimodale.  
**Entrée** : dossiers `performance_task/performance_VR/` et `performance_task/performance_PC/`  
**Sortie** : `performance_task/recap_scores_all.csv`

**Note** : le dataset fusionné complet est désormais exporté par
`rapport/main.py` dans `results/merged_dataset/{with_pruning,without_pruning}/`
avec des fichiers compagnons décrivant le schéma, la missingness et les sources chargées.

---

### 6.10 `analyse_TCI/TCI.py`

**Rôle** : extrait le **c_score** (facteur c d'intelligence collective) et les scores **RME** (Reading the Mind in the Eyes) depuis les données TCI brutes.  
**Entrée** : ZIPs `data_TCI/bim*/`  
**Sortie** : `results/TCI/c_scores.csv`

**C-score (facteur c)** :
- Premier facteur commun d'une ACP sur les performances aux tâches cognitives
- Analogue au facteur g d'intelligence individuelle mais pour les groupes
- Capte la variance partagée entre toutes les tâches cognitives collaboratives

**RME (Reading the Mind in the Eyes)** :
- Mesure de la capacité à détecter les émotions d'autrui par le regard
- Calculé par membre puis agrégé (mean, max, min)
- `rme_mean`, `rme_max`, `rme_min` — stats de groupe

**Résultats TCI (13 groupes)** :
- c_score : distribution centrée sur 0 (z-score ACP)
- rme_mean : M~15–20 (score sur 36)

---

### 6.11 `analyse_TCI/compute_team_indicators_rields.py`

**Rôle** : calcule les indicateurs de collaboration inspirés de Riedl & Woolley (Hackman model).  
**Entrée** : logs d'événements TCI + données de rôles  

**Indicateurs Riedl/Hackman calculés** :

| Variable | Description | Formule |
|---|---|---|
| `effort_task_sum` | Effort total investi dans les tâches | Σ temps actif sur tâches |
| `effort_task_mean` | Effort moyen par membre | effort_task_sum / 3 |
| `effort_task_cv` | Coefficient de variation de l'effort | CV(efforts membres) |
| `effort_task_gini` | Inégalité de distribution de l'effort | Indice de Gini |
| `strategy_ratio_mean` | Proportion de tâches avec stratégie définie | n_stratégie / n_tâches |
| `strategy_norm` | Version normalisée | `strategy_ratio_mean / max_sample(strategy_ratio_mean)` (normalisation par maximum observé, pas un z-score) |
| `skill_mean` | Compétence moyenne du groupe | Moyenne des scores de compétence |
| `skill_max` | Compétence maximale dans le groupe | Max des scores |
| `skill_congruence_mean` | Congruence entre compétences et rôles | Corrélation compétence-assignation |
| `contribution_mean` | Contribution moyenne | — |
| `contribution_cv` | Disparité des contributions | CV |
| `n_members_with_skill` | Nb membres compétents pour la tâche | Comptage |

---

### 6.12 `analyse_questionnaire/main.py`

**Rôle** : calcul des scores de questionnaire TMS/cohésion, fiabilité psychométrique, analyse exploratoire.  
**Entrée** : `results-survey.xlsx`  
**Sorties** : dossier `results/questionnaire/`

**Pipeline** :
1. Chargement + nettoyage (encodage, recodage items inversés)
2. Calcul des scores par dimension (moyenne des items)
3. **Calcul alpha de Cronbach** par dimension
4. **Calcul r.drop** (corrélation item-total corrigée = corr(item, score_sans_item))
5. **Alpha si item supprimé** pour chaque item
6. **Suppression exploratoire itérative** : retire l'item avec le plus faible r.drop si α augmente ; s'arrête si aucune amélioration
7. Exports CSV détaillés + résumé console

**Fichiers produits** :
- `cronbach_alpha_questionnaire.csv` — α par dimension
- `stats_items_questionnaire.csv` — r.drop, α-if-deleted par item
- `items_signales.csv` — items dont r.drop < 0.3
- `exploratory_report.txt` — rapport textuel
- `exploratory_summary.csv` — tableau alpha_initial / alpha_optimisé / items_retirés
- `exploratory_trace_items.csv` — trace complète des suppressions
- `scores_dimension_par_participant.csv` — scores par participant
- `participant_profile_responses.csv` — réponses G3 nettoyées (genre, âge, familiarité VR, familiarité équipe)
- `participant_profile_summary.csv` — résumé global du profil participant
- `participant_profile_category_counts.csv` — comptages / pourcentages par catégorie G3
- `free_comments_long.csv` — commentaires libres non vides
- `free_comments_theme_summary.csv` — synthèse heuristique des thèmes de commentaires

---

### 6.13 `analyse_performance/analyze_performance_effects.py`

**Rôle** : modèles statistiques sur la variable dépendante `Score_perf_tsk`.  
**Entrée** : `results/merged_dataset/{with_pruning,without_pruning}/merged_dataset_complete_all.csv`  
**Sorties** : dossier `results/analyse_performance/`

**Modèles testés** :
- descriptifs par modalité et par scénario
- modèle modalité seule et modèle scénario seul
- modèle factoriel (modalité × scénario)
- modèle ANCOVA additif avec `scenario` comme covariable quand l'objectif est d'estimer l'effet ajusté de la modalité

**État v2** :
- l'ANCOVA additif `Score_perf_tsk ~ modalite + scenario` est l'analyse de référence pour l'effet modalité
- le scénario est maintenu comme covariable, mais retiré des figures comparatives standards
- la règle d'exclusion a été réalignée pour **garder `bim065`** et **exclure `bim065_2`**

**Outputs** :
- `performance_modele_factoriel.csv` — ANOVA 2×2
- `performance_modele_modalite.csv` — effet modalité
- `performance_modele_scenario.csv` — effet scénario
- `performance_emm_modalite_ajustee.csv` — moyennes marginales estimées (ANCOVA)
- `performance_descriptifs_par_condition.csv` — M, SD, n par condition
- Plots : `performance_par_modalite.png`, `performance_par_scenario.png`, `performance_par_condition.png`

---

### 6.14 `rapport/v2/main.py`

**Rôle** : génère les rapports publication `v2`. `scripts/rapport/main.py` reste la version historique `v1`.  
**Entrée** : dossier `results/` complet  
**Sorties** :
- `results/rapport_v2/PC_VR/rapport_principal_PC_VR.pdf`
- `results/rapport_v2/VR_only/rapport_principal_VR.pdf`
- `results/rapport_v2/VR_only/rapport_INV_VR.pdf`
- `results/rapport_v2/VR_only/rapport_PCA_VR.pdf`
- bundles `merged_dataset` et fichiers compagnons de documentation

**Source de vérité INV** :
- `scripts/config/inv_features_config.py` centralise les familles, priorités, `core/core_hl` et les flags `report_preferred` / `regression_preferred`

**Architecture actuelle** :
- `scripts/rapport/v2/main.py` = orchestrateur publication
- `scripts/rapport/main.py` = pipeline historique / exploratoire
- `scripts/rapport/v2/py/` = sous-modules thématiques (`data.py`, `questionnaire.py`, `regression.py`, etc.)

**Structure du rapport (sections)** :
1. Profil questionnaire et performance
   - 1.2 Profil des participants questionnaire (`G3Q00001–G3Q00005`)
   - 1.2b Commentaires libres questionnaire (`G3Q00006–G3Q00007`)
   - 1.3 Performance (Score_perf_tsk, M1–M4, z-scores, plots)
   - 1.3 Performance par condition
   - 1.3b Fiabilité du score de performance (M1+M3)
   - 1.3c Analyses de régression — Performance
   - 1.3d Analyse statistique (effets modalité × scénario)
2. Questionnaire (TMS : COR, CRE, SPE, SOC, TSK, COM — statistiques descriptives, comparaisons)
   - exclusion explicite des groupes à accord inter-membres `poor` (`ICC2k < 0.50`) pour toutes les analyses questionnaire aval
   - 1.4.5c Analyse approfondie de la fiabilité TMS (alpha if deleted, corrélations inter-dimensions, pistes d'amélioration)
3. TCI (c_score, RME — corrélations, scatter plots)
4. Corrélations globales (heatmap Riedl × TCI × questionnaire × INV)
5. **Section 2.5 Régressions TMS/Cohésion → Outcomes** : graphiques de régression bivariée TMS/Cohésion → Performance et C-factor
6. **Réseau de corrélations globales** (`global_correlation_network.png`) — graphe networkx des associations fortes (|rho| ≥ 0.55, p ≤ 0.05)
7. **Diagramme théorique CI** (modèle Riedl-Woolley-Hackman) avec pondération empirique
8. INV — indicateurs audio, face, gaze
9. Analyse longitudinale (T1 vs T2)

**Nouvelles fonctionnalités (v8)** :
- Réorganisation des sections 1.3b/1.3c/1.3d (fiabilité et régression après graphique de composition)
- Graphiques de régression TMS/Cohésion → Performance et C-factor (section 2.5)
- Analyse approfondie de la fiabilité TMS avec recommandations (section 1.4.5c)
- Correction FDR : arrondissement cohérent des p-valeurs initiales
- Diagnostic du filtrage speech par modalité (messages informatifs si données manquantes)

**Paramètres réseau de corrélations** :
- `rho_threshold = 0.55`
- `p_threshold = 0.05`
- `min_n = 5`

**Intégration du pruning INV** :
- la PCA charge `inv_pruned_features.csv` et `inv_features_used.csv`
- la régression `v2` utilise `inv_pruned_features.csv` comme filtre `kept=1`, puis intersecte cette liste avec `REGRESSION_RETAINED_INV_FEATURES` et réinjecte `REGRESSION_FORCE_INCLUDE`
- les exclusions analytiques `v2` sont centralisées dans `scripts/config/inv_features_config.py` :
  - suppression des `*_median`
  - suppression des `tms_*_idx`
  - questionnaire analysé comme `COR`, `CRE`, `SPE`, `Cohesion_questionnaire_score`
  - performance analysée comme `Score_perf_tsk`, `Score_perf_tsk_z`

**Familles de variables dans le réseau** :
- `performance` : Score_perf_*, M1–M4
- `tci` : c_score, rme_*
- `riedl` : skill_*, effort_*, strategy_*, contribution_*
- `questionnaire` : COR, CRE, SPE, SOC, TSK, COM
- `audio` : audio_*, pause_*, overlap_*
- `gaze` : gaze_*, shared_obj_*, mutual_gaze_*
- `face` : face_*
- `face` : face_*, joy_*, sad_*, affect_*

---

### 6.15 `common/` — Modules partagés

| Module | Contenu |
|---|---|
| `constants.py` | Constantes globales (timepoints, scénarios, rôles) |
| `io_utils.py` | Lecture/écriture CSV, normalisation colonnes, encodage |
| `metadata.py` | Parsing des identifiants groupe/condition/scénario |
| `stats.py` | Fonctions stats (corrélations, alpha, bootstrap) |
| `temporal.py` | Fonctions sur les séries temporelles (alignement, segmentation) |

---

## 7. Résultats produits

### `results/INV/`

| Fichier | Description | Shape |
|---|---|---|
| `speech_metrics.csv` | Métriques audio brutes (diarisation) | 23 × 70 |
| `audio_features.csv` | Métriques audio enrichies (TMS, z-scores, totaux) | 23 × 101 |
| `face_emotion_metrics_all.csv` | Émotions faciales (AU, synchronie) | N × ~100 |
| `high_level_features.csv` | HLF multimodaux intégrés | 19 × 42 |
| `high_level_features_audit.csv` | HLF + colonnes `_source` + z-scores | 19 × 151 |
| `high_level_features_missingness.csv` | Rapport NaN par colonne | — |
| `gaze/bim*/` | Métriques gaze par groupe | — |
| `gaze/ALL_metrics_overall.csv` | Agrégation gaze toutes sessions | — |

### `results/results_inv_structure/`

> **Structure subdivisée** : depuis v9, les outputs sont organisés dans deux sous-dossiers selon le mode d'analyse. Le fichier `ANALYSIS_MODES_README.txt` à la racine du dossier documente automatiquement les deux modes.

```
results/results_inv_structure/
├── ANALYSIS_MODES_README.txt          # Récapitulatif global des deux modes
├── with_pruning/                      # Hard pruning actif (|r| > 0.80) → nombre variable selon scope
│   └── [tous les fichiers ci-dessous]
└── without_pruning/                   # Pas de pruning → nombre variable selon scope
    └── [tous les fichiers ci-dessous]
```

**Fichiers présents dans chaque sous-dossier** :

| Fichier | Description |
|---|---|
| `analysis_summary.csv` | Résumé des paramètres d'analyse (mode, n_features_initial, n_features_final, apply_pruning) |
| `analysis_report.txt` | Rapport textuel complet |
| `inv_features_used.csv` | Liste des features utilisées pour la PCA |
| `inv_correlation_matrix.csv` | Matrice de corrélation complète |
| `inv_pruned_features.csv` | Rapport de pruning (kept/dropped + raisons ; `applied=1` si pruning effectif) |
| `inv_correlation_matrix_pruned.csv` | Matrice de corrélation après pruning (mode `with_pruning/` uniquement) |
| `pca_explained_variance.csv` | Variance expliquée PCA globale |
| `pca_loadings.csv` | Loadings PCA globale |
| `pca_loadings_full_table.csv` | Loadings triés par importance (valeur absolue max) |
| `pca_loadings_sorted_PC*.csv` | Loadings triés par importance pour chaque PC |
| `pca_loadings_heatmap.png` | Heatmap des loadings (triée, mise en évidence \|loading\|>0.5) |
| `pca_scree_plot.png` | Scree plot (critère Kaiser) |
| `pca_projection_groups.png` | Projection des groupes PC1×PC2 (couleurs par scénario) |
| `corr_matrix_inv.png` | Heatmap corrélations |
| `inv_dimensions.csv` | Dimensions retenues + top features |
| `feature_dendrogram.png` | Dendrogramme hiérarchique |
| `pca_loadings_{audio\|face\|gaze}.csv` | Loadings par famille INV |
| `pca_explained_variance_{modality}.csv` | Variance par famille |
| `pca_scree_{modality}.png` | Scree par famille |
| `pca_bootstrap_loading_stability_{modality}.csv` | Stabilité bootstrap des loadings |
| `pca_bootstrap_component_stability_{modality}.csv` | Stabilité cosine des composantes |
| `inv_modality_dimensions.csv` | Dimensions par famille (résumé) |
| `inv_modality_redundancy.csv` | Paires redondantes (\|r\|>0.80) |
| `pca_variance_by_modality.png` | Variance PC vs VR |
| `pca_variance_by_inv_modality.png` | Variance audio/face/gaze |
| `pca_loadings_PC.csv`, `pca_loadings_VR.csv` | Loadings par condition |
| `inv_cluster_representatives.csv` | Feature représentative par cluster |
| `inv_final_selected_features.csv` | Liste finale des features sélectionnées |
| `inv_modality_redundancy_correlations.csv` | Corrélations redondantes par modalité INV |

### `results/TCI/`

| Fichier | Description |
|---|---|
| `c_scores.csv` | c_score + RME par groupe (13 groupes, 5 colonnes) |
| `c_scores_with_tasks.csv` | c_score + scores par tâche cognitive (13 × 13) |
| `c_scores_rme_task_correlations.csv` | Corrélations RME × tâches |
| `rme_vs_c.png` | Scatter RME vs c_score |
| `rme_task_corr_heatmap.png` | Heatmap corrélations RME-tâches |
| `group_profiles.png` | Profils TCI par groupe |

### `results/questionnaire/`

```
questionnaire/
├── global/
│   ├── cronbach_alpha_questionnaire.csv     # α par dimension
│   ├── stats_items_questionnaire.csv        # r.drop, α-if-deleted
│   ├── items_signales.csv                   # Items problématiques
│   ├── exploratory_summary.csv             # Tableau avant/après optimisation
│   ├── exploratory_trace_items.csv         # Trace des suppressions
│   ├── exploratory_report.txt             # Rapport textuel
│   ├── scores_dimension_par_participant.csv
│   ├── desc_dim_questionnaire.csv          # Descriptifs par dimension
│   └── table_moyennes_pivot.csv
├── analyse/
│   ├── descriptifs_scenario_modalite.csv
│   ├── modeles_scenario_modalite_par_dimension.csv
│   ├── plot_means_ic95_scenario_modalite.pdf
│   ├── analyse_perf.pdf
│   ├── stats_par_dimension_role_perf.csv
│   └── plots_dimensions_roles_perf.pdf
└── pruned/                                 # Scores avec items optimisés
```

### `results/analyse_performance/`

| Fichier | Description |
|---|---|
| `performance_modele_factoriel.csv` | ANOVA 2×2 (modalité × scénario) |
| `performance_modele_modalite.csv` | Effet modalité isolé |
| `performance_modele_scenario.csv` | Effet scénario isolé |
| `performance_emm_modalite_ajustee.csv` | Moyennes marginales ANCOVA |
| `performance_descriptifs_par_condition.csv` | M, SD, n par condition |
| `performance_descriptifs_par_modalite.csv` | M, SD par modalité |
| `performance_descriptifs_par_scenario.csv` | M, SD par scénario |
| `rapport_analyse_performance.txt` | Rapport textuel complet |
| `performance_par_condition.png` | Plot scores par condition |

### `results/rapport_ci/`

| Fichier | Description |
|---|---|
| `report_all.pdf` | Rapport complet (tous groupes) |
| `report_all.md` | Source markdown |
| `report_all.html` | Source HTML |
| `figs_all/` | Figures pour rapport global |
| `rapport/` | Sous-dossier de rapports par mode |
| `rapport_no_redon/` | Rapports avec features redondantes filtrées |

### `results/merged_dataset/{with_pruning,without_pruning}/merged_dataset_complete_all.csv`

Dataset fusionné exporté par le rapport, avec un sous-dossier séparé selon
`--inv-analysis-mode` :
- `with_pruning/` : export aligné sur `pruning`
- `without_pruning/` : export aligné sur `no-pruning`

Le dossier contient désormais plusieurs artefacts :
- `merged_dataset_complete_all.csv` — table fusionnée principale
- `merged_dataset_columns_all.csv` — inventaire détaillé des colonnes
- `merged_dataset_blocks_all.csv` — résumé par bloc conceptuel
- `merged_dataset_unit_counts_all.csv` — répartition des lignes par condition/scénario/timepoint
- `merged_dataset_sources_all.csv` — résumé des tables chargées
- `merged_dataset_report_all.md` — note de synthèse sur le contenu exporté

Dataset fusionné — structure conceptuelle principale :

| Famille | Variables |
|---|---|
| Identifiants | `group_id`, `group_base_id`, `timepoint`, `scenario`, `modalite` |
| Performance | `Score_perf_tsk`, `M1_consignes_%`, `M2_nombre_%`, `M3_precision_%`, `M4_temps_%`, `Score_perf_tsk_z` |
| Riedl/Hackman | `effort_*`, `strategy_*`, `skill_*`, `contribution_*`, `n_tasks*` |
| TCI | `c_score`, `rme_mean`, `rme_max`, `rme_min` |
| Questionnaire | `COM`, `COR`, `CRE`, `SOC`, `SPE`, `TSK`, `Cohesion_questionnaire_score` |
| INV audio | `audio_total_speaking_turns`, `audio_avg_speaking_turn_duration_s`, `audio_floor_exchange_pause_mean_s`, `audio_overlap_speaking_ratio`, `audio_successful_interruption_ratio`, `audio_backchannel_rate_per_min` |
| INV vocal brut | `overlap_ratio`, `pause_ratio`, `mean_turn_s` |
| INV face | `face_smile_ratio`, `face_negative_affect_ratio`, `joy_tri_rate_per_min`, `joy_tri_occupancy`, `sad_tri_rate_per_min`, `sad_tri_occupancy` |
| INV gaze | `shared_obj_ratio`, `shared_obj_dur_mean_s`, `gaze_entropy_mean_participants`, `gaze_mutual_gaze_ratio`, `gaze_to_speaker_ratio_final`, `gaze_shared_visual_attention_ratio`, `gaze_entropy` |

**Note analytique v2** :
- le dataset peut contenir plus de colonnes que celles réellement retenues dans les analyses
- les analyses `v2` filtrent ensuite :
  - performance : `Score_perf_tsk`, `Score_perf_tsk_z`
  - questionnaire : `COR`, `CRE`, `SPE`, `Cohesion_questionnaire_score`
  - INV : exclusion des `*_median` et `tms_*_idx`

---

## 8. Cadre théorique

### Modèle de Collective Intelligence (CI)

Le projet s'inscrit dans le cadre du modèle de **CI de Riedl, Woolley & Hackman** :

```
INPUTS                    MEDIATEURS                 OUTPUTS
─────────────────────────────────────────────────────────────────
Team composition     →    TMS (Transactive Memory)  → Performance objective
Context/Environment  →    TAS (Task Attention)      → Résultats subjectifs
Task characteristics →    TRS (Turn Regulation)     → Synergy
C-factor / TCI       →    Coordination
Initial potential    →    Cohesion
                     →    Collective effort
                     →    Specialization
                     →    Credibility
                     →    Strategies
                     →    Utilization of knowledge
```

### Transactive Memory System (TMS) — Lewis (2003)

Trois dimensions mesurées par questionnaire :
- **Coordination** (COR) : efficacité de coordination du travail collectif
- **Crédibilité** (CRE) : confiance dans l'expertise des autres membres
- **Spécialisation** (SPE) : connaissance des domaines de compétence de chacun

### Mapping variables empiriques → composants théoriques

| Variable empirique | Composant théorique |
|---|---|
| `c_score` | C-factor |
| `rme_mean` | TRS (sensibilité sociale) |
| `skill_mean` | Utilization of knowledge |
| `skill_max` | Specialization |
| `strategy_ratio_mean` | Strategies |
| `effort_task_sum` | Collective effort |
| `face_facial_synchrony` | Cohesion |
| `shared_obj_ratio` | Coordination |
| `gaze_shared_visual_attention_ratio` | TAS |
| `audio_avg_speaking_turn_duration_s` | Coordination |
| `CRE` | Credibility |
| `SOC` | Cohesion (sociale) |
| `SPE` | Specialization |
| `TSK` | Coordination (cohésion de tâche) |

---

## 9. Variables clés et définitions

### Score de performance composite

```
Score_perf_tsk = 0.50 × M1 + 0.00 × M2 + 0.50 × M3 + 0.00 × M4
```

> Note : M2 et M4 sont calculés en amont mais non utilisés dans le score composite final (voir `FORMULES_VARIABLES_CLES.md`).

### C-score (facteur c)

- Premier composant principal d'une PCA sur les performances aux tâches TCI cognitives
- Variance commune capturée : reflète une capacité collective générale
- Corrélations attendues avec : skill_max, rme_mean, performance BIM

### Alpha de Cronbach

```
α = (k/(k-1)) × (1 - Σσᵢ² / σ²_total)
```
Interprétation : α < 0.6 = insuffisant ; 0.6–0.7 = acceptable ; > 0.7 = bon ; > 0.9 = excellent

### R.drop (corrélation item-total corrigée)

```
r_drop(i) = corr(item_i, score_total_sans_item_i)
```
Seuil signal : r.drop < 0.3 → item problématique (peu discriminant)

### Participation entropy

```
H = -Σ p_i × log2(p_i)   où p_i = fraction de parole du membre i
```
H max = log2(3) ≈ 1.585 (participation parfaitement équilibrée)

### Gaze joint attention index

Proportion de temps où au moins 2 membres regardent simultanément le même objet virtuel (VR uniquement).

---

## 10. Acquisition des données INV — paramètres techniques

Cette section documente les conditions d'enregistrement et de traitement pour chaque modalité INV.

---

### 10.1 Audio / Parole

#### Enregistrement

| Paramètre | Valeur | Notes |
|---|---|---|
| **Taux d'échantillonnage** | **16 000 Hz** | Utilisé dans `analyze_audio.py` (sr=16000) |
| **Format de sortie** | `.wav` (PCM) | Extrait depuis `.mp4`/`.mkv` via `processed_openface/` |
| **Nb de canaux** | 1 (mono) | Par participant (fichiers séparés par rôle) |
| **Fichiers** | `{role}__{hash}__audio.wav` | Un fichier audio par rôle |

#### Traitement signal

| Paramètre | Valeur | Signification |
|---|---|---|
| `FRAME_LENGTH` | 1024 samples | Fenêtre d'analyse FFT (~64 ms @ 16 kHz) |
| `HOP_LENGTH` | 256 samples | Pas de déplacement (~16 ms @ 16 kHz) |
| `DISP_SR` | 200 Hz | Fréquence de ré-échantillonnage pour affichage |
| Seuil VAD | par défaut dB | Voice Activity Detection : détecte les segments de parole actifs |

#### Ce qui est mesuré

La diarisation (segmentation locuteur) produit pour chaque session :

- **Segments de parole** : onset, offset, durée par rôle (calculateur, modelisateur, lecteur)
- **Chevauchements** (`overlap_s`, `overlap_ratio`) : durée et proportion où ≥2 membres parlent simultanément
- **Pauses** (`pause_time_s`, `pause_ratio`, `mean_pause_s`) : silences entre tours
- **Tours de parole** (`turns_{role}_n`, `mean_turn_{role}_s`) : nombre et durée moyenne par rôle
- **Backchannels** (`backchannels_{role}_n`) : signaux courts d'acquiescement par rôle
- **Interruptions** (`interruptions_total`, `interruptions_rate_per_min`) : chevauchements initiés par un autre locuteur
- **Synchronie vocale pairwise** (`sync_{role1}_{role2}`) : corrélation croisée des signaux d'activité vocale
- **Indicateurs TMS dérivés** (`tms_spec_*`, `tms_coord_*`, `tms_cred_*`) : proxys de spécialisation, coordination et crédibilité calculés depuis la distribution de parole

---

### 10.2 Regard / Gaze (VR uniquement)

#### Dispositif

Les données de gaze sont **uniquement disponibles en modalité VR**. Le casque VR intègre un **eye-tracker binoculaire** qui enregistre la direction du regard dans l'espace 3D virtuel.

> En modalité PC, aucun dispositif de tracking du regard n'est utilisé → toutes les métriques gaze sont NaN pour les groupes PC.

#### Paramètres d'acquisition et d'analyse

| Paramètre | Valeur | Notes |
|---|---|---|
| **Fréquence de ré-échantillonnage (grille)** | **20 Hz** (défaut) | `fs_grid` dans `analyze_gaze.py` — paramétrable via `--fs-grid` |
| **Fenêtre d'analyse glissante** | 30 s | `win=30.0` — taille de fenêtre pour les métriques temporelles |
| **Pas de la fenêtre** | 30 s | `step=30.0` — fenêtres non chevauchantes |
| **Durée minimale d'une fixation** | 200 ms | `min_fix=0.20` — seuil pour considérer un épisode de regard stable |
| **Tolérance chevauchement gaze** | 100 ms | `overlap_min_s` — durée minimale pour comptabiliser un shared gaze |
| **Tolérance lead/lag synchronie** | 0 ms | `sync_lag_ms` — pas de décalage autorisé pour la synchronie |
| **Fenêtre gaze→speech** | 1000 ms (tau) | `tau_ms` — fenêtre pour calculer gaze-to-speaker ratio |

#### Ce qui est mesuré

**Regard partagé sur un objet (shared object gaze)** :
- `shared_obj_ratio` — proportion du temps où ≥2 membres regardent le même objet virtuel simultanément
- `shared_obj_n_episodes` — nombre d'épisodes de regard partagé
- `shared_obj_dur_mean_s` — durée moyenne d'un épisode
- `shared_obj_episode_rate_per_min_ref` — taux d'épisodes par minute (normalisé par durée de référence)

**Regard mutuel (mutual gaze)** — paires de participants :
- `mutual_gaze_ratio_mean_pairs` — ratio moyen de regard mutuel entre paires
- `mutual_gaze_n_episodes_sum_pairs` — nombre total d'épisodes de regard mutuel (somme des paires)
- `mutual_gaze_dur_total_s_sum_pairs` — durée totale de regard mutuel
- `mutual_gaze_episode_rate_per_min_ref` — taux d'épisodes normalisé

**Entropie et distribution du regard** :
- `gaze_entropy` — entropie de Shannon sur la distribution des zones/objets regardés (mesure la dispersion de l'attention)
- `gaze_entropy_mean_participants` — moyenne de l'entropie individuelle des participants
- `gaze_focus_proxy` — proxy de focalisation du groupe (inverse de l'entropie)

**Couplage regard–locuteur** :
- `gaze_to_speaker_ratio` — proportion du temps où les membres regardent le locuteur actif
- `gaze_speaker_coupling_idx` — indice de couplage entre orientation du regard et prise de parole

**Attention visuelle partagée** :
- `gaze_shared_visual_attention_ratio` — ratio d'attention visuelle simultanée sur la même zone
- `gaze_joint_attention_idx_raw` — indice brut d'attention conjointe

**Métriques pairwise** (par paire de participants) :
- `pair_shared_obj_ratio_mean` — ratio moyen de regard sur objet partagé entre paires
- `pair_mutual_gaze_ratio_mean` — ratio moyen de regard mutuel entre paires

---

### 10.3 Visage / Émotions faciales (OpenFace)

#### Enregistrement vidéo

| Paramètre | Valeur | Notes |
|---|---|---|
| **Fréquence d'images** | **30 fps** | Vidéos encodées H.264, résolution 1280×720, format yuv420p |
| **Format brut** | `.mp4` / `.mkv` | Un fichier par rôle (`calculateur.mp4`, `lecteur.mkv`, etc.) |
| **Format prétraité** | `{role}__{hash}__openface.mp4` | Vidéo recadée/normalisée pour OpenFace |

#### Pipeline OpenFace

Les vidéos sont traitées par **OpenFace 2.x** qui produit par frame :

| Sortie OpenFace | Description |
|---|---|
| `{role}__openface.csv` | Toutes les AU intensités + présence + pose tête + landmarks par frame |
| `{role}__openface__FACS.csv` | Sous-ensemble AU FACS filtrées (Action Units sélectionnées) |
| `{role}__openface_of_details.txt` | Métadonnées OpenFace (version, paramètres, taux de succès) |

#### Action Units mesurées et mapping émotionnel

| AU | Nom FACS | Émotion associée |
|---|---|---|
| **AU1** | Inner Brow Raiser | Tristesse / Surprise |
| **AU4** | Brow Lowerer | Tristesse / Colère |
| **AU6** | Cheek Raiser | Joie (composante) |
| **AU12** | Lip Corner Puller | Joie (sourire) |
| **AU15** | Lip Corner Depressor | Tristesse |
| **AU17** | Chin Raiser | Tristesse |

**Composites calculés** (dans `analyze_aus_group.py`) :

```python
joy_intensity  = mean(AU6_intensity, AU12_intensity)   # Joie
sad_intensity  = mean(AU1_intensity, AU4_intensity, AU15_intensity)  # Tristesse
```

#### Seuillage et agrégation

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| **Méthode de seuillage** | quantile 90% | Seuil adaptatif pour binarisation joy/sad actif/inactif |
| Alternatives disponibles | zscore, absolu | Paramétrable dans le script |
| **Fenêtre Pearson** | glissante (sliding_windows) | Pour calcul de synchronie temporelle entre membres |

#### Ce qui est mesuré — métriques groupe

**Activité émotionnelle en triade** :
- `joy_tri_rate_per_min` — fréquence des épisodes de joie simultanés (≥2 membres) par minute
- `joy_tri_occupancy` — proportion du temps avec joie triasique active
- `sad_tri_rate_per_min` — idem pour la tristesse
- `sad_tri_occupancy` — idem

**Synchronie émotionnelle pairwise** :
- `joy_sync_jaccard_mean` — synchronie joy par Jaccard moyen des paires (chevauchement temporel)
- `joy_sync_pearson_mean` — synchronie joy par corrélation de Pearson (signaux continus)
- `joy_active_sync_pearson_mean` — Pearson calculé uniquement sur les frames où au moins un membre est actif
- Idem pour `sad_sync_*`

**Intensité et niveau d'activation** :
- `joy_mean_mean` — intensité moyenne de joy (AU composite) moyennée sur les membres
- `joy_active_pct_mean` — % de frames où joy est actif, moyenné sur les membres
- `joy_mean_median`, `joy_active_pct_median` — versions médianes

**Alignement affectif global** :
- `affect_alignment_idx` — corrélation entre les profils d'affect des membres (mesure à quel point les membres vivent les mêmes variations émotionnelles)
- `affect_balance_occ` — équilibre joy/sad en termes d'occurrences
- `affect_balance_rate` — équilibre joy/sad en termes de taux
- `affect_sync_jaccard_contrast` — contraste de synchronie (joy vs sad, Jaccard)
- `affect_sync_pearson_contrast` — contraste de synchronie (Pearson)
- `pos_neg_occ_ratio` — ratio occurrences positives / négatives
- `pos_neg_rate_ratio` — ratio taux positif / négatif

**Synchronie faciale générale** :
- `face_smile_ratio` — proportion de frames avec sourire actif (AU6 + AU12)
- `face_negative_affect_ratio` — proportion de frames avec affect négatif actif
- `face_facial_synchrony` — synchronie faciale globale (corrélation croisée des signaux AUs)
- `face_sync_pearson_global_idx` — indice global de synchronie faciale Pearson

---

### 10.4 Récapitulatif des modalités et disponibilité

| Modalité | Dispositif | Fréquence | PC | VR |
|---|---|---|---|---|
| **Audio** | Microphone (par participant) | 16 000 Hz | ✅ | ✅ |
| **Visage / OpenFace** | Caméra webcam (30 fps) | 30 fps | ✅ | ✅ |
| **Regard / Gaze** | Eye-tracker intégré casque VR | ~20 Hz (grille) | ❌ | ✅ |

> **Note VR** : en modalité VR, la caméra est frontale (caméra du casque) donc la qualité de l'analyse OpenFace peut être inférieure à la modalité PC (caméra fixe externe). Les vidéos VR ont généralement une résolution et un angle moins optimaux pour la détection AU.

---



## 11. Notes techniques et anomalies connues

### Données manquantes

| Source | Taux NaN | Cause |
|---|---|---|
| Métriques gaze (toutes) | ~42% sur l'ensemble | PC sans tracking gaze |
| `gaze_to_speaker_ratio_final` | 100% | Variable pas encore calculée |
| `gaze_speaker_coupling_idx` | 100% | Idem |
| `audio_successful_interruption_ratio` | 100% dans les anciens exports, disponible après régénération audio | Désormais calculé dans `analyze_audio.py` puis propagé via `compute_audio_features.py` |
| `tms_credibility_idx`, `tms_coordination_idx` | 42% | Calcul TMS-audio dépend de métriques audio manquantes |

### Groupes avec données incomplètes

- **bim002** : score performance = 0 (aucune règle passée en S1/VR)
- Groupes `_2` (ex. `bim066_2`, `bim073_2`) : sessions en doublon (second passage de la même triade)
- **bim009**, **bim032** etc. : 1/3 RMET seulement (TCI incomplet)

### Conventions de nommage

- `group_id` = identifiant de session (ex. `bim066_VR_S1`)
- `group_base_id` = identifiant de triade de base sans modalité/session (ex. `bim066`)
- `timepoint` = T1 (première session) ou T2 (deuxième session)
- `condition` = `PC` ou `VR`
- `scenario` = `S1` ou `S2`

### Bootstrap PCA — Note sklearn ≥ 1.1

Les bootstrap samples avec toutes les valeurs d'une colonne identiques (std=0) provoquaient des erreurs `ValueError: shapes not aligned`. Résolu en utilisant l'imputer fitté sur les données complètes (`transform()` au lieu de `fit_transform()` à chaque itération bootstrap).

### Audio — Hiérarchie des fichiers sources

```
audio_features.csv  (superset, recommandé pour HLF)
    └── contient : total_turns, participation_entropy, interruptions_rate_per_min,
                   tms_*, z_*, backchannels_per_role
speech_metrics.csv  (sous-ensemble, diarisation brute)
    └── contient : overlap_ratio, pause_ratio, mean_turn_s, turns_per_role
```

---

## 12. Commandes d'exécution

```bash
# Aller dans le dossier scripts
cd D:\Analyse_donnee\Longitudinale\scripts

# === Pipeline complet INV (speech + gaze + face + HLF) ===
python run_inv.py --data-dir D:/data_e2 --out-dir D:/Analyse_donnee/Longitudinale/results/INV

# === Pipeline INV — modalité spécifique ===
python run_inv.py --data-dir D:/data_e2 --inv speech --out-dir D:/Analyse_donnee/Longitudinale/results/INV
python run_inv.py --data-dir D:/data_e2 --inv speech gaze --out-dir D:/Analyse_donnee/Longitudinale/results/INV

# === Recalculer uniquement les HLF (sans relancer speech/gaze/face) ===
python run_inv.py --hlf-only \
  --speech-csv D:/Analyse_donnee/Longitudinale/results/INV/audio_features.csv \
  --gaze-group D:/Analyse_donnee/Longitudinale/results/INV/gaze/ALL_metrics_overall.csv \
  --face D:/Analyse_donnee/Longitudinale/results/INV/face_emotion_metrics_all.csv \
  --out-dir D:/Analyse_donnee/Longitudinale/results/INV

# === Scoring performance BIM ===
cd evaluation_performance_scenario
python score_groupes.py

# === Pipeline performance + fusion datasets ===
cd ..
python run_performance.py

# === TCI (c_score, RME, Riedl) ===
cd analyse_TCI
python TCI.py --data D:/data_e2/data_TCI --out D:/Analyse_donnee/Longitudinale/results/TCI
python compute_team_indicators_rields.py

# === Questionnaire (fiabilité + scoring) ===
cd ../analyse_questionnaire
python main.py --data D:/data_e2/results-survey.xlsx \
               --out ../../results/questionnaire --mode all

# === PCA structurelle INV ===
cd ../analyse_inv
python analyze_inv_structure.py \
  --data D:/Analyse_donnee/Longitudinale/results/INV/high_level_features_audit.csv
# Options supplémentaires :
#   --prune-threshold 0.80   # Seuil de corrélation pour le hard pruning (défaut: 0.80)
#   --perf D:/Analyse_donnee/Longitudinale/results/performance_task/recap_scores_all.csv  # Pour sélection par cluster

# === Analyse des effets sur la performance ===
cd ../analyse_performance
python analyze_performance_effects.py \
  --perf D:/Analyse_donnee/Longitudinale/results/performance_task/recap_scores_all.csv \
  --merged D:/Analyse_donnee/Longitudinale/results/merged_dataset/with_pruning/merged_dataset_complete_all.csv \
  --out D:/Analyse_donnee/Longitudinale/results/analyse_performance

# === Génération du rapport PDF (toutes conditions) ===
cd ../rapport
python main.py \
  --results-dir D:/Analyse_donnee/Longitudinale/results \
  --out-dir D:/Analyse_donnee/Longitudinale/results/rapport_ci

# === Modes de rapport spécifiques ===
python main.py --results-dir D:/Analyse_donnee/Longitudinale/results \
  --out-dir D:/Analyse_donnee/Longitudinale/results/rapport_ci --mode pc
python main.py --results-dir D:/Analyse_donnee/Longitudinale/results \
  --out-dir D:/Analyse_donnee/Longitudinale/results/rapport_ci --mode vr
```

---

*Documentation générée automatiquement à partir de l'analyse du code et des données. À mettre à jour lors des changements majeurs de pipeline ou de structure de données.*
