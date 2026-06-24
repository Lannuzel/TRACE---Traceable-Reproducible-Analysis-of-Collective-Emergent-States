# PROJECT_CONTEXT.md

> **CONSIGNE POUR COPILOT / CLAUDE CODE**
> Ce fichier est la **mémoire projet principale** et doit être RELU avant toute modification importante du code.
> Après toute évolution structurelle (nouveaux modes d'analyse, refactoring de pipeline, nouvelle option, changement de logique métier), ce fichier doit être MIS À JOUR.
> Il contient les décisions stables, les conventions, les règles métier et l'historique des choix structurants du projet.

🔄 Synchronisé le 2026-03-27 — Analyse globale de cohérence variables/pipeline/rapport réalisée ; 2 incohérences critiques identifiées, dimensions CSV mises à jour, structure results_inv_structure/ subdivisée documentée.
🔄 Mis à jour le 2026-03-31 — Ajout du module `visualisation_sociale/mirage_sociogram.py` (sociogramme multimodal offline, inspiré MIRAGE). Nettoyage : backups et archives supprimés. PROJECT_CONTEXT.md canonique (doublon D:/ supprimé).
🔄 Mis à jour le 2026-04-03 — Rapport `v2` stabilisé : config INV unifiée dans `scripts/config/inv_features_config.py`, shim `v2` supprimé, régressions stepwise alignées sur les features prunées, alignement des exclusions performance (`bim065` conservé, `bim065_2` exclu). Les comptages PCA ne doivent plus être supposés fixes : lire `inv_features_used.csv` et `analysis_summary.csv`.
🔄 Mis à jour le 2026-04-07 — Logiques de sélection désormais explicitement dissociées : rapport (`core/core_hl` + `report_preferred`), PCA (`priority` via `FEATURE_PRIORITY`), régression `v2` (`REGRESSION_RETAINED_INV_FEATURES` + `REGRESSION_FORCE_INCLUDE` après filtre `kept=1`). Les analyses questionnaire `v2` excluent les groupes à ICC2k `poor`, et le rapport ajoute le profil participant / les commentaires libres issus de `G3Q00001–G3Q00007`.

---

## 1. Objectif général du projet

Ce projet étudie l'**intelligence collective (CI)** dans des **triades** (groupes de 3 personnes) réalisant des tâches de **collaboration BIM** (Building Information Modeling) dans deux modalités :
- **VR** : Réalité virtuelle immersive
- **PC** : Interface 2D Revit sur ordinateur

Objectifs scientifiques :
- Identifier les **indicateurs non-verbaux (INV)** — vocaux, faciaux, de regard — associés à la **performance collective**
- Analyser les liens entre INV, **TCI** (Team Collective Intelligence / c-factor), **TMS** (Transactive Memory System), **questionnaires**, et **performance objective**
- Design **longitudinal** (T1/T2) avec deux scénarios de tâche (S1, S2)

**Note v2 (rapport publication)** :
- Les analyses de performance s'appuient prioritairement sur `Score_perf_tsk` et `Score_perf_tsk_z`
- Les composantes M1–M4 restent calculées en amont, mais ne constituent plus le cœur des analyses `v2`

Les scripts orchestrent :
- L'extraction des INV depuis données brutes (audio, vidéo, eye-tracking)
- L'analyse factorielle des INV (PCA, clustering)
- L'évaluation de la performance de tâche (score composite M1–M4)
- La génération de rapports PDF synthétiques intégrant toutes les dimensions

---

## 2. Principes métier stables

### 2.1 Cohérence analyse–rapport

**Règle fondamentale** : les sélections de variables doivent être **cohérentes avec leur usage**, mais elles ne sont plus strictement identiques entre rapport, PCA et régression `v2`.

- **Rapport** : piloté par `core` / `core_hl`, complétés par les flags documentaires `report_preferred`
- **PCA** : pilotée par `priority` via `FEATURE_PRIORITY` dans `scripts/config/inv_features_config.py`
- **Régression `v2`** : pilotée par `REGRESSION_RETAINED_INV_FEATURES` + `REGRESSION_FORCE_INCLUDE`, avec `inv_pruned_features.csv` utilisé comme filtre `kept=1` après intersection avec la whitelist métier
- Le mode `pruning` / `no-pruning` contrôle toujours les sorties PCA chargées par le rapport, mais n'est plus à lui seul la source de vérité pour la régression `v2`

### 2.2 Distinction entre types de features

| Type | Description | Usage |
|------|-------------|-------|
| **Features brutes** | Toutes les features numériques après filtrage technique (NaN < 20%) | Base pour l'analyse |
| **Features pruned** | Features après suppression des redondances (&#124;r&#124; > 0.80) | Mode `with_pruning/` — le nombre dépend du scope et des exclusions analytiques ; lire `inv_features_used.csv` |
| **Features full** | Toutes les features valides conservées | Mode `without_pruning/` — le nombre dépend du scope et de la disponibilité réelle des données |
| **Core features** | Features prioritaires pour affichage dans le rapport | Définies par `core` / `core_hl` dans `inv_features_config.py` |
| **Report preferred** | Features privilégiées pour les blocs rapport `v2` | Flags `report_preferred=True` |
| **Regression preferred** | Features candidates prioritaires pour la stepwise `v2` | `REGRESSION_RETAINED_INV_FEATURES` + `REGRESSION_FORCE_INCLUDE` |

### 2.3 Cadre théorique

Le projet s'inscrit dans le modèle **CI de Riedl, Woolley & Hackman** :

```
INPUTS                  MEDIATEURS                OUTPUTS
─────────────────────────────────────────────────────────
Composition            TMS (Transactive Memory)   Performance objective
Contexte/Environnement TAS (Task Attention)       Synergies
Caractéristiques tâche TRS (Turn Regulation)      Résultats subjectifs
C-factor               Coordination
                       Cohésion
                       Effort collectif
                       Spécialisation
                       Crédibilité
                       Stratégies
```

Les INV sont des **proxys comportementaux** de ces médiateurs.

---

## 3. Architecture logique du pipeline

```
[1] Données brutes
    ├── Audio (16 kHz, WAV) → Speech features (tours de parole, pauses, overlaps)
    ├── Vidéo (30 fps, MP4) → OpenFace → Face features (AUs, émotions, synchronie faciale)
    └── Eye-tracking VR (20 Hz) → Gaze features (shared attention, entropie, regard mutuel)

[2] Fusion des features
    → high_level_features_audit.csv (151 colonnes × 19 observations)  [mis à jour 2026-03-27]
    → high_level_features.csv (42 colonnes × 19 observations)          [mis à jour 2026-03-27]

[3] Analyse factorielle INV (analyze_inv_structure.py)
    ├── Sélection features (filtrage technique : NaN < 20%)
    ├── Standardisation (z-score)
    ├── Matrice de corrélation
    ├── [BRANCHEMENT] Deux pipelines parallèles :
    │   ├─ with_pruning/    : Hard pruning (|r| > 0.80) → PCA sur un nombre de features dépendant du scope
    │   └─ without_pruning/ : Pas de pruning → PCA sur toutes les features valides disponibles
    └── Outputs dans results/results_inv_structure/{with_pruning, without_pruning}/

[4] Évaluation performance
    → score_groupes.py : Calcul M1–M4 + Score_perf_tsk composite

[5] Génération rapport PDF (rapport/main.py)
    ├── Argument --inv-analysis-mode détermine quel sous-dossier charger
    ├── Fusion datasets (performance, TCI, questionnaires, INV)
    ├── Corrélations de Spearman (+ correction FDR optionnelle)
    └── Génération PDF/MD/HTML dans results/rapport_ci/
```

---

## 4. Fichiers centraux

### 4.1 `analyze_inv_structure.py`

**Rôle** : Analyse factorielle exploratoire des INV — PCA, clustering, projections.

**Nouveauté v9+** : Double pipeline avec deux modes d'analyse distincts.

**Fonction clé** : `run_inv_analysis_pipeline()` — encapsule le pipeline complet.

**Arguments** :
- `--only-pruning-mode {with, without, both}` : Contrôle quels pipelines exécuter (défaut : `both`)
- `--prune-threshold 0.80` : Seuil de corrélation pour le pruning (défaut : 0.80)
- `--max-missing 0.20` : Seuil max NaN pour filtrage technique
- `--min-cumvar 0.70` : Variance cumulée min pour rétention PCA

**Outputs** (dans chaque sous-dossier `with_pruning/` et `without_pruning/`) :
- `inv_features_used.csv` — Liste des features utilisées pour la PCA
- `inv_correlation_matrix.csv` — Matrice de corrélation complète
- `inv_pruned_features.csv` — Rapport de pruning (appliqué ou diagnostic)
- `pca_*.csv`, `pca_*.png` — Résultats PCA (scree plot, loadings, projections)
- `analysis_summary.csv` — Résumé des paramètres d'analyse (mode, n_features, apply_pruning)
- `analysis_report.txt` — Rapport textuel complet

### 4.2 `scripts/rapport/v2/main.py` (pipeline publication)

**Rôle** : Génération du rapport publication `v2` intégrant performance, INV, TCI, questionnaires.

**Argument critique** : `--inv-analysis-mode {pruning, no-pruning}` (défaut : `pruning`)
- `pruning` → Charge `results_inv_structure/with_pruning/`
- `no-pruning` → Charge `results_inv_structure/without_pruning/`

**Rétrocompatibilité** : `--no-pruning` (déprécié) est maintenu comme alias mais émet un warning.

**Logique de résolution de dossier** :
```python
inv_base_subdir = "results_inv_structure_vr_only" if modality_filter == "VR" else "results_inv_structure"
inv_pruning_subdir = "with_pruning" if inv_analysis_mode == "pruning" else "without_pruning"
inv_subdir = f"{inv_base_subdir}/{inv_pruning_subdir}"
```

**Fallback** : Si le sous-dossier n'existe pas, tentative sur l'ancien format (sans sous-dossier).

**Important** :
- `scripts/rapport/main.py` reste le pipeline historique `v1`
- `scripts/rapport/v2/main.py` est désormais le point d'entrée principal pour les sorties publication
- la régression `v2` n'est plus une simple projection des features prunées PCA : elle applique une whitelist métier dédiée

### 4.3 `scripts/config/inv_features_config.py`

**Rôle** : Configuration centralisée des features INV — source unique de vérité.

**Contenu** :
- `INV_FEATURES` : Dictionnaire complet avec priorité, famille, core flags, redondances documentées
- `AUDIO_ALIAS_PAIRS` : Mapping canonique ↔ alias (ex. `audio_overlap_speaking_ratio` ↔ `overlap_ratio`)
- `CORE_AUDIO`, `CORE_FACE`, `CORE_GAZE`, `CORE_HL` : Listes de features affichées dans le rapport
- `FEATURE_PRIORITY` : Ordre de priorité pour le pruning (1 = conservé, >50 = alias/redondant)
- Règles analytiques partagées `v2` : exclusions `*_median`, exclusion des `tms_*_idx`, sélection questionnaire/performance, `CORE_*_V2`
- `filter_inv_dataframe()` / `is_excluded_inv_feature()` : helpers de filtrage communs utilisés par le rapport `v2`

**Familles** : `audio`, `face`, `gaze`, `tms`

**Important** :
- ce fichier `scripts/config/inv_features_config.py` est la config canonique pour les features INV et les règles analytiques partagées

### 4.4 Autres fichiers clés

- `run_inv.py` : Pipeline complet INV (speech → gaze → face → high-level fusion)
- `run_performance.py` : Agrégation des scores de performance par groupe
- `score_groupes.py` : Calcul composite M1–M4 → Score_perf_tsk
- `common/` : Package partagé (constants, metadata, io_utils, stats, temporal)
- `config/` : Configurations centralisées (inv_features_config.py)

---

## 5. Conventions de nommage

### 5.1 Colonnes de métadonnées

**Identifiants** :
- `group_id` : ID complet avec condition (ex. `bim006_VR_T1`)
- `group_base_id` : ID de base (ex. `bim006`)
- `timepoint` : `T1` ou `T2`
- `scenario` : `S1` (BSI_A1/FISA_A3) ou `S2` (FISA_A5)
- `modalite` : `VR` ou `PC`
- `condition` : Combinaison modalité × timepoint (ex. `VR_T1`)

### 5.2 Familles de features INV

| Préfixe | Famille | Disponibilité |
|---------|---------|---------------|
| `audio_*` | Audio / Speech | PC + VR |
| `face_*`, `joy_*`, `sad_*`, `affect_*` | Émotions faciales | PC + VR |
| `gaze_*`, `shared_obj_*`, `mutual_gaze_*` | Regard / Attention | VR uniquement |

**Alias audio** : Les features brutes (`overlap_ratio`, `mean_turn_s`, etc.) sont des alias des versions canoniques (`audio_overlap_speaking_ratio`, `audio_avg_speaking_turn_duration_s`). Le HLF utilise `first_valid_series()` pour les résoudre par priorité.

**Alias gaze** : `gaze_mutual_gaze_ratio` est l'alias canonique de `mutual_gaze_ratio_mean_pairs`.

### 5.3 Modes d'analyse INV

| Mode | Dossier | Description |
|------|---------|-------------|
| `with_pruning` | `results_inv_structure/with_pruning/` | PCA sur features après hard pruning (&#124;r&#124; > 0.80) — effectif variable selon scope |
| `without_pruning` | `results_inv_structure/without_pruning/` | PCA sur toutes features valides — effectif variable selon scope |

**Fichier récapitulatif global** : `results_inv_structure/ANALYSIS_MODES_README.txt`

### 5.4 Outputs INV structure

- `inv_features_used.csv` : Liste explicite des features utilisées pour la PCA
- `analysis_summary.csv` : Résumé analytique (paramètres, n_features_initial, n_features_final, apply_pruning)
- `inv_pruned_features.csv` : Rapport de pruning (`applied=1` si pruning effectif, `applied=0` si diagnostic)

---

## 6. Règles de sélection de variables

### 6.1 Filtrage technique (toujours appliqué)

**Critères de base** :
- Exclusion des colonnes ID (`group_id`, `condition`, etc.)
- Exclusion des suffixes `_source`, préfixes `z_`
- Exclusion des features avec **>20% de valeurs manquantes** (paramètre `--max-missing`)

**Imputation** : Médiane sur les valeurs restantes après sélection.

**Note critique** : les métriques gaze ont ~42% de NaN sur l'ensemble PC+VR → exclues du filtrage global mais incluses dans les analyses VR-only.

### 6.2 Hard pruning (optionnel, mode `with_pruning/` uniquement)

**Définition** : Une feature A est **redondante** avec B si **|r(A, B)| > 0.80** (paramètre `--prune-threshold`).

**Règle de suppression** : Parmi deux features redondantes, supprimer celle qui a :
1. Le plus de valeurs manquantes
2. La priorité métier la plus basse (rang `priority` dans `inv_features_config.py`)
3. Le nom le plus long (moins canonique)

**Features conservées prioritairement** :
- `audio_*` canoniques (priority 1–6)
- `face_facial_synchrony`, `face_negative_affect_ratio`, `affect_alignment_idx` (priority 20–24)
- `gaze_entropy_mean_participants`, `shared_obj_ratio`, `shared_obj_dur_mean_s` (priority 30–32)

**Note** : La redondance observée empiriquement peut différer de celle documentée dans `inv_features_config.py` (redondance documentée = a priori conceptuel, redondance empirique = corrélation mesurée sur les 19 obs). Les deux sont utiles mais ne doivent pas être confondues.

### 6.3 Différence entre full et pruned

| Aspect | `without_pruning/` (full) | `with_pruning/` (pruned) |
|--------|---------------------------|--------------------------|
| **N features PCA** | Variable selon dataset et disponibilité réelle | Variable selon dataset, pruning et exclusions analytiques |
| **Objectif** | Décrire la **structure complète** de l'espace INV | Extraire des **dimensions indépendantes** pour analyses stats |
| **Redondances** | Conservées | Supprimées (&#124;r&#124; > 0.80) |
| **Usage** | Exploration, description riche | Régression, corrélations, modèles prédictifs |

**Règle de lecture** :
- ne pas figer les nombres `17` / `51` dans l'interprétation
- pour un run donné, utiliser `analysis_summary.csv` et `inv_features_used.csv`
- exemple actuel `rapport_v2` VR-only avec pruning : 25 features effectivement utilisées en PCA, et les mêmes 25 features prunées servent de base à la régression stepwise

---

## 7. Paramètres analytiques importants

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| **Seuil de corrélation pour pruning** | 0.80 | `--prune-threshold` |
| **Seuil NaN technique** | 0.20 (20%) | `--max-missing` |
| **Variance cumulée PCA** | 0.70 (70%) | `--min-cumvar` |
| **Rotation PCA** | `"none"` | `--rotation {none, varimax}` |
| **Imputation** | Médiane | Stratégie pour remplacer les NaN restants |
| **Standardisation** | Z-score | Toutes features sont standardisées avant PCA |

**Règle PCA — Rétention des composantes** :
- Eigenvalue > 1 (critère de Kaiser)
- OU variance cumulée ≥ 70% (paramètre `min_cumvar`)

---

## 8. Décisions de conception

### [2026-03-24] Ajout du double pipeline INV (with_pruning / without_pruning)

**Solution implémentée** :
- Créer **deux pipelines parallèles** complets dans `analyze_inv_structure.py`
- Organiser les outputs dans **deux sous-dossiers distincts** : `with_pruning/` et `without_pruning/`
- Ajouter l'argument `--only-pruning-mode {with, without, both}` (défaut : `both`)
- Modifier `rapport/main.py` pour accepter `--inv-analysis-mode {pruning, no-pruning}`
- Assurer que le rapport charge les bons fichiers et affiche le bon mode explicitement

### [2026-03-24] Extension de `inv_features_config.py`

**Solution implémentée** :
- Enrichir la config avec les métriques analytiques manquantes présentes dans `high_level_features_audit.csv`
- Réactiver `tms_coordination_idx`, `tms_specialization_idx`, `tms_credibility_idx` (famille `audio`)
- Étendre `MODALITY_PREFIX` pour mieux reconnaître les métriques brutes et agrégées
- Définir `CORE_SPEECH` distincte de `CORE_AUDIO`

### [2026-03-24] Corrections de cohérence inv_features_config / analyze_inv_structure / rapport

**Solution implémentée** :
- Remplacer dans `analyze_inv_structure.py` la logique locale d'assignation par l'appel à `infer_family_from_name()` depuis la config centrale
- Corriger `rapport/main.py` pour que la section `Speech` utilise explicitement `CORE_SPEECH`
- Enrichir `inv_face` avec les features core face disponibles dans `high_level_features.csv`
- Remplacer `feature_clusters.csv` par `feature_dendrogram.png` dans la section PCA du rapport

### [2026-03-26/27] Corrections de cohérence critiques

- `e_score` supprimé de `rapport/py/network.py` (variable fantôme dans la famille TCI)
- `CRE_pruned` → `CRE` corrigé dans `ci_multimodal_features_dimensions.txt` et `theory_diagrams.py`
- Backups `backup_inv_fix_*/` et fichiers `.pre_*_backup` supprimés du projet (nettoyage 2026-03-31)

### [2026-03-31] Ajout du module visualisation_sociale (sociogramme MIRAGE)

**Scripts impactés** :
- `scripts/visualisation_sociale/mirage_sociogram.py` — nouveau script

**Logique ajoutée** :
- Sociogramme multimodal offline inspiré de MIRAGE (SaacPSI/saac) adapté au pipeline du projet
- Fenêtre glissante configurable (défaut 20 s, pas 1 s) sur sessions enregistrées
- Couches : parole (PC+VR), synchronie faciale (PC+VR), regard/JVA/positions (VR uniquement)
- Export snapshot PNG, frames PNG, GIF, CSV analytiques

**Nouvelles options** :
- `--window-s`, `--step-s`, `--snapshot-at`, `--export-frames`, `--export-gif`
- `--frame-stride N` : sous-échantillonnage GIF (réduction ~10× taille si stride=10)
- `--show-live` + `--playback-speed` : lecture interactive Matplotlib

**Impact sur les entrées/sorties** :
- Dossier créé : `results/visualisation_sociale/<group_id>/`
- Fichiers générés : `mirage_snapshot.png`, `window_summary.csv`, `node_metrics.csv`, `edge_metrics.csv`, `run_metadata.json`

**Justification** :
- Fournir une visualisation dynamique des dynamiques de groupe comparable à MIRAGE pour des sessions enregistrées
- Réutilise les fonctions du pipeline existant (analyze_aus_group, analyze_gaze, io_utils, metadata)

**Rétrocompatibilité** : sans objet (nouveau module)

### [2026-04-03] Consolidation du rapport `v2`

**Solution implémentée** :
- Centraliser toute la logique INV analytique dans `scripts/config/inv_features_config.py`
- Supprimer le shim de compatibilité `v2` devenu inutile après la centralisation dans `scripts/config/inv_features_config.py`
- Introduire une whitelist métier de régression `v2` (`REGRESSION_RETAINED_INV_FEATURES`) et une réinjection forcée (`REGRESSION_FORCE_INCLUDE`) pour ne plus dépendre uniquement du pruning PCA
- Conserver `inv_pruned_features.csv` comme filtre `kept=1` en amont de la régression, mais seulement après intersection avec la whitelist métier
- Réaligner l'analyse de performance sur la règle d'exclusion correcte : garder `bim065`, exclure `bim065_2`

### [2026-04-07] Questionnaire `v2` et documentation de contexte

**Solution implémentée** :
- Exclure des analyses questionnaire `v2` tous les groupes dont l'accord inter-membres est classé `poor` (`ICC2k < 0.50`)
- Recalculer les sorties questionnaire utilisées par le rapport sur le sous-échantillon retenu après exclusion ICC
- Exploiter les questions `G3Q00001–G3Q00007` pour documenter le profil des participants, la familiarité VR et les commentaires libres en tête de rapport
- Documenter explicitement la séparation entre logique rapport, logique PCA et logique de régression `v2`

**État analytique `v2`** :
- Performance : `Score_perf_tsk`, `Score_perf_tsk_z`
- Questionnaire : `COR`, `CRE`, `SPE`, `Cohesion_questionnaire_score`
- Scénario : conservé comme covariable ANCOVA, mais retiré des figures comparatives courantes
- Régression INV : meilleurs modèles significatifs par nombre de prédicteurs, puis validation croisée 10-run 5-fold

---

## 9. ⚠️ Incohérences critiques identifiées (2026-03-27)

### CRITIQUE 1 — `e_score` référencé mais inexistant

**Localisation** : `scripts/rapport/py/network.py` (règle de classification `_FAMILY_RULES`, famille `"tci"`)

**Problème** : `e_score` est listé comme membre de la famille `tci` dans le réseau de corrélations, mais n'est **jamais calculé** nulle part dans le projet. Le fichier `results/TCI/c_scores.csv` contient uniquement `c_score`, `rme_mean`, `rme_max`, `rme_min`.

**Action requise** : Supprimer `"e_score"` de `_FAMILY_RULES` dans `network.py` et de toute documentation qui le référence (PROJECT_DOCUMENTATION.md §6.14 et §7).

### CRITIQUE 2 — documentation `CRE` et questionnaire partiellement obsolète

**Localisation** : `ci_multimodal_features_dimensions.*`, `PROJECT_DOCUMENTATION.md`, README du rapport `v2`

**Problème** : plusieurs documents faisaient encore référence à `CRE_pruned`, à la suppression simultanée `CRE02/CRE04`, ou omettaient le recalcul des analyses questionnaire après exclusion des groupes à ICC `poor`. L'état courant du pipeline utilise un score exporté sous le nom `CRE`, avec override exploratoire limité à `CRE04`, puis recalcul `v2` sur le sous-échantillon questionnaire retenu.

**Action requise** : maintenir la documentation alignée sur le pipeline courant : `CRE`, exclusions ICC explicites, et distinction claire entre pruning exploratoire questionnaire et analyses finales `v2`.

---

## 10. Points de vigilance (2026-03-27)

### Dimensions réelles des CSV (mises à jour)

| Fichier | Dimensions RÉELLES | Dimensions ancienne doc |
|---|---|---|
| `high_level_features_audit.csv` | 151 colonnes × 19 obs | 124 cols / 136 cols (incohérent) |
| `high_level_features.csv` | 42 colonnes × 19 obs | "~80 cols" (erroné) |
| `audio_features.csv` | 101 colonnes × 23 obs | 76 cols |
| `speech_metrics.csv` | 70 colonnes × 23 obs | 48 cols |

### Variables Riedl présentes dans les données mais absentes de `CORE_RIEDL_COLS`

Ces variables sont calculées et dans le merged_dataset, mais non incluses dans `CORE_RIEDL_COLS` dans `inv_features_config.py`. Elles n'apparaissent pas dans les corrélations systématiques du rapport :
- `rme_max`, `rme_min` — produites par TCI.py, utilisées dans `theory_diagrams.py` mais absentes de CORE_RIEDL
- `effort_task_mean`, `effort_task_cv`, `effort_task_gini` — documentées dans .txt, non CORE
- `contribution_cv`, `n_tasks`, `n_tasks_strategy_defined`, `n_members_with_skill` — idem

### Variables composites absentes du fichier de référence `.txt`

Ces variables sont actives et utilisées dans le rapport, mais non documentées dans `ci_multimodal_features_dimensions.txt` :
- `gaze_attention_coordination_idx` (famille gaze/HLF)
- `tms_coordination_idx`, `tms_specialization_idx`, `tms_credibility_idx` (famille audio/HLF)
- `effort_task_norm` (famille riedl)

### Gaze — VR uniquement

Toutes les métriques gaze ont ~42% de NaN sur l'ensemble PC+VR (observations PC sans eye-tracker). Toute analyse incluant le gaze sur le dataset complet requiert imputation ou restriction VR-only.

### Alias de nommage gaze

`mutual_gaze_ratio_mean_pairs` (nom dans `analyze_gaze.py`) ↔ `gaze_mutual_gaze_ratio` (alias canonique dans HLF et rapport). Les deux sont dans `high_level_features_audit.csv`. Le `.txt` de référence utilise l'ancien nom.

### Scripts utilitaires non documentés

Non documentés dans `PROJECT_DOCUMENTATION.md` mais présents :
- `scripts/analyse_performance/correlation_metrics_final.py`
- `scripts/analyse_performance/cronbach_alpha_performance.py`
- `scripts/analyse_performance/multiple_regression_perf.py`
- `scripts/utilitaires/build_inv_variable_inventory.py`
- `scripts/evaluation_performance_scenario/score_v2.py` (vide)

### Module visualisation_sociale — Limitations

- Gaze, JVA et positions : VR uniquement. Sur PC, seules audio et synchronie faciale sont actives.
- VAD locale (RMS + seuil adaptatif) utilisée pour la vitesse — moins précise que Whisper.
- Offline uniquement : pas de streaming temps réel contrairement à MIRAGE original.
- Voir `MIRAGE_SOCIOGRAM_NOTES.md` pour les détails techniques complets.

---

## 11. Variables confirmées cohérentes (2026-03-27)

Le pipeline INV complet est fonctionnellement cohérent pour les variables canoniques centrales :

**Audio (canonical)** : `audio_total_speaking_turns`, `audio_avg_speaking_turn_duration_s`, `audio_floor_exchange_pause_mean_s`, `audio_overlap_speaking_ratio`, `audio_backchannel_rate_per_min` — produites, aliasées, présentes dans le rapport.

**Audio (brut)** : `mean_turn_s`, `overlap_ratio`, `pause_ratio` — cohérents comme alias.

**Face** : `face_smile_ratio`, `face_negative_affect_ratio`, `face_facial_synchrony`, `affect_alignment_idx` — calculées, dans les CSV, dans le rapport.

**Performance** : `Score_perf_tsk`, `M1_consignes_%`, `M2_nombre_%`, `M3_precision_%`, `M4_temps_%` — pipeline complet cohérent.

**Questionnaire** : `COR`, `CRE`, `SPE`, `SOC`, `TSK`, `COM` — cohérents ; les analyses `v2` sont recalculées après exclusion des groupes à ICC `poor`.

**TCI** : `c_score`, `rme_mean` — cohérents.

**Riedl** : `skill_mean`, `skill_max`, `skill_congruence_mean`, `strategy_ratio_mean`, `strategy_norm`, `contribution_mean`, `effort_task_sum` — cohérents dans le pipeline.

---

## 12. TODO / Points ouverts

1. ~~CRITIQUE : Supprimer `e_score` de `network.py`~~ — **FAIT 2026-03-27**
2. ~~CRITIQUE : `CRE_pruned` → `CRE` dans les fichiers de référence questionnaire~~ — **FAIT 2026-04-07**
3. **TMS features** : Actuellement commentées dans `inv_features_config.py` — à réactiver si données exploitables
4. **Seuil de pruning** : Actuellement 0.80 — à réévaluer selon analyses stat (tester 0.75, 0.85)
5. **Scope VR-only** : vérifier à chaque mise à jour documentaire que les sorties `results_inv_structure_vr_only/` et les règles spécifiques gaze restent cohérentes avec le rapport `v2`
6. **Correction FDR** : Argument `--fdr` disponible mais pas systématiquement appliqué — définir politique claire
7. **CORE_RIEDL_COLS** : Envisager d'ajouter `rme_max`, `rme_min`, `effort_task_mean` si pertinent pour les corrélations systématiques du rapport
8. **Dimensions CSV** : Mettre à jour `PROJECT_DOCUMENTATION.md` §7 avec les vraies dimensions (voir tableau §10 ci-dessus)

---

## 13. Règles de mise à jour

### Ce fichier doit être relu avant toute nouvelle modification importante

**Quand relire `PROJECT_CONTEXT.md`** :
- Avant d'ajouter une nouvelle feature INV
- Avant de modifier la logique de pruning
- Avant d'ajouter un nouvel argument à `analyze_inv_structure.py` ou `rapport/main.py`
- Avant de refactorer le pipeline d'analyse
- Avant de changer la structure des dossiers de sortie

### Ce fichier doit être mis à jour après toute évolution structurelle

**Quand mettre à jour `PROJECT_CONTEXT.md`** :
- Modification de la logique métier (nouvelle règle de pruning, nouveau calcul)
- Ajout/modification de conventions de nommage
- Changement des paramètres d'analyse par défaut
- Ajout de nouveaux fichiers centraux
- Modification de la structure du pipeline

### Contraintes de contenu

**Ce fichier NE doit PAS contenir** :
- Détails d'implémentation bas-niveau (ex. indices de boucles, noms de variables locales)
- Informations temporaires ou session-spécifiques
- Logs ou erreurs ponctuelles
- Instructions d'utilisation détaillées (→ README.md)
- Code complet (→ fichiers sources)

**Ce fichier DOIT contenir** :
- Principes métier stables
- Règles de conception structurelles
- Conventions respectées dans tout le projet
- Historique des décisions architecturales majeures
- Points d'attention critiques pour les modifications futures

### [2026-04-01] Rapport — construits questionnaire et diagramme théorique

**Scripts impactés** :
- `scripts/rapport/main.py`
- `scripts/rapport/py/regression.py`
- `scripts/rapport/py/theory_diagrams.py`

**Logique modifiée** :
- le rapport ne crée plus de score TMS moyen pour les analyses aval ;
  les dimensions `COR`, `CRE`, `SPE` restent analysées séparément
- un score latent explicite de cohésion questionnaire,
  `Cohesion_questionnaire_score`, est calculé uniquement si l'alpha de
  Cronbach de `SOC`, `TSK`, `COM` est acceptable (`α ≥ 0.70`)
- ce score de cohésion est injecté dans les corrélations questionnaire du
  rapport et dans les vues associées
- dans `theory_diagrams.py`, les liens latent → dimensions questionnaire
  n'apparaissent dans la vue empirique que si l'alpha du construit est
  acceptable ; ils sont alors annotés par `α=...`

**Impact sur les entrées/sorties** :
- pas de nouveau fichier d'entrée requis
- les résultats de corrélation peuvent maintenant inclure
  `Cohesion_questionnaire_score`
- les diagrammes théoriques empiriques peuvent afficher des liens
  latent-dimension validés par alpha

**Justification** :
- éviter d'interpréter un construit latent questionnaire non valide
- distinguer le traitement de la cohésion (score latent acceptable) de celui
  du TMS (dimensions conservées séparément)

**Rétrocompatibilité** :
- les dimensions questionnaire historiques restent inchangées
- les vues de régression continuent de fonctionner, mais n'agrègent le TMS
  que si un score explicite est déjà présent en entrée

### [2026-04-01] PCA du rapport — projections questionnaire détaillées

**Scripts impactés** :
- `scripts/rapport/py/pca_inv.py`
- `scripts/rapport/main.py`

**Logique modifiée** :
- les projections PCA du rapport n'utilisent plus un score TMS agrégé dans la
  section PCA
- le score de cohésion questionnaire est projeté explicitement via
  `Cohesion_questionnaire_score`
- les dimensions de cohésion `SOC`, `TSK`, `COM` sont projetées séparément
- les dimensions TMS `COR`, `CRE`, `SPE` remplacent la projection TMS globale
- dans le PDF, les trois dimensions de cohésion et les trois dimensions TMS
  sont disposées par lignes de 3 figures pour lecture comparative

**Impact sur les sorties** :
- nouveaux PNG attendus côté rapport :
  - `pca_projection_by_Cohesion_questionnaire_score.png`
  - `pca_projection_by_SOC.png`
  - `pca_projection_by_TSK.png`
  - `pca_projection_by_COM.png`
  - `pca_projection_by_COR.png`
  - `pca_projection_by_CRE.png`
  - `pca_projection_by_SPE.png`

**Justification** :
- mieux distinguer le construit latent de cohésion et ses dimensions
- éviter d'interpréter une projection PCA basée sur un score TMS global alors
  que l'analyse questionnaire conserve les dimensions TMS séparées

### [2026-04-01] Diagramme théorie/empirie — refonte hiérarchique + section 8

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`
- `scripts/rapport/main.py`

**Logique modifiée** :
- le diagramme théorie/empirie n'est plus rendu comme un réseau libre mais
  comme une structure hiérarchique `Input -> Mediators -> Outputs`
- `C_factor_Input` et `CI` sont fusionnés dans un nœud unique
  `CI / C-factor`
- ce nœud fusionné alimente théoriquement les médiateurs (`TMS`, `TAS`, `TRS`)
  et conserve un lien direct vers `Performance`
- la couche `Audio INV / Face INV / Gaze INV` est explicitement représentée
  comme couche de mesure multimodale, distincte des construits centraux
- la section 7 du rapport n'affiche plus qu'une seule vue comparée à deux
  panneaux (`A. Modèle théorique` / `B. Théorie + soutien empirique`)
- une nouvelle section 8 relie le diagramme théorie/empirie à un schéma
  computationnel de la chaîne d'opérationnalisation

**Impact sur les sorties** :
- le dossier de figures du rapport contient toujours :
  - `theory_model_only.png`
  - `theory_plus_empirical.png`
  - `theory_empirical_only.png`
  - `theory_comparison.png`
- nouveau fichier généré :
  - `computational_model_articulation.png`

**Justification** :
- rapprocher le rendu des schémas de référence Input / Mediators / Outputs
- clarifier le statut conceptuel du `C-factor` comme potentiel collectif
  initial et mesure empirique agrégée
- expliciter le pont entre instrumentation multimodale, variables calculées
  et construits médiateurs interprétés dans le rapport

**Arbitrages de modélisation** :
- la logique alpha est conservée explicitement pour la cohésion dans la vue
  empirique
- les dimensions TMS sont projetées comme processus médiateurs distincts
  dans la structure hiérarchique, plutôt que comme un score latent unique
  affiché dans le diagramme

### [2026-04-01] Diagrammes sections 7 et 8 — passe visuelle fine

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`
- `scripts/rapport/main.py`

**Logique modifiée** :
- espacement accru des nœuds médiateurs pour limiter les croisements au centre
- réduction forte de la surcharge visuelle des labels d'arêtes
- repositionnement des dimensions de cohésion (`SOC`, `TSK`, `COM`) dans une
  zone plus compacte et plus lisible
- ajout d'un nœud `TMSQ` relié à `Specialization`, `Coordination` et
  `Credibility` pour rendre explicite le latent questionnaire TMS
- ajout des liens directs `CI / C-factor -> Skill congruence`, `Effort` et
  `Strategies`
- la figure computationnelle de la section 8 est rapprochée du schéma de
  référence avec un bloc central "Mediator Dynamics", un contour rouge pour
  les composantes explicitement mesurées et une hiérarchie visuelle plus nette

**Impact sur les sorties** :
- pas de nouveaux fichiers de sortie
- amélioration attendue de la lisibilité des PNG/PDF :
  - `theory_comparison.*`
  - `computational_model_articulation.*`

**Justification** :
- rendre les sections 7 et 8 plus proches des schémas de travail de thèse
- faire apparaître plus explicitement les relations conceptuelles demandées
  sans regonfler la complexité analytique du pipeline

### [2026-04-01] Diagrammes sections 7 et 8 — tailles de blocs dynamiques

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`

**Logique modifiée** :
- les tailles `wh` des nœuds du diagramme théorique sont désormais traitées
  comme des tailles minimales
- le rendu calcule une largeur/hauteur ajustée au label avec bornes de
  croissance contrôlées
- la même logique est appliquée aux boîtes de la figure computationnelle de
  section 8

**Justification** :
- éviter les débordements ou compressions lorsque les labels deviennent plus
  longs
- conserver un layout stable tout en rendant les blocs plus robustes aux
  changements de libellés

---

### [2026-04-01] Diagramme théorie/empirie — empilement vertical des panneaux

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`
- `scripts/rapport/main.py`

**Logique modifiée** :
- La vue comparée de la section 7 conserve une seule figure `theory_comparison`, mais les panneaux `A. Modèle théorique` et `B. Théorie + soutien empirique` sont maintenant empilés verticalement au lieu d'être affichés côte à côte.
- Le changement répond à un problème de lisibilité des blocs et des libellés dans la mise en page portrait du rapport.

**Impact sur les entrées/sorties** :
- Fichier généré inchangé : `theory_comparison.png/pdf`
- Mise en page PDF ajustée pour afficher une image plus haute en section 7

**Justification** :
- Le rendu horizontal comprimait trop fortement les blocs, ce qui dégradait la lecture du schéma malgré la refonte théorique précédente.

**Rétrocompatibilité** :
- Nom de sortie conservé : oui
- Appel depuis `main.py` conservé : oui

### [2026-04-01] Diagrammes sections 7 et 8 — géométrie recalée sur les fichiers draw.io

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`

**Logique modifiée** :
- Abandon de l'agrandissement heuristique des boîtes pour les diagrammes théoriques et computationnels.
- Les blocs et les nœuds principaux sont maintenant positionnés et dimensionnés à partir de coordonnées normalisées dérivées des fichiers `model.drawio` et `pipeline.drawio`.
- Le nœud `CI / C-factor` a été conservé avec ce libellé canonique dans la géométrie recalée.

**Impact sur les entrées/sorties** :
- Sorties inchangées : `theory_model_only.*`, `theory_plus_empirical.*`, `theory_empirical_only.*`, `theory_comparison.*`, `computational_model_articulation.*`
- Rendu attendu plus stable et plus proche des schémas de référence, sans croissance dynamique des boîtes.

**Justification** :
- La logique de taille dynamique introduisait des déformations visuelles importantes ; le recalage sur draw.io fournit une base géométrique plus robuste et plus fidèle au modèle source.

**Rétrocompatibilité** :
- API publique inchangée : oui
- Noms de fichiers générés inchangés : oui

### [2026-04-01] Diagrammes sections 7 et 8 — ajustements conceptuels et de lisibilité

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`

**Logique modifiée** :
- Suppression des liens théoriques `CI / C-factor -> TMS/TAS/TRS` dans la section 7.
- Conservation des liens `CI / C-factor -> Skill congruence / Strategies / Effort / Performance`.
- Renommage visuel de `TMSQ` en `TMS (Q)` pour expliciter qu'il s'agit du versant questionnaire.
- Repositionnement horizontal de `SOC (Q)`, `TSK (Q)` et `COM (Q)` sur une même ligne du niveau `Behavioral process`.
- Espacement vertical accru entre la couche `Input` et la couche `Nonverbal cues / INV measures`.
- Section 8 : boîtes `CI / C-factor` et `Feature & indicator layer` élargies, flèches recâblées et simplifiées pour éviter les effets de clipping et les trajectoires visuellement incohérentes.

**Justification** :
- Ces ajustements répondent à une revue visuelle du rendu PDF : meilleure lisibilité théorique du schéma section 7 et stabilisation du modèle computationnel section 8.

**Rétrocompatibilité** :
- API publique inchangée : oui
- Noms de fichiers générés inchangés : oui

### [2026-04-01] Section 7 — agrandissement et ajustements fins de layout

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`
- `scripts/rapport/main.py`

**Logique modifiée** :
- Les trois sous-blocs d'entrée (`Team composition`, `Context / Environment`, `Task characteristics`) sont légèrement séparés verticalement pour améliorer la lisibilité.
- Les boîtes `Face INV`, `Audio INV`, `Gaze INV` sont abaissées à l'intérieur du bandeau `Nonverbal cues / INV measures`, sans déplacer le conteneur global.
- La figure comparée de la section 7 est générée un peu plus grande et insérée plus grande dans le PDF.

**Justification** :
- Le rendu précédent restait trop serré au niveau de l'input et de la couche INV, et la figure section 7 gagnait à être légèrement agrandie dans le rapport final.

### [2026-04-01] Section 7 — simplification visuelle des médiateurs

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`

**Logique modifiée** :
- Les nœuds `TMS_Q` et `Cohesion` sont désormais masqués au rendu sans suppression des liens définis dans le script.
- Le bloc `Mediators` a été agrandi, avec mise à l'échelle des sous-zones internes (`Production`, `Maintenance`, `Transactive sub-system`, `Dynamic indicators`, `Behavioral process`).
- Le bloc `Input` a été agrandi et ses trois sous-blocs ont été espacés pour une lecture plus nette.

**Justification** :
- L'objectif était de conserver la logique de liens déjà modifiée dans le script, tout en allégeant visuellement la figure et en améliorant la lisibilité des zones principales.

### [2026-04-01] Section 7 — réétalement des dimensions questionnaire de cohésion

**Scripts impactés** :
- `scripts/rapport/py/theory_diagrams.py`

**Logique modifiée** :
- Les nœuds `Social_Cohesion`, `Task_Cohesion` et `Communication` ont été réétalés horizontalement pour occuper plus proprement la largeur disponible sur la ligne `Behavioral process`.
- Cette passe ne modifie pas la logique des liens et conserve les commentaires utilisateur déjà présents dans le code autour des arêtes.

## TEMPLATE DE MISE À JOUR

```markdown
### [YYYY-MM-DD] Titre de la modification

**Scripts impactés** :
- script1.py
- script2.py

**Logique modifiée** :
- Brève description de ce qui change dans la logique métier

**Nouvelles options** :
- `--nouvelle-option {valeur1, valeur2}` : Description

**Impact sur les entrées/sorties** :
- Nouveaux fichiers générés : `fichier_nouveau.csv`
- Fichiers modifiés : `fichier_existant.csv` (ajout de colonnes X, Y)
- Dossiers créés : `nouveau_dossier/`

**Mise à jour du contexte nécessaire** : oui / non
- Section(s) à mettre à jour : [Section X, Section Y]

**Justification** :
- Pourquoi cette modification était nécessaire

**Rétrocompatibilité** :
- Ancien argument maintenu ? oui/non
- Fallback disponible ? oui/non
```
