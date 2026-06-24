# rapport/v2

Version v2 du pipeline de rapport CI, dérivée de `../v1/`.

Objectif :
- conserver `v1` comme baseline exploratoire ;
- appliquer uniquement dans `v2/` les changements orientés publication ;
- générer automatiquement les sorties :
  - `PC_VR/rapport_principal_PC_VR.pdf`
  - `VR_only/rapport_principal_VR.pdf`
  - `VR_only/rapport_INV_VR.pdf`
  - `VR_only/rapport_PCA_VR.pdf`

Principes analytiques v2 :
- sélection rapport pilotée par `core/core_hl` et les flags `report_preferred` de `scripts/config/inv_features_config.py` ;
- PCA pilotée par `priority` via `FEATURE_PRIORITY` dans la même config ;
- régression stepwise pilotée par `REGRESSION_RETAINED_INV_FEATURES` + `REGRESSION_FORCE_INCLUDE` ; `inv_pruned_features.csv` sert de filtre `kept=1` après intersection avec cette whitelist métier ;
- suppression analytique des indices `tms_*_idx` ;
- suppression analytique des variables INV `*_median` ;
- score de cohésion agrégé comme variable principale ;
- dimensions TMS conservées séparément (`COR`, `CRE`, `SPE`) ;
- exclusion des groupes questionnaire à accord inter-membres `poor` (`ICC2k < 0.50`) dans toutes les analyses questionnaire aval ;
- ajout en tête du rapport du profil participant et des commentaires libres issus de `G3Q00001–G3Q00007` ;
- scénario conservé uniquement comme covariable ANCOVA ;
- sorties principales recalculées sur le sous-échantillon questionnaire retenu quand des exclusions ICC sont appliquées.

Source de vérité INV :
- canonique : `scripts/config/inv_features_config.py`
