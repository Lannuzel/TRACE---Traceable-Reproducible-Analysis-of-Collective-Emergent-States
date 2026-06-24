# CHANGELOG v2

## 2026-04-02

- création de `v2/` à partir de `v1/`
- ajout initial d'une configuration locale `v2` pour porter les choix analytiques du rapport
- exclusion des variables `tms_*_idx` et `*_median` des analyses v2
- simplification des analyses de performance autour de `Score_perf_tsk` et `Score_perf_tsk_z`
- retrait des figures scénario-dépendantes hors ANCOVA
- ajout d'une table ICC questionnaire avant agrégation groupe
- ajout d'une section d'effet de modalité sur `COR`, `CRE`, `SPE` et `Cohesion_questionnaire_score`
- retrait des dimensions `SOC`, `TSK`, `COM` des analyses corrélationnelles/régressives directes
- ajout de régressions INV stepwise avec diagnostics de résidus
- génération automatique des sorties :
  - `PC_VR/rapport_principal_PC_VR.pdf`
  - `VR_only/rapport_principal_VR.pdf`
  - `VR_only/rapport_INV_VR.pdf`
  - `VR_only/rapport_PCA_VR.pdf`

## 2026-04-03

- centralisation finale des règles INV dans `scripts/config/inv_features_config.py`
- suppression du shim de compatibilité `v2` après centralisation complète dans `scripts/config/inv_features_config.py`
- alignement de la PCA `v2` et des régressions `v2` sur les mêmes exclusions analytiques (`*_median`, `tms_*_idx`)
- régression stepwise `v2` réalignée sur les **features prunées effectivement utilisées**
- distinction explicite entre espace PCA et espace analytique dans la documentation des rapports
- harmonisation de l'analyse performance : `bim065` conservé, `bim065_2` exclu
- nettoyage de la structure ANCOVA dans le rapport principal `v2`
