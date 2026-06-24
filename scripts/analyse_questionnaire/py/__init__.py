"""
py — Analyse des questionnaires TMS / Cohésion (version Python).

Modules :
    config             — Constantes (dimensions, Likert, items inversés, seuils)
    io_read            — Lecture du fichier source (xlsx/csv)
    transform          — Transformation wide → long + recodage Likert + inversion
    reliability        — Alpha de Cronbach + statistiques item (r.drop)
    descriptives       — Descriptifs par dimension
    role_tests         — Tests inter-rôles (ANOVA / Kruskal-Wallis)
    plots              — Exports PDF (distributions items, barplots rôles)
    item_pruning       — Détection / suppression des items faibles
    scenario_modalite  — Analyse Scénario × Modalité (scores, modèles, plots)
"""
