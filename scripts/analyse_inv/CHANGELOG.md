# CHANGELOG — pipeline analyse_inv

## 2026-05-09 — MOD-11 — Audit homogeneite fallbacks + clarification face_facial_synchrony

**Fichiers modifies :** aucun (audit de confirmation — pas de modification de code)
**Rapport d'audit :** `hlf/fallback_audit_report.md`

### MOD-11a — Audit homogeneite sources HLF

Audit de la distribution des colonnes `_source` sur les 19 groupes face / 12 groupes gaze VR.

**Resultats :**
- `face_negative_affect_ratio_source` : 19/19 groupes = "au15_au17_coactive_pct_mean" (niveau 1). Chaine de fallback jamais activee.
- `face_smile_ratio_source` : 19/19 groupes = "au6_au12_coactive_pct_mean" (niveau 1). Chaine de fallback jamais activee.
- `gaze_mutual_gaze_ratio_source` : 12/12 groupes avec donnees gaze = "mutual_gaze_ratio_mean_pairs" (niveau 1). 7 groupes NaN = donnees gaze absentes (structurel, pas un fallback).
- `face_facial_synchrony_source` : 19/19 groupes = "au_sync_pearson_mean" (post-MOD-5, attendu).

**Conclusion :** les trois variables ciblees sont deja homogenes. L'etape de correction (Strategie A) est sans objet.

### MOD-11b — Clarification operationnalisation face_facial_synchrony

Audit du code source `face/analyze_aus_group.py::compute_synchrony_pearson` :
- Operationnalisation confirmee : **Option B** (synchronie par AU individuel, puis agregation)
- Pour chaque paire (a,b) et chaque AU k in {au6_active_pct, au12_active_pct, au15_active_pct, au17_active_pct, au6_au12_coactive_pct, au15_au17_coactive_pct} :
  `r_(ab,k) = Pearson(AU_k_a(fenetre), AU_k_b(fenetre))`
- `au_sync_pearson_mean = mean(r_(ab,k))` sur 6 metriques x 3 paires

**Reference correcte :** Prochazkova & Kret (2017), Neuroscience & Biobehavioral Reviews.
**Reference incorrecte retiree :** Hess & Fischer (2013) correspondait a l'Option A (synchronie globale), non implementee.

**Fichiers mis a jour :** `INV_METHODES_CALCUL.md` (sections face_facial_synchrony, face_negative_affect_ratio, face_smile_ratio, gaze_mutual_gaze_ratio)

---

## 2026-05-07 — MOD-1 a MOD-10

### MOD-1 — Agregation IPU->tours CA
**Fichiers modifies :** `speech/analyze_audio.py`

Nouvelle fonction `aggregate_ipus_to_ca_turns(segs_by_role, turn_min_s)` ajoutee avant `process_group`.
Implementation : Sacks, Schegloff & Jefferson (1974), Conversation Analysis ;
Levitan & Hirschberg (2011).

Les segments IPU consecutifs du meme role sont fusionnes en un seul tour CA.
Le tour se ferme lors d'une alternance de role. Filtre final : duree >= turn_min_s.

**Variables ajoutees :**
- `turns_{r}_n` : nombre de tours CA par role (ecrase l'ancien IPU-based)
- `turns_{r}_n_old` : ancien comptage IPU-based
- `mean_turn_{r}_s` : duree moyenne tours CA par role (ecrase)
- `mean_turn_{r}_s_old` : ancien calcul IPU-based
- `speaking_turn_time_{r}_s` : temps de parole en tour CA (ecrase)
- `mean_turn_s` : duree moyenne tours CA tous roles (ecrase)
- `mean_turn_s_old` : ancien calcul IPU-based
- `floor_exchange_pause_mean_s` : pause moyenne echange de tour CA-based (ecrase)
- `floor_exchange_pause_mean_s_old` : ancien calcul IPU-based
- `n_floor_exchanges` : nombre echanges CA-based (ecrase)
- `n_floor_exchanges_old` : ancien comptage IPU-based

**Assertion anti-regression :** `total_turns_NEW <= total_turns_OLD + 0.5`
(levee dans `compute_audio_features.py::build_features`)


### MOD-2 — Decomposition Occupancy multi-locuteurs
**Fichiers modifies :** `speech/analyze_audio.py`

Nouvelle fonction `compute_speaker_occupancy(segs_by_role, total_s, fs_grid=100.0)`.
Implementation : Cetin & Shriberg (2006), ICSI multi-party occupancy decomposition.

**Variables ajoutees :**
- `audio_pct_time_0_speakers` : fraction temps sans locuteur actif
- `audio_pct_time_1_speaker` : fraction temps 1 seul locuteur
- `audio_pct_time_2_speakers` : fraction temps 2 locuteurs simultanement
- `audio_pct_time_3_speakers` : fraction temps 3 locuteurs simultanement
- `audio_overlap_speaking_ratio_from_occupancy` : overlap ratio = pct_2 + pct_3

**Colonnes modifiees :**
- `overlap_ratio` : recalcule depuis occupancy (ecrase, ancienne valeur en `overlap_ratio_old`)
- `overlap_ratio_old` : ancien calcul overlap_s/total
- `audio_overlap_speaking_ratio` : alias de `overlap_ratio`

**Assertion interne :** `sum(pct_time_*) == 1.0 +/- 1e-3` (AssertionError avec valeur explicite)


### MOD-3 — Backchannels stricts
**Fichiers modifies :** `speech/analyze_audio.py`

Nouvelle fonction `is_backchannel_strict(...)`.
Implementation : Truong & Heylen (2010), 4 filtres en cascade.

**Filtres appliques :**
1. Duree dans [bc_min_dur=0.10s, bc_max_dur=backch_max]
2. Chevauchement >= bc_min_ovl avec un autre role
3. Pas une continuation : exclure si precede de < 200ms un segment du meme role
4. Pas un tour CA propre : exclure si debut a < 500ms d'un tour CA du meme role

**Filtre F0 omis :** Le filtre F0 (detection melodique descendante) prescrit par Truong &
Heylen (2010) a ete deliberement omis. Raison : l'extraction F0 via librosa.yin
multiplierait le temps de run total par 3-5x (traitement frame-by-frame sur toutes
les pistes audio) sans apport methodologique majeur, les 4 filtres temporels et
structurels couvrant l'essentiel du construit. A reevaluer si un pipeline prosodique
est integre ulterieurement.

**Variables ajoutees :**
- `backchannels_{r}_n` : backchannels stricts par role (ecrase l'ancien)
- `backchannels_{r}_n_old` : ancien comptage laxiste


### MOD-4 — Renommages canoniques audio
**Fichiers modifies :** `speech/compute_audio_features.py`

**Variables ajoutees dans `build_features` :**
- `audio_overlap_takeover_ratio` : alias de `successful_interruption_ratio` (Truong 2013)
- `audio_successful_interruption_ratio_old` : copie de l'ancienne valeur
- `audio_avg_speaking_turn_duration_s` : alias de `mean_turn_s`
- `audio_avg_speaking_turn_duration_s_old` : alias de `mean_turn_s_old`
- `audio_total_speaking_turns` : alias de `total_turns`
- `total_turns_old` : somme des `turns_{r}_n_old`
- `audio_total_speaking_turns_old` : alias de `total_turns_old`
- `audio_floor_exchange_pause_mean_s` : alias de `floor_exchange_pause_mean_s`
- `audio_floor_exchange_pause_mean_s_old` : alias de `floor_exchange_pause_mean_s_old`
- `audio_backchannel_rate_per_min` : alias de `backchannel_rate_per_min`
- `n_backchannels_old` : somme des `backchannels_{r}_n_old`
- `backchannel_rate_per_min_old` : taux/min backchannels ancienne methode
- `audio_backchannel_rate_per_min_old` : alias de `backchannel_rate_per_min_old`


### MOD-5 — Synchronie faciale Pearson uniquement
**Fichiers modifies :** `hlf/compute_high_level_features.py`

**Rationalisation :** Ancien calcul = fallback multi-source (au_sync_mean > jaccard > pearson >
moyenne joy/sad sync). Nouveau calcul = Pearson uniquement, source unique.
Reference : Hess & Fischer (2013), Emotional mimicry as social regulation, PNAS.

**Variables modifiees/ajoutees :**
- `face_facial_synchrony` : desormais = `au_sync_pearson_mean` uniquement (ou NaN)
- `face_facial_synchrony_old` : ancienne valeur multi-fallback
- `face_facial_synchrony_source` : "au_sync_pearson_mean" ou "missing"


### MOD-6 — Renommage AU15+AU17
**Fichiers modifies :** `hlf/compute_high_level_features.py`

**Rationalisation :** AU15 (depressor anguli oris) + AU17 (chin raiser) = marqueur FACS
de tristesse, pas d'affect negatif global.
Reference : Ekman & Friesen (1978), Facial Action Coding System.

**Variables ajoutees :**
- `face_negative_affect_ratio_old` : copie de la valeur originale (conservee)
- `face_sadness_marker_ratio` : renommage explicite AU15+AU17 = tristesse


### MOD-7 — Documentation face_smile_ratio (Duchenne)
**Fichiers modifies :** `hlf/compute_high_level_features.py`

Ajout commentaire avant le calcul de `face_smile_ratio` :
"Sourire de Duchenne : AU6 (orbicularis oculi) + AU12 (zygomaticus major).
Sourire sincere (Ekman, Davidson & Friesen 1990, JPSP). Exclut le sourire social (AU12 seul)."

Pas de changement de calcul. Pas de colonne `_old`.


### MOD-8 — Conservation z-scores affect_alignment_idx
**Fichiers modifies :** `hlf/compute_high_level_features.py`

Les 4 z-scores composants de `affect_alignment_idx` sont desormais inclus dans la sortie
compacte et dans `_reorder_output_columns` :
- `z_joy_tri_occupancy`
- `z_joy_sync_jaccard_mean`
- `z_sad_tri_occupancy`
- `z_sad_sync_jaccard_mean`

(Avec guard `if c in df.columns` dans `make_compact_output`)


### MOD-10 — Retrait de z_mutual_gaze_ratio_mean_pairs de gaze_attention_coordination_idx
**Fichiers modifies :** `hlf/compute_high_level_features.py`, `config/inv_features_config.py`, `build_schema.py`

Audit 2026-05-18 (N=11 groupes VR) : `mutual_gaze_ratio_mean_pairs` = 0.0 pour tous les groupes
VR, variance = 0. Apres z-score, le terme est structurellement nul → ne contribue pas a l'index.

Nouvelle formule : `nanmean([z_shared_obj_ratio, -1.0 × z_gaze_entropy_mean_participants])`
Ancienne formule : `nanmean([z_shared_obj_ratio, z_mutual_gaze_ratio_mean_pairs, -1.0 × z_gaze_entropy])`

Les valeurs existantes des 11 groupes VR sont inchangees (terme retiré valait deja 0).

---

### MOD-9 — Poids gaze_entropy dans gaze_attention_coordination_idx
**Fichiers modifies :** `hlf/compute_high_level_features.py`

Ancien poids `w.gaze_entropy_penalty = -0.10` (Weights dataclass).
Nouveau poids : `-1.0` applique directement dans le recalcul MOD-9.

Rationale : en l'absence de validation empirique du poids differentiel, poids unitaires
assumes (contribution egale de chaque composante dans la moyenne).

**Variables ajoutees :**
- `gaze_attention_coordination_idx_old` : valeur avec ancien poids -0.10
- `gaze_attention_coordination_idx` : recalcule avec poids -1.0 (ecrase)

Les z-scores composants sont desormais inclus dans la sortie compacte :
- `z_shared_obj_ratio`
- `z_mutual_gaze_ratio_mean_pairs`
- `z_gaze_entropy_mean_participants`


### MOD-10 — Statistiques robustes shared_obj_dur
**Fichiers modifies :** `gaze/analyze_gaze.py`, `hlf/compute_high_level_features.py`

La fonction `shared_object_metrics` retourne desormais 4 statistiques supplementaires
sur la distribution des durees d'episodes d'attention partagee.
Reference : distribution skewed observee empiriquement (SD/mean > 0.85).

**Variables ajoutees dans tous les CSV gaze :**
- `shared_obj_dur_median_s` (overall + paires)
- `shared_obj_dur_q25_s` (overall + paires)
- `shared_obj_dur_q75_s` (overall + paires)
- `shared_obj_dur_iqr_s` (overall + paires)

Propagees dans :
- `shared_object_windows.csv`
- `shared_object_pairs_windows.csv`
- `metrics_overall.csv`
- `metrics_pairs.csv`
- `ALL_metrics_overall.csv`
- `ALL_metrics_pairs.csv`

Les colonnes sont ajoutees dans `load_gaze_group` (numeric_cols + keep) de HLF.

---

## Notes generales

- Toutes les colonnes `_old` sont permanentes dans tous les CSV de sortie.
- Tous les `print()` utilisent ASCII uniquement (contrainte Windows cp1252).
- Les assertions runtime levent `AssertionError` avec group_id et valeurs explicites.
- Les modules `common/` et `face/` n'ont pas ete modifies.
