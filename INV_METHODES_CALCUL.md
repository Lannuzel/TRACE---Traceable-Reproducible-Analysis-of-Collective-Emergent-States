# Méthodes de calcul des variables INV — état actuel

> **Source de vérité :** `scripts/config/inv_features_config.py`  
> **Dernière mise à jour :** 2026-06-24  
> **Pipeline :** MOD-1 à MOD-11 intégrés + analyse directionnelle gaze (voir `scripts/analyse_inv/CHANGELOG.md`)  
> **Périmètre :** variables INV core utilisées dans la régression stepwise et la path analysis VR

---

## Conventions transversales

| Symbole | Définition |
|---------|-----------|
| `safe_div(a,b)` | `a/b` si `b` fini et non nul, sinon `NaN` |
| `z(x)` | `(x − mean(x)) / sd(x)` avec `ddof=0` ; retourne `0.0` si `sd=0` |
| `nanmean([...])` | Moyenne ignorant les `NaN` |
| `first_valid_series([s1,s2,...])` | Premier élément de la liste dont toutes les valeurs sont non-NaN |
| `TURN_MIN_SEC` | 1.0 s (seuil minimal durée tour CA) |

---

## 1. Audio / Speech

### `audio_avg_speaking_turn_duration_s`
**Description :** Durée moyenne des tours de parole (CA)

**Algorithme (3 passes, MOD-1) :**
1. **Fusion IPU** — les IPU VAD consécutifs du même locuteur sont fusionnés en un seul tour CA ; la fermeture survient à chaque alternance de rôle.
2. **Filtre durée** — seuls les tours ≥ `TURN_MIN_SEC` (1.0 s) sont conservés.
3. **Re-fusion post-filtrage** — après filtrage, les tours adjacents du même rôle redevenus voisins (parce qu'un micro-tour séparateur a été supprimé à l'étape 2) sont à nouveau fusionnés.

```
audio_avg_speaking_turn_duration_s = mean( durée(tour_CA) )
```
sur l'ensemble des tours CA des trois rôles, agrégé au niveau groupe.

**Fichiers :** `speech/analyze_audio.py::aggregate_ipus_to_ca_turns` → `speech/compute_audio_features.py::build_features`  
**Référence :** Sacks, Schegloff & Jefferson (1974) ; Levitan & Hirschberg (2011)

---

### `audio_floor_exchange_pause_mean_s`
**Description :** Pause moyenne lors des échanges de tour

```
floor_exchange_pause_mean_s = mean( début_tour_CA_B − fin_tour_CA_A )
```

pour toutes les transitions `A → B` (changement de locuteur) où :
- le gap est ≤ `max_gap = 2.0 s` (transitions valides uniquement) ;
- les deux tours concernés sont des tours CA (≥ 1.0 s).

**Note :** depuis MOD-1, les gaps sont calculés sur les bornes des tours CA agrégés, non sur les IPU bruts.

**Fichiers :** `speech/analyze_audio.py` → `speech/compute_audio_features.py`

---

### `audio_overlap_speaking_ratio`
**Description :** Ratio de chevauchements de parole

```
audio_overlap_speaking_ratio = overlap_s / duration_s
```

où `overlap_s` est la durée totale de l'union des intervalles temporels avec ≥ 2 locuteurs actifs simultanément.

Depuis MOD-2, recalculé par décomposition occupancy (Cetin & Shriberg 2006) :

```
audio_overlap_speaking_ratio = pct_time_2_speakers + pct_time_3_speakers
```

La grille temporelle interne est à `fs_grid = 100 Hz` (paramètre de `compute_speaker_occupancy`, distinct du `fs_grid = 20 Hz` utilisé pour le gaze).

**Fichiers :** `speech/analyze_audio.py::compute_speaker_occupancy`

---

### `audio_backchannel_rate_per_min`
**Description :** Taux de backchannels (signaux d'écoute), en backchannel/min

**Détection stricte (MOD-3, 4 filtres en cascade) :**

| Filtre | Condition |
|--------|-----------|
| F1 — durée | IPU ∈ [0.10 s, 0.70 s] |
| F2 — chevauchement | overlap ≥ 100 ms avec un autre locuteur actif |
| F3 — non-continuation | aucun IPU du même rôle se terminant dans les 200 ms précédents |
| F4 — non-tour-CA | aucun tour CA du même rôle démarrant dans les 500 ms suivants |

> **Filtre F0 (F0 mélodique) omis** : prescrit par Truong & Heylen (2010) mais délibérément écarté — l'extraction frame-by-frame via `librosa.yin` multiplierait le temps de run par 3–5× sans apport méthodologique majeur.

```
audio_backchannel_rate_per_min = n_backchannels / (duration_s / 60)
```

**Fichiers :** `speech/analyze_audio.py::is_backchannel_strict` → `speech/compute_audio_features.py`  
**Référence :** Truong & Heylen (2010)

---

### `audio_successful_interruption_ratio`
**Description :** Ratio d'interruptions réussies (overlap-based)

**Détection :**
- chevauchement minimal ≥ 0.1 s entre les deux locuteurs ;
- maintien post-prise de tour ≥ 0.5 s par l'interrupteur ;
- tour résultant de durée minimale ≥ 1.0 s.

```
audio_successful_interruption_ratio =
    n_successful_interruptions / n_attempted_interruptions
```

Retourne `NaN` si `n_attempted_interruptions = 0`.

**Fichiers :** `speech/analyze_audio.py` → `speech/compute_audio_features.py`

---

### `participation_entropy`
**Description :** Entropie de participation à la parole (Shannon, bits)

Soient les durées de parole des trois rôles `v = [v_calc, v_model, v_lect]` :

```
p_i     = v_i / sum(v)
participation_entropy = − sum( p_i × ln(p_i) )
```

Valeur basse = dominance d'un rôle ; valeur haute = équilibre.

**Fichiers :** `speech/compute_audio_features.py`

---

### Variables occupancy dérivées (MOD-2)

Ces variables sont ajoutées au CSV mais non retenues dans la régression core :

| Variable | Définition |
|----------|-----------|
| `audio_pct_time_0_speakers` | fraction du temps sans locuteur actif |
| `audio_pct_time_1_speaker` | fraction du temps avec exactement 1 locuteur |
| `audio_pct_time_2_speakers` | fraction du temps avec 2 locuteurs simultanés |
| `audio_pct_time_3_speakers` | fraction du temps avec 3 locuteurs simultanés |

Contrainte : `sum(pct_time_*) = 1.0 ± 1e-3`.

---

## 2. Face / Affect

### `face_facial_synchrony`
**Description :** Synchronie faciale interpersonnelle (composite)

**Depuis MOD-5 :** source unique `au_sync_pearson_mean` (rationalisé depuis la chaîne de fallback multi-source antérieure).

```
face_facial_synchrony = au_sync_pearson_mean
```

**Opérationnalisation (Option B — synchronie par AU, puis agrégation) :**

Pour chaque paire de participants `(a, b)` et chaque métrique AU `k` parmi
`{au6_active_pct, au12_active_pct, au15_active_pct, au17_active_pct, au6_au12_coactive_pct, au15_au17_coactive_pct}` :

```
r_(ab,k) = Pearson( AU_k_a(fenêtre), AU_k_b(fenêtre) )
```

Puis agrégation sur les 6 métriques AU × 3 paires (calc–model, calc–lect, model–lect) :

```
au_sync_pearson_mean = mean( r_(ab,k) )  sur toutes les paires et métriques disponibles
```

Ce n'est **pas** une synchronie sur un signal AU moyen global (Option A), mais une corrélation par AU individuel agrégée ensuite.

La colonne `face_facial_synchrony_source` trace la provenance effective ("au_sync_pearson_mean" ou "missing"). En VR : source = `au_sync_pearson_mean` pour 100 % des groupes (audit MOD-11, 2026-05-09).

**Fichiers :** `face/analyze_aus_group.py::compute_synchrony_pearson` → `hlf/compute_high_level_features.py`  
**Référence :** Prochazkova & Kret (2017), *Connecting minds and sharing emotions*, *Neuroscience & Biobehavioral Reviews* — synchronie par-AU inter-paires

---

### `face_negative_affect_ratio`
**Description :** Ratio moyen d'affect négatif (marqueur AU15+AU17)

Le code implémente une chaîne de fallback, mais **l'audit MOD-11 (2026-05-09) confirme que 100 % des groupes VR (N=19) utilisent le niveau 1** — la chaîne de fallback ne s'active jamais en pratique.

```
face_negative_affect_ratio = au15_au17_coactive_pct_mean
```

soit la proportion moyenne de frames où AU15 et AU17 sont co-actifs simultanément.

La chaîne de fallback dans le code (pour mémoire, niveaux 2–5 jamais atteints en VR) :

```
first_valid_series([
    au15_au17_coactive_pct_mean,              ← niveau 1 — utilisé pour 19/19 groupes
    mean(au15_active_pct_mean, au17_active_pct_mean),
    sad_tri_occupancy,
    sad_active_pct_mean,
    sad_active_pct_median
])
```

**Note (MOD-6) :** AU15 (depressor anguli oris) + AU17 (chin raiser) = marqueur FACS de **tristesse**, pas d'affect négatif générique. La variable `face_sadness_marker_ratio` est un alias explicite ajouté pour les publications.

**Fichiers :** `hlf/compute_high_level_features.py`  
**Référence :** Ekman & Friesen (1978), *Facial Action Coding System*

---

### `au4_active_pct_mean` et `au4_au15_coactive_pct_mean`
**Description :** Activation du sourcil froncé (AU4) et co-activation AU4+AU15

Ces deux métriques sont calculées par `face/analyze_aus_group.py` et agrégées à l'échelle groupe (moyenne sur les participants et sur les fenêtres temporelles).

**AU4 = BrowLowerer (abaisseur des sourcils)** — Action Unit FACS impliquée dans les expressions de tristesse, de concentration et de colère.

```
au4_active_pct_mean    = proportion moyenne de frames où AU4 > seuil adaptatif
au4_au15_coactive_pct_mean = proportion moyenne de frames où AU4 et AU15 sont co-actifs
```

AU4 intervient également dans la variable intermédiaire `sad_intensity` (non exportée directement dans les CSV HLF), définie comme :

```
sad_intensity = mean(AU1, AU4, AU15)
             = mean(InnerBrowRaiser, BrowLowerer, LipCornerDepressor)
```

Cette variable intermédiaire sert de signal d'intensité tristesse avant binarisation par seuil adaptatif (même logique que `joy_intensity = mean(AU6, AU12)`). Elle alimente `sad_active_pct` et les métriques de synchronie `sad_sync_jaccard_mean`.

La co-activation `au4_au15_coactive` (BrowLowerer + LipCornerDepressor) est un marqueur FACS de tristesse complémentaire à `au15_au17_coactive` (utilisé dans `face_negative_affect_ratio`).

**Tableau des Action Units utilisées dans le pipeline :**

| AU | Nom FACS | Muscle | Rôle dans le pipeline |
|----|----------|--------|-----------------------|
| AU1 | InnerBrowRaiser | Frontal (pars medialis) | Composant de `sad_intensity` |
| AU4 | BrowLowerer | Corrugateur du sourcil | `au4_active_pct_mean`, `au4_au15_coactive_pct_mean`, `sad_intensity` |
| AU6 | CheekRaiser | Orbicularis oculi | Composant de `joy_intensity`, `face_smile_ratio` |
| AU12 | LipCornerPuller | Zygomaticus major | Composant de `joy_intensity`, `face_smile_ratio` |
| AU15 | LipCornerDepressor | Depressor anguli oris | `face_negative_affect_ratio`, `au4_au15_coactive_pct_mean`, `sad_intensity` |
| AU17 | ChinRaiser | Mentalis | `face_negative_affect_ratio` |

**Fichiers :** `face/analyze_aus_group.py` (lignes 101, 109, 132, 143, 211, 225, 229)  
**Référence :** Ekman & Friesen (1978), *Facial Action Coding System*

---

### `face_smile_ratio`
**Description :** Ratio moyen de sourire (Duchenne)

Le code implémente une chaîne de fallback, mais **l'audit MOD-11 (2026-05-09) confirme que 100 % des groupes VR (N=19) utilisent le niveau 1** — la chaîne de fallback ne s'active jamais en pratique.

```
face_smile_ratio = au6_au12_coactive_pct_mean
```

La chaîne de fallback dans le code (pour mémoire, niveaux 2–5 jamais atteints en VR) :

```
first_valid_series([
    au6_au12_coactive_pct_mean,               ← niveau 1 — utilisé pour 19/19 groupes
    mean(au6_active_pct_mean, au12_active_pct_mean),
    joy_tri_occupancy,
    joy_active_pct_mean,
    joy_active_pct_median
])
```

Dans le meilleur cas :

```
face_smile_ratio = proportion de frames où AU6 et AU12 sont co-actifs
```

**Note (MOD-7) :** AU6 (orbicularis oculi) + AU12 (zygomaticus major) = sourire de Duchenne (sourire sincère). Exclut le sourire social (AU12 seul).

**Fichiers :** `hlf/compute_high_level_features.py`  
**Référence :** Ekman, Davidson & Friesen (1990), JPSP

---

### `affect_alignment_idx`
**Description :** Alignement affectif entre participants (composite)

```
affect_alignment_idx =
    nanmean([
        + z_joy_tri_occupancy,
        + z_joy_sync_jaccard_mean,
        − z_sad_tri_occupancy,
        − z_sad_sync_jaccard_mean
    ])
```

où chaque `z_*` est le z-score de la variable correspondante calculé sur l'échantillon courant.

**Note (MOD-8) :** les 4 z-scores composants (`z_joy_tri_occupancy`, `z_joy_sync_jaccard_mean`, `z_sad_tri_occupancy`, `z_sad_sync_jaccard_mean`) sont inclus dans le CSV de sortie.

**Fichiers :** `hlf/compute_high_level_features.py`

---

## 3. Gaze

> **Note pipeline :** depuis MOD-11 (analyse directionnelle), les variables canoniques sont `gaze_convergence_ratio` et `gaze_entropy_dir_mean`. Les anciennes variables (`shared_obj_ratio`, `gaze_entropy_mean_participants`) sont conservées en legacy dans les CSV pour traçabilité et backward compatibility.

### `gaze_convergence_ratio` *(canonique)*
**Description :** Ratio de temps passé en convergence visuelle directionnelle

Calculé par `analyze_gaze_directional.py` à partir des angles de regard inter-participants. Remplace `shared_obj_ratio` comme source principale dans le HLF.

**Fichiers :** `gaze/analyze_gaze_directional.py`

---

### `shared_obj_ratio` *(legacy — fallback si analyse directionnelle absente)*
**Description :** Ratio de temps passé en co-fixation sur un objet partagé

Grille temporelle à `fs_grid = 20 Hz` :

```
sameobj(t) = 1  si ≥ 2 participants fixent simultanément le même objet
sameobj(t) = 0  sinon
```

Après suppression des micro-épisodes < 0.10 s :

```
shared_obj_ratio = mean_t( sameobj(t) )
```

**Note (MOD-10) :** 4 statistiques robustes dans le CSV (`shared_obj_dur_median_s`, `_q25_s`, `_q75_s`, `_iqr_s`) — non retenues dans la régression core.

**Fichiers :** `gaze/analyze_gaze.py::shared_object_metrics`

---

### `gaze_entropy_dir_mean` *(canonique)*
**Description :** Entropie directionnelle moyenne du regard (Shannon)

Calculée par `analyze_gaze_directional.py` sur la distribution des directions de regard. Remplace `gaze_entropy_mean_participants` comme source principale.

Un proxy dérivé est aussi calculé : `gaze_focus_proxy = 1.0 − gaze_entropy_dir_mean`.

**Fichiers :** `gaze/analyze_gaze_directional.py`

---

### `gaze_entropy_mean_participants` *(legacy — fallback)*
**Description :** Entropie moyenne du regard par objet fixé (Shannon, normalisée)

Pour chaque participant :

```
p_j = dur_fixation_objet_j / sum_j( dur_fixation_objet_j )
H   = − sum_j( p_j × log2(p_j) )
H_normalise = H / log2( N_objets_fixés )
```

Puis au niveau groupe :

```
gaze_entropy_mean_participants = mean( H_normalise )  sur les participants
```

Valeur haute = regard dispersé ; valeur basse = regard focalisé.

**Fichiers :** `gaze/analyze_gaze.py`

---

### `gaze_attention_coordination_idx`
**Description :** Coordination de l'attention (composite gaze)

Formule canonique (MOD-11 — analyse directionnelle) :

```
gaze_attention_coordination_idx =
    nanmean([
        z_gaze_convergence_ratio,          ← priorité 1
        −1.0 × z_gaze_entropy_dir_mean     ← priorité 1
    ])
```

Fallback si colonnes directionnelles absentes :

```
gaze_attention_coordination_idx =
    nanmean([
        z_shared_obj_ratio,
        −1.0 × z_gaze_entropy_mean_participants
    ])
```

Le signe négatif de l'entropie reflète qu'un regard plus focalisé indique une meilleure coordination.

**Note (MOD-9) :** poids entropie = −0.10 → −1.0 (poids unitaires).  
**Note (MOD-10) :** `z_mutual_gaze_ratio_mean_pairs` retiré (variance nulle sur groupes VR).  
**Note (MOD-11) :** passage aux colonnes directionnelles (`gaze_convergence_ratio`, `gaze_entropy_dir_mean`). L'ancienne formule legacy est conservée dans `gaze_attention_coordination_idx_old`.

**Fichiers :** `hlf/compute_high_level_features.py::compute_composites`

---

### `gaze_joint_attention_idx_raw`
**Description :** Indice d'attention conjointe brut (convergence + regard mutuel)

```
gaze_joint_attention_idx_raw = (gaze_convergence_ratio + mutual_gaze_ratio) / 2
```

Fallback : `(shared_obj_ratio + mutual_gaze_ratio_mean_pairs) / 2`

**Fichiers :** `hlf/compute_high_level_features.py::compute_composites`

---

### `gaze_mutual_gaze_ratio`
**Description :** Ratio global de regard mutuel

Chaîne de fallback (priorité décroissante) :

```
first_valid_series([
    mutual_gaze_ratio,            ← analyse directionnelle (canonique)
    mutual_gaze_ratio_mean_pairs, ← legacy (paires dyadiques)
    pair_mutual_gaze_ratio_mean,
    mutual_gaze_dur_total_ratio_ref
])
```

Variable VR-only. La colonne `gaze_mutual_gaze_ratio_source` trace la provenance.

**Fichiers :** `hlf/compute_high_level_features.py::add_final_feature_columns`

---

### `gaze_shared_visual_attention_ratio`
**Description :** Ratio d'attention visuelle partagée (alias HLF)

Chaîne de fallback :

```
first_valid_series([
    gaze_convergence_ratio,          ← canonique (directionnelle)
    pair_convergence_ratio_mean,
    shared_obj_ratio,                ← legacy
    shared_obj_dur_total_ratio_ref,
    pair_shared_obj_ratio_mean
])
```

**Fichiers :** `hlf/compute_high_level_features.py::add_final_feature_columns`

---

## 4. Variables non retenues dans la régression core (référence)

Ces variables sont calculées et présentes dans les CSV mais exclues du core set analytique (pruning |r| > 0.80 ou flag `drop_if_redundant`).

| Variable | Redondante avec | Raison |
|----------|----------------|--------|
| `audio_overlap_takeover_ratio` | `audio_successful_interruption_ratio` | alias renommage MOD-4 |
| `audio_total_speaking_turns` | — | compte absolu, non normalisé |
| `gaze_entropy` | `gaze_entropy_dir_mean` (ou `gaze_entropy_mean_participants` legacy) | alias `first_valid_series` |
| `gaze_shared_visual_attention_ratio` | `gaze_convergence_ratio` (ou `shared_obj_ratio` legacy) | composite HLF, `\|r\| ≈ 1.0` |
| `shared_obj_ratio` | `gaze_convergence_ratio` | variable legacy, remplacée par analyse directionnelle |
| `gaze_entropy_mean_participants` | `gaze_entropy_dir_mean` | variable legacy, remplacée par analyse directionnelle |
| `face_sync_pearson_global_idx` | `face_facial_synchrony` | redondance empirique attendue |
| `affect_balance_occ` | `face_smile_ratio` | `\|r\| ≈ 0.99` |
| `face_sadness_marker_ratio` | `face_negative_affect_ratio` | alias explicite MOD-6 |
| `shared_obj_dur_median_s` / `_q25` / `_q75` / `_iqr` | `shared_obj_dur_mean_s` | statistiques robustes ajoutées MOD-10, non core |

---

## 5. Fichiers de référence

| Fichier | Rôle |
|---------|------|
| `scripts/config/inv_features_config.py` | Source de vérité — définitions, flags, descriptions, `calc_method` |
| `scripts/analyse_inv/speech/analyze_audio.py` | Calculs audio bas niveau (tours CA, occupancy, backchannels, interruptions) |
| `scripts/analyse_inv/speech/compute_audio_features.py` | Agrégation groupe, aliases canoniques |
| `scripts/analyse_inv/gaze/analyze_gaze.py` | Calculs gaze (entropie, co-fixation, regard mutuel) |
| `scripts/analyse_inv/hlf/compute_high_level_features.py` | Composites face et gaze (synchronie, affect, coordination) |
| `scripts/analyse_inv/CHANGELOG.md` | Historique MOD-1 à MOD-11 avec références |
