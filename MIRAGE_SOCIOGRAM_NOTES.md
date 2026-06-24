# MIRAGE_SOCIOGRAM_NOTES.md

> Notes techniques sur l'adaptation du sociogramme MIRAGE au projet Longitudinale.
> Ce fichier documente les choix de conception, les adaptations effectuées et les limitations connues.

Dernière mise à jour : 2026-03-31

---

## 1. Contexte et inspiration

Le sociogramme MIRAGE (Multimodal Interaction Representation And Group Evaluation) est une visualisation
temps réel développée dans le cadre d'une thèse sur l'évaluation automatique de la collaboration (repo SaacPSI/saac).

Caractéristiques originales MIRAGE :
- Fenêtre glissante de 20 s, mise à jour toutes les 1 s — plugin PsiStudio (C# / WPF)
- Noeuds = individus, taille proportionnelle au speaking time
- Arêtes dyadiques : speech equality (orange), gaze on peers (vert + flèches), JVA (cyan), synchrony (capsule rose)
- Panneau groupe : leaders par dimension + scores [0,1]

Classes SAAC utilisées comme référence :
- `PersonNode` : Id, SpeakingTime, MovementScore, TaskParticipation
- `PersonEdge` : Proximity, Synchrony, GazeOnPeers12/21, JVAEvent/Intensity, SpeechEquality
- `PersonGroup` : Dominance_score, Ja_score, Engagement_score, Collaboration_score

Le script `scripts/visualisation_sociale/mirage_sociogram.py` est une **reproduction offline** en Python/Matplotlib,
inspirée de MIRAGE mais adaptée aux données et au pipeline du projet. Ce n'est pas un clone temps réel.

---

## 2. Adaptations des couches visuelles

| Couche MIRAGE | Adaptation projet | Source données | Disponibilité |
|---|---|---|---|
| Taille noeud ∝ speaking time | `speech_{role}_ratio` dans la fenêtre (VAD locale) | WAV bruts | PC + VR |
| Halo noeud = task focus | `task_focus_ratio` = fixations sur objets / durée fenêtre | EyeTrackingData | VR uniquement |
| Arête orange = speech equality | `1 - |da - db| / (da + db)` par paire | VAD locale | PC + VR |
| Arête verte + flèches = gaze on peers | Distance hit-point → position pair < 0.45 m | EyeTrackingData + UsersPositions | VR uniquement |
| Courbe bleue = JVA | `shared_object_metrics()` : ratio fixations partagées par paire | EyeTrackingData | VR uniquement |
| Capsule rose = synchrony | Pearson(joy_intensity, sad_intensity) entre rôles, normalisé [0,1] | FaceTrackingData | PC + VR |
| Layout triangle | Distances VR compressées ; triangle fixe si PC | UsersPositions | VR : dynamique / PC : fixe |
| % parole dans noeud | `speaking_ratio_window` affiché en texte blanc dans le cercle | VAD locale | PC + VR |

---

## 3. Scores groupe (panneau bas-gauche)

| Score | Calcul |
|---|---|
| `speech_balance_score` | `mean(1 - |da-db|/(da+db))` sur les 3 paires |
| `joint_attention_score` | Moyenne JVA ratio sur les 3 paires |
| `engagement_score` | Moyenne(task_focus, regarded_ratio normalisé) par rôle, puis groupe |
| `proximity_score` | Normalisé (1 - distance) relativement au min/max de la session |
| `face_synchrony_score` | Moyenne Pearson facial par paire |
| `collaboration_proxy_score` | Moyenne des 5 composantes ci-dessus |

Leaders affichés avec carré coloré ■ (couleur du rôle) :
- **Parle le plus** : rôle avec le plus de parole dans la fenêtre
- **Reçu regard max** : rôle vers lequel les autres ont le plus regardé (VR)
- **Focus tâche max** : rôle avec le plus de fixations sur objets (VR)

---

## 4. Arguments CLI principaux

```powershell
# Snapshot statique
python scripts/visualisation_sociale/mirage_sociogram.py \
  --group-id bim073 --modality VR --scenario S2 --timepoint T1 \
  --data-dir D:/data_e2 --out-dir D:/Analyse_donnee/Longitudinale/results/visualisation_sociale/bim073 \
  --snapshot-at 120 --debug-timings

# Animation GIF allégée (1 frame / 10)
python scripts/visualisation_sociale/mirage_sociogram.py \
  --group-id bim073 --modality VR --scenario S2 --timepoint T1 \
  --data-dir D:/data_e2 --out-dir D:/Analyse_donnee/Longitudinale/results/visualisation_sociale/bim073 \
  --export-gif --frame-stride 10 --frame-dpi 90

# Lecture interactive
python scripts/visualisation_sociale/mirage_sociogram.py \
  --group-id bim073 --modality VR --scenario S2 --timepoint T1 \
  --data-dir D:/data_e2 --show-live --playback-speed 2
```

| Argument | Défaut | Description |
|---|---|---|
| `--window-s` | 20 | Taille fenêtre glissante (s) |
| `--step-s` | 1 | Pas fenêtre (s) |
| `--snapshot-at` | médiane | Instant du snapshot statique (s) |
| `--export-frames` | off | Exporte toutes les frames PNG |
| `--export-gif` | off | Exporte l'animation GIF (Pillow requis) |
| `--frame-stride` | 1 | Sous-échantillonnage GIF (10 = 1 frame/10) |
| `--frame-dpi` | 110 | Résolution des frames |
| `--show-live` | off | Fenêtre Matplotlib interactive |
| `--playback-speed` | 1.0 | Vitesse lecture live |
| `--debug-timings` | off | Timings de chargement |
| `--max-windows` | — | Limite le nombre de fenêtres calculées |
| `--start-at` / `--end-at` | — | Bornes optionnelles de l'analyse |

---

## 5. Performances mesurées (bim073, VR, S2, T1)

| Opération | Durée |
|---|---|
| Chargement audio (3 pistes, VAD rapide) | ~3 s |
| Chargement face (FaceTrackingData) | ~15 s |
| Chargement gaze / positions / fixations | ~7 s |
| Calcul 1 571 fenêtres (step=1 s, window=20 s) | ~90 s |
| Export snapshot PNG | ~1 s |
| Export 157 frames PNG avec stride=10, dpi=90 | ~55 s |
| Export GIF depuis 157 frames | ~5 s |

---

## 6. Limitations connues

- **Gaze, JVA, positions** : VR uniquement. Couches désactivées automatiquement sur PC.
- **VAD locale** : basée sur RMS + seuil adaptatif (rapide mais moins précise que Whisper).
- **Synchronie faciale** : joy/sad uniquement. Pas de synchronie de mouvement (IMU non disponible).
- **Offline uniquement** : pas de mise à jour temps réel. Fenêtre glissante précalculée.
- **Layout PC** : triangle de taille fixe, distances non réelles.
- **Pillow requis pour GIF** : `pip install Pillow`.
