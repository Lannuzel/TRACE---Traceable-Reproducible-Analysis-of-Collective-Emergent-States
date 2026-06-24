"""
Construction des deux schemas pour la conception du feedback VR :
1. Carte empirique IMOI (C-factor + Riedl -> INV / etats emergents -> performance)
2. Cahier de charges feedback VR temps reel
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# Couleurs (alignees sur la palette du systeme)
C_INPUT = "#EEEDFE"; C_INPUT_E = "#534AB7"; C_INPUT_T = "#26215C"
C_INV   = "#E1F5EE"; C_INV_E   = "#1D9E75"; C_INV_T   = "#04342C"
C_EMERG = "#FAEEDA"; C_EMERG_E = "#BA7517"; C_EMERG_T = "#412402"
C_PERF  = "#F1EFE8"; C_PERF_E  = "#5F5E5A"; C_PERF_T  = "#2C2C2A"
C_FB    = "#E6F1FB"; C_FB_E    = "#185FA5"; C_FB_T    = "#042C53"
C_POS   = "#3B6D11"
C_NEG   = "#993C1D"
C_WEAK  = "#888780"
C_BOX_LIGHT = "#F8F7F2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.linewidth": 0.0,
})


def fancy_box(ax, x, y, w, h, fc, ec, lw=0.7, rad=0.04):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)


def arrow(ax, x1, y1, x2, y2, color, lw=1.5, style="-",
          connectionstyle="arc3,rad=0", head=8):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=f"-|>,head_length={head*0.5},head_width={head*0.2}",
                        mutation_scale=head*1.1,
                        color=color, linewidth=lw,
                        linestyle=style,
                        connectionstyle=connectionstyle,
                        zorder=3, capstyle="round", joinstyle="round")
    ax.add_patch(a)


# =================================================================
# SCHEMA 1 : CARTE EMPIRIQUE IMOI
# =================================================================
def schema1():
    fig, ax = plt.subplots(figsize=(11, 8.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ----- Tier 1 : INPUT -----
    ax.text(2, 95, "Niveau 1 — Capital cognitif collectif (Input)",
            fontsize=10, style="italic", color="#5F5E5A")

    fancy_box(ax, 23, 84, 22, 8, C_INPUT, C_INPUT_E)
    ax.text(34, 90, "C-factor / RME", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_INPUT_T)
    ax.text(34, 86.6, "c_score, rme_max", ha="center", va="center",
            fontsize=8.5, color="#3C3489")

    fancy_box(ax, 50, 84, 27, 8, C_INPUT, C_INPUT_E)
    ax.text(63.5, 90, "Indicateurs Riedl", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_INPUT_T)
    ax.text(63.5, 86.6, "skill, strategy, contribution, effort", ha="center", va="center",
            fontsize=8.5, color="#3C3489")

    # Down dashed arrows from input toward mediator block
    arrow(ax, 34, 84, 34, 78, C_WEAK, lw=1, style=(0, (3, 3)))
    arrow(ax, 63.5, 84, 63.5, 78, C_WEAK, lw=1, style=(0, (3, 3)))

    # ----- Tier 2 : Mediator container -----
    ax.text(2, 76.5, "Niveau 2 — Mediateurs : INV ↔ etats emergents",
            fontsize=10, style="italic", color="#5F5E5A")

    container = FancyBboxPatch((2, 32), 96, 43,
                               boxstyle="round,pad=0,rounding_size=0.7",
                               linewidth=0.7, edgecolor="#B4B2A9",
                               facecolor="none", linestyle=(0, (5, 4)), zorder=1)
    ax.add_patch(container)

    # Sous-region INV
    ax.text(4, 71.5, "INV  (mesurables en temps reel)",
            fontsize=9.5, fontweight="bold", color=C_INV_E)
    # Audio
    fancy_box(ax, 5, 60, 24, 9, C_INV, C_INV_E)
    ax.text(17, 65.5, "Audio / Speech", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_INV_T)
    ax.text(17, 62, "tours, interruptions, pauses", ha="center", va="center",
            fontsize=8.5, color="#0F6E56")
    # Regard
    fancy_box(ax, 38, 60, 24, 9, C_INV, C_INV_E)
    ax.text(50, 65.5, "Regard / Gaze", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_INV_T)
    ax.text(50, 62, "attention coord., entropie", ha="center", va="center",
            fontsize=8.5, color="#0F6E56")
    # Face
    fancy_box(ax, 71, 60, 24, 9, C_INV, C_INV_E)
    ax.text(83, 65.5, "Face / FACS", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_INV_T)
    ax.text(83, 62, "affect negatif, sourire", ha="center", va="center",
            fontsize=8.5, color="#0F6E56")

    # Sous-region etats emergents
    ax.text(4, 51, "Etats emergents  (declaratifs questionnaire)",
            fontsize=9.5, fontweight="bold", color=C_EMERG_E)
    # COR
    fancy_box(ax, 5, 39, 17, 8, C_EMERG, C_EMERG_E)
    ax.text(13.5, 44.5, "COR", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_EMERG_T)
    ax.text(13.5, 41.3, "coordination", ha="center", va="center",
            fontsize=8.5, color="#854F0B")
    # CRE
    fancy_box(ax, 27, 39, 17, 8, C_EMERG, C_EMERG_E)
    ax.text(35.5, 44.5, "CRE", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_EMERG_T)
    ax.text(35.5, 41.3, "credibilite", ha="center", va="center",
            fontsize=8.5, color="#854F0B")
    # SPE
    fancy_box(ax, 49, 39, 17, 8, C_EMERG, C_EMERG_E)
    ax.text(57.5, 44.5, "SPE", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_EMERG_T)
    ax.text(57.5, 41.3, "specialisation", ha="center", va="center",
            fontsize=8.5, color="#854F0B")
    # Cohesion
    fancy_box(ax, 71, 39, 24, 8, C_EMERG, C_EMERG_E)
    ax.text(83, 44.5, "Cohesion", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_EMERG_T)
    ax.text(83, 41.3, "SOC, TSK, COM", ha="center", va="center",
            fontsize=8.5, color="#854F0B")

    # ----- Arrows : INV -> Etats emergents (within mediator container) -----
    # Audio -> COR (negative beta=-0.82)
    arrow(ax, 13.5, 60, 13.5, 47, C_NEG, lw=2.4)
    ax.text(15, 53, r"$\beta=-.82$", color=C_NEG, fontsize=9, fontweight="bold")
    ax.text(15, 51.2, "interrupt. → COR", color=C_NEG, fontsize=8)

    # Audio -> CRE (negative beta=-0.78)
    arrow(ax, 22, 60, 32, 47, C_NEG, lw=2.4, connectionstyle="arc3,rad=0.18")
    ax.text(22, 49, r"$-.78$", color=C_NEG, fontsize=9, fontweight="bold")

    # Regard -> SPE (positive rho=+0.65 / +0.76)
    arrow(ax, 53, 60, 56, 47, C_POS, lw=2.0)
    ax.text(40.5, 54, r"$\rho=+.65$", color=C_POS, fontsize=9, fontweight="bold")
    ax.text(40.5, 52.2, "entropie → SPE", color=C_POS, fontsize=8)

    # Face -> Cohesion (positive rho=+0.73)
    arrow(ax, 81, 60, 81, 47, C_POS, lw=2.6)
    ax.text(64, 53.7, r"$\rho=+.73$", color=C_POS, fontsize=9, fontweight="bold")
    ax.text(64, 52, "affect− → cohesion", color=C_POS, fontsize=8)

    # Face -> CRE (positive rho=+0.62)
    arrow(ax, 73, 60, 42, 47, C_POS, lw=1.5, connectionstyle="arc3,rad=0.15")
    ax.text(50, 56, r"$+.62$ (CRE)", color=C_POS, fontsize=8.5)

    # ----- Counter-intuitive INPUT -> INV : c_score -> attention coord. -----
    arrow(ax, 34, 84, 50, 69, C_NEG, lw=2.6,
          connectionstyle="arc3,rad=0.25")
    cx, cy = 13.5, 78
    fancy_box(ax, 5, 75.5, 17, 4.7, "#FAECE7", C_NEG, lw=0.5)
    ax.text(cx, 79, r"⚠ $\rho=-.88$ (contre-intuitif)",
            ha="center", va="center", fontsize=8.5, color=C_NEG, fontweight="bold")
    ax.text(cx, 76.8, "c_score → attention coord.",
            ha="center", va="center", fontsize=8, color=C_NEG)
    ax.plot([22, 38], [77.5, 71], color=C_NEG, lw=0.6, linestyle=":", zorder=2)

    # ----- Tier 3 : Performance -----
    ax.text(2, 27.5, "Niveau 3 — Sortie : performance objective",
            fontsize=10, style="italic", color="#5F5E5A")

    fancy_box(ax, 35, 16, 30, 9, C_PERF, C_PERF_E)
    ax.text(50, 21.5, "Performance", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_PERF_T)
    ax.text(50, 18, "Score_perf_tsk", ha="center", va="center",
            fontsize=9, color="#444441")

    # ----- INV -> Performance bypass -----
    # Audio (turn duration) -> performance, beta = +0.91
    arrow(ax, 16, 60, 38, 25, C_POS, lw=2.8,
          connectionstyle="arc3,rad=-0.35")
    ax.text(2, 30, r"$\beta=+.91$", color=C_POS, fontsize=9.5, fontweight="bold")
    ax.text(2, 28, "turn duration", color=C_POS, fontsize=8.5)

    # Regard (attention coord) -> performance, beta = -0.71
    arrow(ax, 50, 60, 50, 25, C_NEG, lw=2.8,
          connectionstyle="arc3,rad=0")
    ax.text(51.5, 33, r"$\beta=-.71$", color=C_NEG, fontsize=9.5, fontweight="bold")
    ax.text(51.5, 31, "attention coord.", color=C_NEG, fontsize=8.5)

    # ----- Etats -> Performance (dashed weak) -----
    for x_start, x_end in [(13.5, 42), (35.5, 45), (57.5, 55), (83, 60)]:
        arrow(ax, x_start, 39, x_end, 25, C_WEAK, lw=1, style=(0, (3, 3)),
              head=5)
    ax.text(70, 30, r"$\beta \approx 0.0$–$.26$", color=C_WEAK, fontsize=8.5)
    ax.text(70, 28, "(faible)", color=C_WEAK, fontsize=8.5)

    # ----- INPUT -> Performance direct -----
    arrow(ax, 77, 88, 65, 21, C_POS, lw=1.5, style=(0, (5, 3)),
          connectionstyle="arc3,rad=-0.55", head=7)
    ax.text(96, 53, r"$\beta_{\mathrm{direct}}=+.52$", color=C_POS, fontsize=9,
            ha="right", fontweight="bold")
    ax.text(96, 51, "(input → performance)", color=C_POS, fontsize=8.5,
            ha="right")

    # ----- Legend -----
    fancy_box(ax, 2, 1.5, 96, 11, C_BOX_LIGHT, "#D3D1C7", lw=0.6, rad=0.2)
    ax.text(4, 10.5, "Legende", fontsize=10, fontweight="bold", color="#2C2C2A")
    # Positive line
    ax.plot([4, 9], [7.5, 7.5], color=C_POS, lw=2.5)
    ax.text(10, 7.5, "Correlation positive (vert)", fontsize=8.8, va="center",
            color="#444441")
    # Negative line
    ax.plot([34, 39], [7.5, 7.5], color=C_NEG, lw=2.5)
    ax.text(40, 7.5, "Correlation negative (coral)", fontsize=8.8, va="center",
            color="#444441")
    # Weak/dashed
    ax.plot([66, 71], [7.5, 7.5], color=C_WEAK, lw=1.2, linestyle=(0, (3, 3)))
    ax.text(72, 7.5, "Lien faible / direct mediator", fontsize=8.8, va="center",
            color="#444441")
    ax.text(4, 4.8, "Epaisseur de trait : magnitude (|"+r"$\rho$"+r"| ou |$\beta_{\mathrm{std}}$|).",
            fontsize=8.6, color="#444441")
    ax.text(4, 2.7, "Etiquettes : valeurs empiriques, niveau groupe (n = 8–12, sous-echantillon TCI/perf., VR).",
            fontsize=8.6, color="#444441")

    plt.tight_layout()
    return fig


# =================================================================
# SCHEMA 2 : CAHIER DE CHARGES FEEDBACK VR
# =================================================================
def schema2():
    fig, ax = plt.subplots(figsize=(12.5, 11.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(2, 97.5, "Feedback VR temps reel — translation INV → signal de feedback",
            fontsize=13, fontweight="bold", color="#2C2C2A")
    ax.text(2, 95.2,
            "Pour chaque INV mesurable en continu sur Meta Quest Pro : sens de l'effet "
            "empirique sur performance/etats emergents et signal de feedback induit.",
            fontsize=9.2, style="italic", color="#5F5E5A")

    # Headers
    headers = [
        (2, 28, "INV mesure", C_INV, C_INV_E, C_INV_T),
        (32, 36, "Effet empirique observe", C_EMERG, C_EMERG_E, C_EMERG_T),
        (70, 28, "Feedback VR recommande", C_FB, C_FB_E, C_FB_T),
    ]
    for x, w, label, fc, ec, tc in headers:
        fancy_box(ax, x, 90.5, w, 3, fc, ec, lw=0.6)
        ax.text(x + w / 2, 92, label, ha="center", va="center",
                fontsize=10.2, fontweight="bold", color=tc)

    # Rows : (y_top, height, var_name, var_sub, effect_lines, feedback_lines)
    rows = [
        # Row 1 : turn duration
        (
            85.5, 7,
            "Duree moyenne tour",
            "audio_avg_speaking_turn_duration_s\nVAD audio, fenetre 30 s",
            [(C_POS, r"Performance ↑ : $\beta=+.91$ (M2)"),
             ("#854F0B", "Tours longs ⇒ argumentation developpee,"),
             ("#854F0B", "faible fragmentation cognitive du raisonnement")],
            [(C_FB_T, "Sens souhaite : ↑ tours longs", True),
             ("#0C447C", "Si tours < 1.5 s soutenus :", False),
             ("#0C447C", "badge soft \"developpez votre point\"", False),
             ("#0C447C", "ou jauge \"fluidite cognitive\" affichee", False)]
        ),
        # Row 2 : interruptions
        (
            77, 7.5,
            "Ratio interruptions",
            "audio_successful_interruption_ratio\noverlap-based (0.1 s / 0.5 s)",
            [(C_NEG, r"COR ↓ : $\beta=-.82$ (M4)"),
             (C_NEG, r"CRE ↓ : $\beta=-.78$ (M6)"),
             ("#854F0B", "Interruptions reussies ⇒ erosion forte"),
             ("#854F0B", "de la coordination perçue (TMS)")],
            [(C_FB_T, "Sens souhaite : ↓ interruptions", True),
             ("#0C447C", "Halo ambient ou pulse visuel discret", False),
             ("#0C447C", "si ratio > 0.55 sur 60 s ; jauge", False),
             ("#0C447C", "\"tour de parole\" partagee entre roles", False)]
        ),
        # Row 3 : pause floor exchange
        (
            68, 6.5,
            "Pause echange floor",
            "audio_floor_exchange_pause_mean_s\nsilence inter-locuteur",
            [(C_POS, r"CRE ↑ : $\beta=+.53$ (M6, conjoint avec interrupt.)"),
             ("#854F0B", "Pauses mesurees ⇒ tour-taking respectueux,"),
             ("#854F0B", "credibilite mutuelle")],
            [(C_FB_T, "Sens souhaite : pauses moderees", True),
             ("#0C447C", "Co-evalue avec les interruptions ;", False),
             ("#0C447C", "indicateur composite de \"respiration\"", False)]
        ),
        # Row 4 : participation entropy
        (
            60.5, 6.5,
            "Entropie de participation",
            "participation_entropy\nShannon sur tps de parole/role",
            [(C_NEG, r"TSK ↓ : $\beta=-.68$ (M8)"),
             ("#854F0B", "Equilibre parfait ⇏ cohesion-tache ;"),
             ("#854F0B", "leader emerge utile pour la tache BIM")],
            [(C_FB_T, "Sens souhaite : ↓ equilibre parfait", True),
             ("#0C447C", "Eviter le feedback \"egalisation\" ;", False),
             ("#0C447C", "laisser le calculateur prendre la main", False)]
        ),
        # Row 5 : attention coordination (key)
        (
            52, 8.5,
            "Coord. d'attention",
            "gaze_attention_coordination_idx\ncomposite : shared_obj − entropie",
            [(C_NEG, r"⚠ Performance ↓ : $\beta=-.71$ (M1)"),
             (C_NEG, r"⚠ c_score : $\rho=-.88$"),
             ("#854F0B", "Contre-intuitif : forte coord. ↔ verification"),
             ("#854F0B", "mutuelle excessive, specialisation des roles"),
             ("#854F0B", "(Calc/Mod/Lect) non assumee")],
            [(C_FB_T, "Sens souhaite : ↓ coord. (paradoxe)", True),
             ("#0C447C", "Encourager la specialisation des roles", False),
             ("#0C447C", "(highlights par role, vues asymetriques)", False),
             ("#0C447C", "Eviter \"regardez le meme objet\" globalement", False)]
        ),
        # Row 6 : gaze entropy
        (
            42, 6.5,
            "Entropie du regard",
            "gaze_entropy_mean_participants\nShannon sur AOI (~20 Hz)",
            [(C_POS, r"SPE ↑ : $\rho=+.65$ *"),
             (C_POS, r"Performance ↑ : $\rho=+.57$ (p=.055)"),
             ("#854F0B", "Diversification visuelle ⇒ specialisation")],
            [(C_FB_T, "Sens souhaite : ↑ entropie", True),
             ("#0C447C", "Ne pas brider l'exploration ; laisser", False),
             ("#0C447C", "chaque role scruter ses zones propres", False)]
        ),
        # Row 7 : shared object episode rate
        (
            33.5, 6.5,
            "Episodes objet partage",
            "shared_obj_episode_rate_per_min\nco-fixation ≥ 2 participants",
            [(C_POS, r"SPE ↑ : $\rho=+.76$ **"),
             ("#854F0B", "Episodes brefs et frequents : \"checks\""),
             ("#854F0B", "cibles, pas verification continue")],
            [(C_FB_T, "Sens souhaite : ↑ frequence d'episodes", True),
             ("#0C447C", "Highlight ephemere des objets co-fixes", False),
             ("#0C447C", "(150–300 ms) pour favoriser checks", False)]
        ),
        # Row 8 : negative affect
        (
            24, 8,
            "Affect negatif",
            "face_negative_affect_ratio\nAU15 + AU17 co-actifs\n⚠ occlusion HMD : signal partiel",
            [(C_POS, r"Cohesion ↑ : $\rho=+.73$ **"),
             (C_POS, r"CRE ↑ : $\rho=+.62$ *"),
             ("#854F0B", "Reinterpretation : marqueurs de"),
             ("#854F0B", "concentration / engagement cognitif,"),
             ("#854F0B", "non d'affect negatif au sens trivial")],
            [(C_FB_T, "Pas de feedback direct", True),
             ("#0C447C", "Variable confondue avec effort cognitif", False),
             ("#0C447C", "sous HMD ; utiliser comme moderateur,", False),
             ("#0C447C", "pas comme cible d'intervention", False)]
        ),
    ]

    for y_top, h, name, sub, effects, feedbacks in rows:
        # INV col
        fancy_box(ax, 2, y_top - h, 28, h, C_INV, C_INV_E, lw=0.5)
        ax.text(3.5, y_top - 1.2, name, fontsize=10, fontweight="bold",
                color=C_INV_T)
        for i, line in enumerate(sub.split("\n")):
            ax.text(3.5, y_top - 2.6 - 1.5 * i, line, fontsize=8, color="#0F6E56")

        # Effect col
        fancy_box(ax, 32, y_top - h, 36, h, C_EMERG, C_EMERG_E, lw=0.4)
        for i, (color, text) in enumerate(effects):
            fontweight = "bold" if color in (C_POS, C_NEG) else "normal"
            fontsize = 8.5 if color in (C_POS, C_NEG) else 8
            ax.text(33.5, y_top - 1.3 - 1.4 * i, text, fontsize=fontsize,
                    color=color, fontweight=fontweight)

        # Feedback col
        fancy_box(ax, 70, y_top - h, 28, h, C_FB, C_FB_E, lw=0.4)
        for i, (color, text, bold) in enumerate(feedbacks):
            ax.text(71.2, y_top - 1.3 - 1.4 * i, text,
                    fontsize=8.3 if bold else 8.0, color=color,
                    fontweight="bold" if bold else "normal")

        # Connecting lines
        ax.plot([30, 32], [y_top - h / 2, y_top - h / 2], color=C_WEAK,
                lw=0.6, zorder=1)
        ax.plot([68, 70], [y_top - h / 2, y_top - h / 2], color=C_WEAK,
                lw=0.6, zorder=1)

    # Footer notes
    ax.text(2, 14.5,
            "Note 1 — Valeurs issues des regressions stepwise et correlations Spearman sur le sous-echantillon VR (n = 8–12 groupes).",
            fontsize=8.5, style="italic", color="#5F5E5A")
    ax.text(2, 12.7,
            "Note 2 — Les seuils (1.5 s, 0.55, 60 s, 150–300 ms) sont indicatifs et a calibrer sur la distribution observee (mean ± SD du tableau 3.1.1).",
            fontsize=8.5, style="italic", color="#5F5E5A")
    ax.text(2, 10.9,
            "Note 3 — Trois INV \"face\" sont degrades sous HMD (occlusion AU); les indicateurs \"Audio\" et \"Gaze\" sont prioritaires pour le feedback temps reel.",
            fontsize=8.5, style="italic", color="#5F5E5A")
    ax.text(2, 9.1,
            "Note 4 — Recommandation generale : pas plus de deux signaux concurrents en VR (turn duration + interruptions, ou bien attention coord. seul) pour eviter la surcharge.",
            fontsize=8.5, style="italic", color="#5F5E5A")

    # Insight callout box
    fancy_box(ax, 2, 1.2, 96, 7, "#FBEAF0", "#993556", lw=0.6)
    ax.text(4, 6.8, "Insight cle pour la conception du feedback VR",
            fontsize=10.5, fontweight="bold", color="#4B1528")
    ax.text(4, 4.8,
            "Les etats emergents (TMS, Cohesion) ne mediatisent que faiblement la performance ($\\beta < 0.30$). "
            "Le feedback VR doit cibler les INV directement,",
            fontsize=8.8, color="#4B1528")
    ax.text(4, 3.0,
            "pas les construits declaratifs. Les leviers les plus puissants empiriquement : "
            "(1) tours de parole longs, (2) interruptions controlees, (3) attention coord. moderee.",
            fontsize=8.8, color="#4B1528")
    ax.text(4, 1.6,
            "Le paradoxe \"forte coordination du regard ⇒ baisse de performance\" reflete la nature "
            "tripartite et asymetrique de la tache BIM (Calculateur / Moderateur / Lecteur).",
            fontsize=8.8, color="#4B1528")

    plt.tight_layout()
    return fig


# =================================================================
# Build and save
# =================================================================
import os
os.makedirs(".", exist_ok=True)

fig1 = schema1()
fig1.savefig("./schema1_imoi_empirique.pdf",
             bbox_inches="tight", dpi=300)
fig1.savefig("./schema1_imoi_empirique.png",
             bbox_inches="tight", dpi=180)
print("schema1 OK")

fig2 = schema2()
fig2.savefig("./schema2_feedback_vr.pdf",
             bbox_inches="tight", dpi=300)
fig2.savefig("./schema2_feedback_vr.png",
             bbox_inches="tight", dpi=180)
print("schema2 OK")