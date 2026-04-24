"""
llm_plots.py  -  Genererer plots for LLM vs. menneske PSG-sammenligning.

Læser output fra llm_vs_menneske.py (--output rapport.json) og laver:
  1. Emnegruppe agreement – vandret barplot
  2. Felt-for-felt uenighed – top-N felter
  3. Heatmap – agreement per felt × patient
  4. Fritekst token F1 – barplot
  5. LLM-mangler vs. uenige – stacked barplot per gruppe
  6. Numerisk afvigelse – dot-plot for numeriske felter

Brug:
  python llm_plots.py rapport.json
  python llm_plots.py rapport.json --top 20 --ud plots/

Krav:
  pip install matplotlib
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Farver og stil
# ---------------------------------------------------------------------------

PRIMÆR   = "#1a5276"   # mørk blå
ACCENT   = "#e74c3c"   # rød
GRØN     = "#1e8449"
GUL      = "#d4ac0d"
LYSBLÅ   = "#aed6f1"
LILLA    = "#7d3c98"
GRÅ      = "#bdc3c7"
BAGGRUND = "#f8f9fa"

def sæt_stil():
    plt.rcParams.update({
        "figure.facecolor":  BAGGRUND,
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#cccccc",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "x",
        "grid.color":        "#e0e0e0",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.7,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "figure.dpi":        150,
    })

def agreement_farve(rate):
    if rate is None:
        return GRÅ
    if rate >= 0.75:
        return GRØN
    if rate >= 0.5:
        return GUL
    return ACCENT


def gem(fig, sti, filnavn):
    ud = Path(sti) / filnavn
    fig.savefig(ud, bbox_inches="tight", facecolor=BAGGRUND)
    plt.close(fig)
    print("  Gemt: {}".format(ud))
    return ud


# ---------------------------------------------------------------------------
# Plot 1 – Emnegruppe agreement
# ---------------------------------------------------------------------------

def plot_gruppe_agreement(gruppe_stats, ud_mappe):
    grupper = sorted(
        gruppe_stats.items(),
        key=lambda x: (x[1]["agreement_rate"] or 0)
    )
    navne  = [g[0] for g in grupper]
    rater  = [g[1]["agreement_rate"] or 0 for g in grupper]
    farver = [agreement_farve(r) for r in rater]

    fig, ax = plt.subplots(figsize=(10, max(4, len(navne) * 0.55)))
    bars = ax.barh(navne, [r * 100 for r in rater], color=farver,
                   height=0.65, edgecolor="white", linewidth=0.5)

    # Procent-label på barerne
    for bar, rate in zip(bars, rater):
        x = bar.get_width()
        label = "{:.0f} %".format(x)
        ax.text(
            min(x + 1.5, 97), bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=9,
            color="#333333", fontweight="bold"
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Agreement (%)")
    ax.set_title("Agreement per emnegruppe  (LLM vs. menneske)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    legend = [
        mpatches.Patch(color=GRØN,  label="> 75 % – god"),
        mpatches.Patch(color=GUL,   label="50–75 % – middel"),
        mpatches.Patch(color=ACCENT, label="< 50 % – lav"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    return gem(fig, ud_mappe, "1_gruppe_agreement.png")


# ---------------------------------------------------------------------------
# Plot 2 – Top-N felter med mest uenighed
# ---------------------------------------------------------------------------

def plot_top_uenige_felter(felt_stats, ud_mappe, top_n=20):
    uenige = [
        (felt, s) for felt, s in felt_stats.items()
        if s["n_uenige"] > 0 and s["n_sammenlignet"] > 0
    ]
    sorteret = sorted(uenige, key=lambda x: -x[1]["n_uenige"])[:top_n]
    sorteret = list(reversed(sorteret))  # lavest øverst = størst nederst

    felter  = [s[0].split(".")[-1] for s in sorteret]
    grupper = [s[1]["gruppe"] for s in sorteret]
    uenige_n = [s[1]["n_uenige"] for s in sorteret]
    agree_r  = [s[1]["agreement_rate"] or 0 for s in sorteret]

    # Unik farve per gruppe
    alle_grupper = sorted(set(grupper))
    gruppe_farver = {
        g: c for g, c in zip(
            alle_grupper,
            ["#1a5276","#1e8449","#7d3c98","#d4ac0d","#c0392b",
             "#117a65","#784212","#1a237e","#4a235a","#1b4f72","#145a32","#78281f"]
        )
    }
    farver = [gruppe_farver[g] for g in grupper]

    fig, (ax_bar, ax_dot) = plt.subplots(
        1, 2, figsize=(14, max(5, len(sorteret) * 0.45)),
        gridspec_kw={"width_ratios": [1.8, 1]}
    )

    # Venstre: antal uenige
    bars = ax_bar.barh(felter, uenige_n, color=farver, height=0.65,
                       edgecolor="white", linewidth=0.5)
    for bar, n in zip(bars, uenige_n):
        ax_bar.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            str(n), va="center", ha="left", fontsize=8
        )
    ax_bar.set_xlabel("Antal uenige par")
    ax_bar.set_title("Top {} felter med flest uenigheder".format(top_n))

    # Højre: agreement rate som dot
    ax_dot.set_facecolor("white")
    ax_dot.axvline(50,  color=GUL,   linestyle="--", linewidth=1, alpha=0.7)
    ax_dot.axvline(75,  color=GRØN,  linestyle="--", linewidth=1, alpha=0.7)
    for i, (rate, farve) in enumerate(zip(agree_r, farver)):
        ax_dot.scatter(rate * 100, i, color=farve, s=70, zorder=3)
    ax_dot.set_yticks(range(len(felter)))
    ax_dot.set_yticklabels([])
    ax_dot.set_xlim(0, 105)
    ax_dot.set_xlabel("Agreement (%)")
    ax_dot.set_title("Agreement-rate")
    ax_dot.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax_dot.grid(True, axis="x", linestyle="--", color="#e0e0e0")
    ax_dot.spines["left"].set_visible(False)

    # Legende for grupper
    legend_handles = [
        mpatches.Patch(color=gruppe_farver[g], label=g)
        for g in alle_grupper
    ]
    ax_bar.legend(
        handles=legend_handles, loc="lower right",
        fontsize=7, framealpha=0.85, title="Gruppe", title_fontsize=8
    )

    fig.tight_layout()
    return gem(fig, ud_mappe, "2_top_uenige_felter.png")


# ---------------------------------------------------------------------------
# Plot 3 – Heatmap: agreement per felt × patient
# ---------------------------------------------------------------------------

def plot_heatmap(detaljer_per_par, felt_stats, ud_mappe, top_n=30):
    """
    Rækker = felter (de mest uenige), kolonner = patientpar.
    Farve: grøn=enig, rød=uenig, gul=mangler, grå=begge mangler.
    """
    par_navne = list(detaljer_per_par.keys())

    # Vælg de top_n felter med flest uenigheder
    uenige_felter = sorted(
        [(felt, s["n_uenige"]) for felt, s in felt_stats.items() if s["n_uenige"] > 0],
        key=lambda x: -x[1]
    )[:top_n]
    felter = [f[0] for f in uenige_felter]

    # Byg matrix: 1=enig, 0=uenig, 0.5=mangler, -1=begge mangler
    matrix = np.full((len(felter), len(par_navne)), -1.0)
    for j, par_navn in enumerate(par_navne):
        for i, felt in enumerate(felter):
            info = detaljer_per_par[par_navn].get(felt, {})
            status = info.get("status", "begge_mangler")
            enige  = info.get("enige")
            f1     = info.get("f1")

            if status == "begge_mangler":
                matrix[i, j] = -1.0
            elif status in ("llm_mangler", "menneske_mangler"):
                matrix[i, j] = 0.3
            elif enige is True:
                matrix[i, j] = 1.0
            elif enige is False:
                if f1 is not None:
                    matrix[i, j] = f1  # gradient for fritekst
                else:
                    matrix[i, j] = 0.0

    # Normaliser matrix fra [-1,1] til [0,1]
    norm_matrix = (matrix + 1) / 2

    # Custom colormap: grå(0) → rød(0.25) → orange(0.65) → gul(0.8) → grøn(1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "agreement",
        [
            (0.00, "#cccccc"),   # begge mangler – grå
            (0.25, "#e74c3c"),   # uenige – rød
            (0.65, "#e67e22"),   # mangler – orange
            (0.80, "#f1c40f"),   # delvis – gul
            (1.00, "#27ae60"),   # enige – grøn
        ],
        N=256
    )

    kort_felter = [
        (f.split(".")[-2] + "." if "." in f else "") + f.split(".")[-1]
        for f in felter
    ]
    kort_par = [
        re.sub(r"(output-|\.json)", "", p.split("↔")[0].strip())
        for p in par_navne
    ]

    fig_h = max(6, len(felter) * 0.38)
    fig_w = max(8, len(par_navne) * 1.1 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(par_navne)))
    ax.set_xticklabels(kort_par, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(felter)))
    ax.set_yticklabels(kort_felter, fontsize=8)
    ax.set_title("Agreement heatmap  (grøn = enig, rød = uenig, orange = mangler)")

    # Farvebar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 0.15, 0.65, 1])
    cbar.set_ticklabels(["Uenig", "Mangler", "Delvis", "Enig"])
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return gem(fig, ud_mappe, "3_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 4 – Fritekst token F1
# ---------------------------------------------------------------------------

def plot_fritekst_f1(felt_stats, ud_mappe):
    fritekst = [
        (felt, s) for felt, s in felt_stats.items()
        if s["felttype"] == "fritekst" and s["gns_f1"] is not None
    ]
    if not fritekst:
        print("  Ingen fritekstfelter med F1-data – springer plot 4 over.")
        return None

    sorteret = sorted(fritekst, key=lambda x: x[1]["gns_f1"])
    felter   = [f[0].split(".")[-1] for f in sorteret]
    f1_vals  = [f[1]["gns_f1"] for f in sorteret]
    farver   = [agreement_farve(v) for v in f1_vals]

    fig, ax = plt.subplots(figsize=(10, max(3, len(felter) * 0.6)))
    bars = ax.barh(felter, f1_vals, color=farver, height=0.6,
                   edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, f1_vals):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            "{:.2f}".format(v), va="center", ha="left", fontsize=9
        )

    ax.axvline(0.8, color="#555", linestyle="--", linewidth=1, label="Tærskel 0.80")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Gns. token F1")
    ax.set_title("Fritekstfelter – gennemsnitlig token F1")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return gem(fig, ud_mappe, "4_fritekst_f1.png")


# ---------------------------------------------------------------------------
# Plot 5 – Stacked: LLM mangler vs. uenige vs. enige per gruppe
# ---------------------------------------------------------------------------

def plot_stacked_status(gruppe_stats, ud_mappe):
    grupper  = list(gruppe_stats.keys())
    enige    = [gruppe_stats[g]["n_enige"]       for g in grupper]
    uenige   = [gruppe_stats[g]["n_uenige"] - gruppe_stats[g]["n_llm_mangler"] for g in grupper]
    mangler  = [gruppe_stats[g]["n_llm_mangler"] for g in grupper]

    # Sorter efter agreement rate
    order = sorted(range(len(grupper)),
                   key=lambda i: gruppe_stats[grupper[i]]["agreement_rate"] or 0)
    grupper = [grupper[i] for i in order]
    enige   = [enige[i]   for i in order]
    uenige  = [max(0, uenige[i])  for i in order]
    mangler = [mangler[i] for i in order]

    x    = np.arange(len(grupper))
    bred = 0.55

    fig, ax = plt.subplots(figsize=(12, 5))
    p1 = ax.bar(x, enige,   bred, label="Enige",          color=GRØN,   edgecolor="white")
    p2 = ax.bar(x, uenige,  bred, label="Uenige",         color=ACCENT, edgecolor="white", bottom=enige)
    p3 = ax.bar(x, mangler, bred, label="LLM mangler",    color=GUL,    edgecolor="white",
                bottom=[e + u for e, u in zip(enige, uenige)])

    ax.set_xticks(x)
    ax.set_xticklabels(grupper, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Antal felt-sammenligninger")
    ax.set_title("Status per emnegruppe  (enige / uenige / LLM mangler)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", color="#e0e0e0")
    ax.spines["bottom"].set_visible(True)

    fig.tight_layout()
    return gem(fig, ud_mappe, "5_stacked_status.png")


# ---------------------------------------------------------------------------
# Plot 6 – Numerisk afvigelse dot-plot
# ---------------------------------------------------------------------------

def plot_numerisk_afvigelse(detaljer_per_par, felt_stats, ud_mappe, top_n=20):
    # Saml alle numeriske felter med afvigelser
    num_felter = {
        felt: s for felt, s in felt_stats.items()
        if s["felttype"] == "numerisk" and s["gns_afvigelse"] is not None
    }
    if not num_felter:
        print("  Ingen numeriske felter med afvigelser – springer plot 6 over.")
        return None

    sorteret = sorted(num_felter.items(), key=lambda x: -x[1]["gns_afvigelse"])[:top_n]
    sorteret = list(reversed(sorteret))

    felter      = [f[0].split(".")[-1] for f in sorteret]
    gns_afv     = [f[1]["gns_afvigelse"] for f in sorteret]
    tol_rate    = [f[1]["tol_match_rate"] for f in sorteret]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, len(felter) * 0.45)),
                                    gridspec_kw={"width_ratios": [1.5, 1]})

    # Venstre: gns. afvigelse
    farver = [ACCENT if t is not None and t < 0.75 else LYSBLÅ for t in tol_rate]
    bars = ax1.barh(felter, gns_afv, color=farver, height=0.6,
                    edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, gns_afv):
        ax1.text(
            bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
            "{:.3g}".format(v), va="center", ha="left", fontsize=8
        )
    ax1.set_xlabel("Gns. absolut afvigelse")
    ax1.set_title("Numerisk afvigelse (LLM − menneske)")

    # Højre: tolerance match rate
    tol_vals = [(t or 0) * 100 for t in tol_rate]
    tol_farver = [agreement_farve((t or 0)) for t in tol_rate]
    ax2.barh(felter, tol_vals, color=tol_farver, height=0.6,
             edgecolor="white", linewidth=0.5)
    ax2.axvline(75, color="#555", linestyle="--", linewidth=1)
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Tolerance match ±5% (%)")
    ax2.set_title("Inden for ±5% tolerance")
    ax2.set_yticklabels([])
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    fig.tight_layout()
    return gem(fig, ud_mappe, "6_numerisk_afvigelse.png")


# ---------------------------------------------------------------------------
# Samlet oversigts-figur
# ---------------------------------------------------------------------------

def plot_oversigt(gruppe_stats, felt_stats, ud_mappe):
    """Én figur med de vigtigste tal – god til at indsætte i rapport."""
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BAGGRUND)

    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax_grp  = fig.add_subplot(gs[0, 0])
    ax_top  = fig.add_subplot(gs[0, 1])
    ax_ftxt = fig.add_subplot(gs[1, 0])
    ax_sum  = fig.add_subplot(gs[1, 1])

    # --- Gruppe agreement (mini) ---
    grupper = sorted(gruppe_stats.items(), key=lambda x: x[1]["agreement_rate"] or 0)
    navne_g = [g[0][:20] for g in grupper]
    rater_g = [(g[1]["agreement_rate"] or 0) * 100 for g in grupper]
    farver_g = [agreement_farve(r / 100) for r in rater_g]
    ax_grp.barh(navne_g, rater_g, color=farver_g, height=0.6, edgecolor="white")
    ax_grp.set_xlim(0, 110)
    ax_grp.set_xlabel("Agreement %")
    ax_grp.set_title("Agreement per gruppe")
    ax_grp.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    # --- Top 10 uenige felter (mini) ---
    uenige = sorted(
        [(f, s["n_uenige"]) for f, s in felt_stats.items() if s["n_uenige"] > 0],
        key=lambda x: -x[1]
    )[:10]
    uenige = list(reversed(uenige))
    ax_top.barh(
        [u[0].split(".")[-1] for u in uenige],
        [u[1] for u in uenige],
        color=PRIMÆR, height=0.6, edgecolor="white"
    )
    ax_top.set_xlabel("Antal uenige par")
    ax_top.set_title("Top 10 felter med uenighed")

    # --- Fritekst F1 (mini) ---
    fritekst = [
        (f.split(".")[-1], s["gns_f1"])
        for f, s in felt_stats.items()
        if s["felttype"] == "fritekst" and s["gns_f1"] is not None
    ]
    if fritekst:
        fritekst = sorted(fritekst, key=lambda x: x[1])
        ax_ftxt.barh(
            [f[0] for f in fritekst], [f[1] for f in fritekst],
            color=[agreement_farve(v) for _, v in fritekst],
            height=0.6, edgecolor="white"
        )
        ax_ftxt.axvline(0.8, color="#555", linestyle="--", linewidth=1)
        ax_ftxt.set_xlim(0, 1.1)
        ax_ftxt.set_xlabel("Token F1")
        ax_ftxt.set_title("Fritekst token F1")
    else:
        ax_ftxt.text(0.5, 0.5, "Ingen fritekstdata", ha="center", va="center",
                     transform=ax_ftxt.transAxes, color="#999")
        ax_ftxt.set_title("Fritekst token F1")

    # --- Samlet donut ---
    total_s = sum(s["n_sammenlignet"] for s in felt_stats.values())
    total_e = sum(s["n_enige"]        for s in felt_stats.values())
    total_u = sum(s["n_uenige"] - s["n_llm_mangler"] for s in felt_stats.values())
    total_m = sum(s["n_llm_mangler"]  for s in felt_stats.values())
    total_u = max(0, total_u)

    wedges, texts, autotexts = ax_sum.pie(
        [total_e, total_u, total_m],
        labels=["Enige", "Uenige", "LLM\nmangler"],
        colors=[GRØN, ACCENT, GUL],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    # Donut-hul
    centre_circle = plt.Circle((0, 0), 0.5, fc="white")
    ax_sum.add_artist(centre_circle)
    ax_sum.text(0, 0, "{:.0f}%\nenige".format(100 * total_e / total_s if total_s else 0),
                ha="center", va="center", fontsize=11, fontweight="bold", color=PRIMÆR)
    ax_sum.set_title("Samlet fordeling")

    fig.suptitle("LLM vs. Menneske – PSG Analyse", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return gem(fig, ud_mappe, "0_oversigt.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generér plots for LLM vs. menneske PSG-sammenligning"
    )
    p.add_argument("rapport", help="JSON-rapport fra llm_vs_menneske.py (--output)")
    p.add_argument("--top",  type=int, default=20,
                   help="Antal felter i top-N plots (standard: 20)")
    p.add_argument("--ud",   default="plots",
                   help="Output-mappe til PNG-filer (standard: plots/)")
    args = p.parse_args()

    ud = Path(args.ud)
    ud.mkdir(parents=True, exist_ok=True)

    print("Indlæser rapport: {}".format(args.rapport))
    data = json.loads(Path(args.rapport).read_text(encoding="utf-8"))

    gruppe_stats      = data["gruppe_stats"]
    felt_stats        = data["felt_stats"]
    detaljer_per_par  = data.get("detaljer_per_par", {})

    sæt_stil()

    print("\nGenererer plots...")
    plot_oversigt(gruppe_stats, felt_stats, ud)
    plot_gruppe_agreement(gruppe_stats, ud)
    plot_top_uenige_felter(felt_stats, ud, top_n=args.top)
    plot_heatmap(detaljer_per_par, felt_stats, ud, top_n=args.top)
    plot_fritekst_f1(felt_stats, ud)
    plot_stacked_status(gruppe_stats, ud)
    plot_numerisk_afvigelse(detaljer_per_par, felt_stats, ud, top_n=args.top)

    print("\nFærdig! {} plots gemt i: {}".format(
        len(list(ud.glob("*.png"))), ud
    ))


if __name__ == "__main__":
    main()