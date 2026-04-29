"""
model_sammenligning.py  -  Sammenligner N modellers PSG-output mod ground truth.

Matcher JSON-filer på tværs af model-mapper og én ground-truth-mappe og beregner:
  - Samlet agreement per model (rangering)
  - Agreement per emnegruppe per model
  - Felter der er svære for ALLE modeller
  - Felter hvor én model klarer sig markant bedre/dårligere end de andre
  - Plots: model-rangering, gruppe-heatmap, fejlprofil-sammenligning

Brug:
  python model_sammenligning.py \\
      --ground-truth mine-output/ \\
      --modeller gpt4/  claude/  gemini/ \\
      --navne "GPT-4o" "Claude 3.5" "Gemini 1.5"

  python model_sammenligning.py \\
      --ground-truth mine-output/ \\
      --modeller gpt4/  claude/ \\
      --output rapport.json \\
      --plots plots/

Krav:
  pip install matplotlib
"""

import re
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Feltklassifikation
# ---------------------------------------------------------------------------

SKIP_FELTER = {
    "metadata.filnavn",
    "metadata.udfyldt_af",
}

FRITEKST_FELTER = {
    "klinisk_information.klinisk_resume",
    "klinisk_information.kommentar",
    "klinisk_information.patient_oplysninger_ved_optagelse",
    "sammenfatning.soevnmoenster",
    "sammenfatning.soevn_dagtid",
    "sammenfatning.anfald",
    "sammenfatning.beskrivelse",
    "sammenfatning.paroksystisk_aktivitet",
    "sammenfatning.fokal",
    "konklusion_og_plan.konklusion_tekst",
    "konklusion_og_plan.plan",
}

NOMINALE_FELTER = {
    "patient.navn",
    "patient.cpr-nummer",
    "test_oplysninger.dato",
    "test_oplysninger.starttid",
    "test_oplysninger.sluttid",
    "test_oplysninger.henviser",
    "test_oplysninger.henvisningsdiagnose",
    "test_oplysninger.optaget_af",
    "test_oplysninger.montage",
    "test_oplysninger.neurofysiologi_assisstent",
    "klinisk_information.medicin",
    "soevn_opsummering.scoringens_navn",
    "soevn_opsummering.analyse_start_lights_out",
    "soevn_opsummering.analyse_afslutning_lights_on",
    "oget_muskelaktivitet_rem.chin",
    "oget_muskelaktivitet_rem.tib",
    "oget_muskelaktivitet_rem.fds",
    "hjerte.ekg_bemaerkninger",
    "konklusion_og_plan.bedoemt_af",
    "konklusion_og_plan.dato_for_bedoemmelse",
    "konklusion_og_plan.a_diagnose.kode",
    "konklusion_og_plan.a_diagnose.tekst",
    "konklusion_og_plan.b_diagnose.kode",
    "konklusion_og_plan.b_diagnose.tekst",
    "benbevaegelser.lms_efterfulgt_af_arousals",
}

EMNEGRUPPER = {
    "Patient & metadata":       ["patient.", "metadata."],
    "Test oplysninger":         ["test_oplysninger."],
    "Klinisk information":      ["klinisk_information."],
    "Søvn opsummering":         ["soevn_opsummering."],
    "Søvnstadier":              ["soevnstadier."],
    "Øget muskelaktivitet REM": ["oget_muskelaktivitet_rem."],
    "Respirationsanalyse":      ["respirations_analyse."],
    "SpO2 oversigt":            ["spo2_oversigt."],
    "Benbevægelser":            ["benbevaegelser."],
    "Hjerte":                   ["hjerte."],
    "Sammenfatning":            ["sammenfatning."],
    "Konklusion og plan":       ["konklusion_og_plan."],
}

# ---------------------------------------------------------------------------
# Stil
# ---------------------------------------------------------------------------

MODEL_FARVER = [
    "#1a5276", "#1e8449", "#7d3c98", "#c0392b",
    "#d4ac0d", "#117a65", "#784212", "#1a237e",
]
BAGGRUND = "#f8f9fa"
GRØN     = "#27ae60"
GUL      = "#f39c12"
RØD      = "#e74c3c"
GRÅ      = "#bdc3c7"

def sæt_stil():
    plt.rcParams.update({
        "figure.facecolor":  BAGGRUND,
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#cccccc",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#e8e8e8",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.7,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "figure.dpi":        150,
    })

def agreement_farve(rate):
    if rate is None: return GRÅ
    if rate >= 0.75: return GRØN
    if rate >= 0.50: return GUL
    return RØD

def gem(fig, mappe, filnavn):
    ud = Path(mappe) / filnavn
    fig.savefig(ud, bbox_inches="tight", facecolor=BAGGRUND)
    plt.close(fig)
    print("  Gemt: {}".format(ud))
    return str(ud)

# ---------------------------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------------------------

def fladgør(d, prefix=""):
    items = {}
    for k, v in d.items():
        key = "{}.{}".format(prefix, k) if prefix else k
        if isinstance(v, dict):
            items.update(fladgør(v, key))
        else:
            items[key] = v
    return items

def normaliser(værdi):
    if værdi is None: return None
    if isinstance(værdi, bool): return værdi
    if isinstance(værdi, (int, float)): return værdi
    v = str(værdi).strip()
    if v in ("", "-", "N/A", "n/a", "None", "null"): return None
    v = re.sub(
        r"\s*(cm|kg|%|/h|/t|min\.|min|slag per minut|kg/m²|kPa)\s*$",
        "", v, flags=re.IGNORECASE
    ).strip()
    if re.match(r"^-?[\d]+,[\d]+$", v):
        v = v.replace(",", ".")
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except ValueError:
        pass
    dato = re.match(r"^(\d{1,4})[-/\.](\d{1,2})[-/\.](\d{1,4})$", v)
    if dato:
        a, b, c = dato.groups()
        if len(a) == 4:   return "{:02d}-{:02d}-{}".format(int(c), int(b), a)
        elif len(c) == 4: return "{:02d}-{:02d}-{}".format(int(a), int(b), c)
    return v.lower()

def indlæs_json(sti):
    tekst = Path(sti).read_text(encoding="utf-8")
    try:
        data = json.loads(tekst)
    except json.JSONDecodeError as e:
        print("FEJL: Kan ikke parse {} – {}".format(Path(sti).name, e))
        sys.exit(1)
    return {felt: normaliser(v) for felt, v in fladgør(data).items()}

def token_f1(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    ta = set(str(a).lower().split())
    tb = set(str(b).lower().split())
    overlap = ta & tb
    if not overlap: return 0.0
    p = len(overlap) / len(tb)
    r = len(overlap) / len(ta)
    return 2 * p * r / (p + r)

def tolerance_match(a, b, tol=0.05):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if a == b == 0: return True
    if a == 0 or b == 0: return abs(a - b) <= tol
    return abs(a - b) / max(abs(a), abs(b)) <= tol

def felttype_for(felt, model_v, gt_v):
    if felt in FRITEKST_FELTER: return "fritekst"
    if felt in NOMINALE_FELTER: return "nominal"
    if isinstance(model_v, (int, float)) or isinstance(gt_v, (int, float)):
        return "numerisk"
    return "nominal"

def emne_for(felt):
    for gruppe, præfikser in EMNEGRUPPER.items():
        for p in præfikser:
            if felt.startswith(p):
                return gruppe
    return "Andet"

# ---------------------------------------------------------------------------
# Filmatching
# ---------------------------------------------------------------------------

def match_filer(gt_mappe, model_mappe):
    """Returnerer liste af (gt_sti, model_sti) par matchet på filnavn eller nummer."""
    gt_filer    = {f.name: f for f in Path(gt_mappe).glob("*.json")}
    mod_filer   = {f.name: f for f in Path(model_mappe).glob("*.json")}

    par = []
    uparret_gt  = []
    uparret_mod = []

    for navn in sorted(gt_filer):
        if navn in mod_filer:
            par.append((gt_filer[navn], mod_filer[navn]))
        else:
            uparret_gt.append(navn)
    for navn in sorted(mod_filer):
        if navn not in gt_filer:
            uparret_mod.append(navn)

    def num(navn):
        m = re.search(r"(\d+)", navn)
        return int(m.group(1)) if m else None

    gt_num  = {num(n): n for n in uparret_gt  if num(n) is not None}
    mod_num = {num(n): n for n in uparret_mod if num(n) is not None}

    for n in sorted(set(gt_num) & set(mod_num)):
        par.append((gt_filer[gt_num[n]], mod_filer[mod_num[n]]))

    return par

# ---------------------------------------------------------------------------
# Sammenligning: én model mod ground truth
# ---------------------------------------------------------------------------

def sammenlign_model(model_ann, gt_ann):
    """
    Returnerer per-felt dict med enige/uenige/status/score.
    score ∈ [0,1]: 1=enig, 0=uenig, None=ikke sammenlignet.
    """
    alle_felter = (set(model_ann) | set(gt_ann)) - SKIP_FELTER
    resultater  = {}

    for felt in sorted(alle_felter):
        mv = model_ann.get(felt)
        gv = gt_ann.get(felt)
        ft = felttype_for(felt, mv, gv)
        em = emne_for(felt)

        if mv is None and gv is None:
            resultater[felt] = {"felttype": ft, "emne": em, "status": "begge_mangler", "score": None}
            continue
        if mv is None:
            resultater[felt] = {"felttype": ft, "emne": em, "status": "model_mangler",
                                 "score": 0.0, "model": None, "gt": gv}
            continue
        if gv is None:
            resultater[felt] = {"felttype": ft, "emne": em, "status": "gt_mangler",
                                 "score": None, "model": mv, "gt": None}
            continue

        if ft == "fritekst":
            f1 = token_f1(mv, gv)
            resultater[felt] = {"felttype": ft, "emne": em, "status": "udfyldt",
                                 "score": round(f1, 3), "model": mv, "gt": gv, "f1": round(f1,3)}
        elif ft == "numerisk":
            try:
                mf, gf = float(str(mv).replace(",",".")), float(str(gv).replace(",","."))
                exact = (mf == gf)
                tol   = tolerance_match(mf, gf)
                score = 1.0 if exact else (0.7 if tol else 0.0)
                resultater[felt] = {
                    "felttype": ft, "emne": em, "status": "udfyldt",
                    "score": score, "model": mv, "gt": gv,
                    "exact": exact, "tolerance": tol,
                    "afvigelse": round(abs(mf - gf), 4),
                }
            except (TypeError, ValueError):
                exact = str(mv).lower().strip() == str(gv).lower().strip()
                resultater[felt] = {"felttype": ft, "emne": em, "status": "udfyldt",
                                     "score": 1.0 if exact else 0.0, "model": mv, "gt": gv}
        else:
            exact = str(mv).lower().strip() == str(gv).lower().strip()
            resultater[felt] = {"felttype": ft, "emne": em, "status": "udfyldt",
                                 "score": 1.0 if exact else 0.0, "model": mv, "gt": gv}

    return resultater

# ---------------------------------------------------------------------------
# Aggregér per model
# ---------------------------------------------------------------------------

def aggreger_model(alle_par_res):
    """
    Input: {par_id: {felt: {...}}}
    Output: {felt: {n, score_sum, score_n, mangler, ...}, gruppe: {...}}
    """
    felt_stats  = defaultdict(lambda: {
        "felttype": None, "emne": None,
        "n": 0, "score_sum": 0.0, "score_n": 0,
        "n_mangler": 0, "n_begge_mangler": 0,
    })
    gruppe_stats = defaultdict(lambda: {
        "n": 0, "score_sum": 0.0, "score_n": 0, "n_mangler": 0
    })

    for par_res in alle_par_res.values():
        for felt, info in par_res.items():
            s = felt_stats[felt]
            s["felttype"] = info["felttype"]
            s["emne"]     = info["emne"]

            st = info["status"]
            if st == "begge_mangler":
                s["n_begge_mangler"] += 1
            elif st == "model_mangler":
                s["n"] += 1
                s["n_mangler"] += 1
                s["score_sum"] += 0.0
                s["score_n"]   += 1
            elif st == "gt_mangler":
                pass  # GT mangler – tæller ikke imod modellen
            else:
                s["n"] += 1
                if info["score"] is not None:
                    s["score_sum"] += info["score"]
                    s["score_n"]   += 1

            # Gruppe
            g = gruppe_stats[info["emne"]]
            if st not in ("begge_mangler", "gt_mangler"):
                g["n"] += 1
                if st == "model_mangler":
                    g["n_mangler"] += 1
                    g["score_sum"] += 0.0
                    g["score_n"]   += 1
                elif info["score"] is not None:
                    g["score_sum"] += info["score"]
                    g["score_n"]   += 1

    for felt, s in felt_stats.items():
        s["gns_score"] = round(s["score_sum"] / s["score_n"], 3) if s["score_n"] > 0 else None

    for g, gs in gruppe_stats.items():
        gs["gns_score"] = round(gs["score_sum"] / gs["score_n"], 3) if gs["score_n"] > 0 else None

    return dict(felt_stats), dict(gruppe_stats)


def samlet_score(felt_stats):
    """Beregner én samlet score for en model."""
    scores = [s["gns_score"] for s in felt_stats.values() if s["gns_score"] is not None]
    return round(sum(scores) / len(scores), 3) if scores else None

# ---------------------------------------------------------------------------
# Tværgående analyse
# ---------------------------------------------------------------------------

def analyser_tværgående(model_felt_stats, model_navne):
    """
    Finder:
      - Felter svære for ALLE modeller (alle scorer < 0.5)
      - Felter med stor spredning (én model meget bedre end resten)
    """
    alle_felter = set()
    for fs in model_felt_stats.values():
        alle_felter.update(fs.keys())

    svære_for_alle   = []
    stor_spredning   = []
    bedste_per_felt  = {}  # felt -> model-navn med højest score

    for felt in sorted(alle_felter):
        scores = {}
        for navn in model_navne:
            fs = model_felt_stats.get(navn, {})
            s  = fs.get(felt, {}).get("gns_score")
            if s is not None:
                scores[navn] = s

        if len(scores) < 2:
            continue

        vals = list(scores.values())
        gns  = sum(vals) / len(vals)
        sprd = max(vals) - min(vals)

        if gns < 0.5 and len(scores) == len(model_navne):
            svære_for_alle.append({
                "felt": felt,
                "gns_score": round(gns, 3),
                "scores": scores,
            })

        if sprd >= 0.3:
            bedste = max(scores, key=scores.get)
            dårligste = min(scores, key=scores.get)
            stor_spredning.append({
                "felt": felt,
                "spredning": round(sprd, 3),
                "bedste_model": bedste,
                "bedste_score": round(scores[bedste], 3),
                "dårligste_model": dårligste,
                "dårligste_score": round(scores[dårligste], 3),
                "scores": scores,
            })

    svære_for_alle.sort(key=lambda x: x["gns_score"])
    stor_spredning.sort(key=lambda x: -x["spredning"])
    return svære_for_alle, stor_spredning

# ---------------------------------------------------------------------------
# Tekstrapport
# ---------------------------------------------------------------------------

def print_rapport(model_navne, model_felt_stats, model_gruppe_stats,
                  svære_for_alle, stor_spredning, n_par_per_model):
    linje = "=" * 70
    print(linje)
    print("MODEL-SAMMENLIGNING  –  PSG ANALYSE")
    print(linje)

    # ── MODEL RANGERING ──────────────────────────────────────────────────
    print("\n── MODEL RANGERING (samlet score) ───────────────────────────────")
    scores = [(navn, samlet_score(model_felt_stats[navn])) for navn in model_navne]
    scores.sort(key=lambda x: -(x[1] or 0))

    print("  {:>3}  {:<25} {:>10} {:>8}".format("#", "Model", "Score", "Par"))
    print("  " + "─" * 50)
    for rang, (navn, score) in enumerate(scores, 1):
        n = n_par_per_model.get(navn, "?")
        bar_len = int((score or 0) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print("  {:>3}  {:<25} {:>9.1f}%  {:>5}".format(
            rang, navn, (score or 0) * 100, n
        ))
        print("       {}".format(bar))

    # ── GRUPPE-SCORE PER MODEL ────────────────────────────────────────────
    print("\n\n── AGREEMENT PER EMNEGRUPPE ─────────────────────────────────────")
    alle_grupper = sorted({
        g for gs in model_gruppe_stats.values() for g in gs
    })

    # Header
    hdr = "  {:<28}".format("Gruppe")
    for navn in model_navne:
        kort = navn[:10]
        hdr += " {:>10}".format(kort)
    print(hdr)
    print("  " + "─" * (28 + 11 * len(model_navne)))

    for gruppe in alle_grupper:
        række = "  {:<28}".format(gruppe[:28])
        for navn in model_navne:
            gs = model_gruppe_stats.get(navn, {}).get(gruppe, {})
            sc = gs.get("gns_score")
            if sc is None:
                celle = "  N/A"
            else:
                sym = "🟢" if sc >= 0.75 else ("🟡" if sc >= 0.5 else "🔴")
                celle = "{} {:>4.0f}%".format(sym, sc * 100)
            række += " {:>10}".format(celle)
        print(række)

    print("\n  🟢 ≥ 75%   🟡 50–75%   🔴 < 50%")

    # ── SVÆRE FOR ALLE MODELLER ───────────────────────────────────────────
    print("\n\n── FELTER SVÆRE FOR ALLE MODELLER (gns. score < 50%) ───────────")
    if not svære_for_alle:
        print("  Ingen felter er svære for samtlige modeller – godt tegn!")
    else:
        print("  {:<48} {:>8}".format("Felt", "Gns. score"))
        print("  " + "─" * 58)
        for item in svære_for_alle[:20]:
            felt_kort = item["felt"]
            if len(felt_kort) > 46:
                felt_kort = "…" + felt_kort[-45:]
            print("  {:<48} {:>7.0f}%".format(felt_kort, item["gns_score"] * 100))
            for navn, sc in sorted(item["scores"].items(), key=lambda x: -x[1]):
                print("    {:>25}: {:.0f}%".format(navn, sc * 100))

    # ── STOR SPREDNING MELLEM MODELLER ───────────────────────────────────
    print("\n\n── FELTER MED STØRST FORSKEL MELLEM MODELLER ───────────────────")
    if not stor_spredning:
        print("  Ingen felter med markant forskel (≥ 30 pp) mellem modeller.")
    else:
        print("  {:<40} {:>8} {:<20} {:<20}".format(
            "Felt", "Spredning", "Bedst", "Dårligst"
        ))
        print("  " + "─" * 90)
        for item in stor_spredning[:15]:
            felt_kort = item["felt"].split(".")[-1]
            print("  {:<40} {:>7.0f}pp  {:<20} {:<20}".format(
                felt_kort,
                item["spredning"] * 100,
                "{} ({:.0f}%)".format(item["bedste_model"][:12], item["bedste_score"]*100),
                "{} ({:.0f}%)".format(item["dårligste_model"][:12], item["dårligste_score"]*100),
            ))

    print("\n" + linje)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_model_rangering(model_navne, model_felt_stats, model_gruppe_stats, ud_mappe):
    """Vandret barplot med samlet score + per-gruppe scores."""
    scores = [(navn, samlet_score(model_felt_stats[navn]) or 0) for navn in model_navne]
    scores.sort(key=lambda x: x[1])
    navne_s = [s[0] for s in scores]
    vals_s  = [s[1] * 100 for s in scores]
    farver  = [MODEL_FARVER[i % len(MODEL_FARVER)] for i in range(len(navne_s))]

    fig, ax = plt.subplots(figsize=(10, max(3, len(navne_s) * 0.7 + 1.5)))
    bars = ax.barh(navne_s, vals_s, color=farver, height=0.55,
                   edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals_s):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                "{:.1f}%".format(v), va="center", ha="left",
                fontsize=10, fontweight="bold")

    ax.axvline(75, color=GRØN, linestyle="--", linewidth=1.2, alpha=0.7, label="75%")
    ax.axvline(50, color=GUL,  linestyle="--", linewidth=1.2, alpha=0.7, label="50%")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Samlet agreement score (%)")
    ax.set_title("Model rangering – samlet agreement mod ground truth")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return gem(fig, ud_mappe, "1_model_rangering.png")


def plot_gruppe_heatmap(model_navne, model_gruppe_stats, ud_mappe):
    """Heatmap: modeller × emnegrupper."""
    alle_grupper = sorted({
        g for gs in model_gruppe_stats.values() for g in gs
    })

    matrix = np.full((len(alle_grupper), len(model_navne)), np.nan)
    for j, navn in enumerate(model_navne):
        for i, gruppe in enumerate(alle_grupper):
            sc = model_gruppe_stats.get(navn, {}).get(gruppe, {}).get("gns_score")
            if sc is not None:
                matrix[i, j] = sc * 100

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ag", [(0, RØD), (0.5, GUL), (1, GRØN)], N=256
    )

    fig_h = max(5, len(alle_grupper) * 0.55 + 1.5)
    fig_w = max(6, len(model_navne) * 1.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    # Værdier i celler
    for i in range(len(alle_grupper)):
        for j in range(len(model_navne)):
            v = matrix[i, j]
            if not np.isnan(v):
                textcol = "white" if v < 40 or v > 80 else "#333"
                ax.text(j, i, "{:.0f}%".format(v),
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=textcol)

    ax.set_xticks(range(len(model_navne)))
    ax.set_xticklabels(model_navne, fontsize=10)
    ax.set_yticks(range(len(alle_grupper)))
    ax.set_yticklabels(alle_grupper, fontsize=9)
    ax.set_title("Agreement per emnegruppe per model  (% score)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return gem(fig, ud_mappe, "2_gruppe_heatmap.png")


def plot_fejlprofil(model_navne, model_felt_stats, ud_mappe, top_n=20):
    """
    Radar/spider-chart med fejlprofil per felttype og gruppe.
    Fallback til barplot hvis < 3 dimensioner.
    """
    alle_grupper = sorted({
        g for fs in model_felt_stats.values()
        for s in fs.values() for g in [s["emne"]] if s["emne"]
    })

    # Score per model per gruppe (til radar)
    gruppe_scores = {}
    for navn in model_navne:
        fs = model_felt_stats[navn]
        gs = defaultdict(list)
        for felt, s in fs.items():
            if s["gns_score"] is not None:
                gs[s["emne"]].append(s["gns_score"])
        gruppe_scores[navn] = {g: sum(v)/len(v) for g, v in gs.items()}

    N = len(alle_grupper)
    if N < 3:
        print("  Springer radar-plot over (for få grupper).")
        return None

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # luk cirklen

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_facecolor("white")

    for i, navn in enumerate(model_navne):
        vals = [gruppe_scores[navn].get(g, 0) * 100 for g in alle_grupper]
        vals += vals[:1]
        farve = MODEL_FARVER[i % len(MODEL_FARVER)]
        ax.plot(angles, vals, "o-", linewidth=2, color=farve, label=navn, markersize=4)
        ax.fill(angles, vals, alpha=0.08, color=farve)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [g[:18] for g in alle_grupper],
        fontsize=8
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#888")
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_title("Fejlprofil per emnegruppe", pad=20, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)

    fig.tight_layout()
    return gem(fig, ud_mappe, "3_fejlprofil_radar.png")


def plot_svære_felter(svære_for_alle, model_navne, ud_mappe, top_n=15):
    """Grouped barplot: felter svære for alle – score per model."""
    if not svære_for_alle:
        print("  Ingen svære felter for alle modeller – springer plot over.")
        return None

    items = svære_for_alle[:top_n]
    felter = [it["felt"].split(".")[-1] for it in items]
    x      = np.arange(len(felter))
    bred   = 0.8 / len(model_navne)

    fig, ax = plt.subplots(figsize=(12, max(4, len(felter) * 0.45 + 2)))

    for i, navn in enumerate(model_navne):
        vals = [it["scores"].get(navn, 0) * 100 for it in items]
        offset = (i - len(model_navne) / 2 + 0.5) * bred
        ax.bar(x + offset, vals, bred * 0.9,
               label=navn, color=MODEL_FARVER[i % len(MODEL_FARVER)],
               edgecolor="white", linewidth=0.5)

    ax.axhline(50, color="#888", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(felter, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Felter svære for alle modeller – score per model")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return gem(fig, ud_mappe, "4_svaere_felter.png")


def plot_spredning(stor_spredning, ud_mappe, top_n=15):
    """Dot-plot: felter med stor forskel mellem modeller."""
    if not stor_spredning:
        print("  Ingen felter med stor spredning – springer plot over.")
        return None

    items    = stor_spredning[:top_n]
    items_r  = list(reversed(items))
    felter   = [it["felt"].split(".")[-1] for it in items_r]
    spredning = [it["spredning"] * 100 for it in items_r]

    model_navne_i = list({
        m for it in items for m in it["scores"]
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, max(4, len(felter) * 0.5 + 1.5)),
        gridspec_kw={"width_ratios": [1, 2]}
    )

    # Venstre: spredning
    farver = [RØD if s >= 50 else GUL for s in spredning]
    ax1.barh(felter, spredning, color=farver, height=0.6, edgecolor="white")
    ax1.set_xlabel("Spredning (pp)")
    ax1.set_title("Forskel mellem\nbedste og dårligste model")

    # Højre: alle modellers scores per felt
    ax2.set_facecolor("white")
    ax2.axvline(50, color=GUL,  linestyle="--", linewidth=1, alpha=0.6)
    ax2.axvline(75, color=GRØN, linestyle="--", linewidth=1, alpha=0.6)

    for j, it in enumerate(items_r):
        for i, (navn, score) in enumerate(it["scores"].items()):
            farve = MODEL_FARVER[
                list({m for item in items for m in item["scores"]}.intersection(
                    {navn}
                ) and [model_navne_i.index(navn)] or [0])[0] % len(MODEL_FARVER)
            ]
            ax2.scatter(score * 100, j, color=farve, s=80, zorder=3,
                        label=navn if j == 0 else "")

        # Linje fra min til max
        vals = list(it["scores"].values())
        ax2.plot(
            [min(vals)*100, max(vals)*100], [j, j],
            color="#ccc", linewidth=2, zorder=2
        )

    ax2.set_yticks(range(len(felter)))
    ax2.set_yticklabels(felter, fontsize=8)
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Score (%)")
    ax2.set_title("Score per model")
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("Felter med størst forskel mellem modeller", fontweight="bold")
    fig.tight_layout()
    return gem(fig, ud_mappe, "5_spredning.png")


def plot_oversigt(model_navne, model_felt_stats, model_gruppe_stats,
                  svære_for_alle, stor_spredning, ud_mappe):
    """Samlet overbliksfigur – 2×2 grid."""
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor(BAGGRUND)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # 1: Rangering
    ax1 = fig.add_subplot(gs[0, 0])
    scores = sorted(
        [(n, (samlet_score(model_felt_stats[n]) or 0) * 100) for n in model_navne],
        key=lambda x: x[1]
    )
    farver = [MODEL_FARVER[i % len(MODEL_FARVER)] for i in range(len(scores))]
    bars = ax1.barh([s[0] for s in scores], [s[1] for s in scores],
                    color=farver, height=0.55, edgecolor="white")
    for bar, (_, v) in zip(bars, scores):
        ax1.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                 "{:.0f}%".format(v), va="center", ha="left", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, 110)
    ax1.axvline(75, color=GRØN, linestyle="--", linewidth=1, alpha=0.6)
    ax1.axvline(50, color=GUL,  linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_xlabel("Score %")
    ax1.set_title("Model rangering")
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    # 2: Top svære felter for alle
    ax2 = fig.add_subplot(gs[0, 1])
    if svære_for_alle:
        top_sv = svære_for_alle[:8]
        ax2.barh(
            [it["felt"].split(".")[-1] for it in reversed(top_sv)],
            [it["gns_score"]*100 for it in reversed(top_sv)],
            color=RØD, height=0.55, edgecolor="white"
        )
        ax2.set_xlim(0, 60)
        ax2.set_xlabel("Gns. score %")
    else:
        ax2.text(0.5, 0.5, "Ingen felter\nsvære for alle", ha="center", va="center",
                 transform=ax2.transAxes, color="#999", fontsize=11)
    ax2.set_title("Svære felter (alle modeller)")

    # 3: Gruppe mini-heatmap
    ax3 = fig.add_subplot(gs[1, :])
    alle_grupper = sorted({
        g for gs_d in model_gruppe_stats.values() for g in gs_d
    })
    matrix = np.full((len(model_navne), len(alle_grupper)), np.nan)
    for i, navn in enumerate(model_navne):
        for j, gruppe in enumerate(alle_grupper):
            sc = model_gruppe_stats.get(navn, {}).get(gruppe, {}).get("gns_score")
            if sc is not None:
                matrix[i, j] = sc * 100

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ag", [(0,RØD),(0.5,GUL),(1,GRØN)], N=256)
    im = ax3.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    for i in range(len(model_navne)):
        for j in range(len(alle_grupper)):
            v = matrix[i, j]
            if not np.isnan(v):
                tc = "white" if v < 35 or v > 80 else "#333"
                ax3.text(j, i, "{:.0f}%".format(v), ha="center", va="center",
                         fontsize=8, fontweight="bold", color=tc)
    ax3.set_xticks(range(len(alle_grupper)))
    ax3.set_xticklabels([g[:16] for g in alle_grupper], rotation=25, ha="right", fontsize=8)
    ax3.set_yticks(range(len(model_navne)))
    ax3.set_yticklabels(model_navne, fontsize=9)
    ax3.set_title("Agreement per gruppe per model")
    fig.colorbar(im, ax=ax3, fraction=0.015, pad=0.01).ax.tick_params(labelsize=7)

    fig.suptitle("Model-sammenligning – PSG Analyse", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return gem(fig, ud_mappe, "0_oversigt.png")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Sammenligner N modellers PSG-output mod ground truth (menneskelige annotationer)"
    )
    p.add_argument("--ground-truth", "-g", required=True,
                   help="Mappe med menneskelige (ground truth) JSON-filer")
    p.add_argument("--modeller", "-m", nargs="+", required=True,
                   help="En eller flere mapper med model-output")
    p.add_argument("--navne", "-n", nargs="+",
                   help="Navne til modellerne (samme rækkefølge som --modeller). "
                        "Standard: mappenavnene bruges.")
    p.add_argument("--output", "-o",
                   help="Gem fuld rapport som JSON")
    p.add_argument("--plots", "-p",
                   help="Mappe til PNG-plots (udelad for ingen plots)")
    p.add_argument("--top", type=int, default=20,
                   help="Antal felter i top-N lister (standard: 20)")
    args = p.parse_args()

    # Navne til modeller
    if args.navne:
        if len(args.navne) != len(args.modeller):
            print("FEJL: --navne skal have samme antal argumenter som --modeller.")
            sys.exit(1)
        model_navne = args.navne
    else:
        model_navne = [Path(m).name for m in args.modeller]

    print("Ground truth: {}".format(args.ground_truth))
    print("Modeller:")
    for navn, mappe in zip(model_navne, args.modeller):
        print("  {:20} → {}".format(navn, mappe))
    print()

    # Indlæs og sammenlign per model
    model_alle_par      = {}   # model_navn -> {par_id: {felt: info}}
    model_felt_stats    = {}
    model_gruppe_stats  = {}
    n_par_per_model     = {}

    for navn, mappe in zip(model_navne, args.modeller):
        par = match_filer(args.ground_truth, mappe)
        if not par:
            print("ADVARSEL: Ingen matchede par for model '{}'.".format(navn))
            continue

        print("Model '{}': {} par fundet.".format(navn, len(par)))
        n_par_per_model[navn] = len(par)

        alle_par_res = {}
        for gt_fil, mod_fil in par:
            gt_ann  = indlæs_json(gt_fil)
            mod_ann = indlæs_json(mod_fil)
            par_id  = "{} ↔ {}".format(gt_fil.name, mod_fil.name)
            alle_par_res[par_id] = sammenlign_model(mod_ann, gt_ann)

        model_alle_par[navn]     = alle_par_res
        fs, gs = aggreger_model(alle_par_res)
        model_felt_stats[navn]   = fs
        model_gruppe_stats[navn] = gs

    if not model_felt_stats:
        print("FEJL: Ingen data at analysere.")
        sys.exit(1)

    # Tværgående analyse
    svære_for_alle, stor_spredning = analyser_tværgående(model_felt_stats, model_navne)

    # Tekstrapport
    print()
    print_rapport(
        model_navne, model_felt_stats, model_gruppe_stats,
        svære_for_alle, stor_spredning, n_par_per_model
    )

    # Plots
    if args.plots:
        sæt_stil()
        Path(args.plots).mkdir(parents=True, exist_ok=True)
        print("\nGenererer plots → {}".format(args.plots))
        plot_oversigt(model_navne, model_felt_stats, model_gruppe_stats,
                      svære_for_alle, stor_spredning, args.plots)
        plot_model_rangering(model_navne, model_felt_stats, model_gruppe_stats, args.plots)
        plot_gruppe_heatmap(model_navne, model_gruppe_stats, args.plots)
        plot_fejlprofil(model_navne, model_felt_stats, args.plots)
        plot_svære_felter(svære_for_alle, model_navne, args.plots, top_n=args.top)
        plot_spredning(stor_spredning, args.plots, top_n=args.top)
        print("Færdig! {} plots gemt.".format(len(list(Path(args.plots).glob("*.png")))))

    # JSON output
    if args.output:
        ud = {
            "model_navne": model_navne,
            "n_par_per_model": n_par_per_model,
            "model_felt_stats": {
                navn: {felt: {k: v for k, v in s.items()} for felt, s in fs.items()}
                for navn, fs in model_felt_stats.items()
            },
            "model_gruppe_stats": {
                navn: dict(gs) for navn, gs in model_gruppe_stats.items()
            },
            "svaere_for_alle": svære_for_alle,
            "stor_spredning": stor_spredning,
        }
        Path(args.output).write_text(
            json.dumps(ud, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\nFuld rapport gemt: {}".format(args.output))


if __name__ == "__main__":
    main()