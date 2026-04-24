"""
llm_eval.py  -  Sammenligner LLM-udfyldninger med menneskelige annotations.

Matcher JSON-filer på tværs af to mapper og opsummerer:
  - Felt-for-felt uenighed sorteret efter hyppighed
  - Emne-niveau opsummering (hvilke sektioner er sværest for LLM'en)
  - Numerisk nøjagtighed med tolerance
  - Token F1 for fritekstfelter
  - En samlet "LLM-score" per emnegruppe

Brug:
  python llm_eval.py --llm <llm-mappe> --menneske <menneske-mappe>
  python llm_eval.py --llm <llm-mappe> --menneske <menneske-mappe> --output rapport.json
  python llm_eval.py --llm <llm-mappe> --menneske <menneske-mappe> --vis-enige

Krav:
  pip install krippendorff
"""

import re
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import krippendorff
except ImportError:
    print("FEJL: krippendorff er ikke installeret.")
    print("Kør:  pip install krippendorff")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Feltklassifikation  (samme logik som psg_agreement.py)
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

# Emnegrupper – bruges til opsummering på sektionsniveau
EMNEGRUPPER = {
    "Patient & metadata":       ["patient.", "metadata."],
    "Test oplysninger":         ["test_oplysninger."],
    "Klinisk information":      ["klinisk_information."],
    "Søvn opsummering":         ["soevn_opsummering."],
    "Søvnstadier":              ["soevnstadier."],
    "Øget muskelaktivitet REM": ["oget_muskelaktivitet_rem."],
    "Respirationsanalyse":      ["respirations_analyse."],
    "SpO2 oversigt":            ["spo2_oversigt."],
    "Benb\u00e6vegelser":       ["benbevaegelser."],
    "Hjerte":                   ["hjerte."],
    "Sammenfatning":            ["sammenfatning."],
    "Konklusion og plan":       ["konklusion_og_plan."],
}


# ---------------------------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------------------------

def emnegruppe_for_felt(felt):
    for gruppe, præfikser in EMNEGRUPPER.items():
        for præfiks in præfikser:
            if felt.startswith(præfiks):
                return gruppe
    return "Andet"


def fladgør(d, prefix=""):
    items = {}
    for k, v in d.items():
        key = "{}.{}".format(prefix, k) if prefix else k
        if isinstance(v, dict):
            items.update(fladgør(v, key))
        else:
            items[key] = v
    return items


def normaliser_værdi(værdi):
    if værdi is None:
        return None
    if isinstance(værdi, bool):
        return værdi
    if isinstance(værdi, (int, float)):
        return værdi

    v = str(værdi).strip()
    if v in ("", "-", "N/A", "n/a", "None", "null"):
        return None

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

    dato_match = re.match(r"^(\d{1,4})[-/\.](\d{1,2})[-/\.](\d{1,4})$", v)
    if dato_match:
        a, b, c = dato_match.groups()
        if len(a) == 4:
            return "{:02d}-{:02d}-{}".format(int(c), int(b), a)
        elif len(c) == 4:
            return "{:02d}-{:02d}-{}".format(int(a), int(b), c)

    return v.lower()


def token_f1(a, b):
    if not a and not b:
        return 1.0, 1.0, 1.0
    if not a or not b:
        return 0.0, 0.0, 0.0
    tokens_a = set(str(a).lower().split())
    tokens_b = set(str(b).lower().split())
    overlap = tokens_a & tokens_b
    if not overlap:
        return 0.0, 0.0, 0.0
    precision = len(overlap) / len(tokens_b)
    recall    = len(overlap) / len(tokens_a)
    f1        = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def tolerance_match(a, b, tolerance=0.05):
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) <= tolerance
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance


def indlæs_json(sti):
    tekst = Path(sti).read_text(encoding="utf-8")
    filnavn = Path(sti).name
    try:
        data = json.loads(tekst)
    except json.JSONDecodeError as e:
        linjer = tekst.split("\n")
        fejl_linje = e.lineno - 1
        start = max(0, fejl_linje - 2)
        slut  = min(len(linjer), fejl_linje + 3)
        kontekst = "\n".join(
            "  {:3d}: {}".format(i + 1, linjer[i])
            for i in range(start, slut)
        )
        print("\nFEJL: Kan ikke parse {}".format(filnavn))
        print("Fejl: {} (linje {}, kolonne {})".format(e.msg, e.lineno, e.colno))
        print("Kontekst:\n{}".format(kontekst))
        sys.exit(1)
    flad = fladgør(data)
    return {felt: normaliser_værdi(v) for felt, v in flad.items()}


def match_filer(llm_mappe, menneske_mappe):
    """
    Matcher filer fra de to mapper. Forsøger:
      1. Eksakt filnavn-match
      2. Match på fælles nummer i filnavnet (trial1, patient3, osv.)
    """
    llm_filer     = {f.name: f for f in Path(llm_mappe).glob("*.json")}
    menneske_filer = {f.name: f for f in Path(menneske_mappe).glob("*.json")}

    par = []
    uparrede_llm      = []
    uparrede_menneske = []

    # 1. Eksakt match
    for navn in sorted(llm_filer):
        if navn in menneske_filer:
            par.append((llm_filer[navn], menneske_filer[navn]))
        else:
            uparrede_llm.append(navn)

    for navn in sorted(menneske_filer):
        if navn not in llm_filer:
            uparrede_menneske.append(navn)

    # 2. Nummer-baseret match for uparrede filer
    def udtræk_nummer(navn):
        m = re.search(r"(\d+)", navn)
        return int(m.group(1)) if m else None

    llm_num      = {udtræk_nummer(n): n for n in uparrede_llm if udtræk_nummer(n) is not None}
    menneske_num = {udtræk_nummer(n): n for n in uparrede_menneske if udtræk_nummer(n) is not None}

    stadig_uparret_llm      = list(uparrede_llm)
    stadig_uparret_menneske = list(uparrede_menneske)

    for num in sorted(set(llm_num) & set(menneske_num)):
        ln = llm_num[num]
        mn = menneske_num[num]
        par.append((llm_filer[ln], menneske_filer[mn]))
        if ln in stadig_uparret_llm:
            stadig_uparret_llm.remove(ln)
        if mn in stadig_uparret_menneske:
            stadig_uparret_menneske.remove(mn)

    return par, stadig_uparret_llm, stadig_uparret_menneske


# ---------------------------------------------------------------------------
# Sammenligning af ét par
# ---------------------------------------------------------------------------

def sammenlign_par(llm_ann, menneske_ann, llm_navn, menneske_navn):
    """
    Sammenligner LLM vs. menneske for ét filpar.
    Returnerer en dict med resultater per felt.
    """
    alle_felter = (set(llm_ann) | set(menneske_ann)) - SKIP_FELTER

    resultater = {}

    for felt in sorted(alle_felter):
        llm_v      = llm_ann.get(felt)
        menneske_v = menneske_ann.get(felt)
        gruppe     = emnegruppe_for_felt(felt)

        # Bestem felttype
        if felt in FRITEKST_FELTER:
            felttype = "fritekst"
        elif felt in NOMINALE_FELTER:
            felttype = "nominal"
        elif isinstance(llm_v, (int, float)) or isinstance(menneske_v, (int, float)):
            felttype = "numerisk"
        else:
            felttype = "nominal"

        # Begge mangler
        if llm_v is None and menneske_v is None:
            resultater[felt] = {
                "gruppe":    gruppe,
                "felttype":  felttype,
                "llm":       None,
                "menneske":  None,
                "status":    "begge_mangler",
                "enige":     None,
                "f1":        None,
                "tolerance": None,
            }
            continue

        # LLM mangler, menneske har udfyldt
        if llm_v is None:
            resultater[felt] = {
                "gruppe":    gruppe,
                "felttype":  felttype,
                "llm":       None,
                "menneske":  menneske_v,
                "status":    "llm_mangler",
                "enige":     False,
                "f1":        None,
                "tolerance": None,
            }
            continue

        # Menneske mangler, LLM har udfyldt
        if menneske_v is None:
            resultater[felt] = {
                "gruppe":    gruppe,
                "felttype":  felttype,
                "llm":       llm_v,
                "menneske":  None,
                "status":    "menneske_mangler",
                "enige":     False,
                "f1":        None,
                "tolerance": None,
            }
            continue

        # Begge har udfyldt – sammenlign
        if felttype == "fritekst":
            _, _, f1 = token_f1(llm_v, menneske_v)
            enige = f1 >= 0.8  # Tærskel for "enige" på fritekst
            resultater[felt] = {
                "gruppe":    gruppe,
                "felttype":  felttype,
                "llm":       llm_v,
                "menneske":  menneske_v,
                "status":    "udfyldt",
                "enige":     enige,
                "f1":        round(f1, 3),
                "tolerance": None,
            }

        elif felttype == "numerisk":
            try:
                llm_f  = float(str(llm_v).replace(",", "."))
                men_f  = float(str(menneske_v).replace(",", "."))
                exact  = (llm_f == men_f)
                tol    = tolerance_match(llm_f, men_f)
                resultater[felt] = {
                    "gruppe":    gruppe,
                    "felttype":  felttype,
                    "llm":       llm_v,
                    "menneske":  menneske_v,
                    "status":    "udfyldt",
                    "enige":     exact,
                    "f1":        None,
                    "tolerance": tol,
                    "afvigelse": round(abs(llm_f - men_f), 4),
                }
            except (TypeError, ValueError):
                enige = str(llm_v).lower().strip() == str(menneske_v).lower().strip()
                resultater[felt] = {
                    "gruppe":   gruppe,
                    "felttype": felttype,
                    "llm":      llm_v,
                    "menneske": menneske_v,
                    "status":   "udfyldt",
                    "enige":    enige,
                    "f1":       None,
                    "tolerance": None,
                }

        else:  # nominal
            enige = str(llm_v).lower().strip() == str(menneske_v).lower().strip()
            resultater[felt] = {
                "gruppe":    gruppe,
                "felttype":  felttype,
                "llm":       llm_v,
                "menneske":  menneske_v,
                "status":    "udfyldt",
                "enige":     enige,
                "f1":        None,
                "tolerance": None,
            }

    return resultater


# ---------------------------------------------------------------------------
# Aggregér på tværs af alle par
# ---------------------------------------------------------------------------

def aggreger(alle_par_resultater):
    """
    Samler resultater fra alle filpar til:
      - Per-felt statistik
      - Per-gruppe statistik
    """
    felt_stats  = defaultdict(lambda: {
        "felttype": None,
        "gruppe": None,
        "n_sammenlignet": 0,
        "n_enige": 0,
        "n_uenige": 0,
        "n_llm_mangler": 0,
        "n_menneske_mangler": 0,
        "n_begge_mangler": 0,
        "f1_sum": 0.0,
        "f1_count": 0,
        "tol_match": 0,
        "tol_total": 0,
        "afvigelser": [],
    })

    for par_navn, resultater in alle_par_resultater.items():
        for felt, info in resultater.items():
            s = felt_stats[felt]
            s["felttype"] = info["felttype"]
            s["gruppe"]   = info["gruppe"]

            status = info["status"]
            if status == "begge_mangler":
                s["n_begge_mangler"] += 1
            elif status == "llm_mangler":
                s["n_llm_mangler"] += 1
                s["n_uenige"] += 1
                s["n_sammenlignet"] += 1
            elif status == "menneske_mangler":
                s["n_menneske_mangler"] += 1
                s["n_uenige"] += 1
                s["n_sammenlignet"] += 1
            else:
                s["n_sammenlignet"] += 1
                if info["enige"]:
                    s["n_enige"] += 1
                else:
                    s["n_uenige"] += 1

                if info["f1"] is not None:
                    s["f1_sum"]   += info["f1"]
                    s["f1_count"] += 1

                if info["tolerance"] is not None:
                    s["tol_total"] += 1
                    if info["tolerance"]:
                        s["tol_match"] += 1

                if info.get("afvigelse") is not None:
                    s["afvigelser"].append(info["afvigelse"])

    # Beregn agreement-rate per felt
    for felt, s in felt_stats.items():
        n = s["n_sammenlignet"]
        s["agreement_rate"] = round(s["n_enige"] / n, 3) if n > 0 else None
        s["uenighed_rate"]  = round(s["n_uenige"] / n, 3) if n > 0 else None
        s["gns_f1"]         = round(s["f1_sum"] / s["f1_count"], 3) if s["f1_count"] > 0 else None
        s["tol_match_rate"] = (
            round(s["tol_match"] / s["tol_total"], 3) if s["tol_total"] > 0 else None
        )
        s["gns_afvigelse"]  = (
            round(sum(s["afvigelser"]) / len(s["afvigelser"]), 4)
            if s["afvigelser"] else None
        )

    # Gruppe-aggregering
    gruppe_stats = defaultdict(lambda: {
        "n_felter": 0,
        "n_sammenlignet": 0,
        "n_enige": 0,
        "n_uenige": 0,
        "n_llm_mangler": 0,
        "f1_sum": 0.0,
        "f1_count": 0,
    })

    for felt, s in felt_stats.items():
        g = gruppe_stats[s["gruppe"]]
        g["n_felter"]       += 1
        g["n_sammenlignet"] += s["n_sammenlignet"]
        g["n_enige"]        += s["n_enige"]
        g["n_uenige"]       += s["n_uenige"]
        g["n_llm_mangler"]  += s["n_llm_mangler"]
        if s["f1_count"] > 0:
            g["f1_sum"]   += s["f1_sum"]
            g["f1_count"] += s["f1_count"]

    for gruppe, g in gruppe_stats.items():
        n = g["n_sammenlignet"]
        g["agreement_rate"] = round(g["n_enige"] / n, 3) if n > 0 else None
        g["gns_f1"]         = (
            round(g["f1_sum"] / g["f1_count"], 3) if g["f1_count"] > 0 else None
        )

    return dict(felt_stats), dict(gruppe_stats)


# ---------------------------------------------------------------------------
# Rapport-print
# ---------------------------------------------------------------------------

UENIGHED_SYMBOL = {
    (0.0,  0.5):  "🔴",
    (0.5,  0.75): "🟡",
    (0.75, 1.01): "🟢",
}

def symbol(rate):
    if rate is None:
        return "⚪"
    for (lo, hi), sym in UENIGHED_SYMBOL.items():
        if lo <= rate < hi:
            return sym
    return "⚪"


def print_rapport(
    felt_stats, gruppe_stats, alle_par_resultater,
    n_par, vis_enige=False
):
    linje  = "=" * 70
    streg  = "─" * 70

    print(linje)
    print("LLM vs. MENNESKE  –  PSG UDFYLDNINGS-RAPPORT")
    print("{} filpar analyseret".format(n_par))
    print(linje)

    # ── EMNEGRUPPE-OPSUMMERING ──────────────────────────────────────────
    print("\n── EMNEGRUPPE-OPSUMMERING " + "─" * 44)
    print("  {:<28} {:>8} {:>10} {:>10} {:>10}".format(
        "Emne", "Sym", "Agreement", "Uenige", "LLM man."
    ))
    print("  " + "─" * 66)

    # Sorter grupper efter agreement rate (lavest øverst)
    sorterede_grupper = sorted(
        gruppe_stats.items(),
        key=lambda x: (x[1]["agreement_rate"] or 1.0)
    )

    for gruppe, g in sorterede_grupper:
        rate = g["agreement_rate"]
        sym  = symbol(rate)
        rate_str = "{:.0f} %".format(rate * 100) if rate is not None else "N/A"
        uenige   = g["n_uenige"]
        llm_man  = g["n_llm_mangler"]
        print("  {:<28} {:>4} {:>10} {:>10} {:>10}".format(
            gruppe[:28], sym, rate_str, uenige, llm_man
        ))

    print()
    print("  🔴 < 50 %   🟡 50–75 %   🟢 > 75 %   ⚪ ikke sammenlignet")

    # ── FELT-FOR-FELT UENIGHED ──────────────────────────────────────────
    print("\n\n── FELT-FOR-FELT UENIGHED (sorteret efter hyppighed) " + "─" * 17)

    uenige_felter = {
        felt: s for felt, s in felt_stats.items()
        if s["n_uenige"] > 0
    }
    sorteret = sorted(
        uenige_felter.items(),
        key=lambda x: (-x[1]["n_uenige"], x[1]["agreement_rate"] or 1.0)
    )

    if not sorteret:
        print("  Ingen uenigheder fundet! 🎉")
    else:
        print("  {:<48} {:>5} {:>8} {:>8} {:>7}".format(
            "Felt", "Uenige", "Agree%", "F1", "Tol%"
        ))
        print("  " + "─" * 76)

        for felt, s in sorteret:
            agree_str = (
                "{:.0f} %".format(s["agreement_rate"] * 100)
                if s["agreement_rate"] is not None else "N/A"
            )
            f1_str  = "{:.2f}".format(s["gns_f1"])   if s["gns_f1"]         is not None else "-"
            tol_str = "{:.0f} %".format(s["tol_match_rate"] * 100) if s["tol_match_rate"] is not None else "-"
            sym     = symbol(s["agreement_rate"])

            # Forkort feltnavn
            felt_kort = felt
            if len(felt) > 46:
                felt_kort = "…" + felt[-45:]

            print("  {} {:<46} {:>5} {:>8} {:>8} {:>7}".format(
                sym, felt_kort,
                s["n_uenige"],
                agree_str, f1_str, tol_str
            ))

            # Vis detaljer for de mest problematiske felter
            if s["n_uenige"] >= 2 or s["agreement_rate"] is not None and s["agreement_rate"] < 0.5:
                if s["n_llm_mangler"] > 0:
                    print("    ↳ LLM manglede at udfylde: {} gange".format(s["n_llm_mangler"]))
                if s["n_menneske_mangler"] > 0:
                    print("    ↳ Menneske manglede at udfylde: {} gange".format(s["n_menneske_mangler"]))
                if s["gns_afvigelse"] is not None:
                    print("    ↳ Gns. numerisk afvigelse: {}".format(s["gns_afvigelse"]))

    # ── FELTER HVOR LLM KONSEKVENT MANGLER ─────────────────────────────
    print("\n\n── FELTER HVOR LLM KONSEKVENT MANGLER ─────────────────────────────")
    llm_mangler = sorted(
        [(felt, s) for felt, s in felt_stats.items() if s["n_llm_mangler"] == n_par],
        key=lambda x: x[0]
    )
    if llm_mangler:
        print("  (LLM udfyldte ALDRIG disse felter, mens menneske gjorde det i alle par)")
        for felt, s in llm_mangler:
            print("  • {}".format(felt))
    else:
        print("  Ingen felter mangler konsekvent.")

    # ── FRITEKST DETALJER ───────────────────────────────────────────────
    fritekst_felter = {
        felt: s for felt, s in felt_stats.items()
        if s["felttype"] == "fritekst" and s["gns_f1"] is not None
    }
    if fritekst_felter:
        print("\n\n── FRITEKST TOKEN F1 (lavest øverst) " + "─" * 32)
        print("  {:<50} {:>8} {:>8}".format("Felt", "Gns. F1", "N"))
        print("  " + "─" * 66)
        for felt, s in sorted(fritekst_felter.items(), key=lambda x: x[1]["gns_f1"] or 1.0):
            sym = symbol(s["gns_f1"])
            print("  {} {:<48} {:>8.3f} {:>8}".format(
                sym, felt[-48:], s["gns_f1"], s["f1_count"]
            ))

    # ── ENIGE FELTER ────────────────────────────────────────────────────
    if vis_enige:
        enige = [
            felt for felt, s in felt_stats.items()
            if s["agreement_rate"] is not None and s["agreement_rate"] == 1.0
        ]
        if enige:
            print("\n\n── FELTER MED PERFEKT AGREEMENT ─────────────────────────────────")
            for felt in sorted(enige):
                print("  ✓ {}".format(felt))

    # ── SAMLET OPSUMMERING ──────────────────────────────────────────────
    print("\n\n── SAMLET OPSUMMERING " + "─" * 48)
    total_sammenlignet = sum(s["n_sammenlignet"] for s in felt_stats.values())
    total_enige        = sum(s["n_enige"]        for s in felt_stats.values())
    total_uenige       = sum(s["n_uenige"]       for s in felt_stats.values())
    total_llm_man      = sum(s["n_llm_mangler"]  for s in felt_stats.values())

    fritekst_f1_vals = [
        s["gns_f1"] for s in felt_stats.values()
        if s["gns_f1"] is not None
    ]

    print("  Sammenlignet (felt × par):     {:>6}".format(total_sammenlignet))
    print("  Enige:                         {:>6}  ({:.1f} %)".format(
        total_enige,
        100 * total_enige / total_sammenlignet if total_sammenlignet else 0
    ))
    print("  Uenige:                        {:>6}  ({:.1f} %)".format(
        total_uenige,
        100 * total_uenige / total_sammenlignet if total_sammenlignet else 0
    ))
    print("  LLM manglede at udfylde:       {:>6}".format(total_llm_man))
    if fritekst_f1_vals:
        print("  Gns. fritekst token F1:        {:>8.3f}".format(
            sum(fritekst_f1_vals) / len(fritekst_f1_vals)
        ))
    print()
    print(linje)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Sammenligner LLM PSG-output med menneskelige annotationer"
    )
    p.add_argument("--llm",      required=True, help="Mappe med LLM JSON-filer")
    p.add_argument("--menneske", required=True, help="Mappe med menneskelige JSON-filer")
    p.add_argument("--output",   "-o", help="Gem fuld rapport som JSON")
    p.add_argument(
        "--vis-enige", action="store_true",
        help="Vis også felter med perfekt agreement"
    )
    args = p.parse_args()

    # Match filer
    par, uparret_llm, uparret_menneske = match_filer(args.llm, args.menneske)

    if not par:
        print("FEJL: Ingen matchede filpar fundet.")
        print("LLM-mappe:      {}".format(args.llm))
        print("Menneske-mappe: {}".format(args.menneske))
        sys.exit(1)

    print("Fandt {} matchede filpar:\n".format(len(par)))
    for llm_fil, men_fil in par:
        print("  {} ↔ {}".format(llm_fil.name, men_fil.name))

    if uparret_llm:
        print("\nAdvarsel: Følgende LLM-filer har ingen match:")
        for f in uparret_llm:
            print("  - {}".format(f))
    if uparret_menneske:
        print("\nAdvarsel: Følgende menneske-filer har ingen match:")
        for f in uparret_menneske:
            print("  - {}".format(f))

    print()

    # Sammenlign alle par
    alle_par_resultater = {}
    for llm_fil, men_fil in par:
        llm_ann      = indlæs_json(llm_fil)
        menneske_ann = indlæs_json(men_fil)
        par_navn     = "{} ↔ {}".format(llm_fil.name, men_fil.name)
        alle_par_resultater[par_navn] = sammenlign_par(
            llm_ann, menneske_ann, llm_fil.name, men_fil.name
        )

    # Aggregér og print
    felt_stats, gruppe_stats = aggreger(alle_par_resultater)
    print_rapport(
        felt_stats, gruppe_stats, alle_par_resultater,
        n_par=len(par),
        vis_enige=args.vis_enige
    )

    # Gem JSON-output
    if args.output:
        output = {
            "n_par": len(par),
            "par": [
                {"llm": str(llm_fil), "menneske": str(men_fil)}
                for llm_fil, men_fil in par
            ],
            "felt_stats": {
                felt: {k: v for k, v in s.items() if k != "afvigelser"}
                for felt, s in felt_stats.items()
            },
            "gruppe_stats": dict(gruppe_stats),
            "detaljer_per_par": alle_par_resultater,
        }
        Path(args.output).write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print("\nFuld rapport gemt: {}".format(args.output))


if __name__ == "__main__":
    main()