"""
psg_agreement.py  -  Inter-annotator agreement for PSG JSON-filer.

Sammenligner to eller flere JSON-filer med samme struktur og beregner:
  - Krippendorff's alpha (nominal og interval) - samlet og parvis
  - Token F1 for fritekstfelter
  - Tolerance-match og exact match for numeriske felter
  - Coverage (andel udfyldte felter)
  - Felt-for-felt oversigt over uoverensstemmelser

Krav:
  pip install krippendorff

Brug:
  python psg_agreement.py a.json b.json
  python psg_agreement.py a.json b.json c.json
  python psg_agreement.py a.json b.json --output rapport.json
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

ALPHA_MIN_OBSERVATIONER = 10


# ---------------------------------------------------------------------------
# Feltklassifikation
# ---------------------------------------------------------------------------

SKIP_FELTER = {
    "metadata.filnavn",
    "metadata.udfyldt_af",
    "metadata.model_parametre",
}

FRITEKST_FELTER = {
    "klinisk_information.klinisk_resume",
    "klinisk_information.kommentar",
    "klinisk_information.patient_oplysninger_ved_optagelse",
    "konklusion_og_plan.konklusion_tekst",
    "konklusion_og_plan.plan",
    "konklusion_og_plan.sammenfatning_soevmnoenster",
    "konklusion_og_plan.sammenfatning_soevn_dagtid",
}

NOMINALE_FELTER = {
    "patient.koen",
    "patient.navn",
    "patient.cpr-nummer",
    "patient.foedselsdag",
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


# ---------------------------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------------------------

def fladgør(d, prefix=""):
    """Konverterer nested dict til flad dict med punktum-separerede nøgler."""
    items = {}
    for k, v in d.items():
        key = "{}.{}".format(prefix, k) if prefix else k
        if isinstance(v, dict):
            items.update(fladgør(v, key))
        else:
            items[key] = v
    return items


def normaliser_værdi(værdi):
    """
    Normaliserer en værdi inden sammenligning.
    Kører ens på alle annotatorer - ingen bias.
    """
    if værdi is None:
        return None
    if isinstance(værdi, bool):
        return værdi
    if isinstance(værdi, (int, float)):
        return værdi

    v = str(værdi).strip()

    if v in ("", "-", "N/A", "n/a", "None", "null"):
        return None

    # Fjern enheder
    v = re.sub(
        r"\s*(cm|kg|%|/h|/t|min\.|min|slag per minut|kg/m²|kPa)\s*$",
        "", v, flags=re.IGNORECASE
    ).strip()

    # Komma -> punktum hvis det ligner et tal
    if re.match(r"^-?[\d]+,[\d]+$", v):
        v = v.replace(",", ".")

    # Forsøg konvertering til tal
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except ValueError:
        pass

    # Normaliser datoformat til DD-MM-ÅÅÅÅ
    dato_match = re.match(r"^(\d{1,4})[-/\.](\d{1,2})[-/\.](\d{1,4})$", v)
    if dato_match:
        a, b, c = dato_match.groups()
        if len(a) == 4:
            return "{:02d}-{:02d}-{}".format(int(c), int(b), a)
        elif len(c) == 4:
            return "{:02d}-{:02d}-{}".format(int(a), int(b), c)

    return v.lower()


def normaliser_annotation(annotation):
    """Kører normaliser_værdi på alle felter i en flad annotation-dict."""
    return {felt: normaliser_værdi(v) for felt, v in annotation.items()}


def token_f1(a, b):
    """Beregner F1-score på token-niveau mellem to strenge."""
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
    """Returnerer True hvis to tal er inden for relativ tolerance."""
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) <= tolerance
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance


def alpha_med_pålidelighedstjek(værdier, niveau, felt_type):
    """
    Beregner Krippendorff's alpha og returnerer både værdien
    og metadata om pålidelighed.
    """
    transposed = list(zip(*værdier))
    par_med_data = [
        row for row in transposed
        if any(v is not None for v in row)
    ]
    par_begge_data = [
        row for row in transposed
        if all(v is not None for v in row)
    ]
    par_kun_null = [
        row for row in transposed
        if all(v is None for v in row)
    ]

    n_observationer = len(par_begge_data)
    n_kun_null      = len(par_kun_null)
    n_en_mangler    = len(par_med_data) - n_observationer
    pålidelig       = n_observationer >= ALPHA_MIN_OBSERVATIONER

    if niveau == "nominal":
        data = []
        for annotator in værdier:
            row = [str(v).lower() if v is not None else "*" for v in annotator]
            data.append(row)
        try:
            alpha_værdi = krippendorff.alpha(
                reliability_data=data,
                level_of_measurement="nominal",
                missing_items={"*"}
            )
        except Exception:
            alpha_værdi = None

    elif niveau == "interval":
        data = []
        for annotator in værdier:
            row = [
                float(str(v).replace(",", ".")) if v is not None else float("nan")
                for v in annotator
            ]
            data.append(row)
        try:
            alpha_værdi = krippendorff.alpha(
                reliability_data=data,
                level_of_measurement="interval"
            )
        except Exception:
            alpha_værdi = None

    return {
        "alpha": round(alpha_værdi, 3) if alpha_værdi is not None else None,
        "n_par_begge_udfyldt": n_observationer,
        "n_par_kun_null": n_kun_null,
        "n_par_en_mangler": n_en_mangler,
        "paalidelig": pålidelig,
        "advarsel": (
            "Baseret på kun {} par med data – alpha er upålidelig (minimum {})".format(
                n_observationer, ALPHA_MIN_OBSERVATIONER
            ) if not pålidelig else None
        )
    }


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def analyser_par(annotationer, filnavne, numeriske, nominale, fritekst):
    """Beregner agreement metrics for ét sæt annotatorer."""

    alle_felter = set()
    for ann in annotationer:
        alle_felter.update(ann.keys())
    alle_felter -= SKIP_FELTER

    rapport = {
        "filer": filnavne,
        "coverage": {},
        "numeriske": {},
        "nominale": {},
        "fritekst": {},
        "opsummering": {}
    }

    # Coverage
    for ann, navn in zip(annotationer, filnavne):
        udfyldte = sum(1 for f in alle_felter if ann.get(f) is not None)
        rapport["coverage"][navn] = {
            "udfyldte": udfyldte,
            "total": len(alle_felter),
            "procent": round(100 * udfyldte / len(alle_felter), 1)
        }

    # Numeriske
    num_exact_total = 0
    num_exact_match = 0
    num_tol_match   = 0
    num_alpha_data  = defaultdict(list)

    for felt in numeriske:
        værdier = [ann.get(felt) for ann in annotationer]
        værdier = [
            float(re.sub(r"[^\d\.\-]", "", str(v).replace(",", "."))) 
            if isinstance(v, str) else v
            for v in værdier
        ]

        par = [
            (værdier[i], værdier[j])
            for i in range(len(værdier))
            for j in range(i+1, len(værdier))
        ]

        felt_exact = []
        felt_tol   = []
        for a, b in par:
            if a is None and b is None:
                continue
            num_exact_total += 1
            exact = (a == b)
            tol   = tolerance_match(a, b)
            felt_exact.append(exact)
            felt_tol.append(tol if tol is not None else False)
            if exact: num_exact_match += 1
            if tol:   num_tol_match   += 1

        rapport["numeriske"][felt] = {
            "værdier": {filnavne[i]: værdier[i] for i in range(len(filnavne))},
            "exact_match": all(felt_exact) if felt_exact else None,
            "tolerance_match_5pct": all(felt_tol) if felt_tol else None,
        }

        for i, v in enumerate(værdier):
            num_alpha_data[i].append(v)

    if len(num_alpha_data) >= 2:
        rapport["opsummering"]["alpha_interval"] = alpha_med_pålidelighedstjek(
            [num_alpha_data[i] for i in range(len(annotationer))],
            niveau="interval", felt_type="numerisk"
        )

    # Nominale
    nom_exact_total = 0
    nom_exact_match = 0
    nom_alpha_data  = defaultdict(list)

    for felt in nominale:
        værdier = [ann.get(felt) for ann in annotationer]
        værdier_norm = [
            str(v).lower().strip() if v is not None else None
            for v in værdier
        ]

        par = [
            (værdier_norm[i], værdier_norm[j])
            for i in range(len(værdier_norm))
            for j in range(i+1, len(værdier_norm))
        ]

        felt_exact = []
        for a, b in par:
            if a is None and b is None:
                continue
            nom_exact_total += 1
            exact = (a == b)
            felt_exact.append(exact)
            if exact: nom_exact_match += 1

        rapport["nominale"][felt] = {
            "værdier": {filnavne[i]: værdier[i] for i in range(len(filnavne))},
            "exact_match": all(felt_exact) if felt_exact else None
        }

        for i, v in enumerate(værdier_norm):
            nom_alpha_data[i].append(v)

    if len(nom_alpha_data) >= 2:
        rapport["opsummering"]["alpha_nominal"] = alpha_med_pålidelighedstjek(
            [nom_alpha_data[i] for i in range(len(annotationer))],
            niveau="nominal", felt_type="nominal"
        )

    # Fritekst
    fritekst_f1_total = []
    for felt in fritekst:
        værdier = [ann.get(felt) for ann in annotationer]
        par_f1 = []
        for i in range(len(værdier)):
            for j in range(i+1, len(værdier)):
                if værdier[i] is None and værdier[j] is None:
                    continue
                _, _, f1 = token_f1(værdier[i], værdier[j])
                par_f1.append(f1)
                fritekst_f1_total.append(f1)

        rapport["fritekst"][felt] = {
            "værdier": {filnavne[i]: værdier[i] for i in range(len(filnavne))},
            "token_f1": round(sum(par_f1) / len(par_f1), 3) if par_f1 else None
        }

    rapport["opsummering"].update({
        "numerisk_exact_pct": round(
            100 * num_exact_match / num_exact_total, 1
        ) if num_exact_total else None,
        "numerisk_tolerance_5pct_pct": round(
            100 * num_tol_match / num_exact_total, 1
        ) if num_exact_total else None,
        "nominal_exact_pct": round(
            100 * nom_exact_match / nom_exact_total, 1
        ) if nom_exact_total else None,
        "fritekst_gennemsnit_token_f1": round(
            sum(fritekst_f1_total) / len(fritekst_f1_total), 3
        ) if fritekst_f1_total else None,
    })

    return rapport


def analyser(json_stier):
    """
    Indlæser JSON-filer og beregner agreement metrics –
    både for alle annotatorer samlet og parvis.
    """
    annotationer = []
    filnavne = []
    for sti in json_stier:
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
            print("Fejl: {} (linje {}, kolonne {})".format(
                e.msg, e.lineno, e.colno))
            print("Kontekst:\n{}".format(kontekst))
            print("\nFilen skal rettes manuelt inden den kan bruges.")
            sys.exit(1)

        flad = fladgør(data)
        annotationer.append(normaliser_annotation(flad))
        filnavne.append(filnavn)

    # Klassificér felter
    alle_felter = set()
    for ann in annotationer:
        alle_felter.update(ann.keys())
    alle_felter -= SKIP_FELTER

    numeriske = []
    nominale  = []
    fritekst  = []

    for felt in sorted(alle_felter):
        if felt in FRITEKST_FELTER:
            fritekst.append(felt)
        elif felt in NOMINALE_FELTER:
            nominale.append(felt)
        else:
            værdier = [ann.get(felt) for ann in annotationer]
            if any(isinstance(v, (int, float)) for v in værdier):
                numeriske.append(felt)
            else:
                nominale.append(felt)

    # Samlet analyse
    samlet = analyser_par(annotationer, filnavne, numeriske, nominale, fritekst)

    # Parvis analyse
    parvis = {}
    for i in range(len(annotationer)):
        for j in range(i+1, len(annotationer)):
            nøgle = "{} vs {}".format(filnavne[i], filnavne[j])
            parvis[nøgle] = analyser_par(
                [annotationer[i], annotationer[j]],
                [filnavne[i], filnavne[j]],
                numeriske, nominale, fritekst
            )

    return {"samlet": samlet, "parvis": parvis}


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def print_rapport(rapport):
    linje = "=" * 60

    print(linje)
    print("PSG INTER-ANNOTATOR AGREEMENT RAPPORT")
    print(linje)

    samlet = rapport["samlet"]
    print("Filer:")
    for f in samlet["filer"]:
        print("  - {}".format(f))
    print()

    # Coverage
    print("── COVERAGE ─────────────────────────────────────────────")
    for navn, c in samlet["coverage"].items():
        print("  {}: {}/{} felter udfyldt ({} %)".format(
            navn, c["udfyldte"], c["total"], c["procent"]
        ))
    print()

    # Alpha-tabel
    print("── KRIPPENDORFF ALPHA ───────────────────────────────────")
    print("  {:<35} {:>10} {:>10}".format("", "Interval", "Nominal"))
    print("  " + "-" * 55)

    def alpha_str(rapport_del, niveau_key):
        o = rapport_del["opsummering"]
        if niveau_key not in o or o[niveau_key] is None:
            return "N/A"
        a = o[niveau_key]
        if a["alpha"] is None:
            return "N/A"
        s = "{:.3f}".format(a["alpha"])
        if not a["paalidelig"]:
            s += " ⚠"
        return s

    label = "Alle ({} annotatorer)".format(len(samlet["filer"]))
    print("  {:<35} {:>10} {:>10}".format(
        label,
        alpha_str(samlet, "alpha_interval"),
        alpha_str(samlet, "alpha_nominal")
    ))

    for par_navn, par_data in rapport["parvis"].items():
        navne = [
            f.replace("output-", "").replace(".json", "")
            for f in par_data["filer"]
        ]
        label = "{} vs {}".format(navne[0], navne[1])
        interval = alpha_str(par_data, "alpha_interval")
        nominal  = alpha_str(par_data, "alpha_nominal")

        advarsel = ""
        for key in ["alpha_interval", "alpha_nominal"]:
            o = par_data["opsummering"]
            if key in o and o[key] and o[key]["alpha"] is not None:
                if o[key]["alpha"] < 0.6:
                    advarsel = "  ← lav"
                    break

        print("  {:<35} {:>10} {:>10}{}".format(
            label, interval, nominal, advarsel
        ))

    print("  (⚠ = færre end {} par med data)".format(ALPHA_MIN_OBSERVATIONER))
    print()

    # Opsummering
    print("── OPSUMMERING (ALLE) ───────────────────────────────────")
    o = samlet["opsummering"]
    if o.get("numerisk_exact_pct") is not None:
        print("  Numerisk exact match:           {} %".format(
            o["numerisk_exact_pct"]))
    if o.get("numerisk_tolerance_5pct_pct") is not None:
        print("  Numerisk tolerance match (±5%): {} %".format(
            o["numerisk_tolerance_5pct_pct"]))
    if o.get("nominal_exact_pct") is not None:
        print("  Nominal exact match:            {} %".format(
            o["nominal_exact_pct"]))
    if o.get("fritekst_gennemsnit_token_f1") is not None:
        print("  Fritekst gns. token F1:         {:.3f}".format(
            o["fritekst_gennemsnit_token_f1"]))
    print()

    # Numeriske uoverensstemmelser
    uenige_num = {
        felt: info for felt, info in samlet["numeriske"].items()
        if info["exact_match"] is False
    }
    if uenige_num:
        print("── NUMERISKE UOVERENSSTEMMELSER ─────────────────────────")
        for felt, info in sorted(uenige_num.items()):
            værdier_str = ", ".join(
                "{}: {}".format(k, v) for k, v in info["værdier"].items()
            )
            tol = "OK (±5%)" if info["tolerance_match_5pct"] else "AFVIGER"
            print("  {} [{}]".format(felt, tol))
            print("    {}".format(værdier_str))
        print()

    # Nominale uoverensstemmelser
    uenige_nom = {
        felt: info for felt, info in samlet["nominale"].items()
        if info["exact_match"] is False
    }
    if uenige_nom:
        print("── NOMINALE UOVERENSSTEMMELSER ──────────────────────────")
        for felt, info in sorted(uenige_nom.items()):
            værdier_str = ", ".join(
                "{}: {}".format(k, v) for k, v in info["værdier"].items()
            )
            print("  {}".format(felt))
            print("    {}".format(værdier_str))
        print()

    print(linje)

def analyser_mapper(mapper):
    """
    Finder matchede JSON-filer på tværs af 3 annotator-mapper.
    Matcher på trial-nummer: output-trial{N}*.json
    """
    import glob

    # Find alle trial-numre per mappe
    alle_trial_numre = []
    mappe_filer = {}

    for mappe in mapper:
        filer = list(Path(mappe).glob("output-trial*.json"))
        trial_map = {}
        for fil in filer:
            # Udtræk tal efter "trial"
            m = re.match(r"output-trial(\d+)", fil.name)
            if m:
                trial_map[int(m.group(1))] = fil
        mappe_filer[mappe] = trial_map
        alle_trial_numre.append(set(trial_map.keys()))

    # Kun trial-numre der findes i alle mapper
    fælles = sorted(set.intersection(*alle_trial_numre))
    kun_i_nogle = set.union(*alle_trial_numre) - set(fælles)

    if kun_i_nogle:
        print("ADVARSEL: Følgende trial-numre mangler i mindst én mappe og springes over:")
        for n in sorted(kun_i_nogle):
            for mappe, trial_map in mappe_filer.items():
                if n not in trial_map:
                    print("  Trial {}: mangler i {}".format(n, mappe))

    print("Fandt {} matchede trial-sæt.\n".format(len(fælles)))

    # Kør agreement per trial-sæt og saml resultater
    alle_resultater = {}
    for n in fælles:
        stier = [str(mappe_filer[mappe][n]) for mappe in mapper]
        print("Trial {}:".format(n))
        for sti in stier:
            print("  {}".format(sti))
        alle_resultater[n] = analyser(stier)

    return alle_resultater


def print_mapper_rapport(alle_resultater):
    """Printer en samlet rapport på tværs af alle trial-sæt."""
    linje = "=" * 60

    print(linje)
    print("PSG INTER-ANNOTATOR AGREEMENT – SAMLET PÅ TVÆRS AF TRIALS")
    print(linje)

    # Saml alpha-værdier på tværs af trials
    interval_værdier = []
    nominal_værdier  = []

    for n, rapport in sorted(alle_resultater.items()):
        samlet = rapport["samlet"]
        o = samlet["opsummering"]

        interval = o.get("alpha_interval", {})
        nominal  = o.get("alpha_nominal", {})

        i_alpha = interval.get("alpha") if interval else None
        n_alpha = nominal.get("alpha")  if nominal  else None

        advarsel_i = " ⚠" if interval and not interval.get("paalidelig") else ""
        advarsel_n = " ⚠" if nominal  and not nominal.get("paalidelig")  else ""

        print("  Trial {:>2}   Interval: {}   Nominal: {}".format(
            n,
            "{:.3f}{}".format(i_alpha, advarsel_i) if i_alpha is not None else "N/A",
            "{:.3f}{}".format(n_alpha, advarsel_n) if n_alpha is not None else "N/A",
        ))

        if i_alpha is not None:
            interval_værdier.append(i_alpha)
        if n_alpha is not None:
            nominal_værdier.append(n_alpha)

    print()

    if interval_værdier:
        print("  Gns. interval alpha: {:.3f}  (over {} trials)".format(
            sum(interval_værdier) / len(interval_værdier),
            len(interval_værdier)
        ))
    if nominal_værdier:
        print("  Gns. nominal alpha:  {:.3f}  (over {} trials)".format(
            sum(nominal_værdier) / len(nominal_værdier),
            len(nominal_værdier)
        ))

    print()
    print("(⚠ = færre end {} par med data)".format(ALPHA_MIN_OBSERVATIONER))
    print(linje)

    print("── SAMLET STATISTIK PÅ TVÆRS AF TRIALS ─────────────────")

    # Header med trial-numre
    trial_numre = sorted(alle_resultater.keys())
    header = "  {:<45}".format("Metrik")
    for n in trial_numre:
        header += " {:>8}".format("T{}".format(n))
    header += " {:>8}".format("Gns.")
    print(header)
    print("  " + "-" * (45 + 9 * len(trial_numre) + 9))

    def print_række(label, værdier_per_trial, fmt="{:>8.1f}"):
        række = "  {:<45}".format(label)
        værdier = []
        for n in trial_numre:
            v = værdier_per_trial.get(n)
            if v is not None:
                række += fmt.format(v)
                værdier.append(v)
            else:
                række += " {:>8}".format("N/A")
        if værdier:
            række += fmt.format(sum(værdier) / len(værdier))
        else:
            række += " {:>8}".format("N/A")
        print(række)

    # Byg værdier per trial
    def hent(rapport, nøgle):
        return rapport["samlet"]["opsummering"].get(nøgle)

    def hent_alpha(rapport, nøgle):
        o = rapport["samlet"]["opsummering"]
        if nøgle not in o or o[nøgle] is None:
            return None
        a = o[nøgle]
        if a["alpha"] is None:
            return None
        suffix = " ⚠" if not a["paalidelig"] else ""
        return (a["alpha"], suffix)

    # Alpha rækker (special format pga. ⚠)
    for label, nøgle in [
        ("Krippendorff alpha (interval)", "alpha_interval"),
        ("Krippendorff alpha (nominal)",  "alpha_nominal"),
    ]:
        række = "  {:<45}".format(label)
        værdier = []
        for n in trial_numre:
            result = hent_alpha(alle_resultater[n], nøgle)
            if result is not None:
                alpha, advarsel = result
                celle = "{:.3f}{}".format(alpha, advarsel)
                række += " {:>8}".format(celle)
                værdier.append(alpha)
            else:
                række += " {:>8}".format("N/A")
        if værdier:
            række += " {:>8.3f}".format(sum(værdier) / len(værdier))
        else:
            række += " {:>8}".format("N/A")
        print(række)

    # Parvis alpha rækker
    # Find alle par på tværs af alle trials
    alle_par = []
    for n in trial_numre:
        for par_navn in alle_resultater[n]["parvis"].keys():
            navne = [
                f.replace("output-", "").replace(".json", "")
                for f in alle_resultater[n]["parvis"][par_navn]["filer"]
            ]
            par_label = "{} vs {}".format(navne[0], navne[1])
            if par_label not in alle_par:
                alle_par.append(par_label)

    for par_label in alle_par:
        for alpha_type, alpha_key in [
            ("interval", "alpha_interval"),
            ("nominal",  "alpha_nominal"),
        ]:
            række = "  {:<45}".format("  {} ({})".format(par_label, alpha_type))
            værdier = []
            for n in trial_numre:
                # Find det rigtige par i denne trial
                par_data = None
                for par_navn, data in alle_resultater[n]["parvis"].items():
                    navne = [
                        f.replace("output-", "").replace(".json", "")
                        for f in data["filer"]
                    ]
                    if "{} vs {}".format(navne[0], navne[1]) == par_label:
                        par_data = data
                        break

                if par_data is None:
                    række += " {:>8}".format("N/A")
                    continue

                o = par_data["opsummering"]
                if alpha_key not in o or o[alpha_key] is None or o[alpha_key]["alpha"] is None:
                    række += " {:>8}".format("N/A")
                    continue

                alpha = o[alpha_key]["alpha"]
                advarsel = " ⚠" if not o[alpha_key]["paalidelig"] else ""
                celle = "{:.3f}{}".format(alpha, advarsel)
                række += " {:>8}".format(celle)
                værdier.append(alpha)

            if værdier:
                række += " {:>8.3f}".format(sum(værdier) / len(værdier))
            else:
                række += " {:>8}".format("N/A")
            print(række)

    # Øvrige metrics
    print_række(
        "Numerisk exact match (%)",
        {n: hent(alle_resultater[n], "numerisk_exact_pct") for n in trial_numre}
    )
    print_række(
        "Numerisk tolerance match ±5% (%)",
        {n: hent(alle_resultater[n], "numerisk_tolerance_5pct_pct") for n in trial_numre}
    )
    print_række(
        "Nominal exact match (%)",
        {n: hent(alle_resultater[n], "nominal_exact_pct") for n in trial_numre}
    )
    print_række(
        "Fritekst gns. token F1",
        {n: hent(alle_resultater[n], "fritekst_gennemsnit_token_f1") for n in trial_numre},
        fmt="{:>8.3f}"
    )

    print("  (⚠ = færre end {} par med data)".format(ALPHA_MIN_OBSERVATIONER))
    print()


    # Detaljeret rapport per trial
    for n, rapport in sorted(alle_resultater.items()):
        print("\n" + "─" * 60)
        print("TRIAL {}".format(n))
        print("─" * 60)
        print_rapport(rapport)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Beregn inter-annotator agreement mellem PSG JSON-filer"
    )
    p.add_argument("filer", nargs="+",
                   help="To eller flere .json filer, ELLER tre mapper med --mapper flag")
    p.add_argument("--mapper", "-m", action="store_true",
                   help="Treat arguments as directories and match trials automatically")
    p.add_argument("--output", "-o",
                   help="Gem fuld rapport som JSON")
    args = p.parse_args()

    if args.mapper:
        if len(args.filer) != 3:
            print("FEJL: Angiv præcis 3 mapper med --mapper.")
            sys.exit(1)
        alle_resultater = analyser_mapper(args.filer)
        print_mapper_rapport(alle_resultater)

        if args.output:
            # Konvertér int-nøgler til strenge for JSON
            Path(args.output).write_text(
                json.dumps(
                    {"trial_{}".format(k): v for k, v in alle_resultater.items()},
                    ensure_ascii=False, indent=2
                ),
                encoding="utf-8"
            )
            print("Fuld rapport gemt: {}".format(args.output))

    else:
        if len(args.filer) < 2:
            print("FEJL: Angiv mindst to JSON-filer.")
            sys.exit(1)
        rapport = analyser(args.filer)
        print_rapport(rapport)

        if args.output:
            Path(args.output).write_text(
                json.dumps(rapport, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print("Fuld rapport gemt: {}".format(args.output))


if __name__ == "__main__":
    main()