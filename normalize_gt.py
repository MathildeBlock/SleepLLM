#!/usr/bin/env python3
"""
Normaliser PSG ground truth JSON-filer.

Regler:
- oget_muskelaktivitet_rem (chin/tib/fds): konverter til boolean (true/false/null)
- soevnstadier latens_min + varighed_min: konverter hh:mm:ss til minutter (float)
- sammenfatning (anfald/fokal/paroksystisk_aktivitet): konverter boolean til string
"""

import json
import os
import sys
import re
from pathlib import Path


# ── Hjælpefunktioner ─────────────────────────────────────────────────────────

def hms_to_minutes(value) -> float | None:
    """Konverter 'hh:mm:ss' eller 'hh:mm' til minutter som float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # Match hh:mm:ss
    m = re.fullmatch(r'(\d+):(\d{2}):(\d{2})', s)
    if m:
        h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
        total = h * 60 + mi + sec / 60
        return round(total, 4)
    # Match hh:mm
    m = re.fullmatch(r'(\d+):(\d{2})', s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        return round(h * 60 + mi, 4)
    # Prøv direkte tal-parsing
    try:
        return float(s)
    except ValueError:
        return value  # returner uændret hvis ukendt format


def to_boolean(value) -> bool | None:
    """Konverter 'ja'/'nej'/True/False/1/0 til bool eller None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ('ja', 'yes', 'true', '1', 'j'):
        return True
    if s in ('nej', 'no', 'false', '0', 'n'):
        return False
    return None  # ukendt → null


def to_string(value) -> str | None:
    """Konverter bool/int til 'ja'/'nej', bevar string-værdier."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 'ja' if value else 'nej'
    if isinstance(value, int):
        return 'ja' if value else 'nej'
    return str(value)  # bevar fritekst som den er


# ── Normalisering ─────────────────────────────────────────────────────────────

def normalize(data: dict) -> dict:
    changes = []

    # 1. oget_muskelaktivitet_rem → boolean
    rem = data.get('oget_muskelaktivitet_rem', {})
    if isinstance(rem, dict):
        for felt in ('chin', 'tib', 'fds'):
            orig = rem.get(felt)
            ny = to_boolean(orig)
            if orig != ny:
                changes.append(f"  rem.{felt}: {repr(orig)} → {repr(ny)}")
                rem[felt] = ny

    # 2. soevnstadier latens_min + varighed_min → minutter som tal
    stadier = data.get('soevnstadier', {})
    if isinstance(stadier, dict):
        for stadie, felter in stadier.items():
            if not isinstance(felter, dict):
                continue
            for felt in ('latens_min', 'varighed_min'):
                orig = felter.get(felt)
                ny = hms_to_minutes(orig)
                if orig != ny and ny is not None:
                    changes.append(f"  soevnstadier.{stadie}.{felt}: {repr(orig)} → {repr(ny)}")
                    felter[felt] = ny

    # 3. soevn_opsummering tidsfelter → minutter
    opsummering = data.get('soevn_opsummering', {})
    if isinstance(opsummering, dict):
        for felt in ('soevnlatens_min', 'rem_latens_min', 'soveperiode_min',
                     'tid_i_seng_trt_min', 'total_soevntid_tst_min'):
            orig = opsummering.get(felt)
            ny = hms_to_minutes(orig)
            if orig != ny and ny is not None:
                changes.append(f"  soevn_opsummering.{felt}: {repr(orig)} → {repr(ny)}")
                opsummering[felt] = ny

    # 4. sammenfatning anfald/fokal/paroksystisk_aktivitet → string
    sammenfatning = data.get('sammenfatning', {})
    if isinstance(sammenfatning, dict):
        for felt in ('anfald', 'fokal', 'paroksystisk_aktivitet'):
            orig = sammenfatning.get(felt)
            ny = to_string(orig)
            if orig != ny:
                changes.append(f"  sammenfatning.{felt}: {repr(orig)} → {repr(ny)}")
                sammenfatning[felt] = ny

    return data, changes


# ── Fil-håndtering ────────────────────────────────────────────────────────────

def process_file(input_path: Path, output_path: Path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data, changes = normalize(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return changes


def process_folder(input_folder: str, output_folder: str):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    json_files = sorted(input_path.glob('**/*.json'))
    if not json_files:
        print(f"Ingen JSON-filer fundet i: {input_folder}")
        return

    total_changes = 0
    for f in json_files:
        rel = f.relative_to(input_path)
        out = output_path / rel
        changes = process_file(f, out)
        if changes:
            print(f"\n{rel}:")
            for c in changes:
                print(c)
            total_changes += len(changes)
        else:
            print(f"{rel}: ingen ændringer")

    print(f"\n✓ Færdig. {len(json_files)} filer behandlet, {total_changes} felter ændret.")
    print(f"  Output: {output_folder}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) == 3:
        process_folder(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        # Enkelt fil → overskriv in-place
        p = Path(sys.argv[1])
        changes = process_file(p, p)
        if changes:
            print(f"{p.name}:")
            for c in changes:
                print(c)
        else:
            print(f"{p.name}: ingen ændringer")
    else:
        print("Brug:")
        print("  python normalize_ground_truth.py <input_mappe> <output_mappe>")
        print("  python normalize_ground_truth.py <enkelt_fil.json>")
        sys.exit(1)