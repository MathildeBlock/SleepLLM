import re
from pathlib import Path
import argparse


# ---------------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------------

def deduplicate_lines(text):
    seen = set()
    result = []

    for line in text.split("\n"):
        clean = line.strip()
        if not clean:
            continue

        if clean not in seen:
            seen.add(clean)
            result.append(clean)

    return "\n".join(result)


def remove_noise(text):
    lines = []

    for line in text.split("\n"):
        l = line.lower().strip()

        # Fjern tydelig støj
        if any(x in l for x in [
            "sidehoved", "sidefod", "=== psg rapport ===", "=== slut"
        ]):
            continue

        # Fjern linjer med kun symboler
        if re.match(r"^[\W_]+$", l):
            continue

        lines.append(line)

    return "\n".join(lines)


def normalize_spacing(text):
    # Fjern mange tomme linjer
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------

def split_inline_values(text):
    """
    Gør:
    AHI: 18.4 | ODI: 14.2
    ->
    AHI: 18.4
    ODI: 14.2
    """
    lines = []

    for line in text.split("\n"):
        if "|" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if p:
                    lines.append(p)
        else:
            lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION TAGGING (KEY FEATURE 🔥)
# ---------------------------------------------------------------------------

def add_sections(text):
    tagged = []

    for line in text.split("\n"):
        l = line.lower()

        if any(k in l for k in ["ahi", "apnø", "hypopnoe", "respir"]):
            tagged.append("[RESPIRATION] " + line)

        elif any(k in l for k in ["rem", "n1", "n2", "n3", "søvn"]):
            tagged.append("[SLEEP] " + line)

        elif any(k in l for k in ["spo2", "oxygen", "desaturation"]):
            tagged.append("[OXYGEN] " + line)

        elif any(k in l for k in ["puls", "hjerte", "ekg"]):
            tagged.append("[HEART] " + line)

        else:
            tagged.append(line)

    return "\n".join(tagged)


# ---------------------------------------------------------------------------
# TRUNCATE
# ---------------------------------------------------------------------------

def truncate(text, max_chars=12000):
    return text[:max_chars]


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def optimize_for_llm(text):
    text = deduplicate_lines(text)
    text = remove_noise(text)
    text = split_inline_values(text)
    text = normalize_spacing(text)
    text = add_sections(text)
    text = truncate(text)

    return text


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="txt fil eller mappe")
    parser.add_argument("--output-dir", default="llm_ready")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.txt"))

    if args.max_files:
        files = files[:args.max_files]

    print(f"Behandler {len(files)} filer...")

    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file.name}")

        text = file.read_text(encoding="utf-8")
        cleaned = optimize_for_llm(text)

        out_path = output_dir / file.name
        out_path.write_text(cleaned, encoding="utf-8")

    print(f"\n✅ Færdig! LLM-ready filer i: {output_dir}")


if __name__ == "__main__":
    main()