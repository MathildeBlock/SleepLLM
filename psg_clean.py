import re
from pathlib import Path
import argparse


# ---------------------------------------------------------------------------
# SAFE DEDUP (kun 100% ens + beskyt tabeldata)
# ---------------------------------------------------------------------------

def deduplicate_safe(lines):
    seen = set()
    result = []

    for line in lines:
        clean = line.strip()

        if not clean:
            continue

        # 🔥 BEVAR korte linjer (typisk tal i tabeller)
        if len(clean) < 15:
            result.append(clean)
            continue

        # Fjern kun hvis 100% identisk OG lang nok til at være støj
        if clean not in seen:
            seen.add(clean)
            result.append(clean)

    return result


# ---------------------------------------------------------------------------
# REMOVE HEADER/FOOTER SPAM (men forsigtigt)
# ---------------------------------------------------------------------------

def remove_repeated_blocks(lines, min_repeats=5):
    counts = {}

    for line in lines:
        counts[line] = counts.get(line, 0) + 1

    result = []
    for line in lines:
        # Fjern kun hvis det er LANG tekst der gentages mange gange
        if len(line.strip()) > 30 and counts[line] >= min_repeats:
            continue
        result.append(line)

    return result


# ---------------------------------------------------------------------------
# SPLIT TABLES (bevar værdier!)
# ---------------------------------------------------------------------------

def split_tables(lines):
    new_lines = []

    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            new_lines.extend(parts)
        else:
            new_lines.append(line)

    return new_lines


# ---------------------------------------------------------------------------
# NORMALIZE
# ---------------------------------------------------------------------------

def normalize(text):
    text = re.sub(r"[^\S\n]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def clean_text(text):
    lines = text.split("\n")

    # 1. split tabeller
    lines = split_tables(lines)

    # 2. fjern header/footer spam
    lines = remove_repeated_blocks(lines)

    # 3. SAFE dedup
    lines = deduplicate_safe(lines)

    # 4. join
    text = "\n".join(lines)
    text = normalize(text)

    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="txt fil eller mappe")
    parser.add_argument("--output-dir", default="cleaned")
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

    print(f"Cleaner {len(files)} filer...")

    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file.name}")

        text = file.read_text(encoding="utf-8")
        cleaned = clean_text(text)

        (output_dir / file.name).write_text(cleaned, encoding="utf-8")

    print(f"\n✅ Done! Cleaned data i: {output_dir}")


if __name__ == "__main__":
    main()