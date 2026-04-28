"""
psg_extract_pipeline.py  -  Trin 2: Udtræk struktureret JSON fra PSG-rapport via lokal LLM.

Optimeringer vs. forrige version:
  - Producer-consumer pipeline: CPU tokeniserer næste fil MENS GPU genererer på nuværende
  - Prefetch-kø med konfigurerbar dybde (standard 2) — holder GPU beskæftiget
  - Pin memory + non-blocking .to(device) for hurtigere CPU→GPU transfer
  - Timing-statistik per fil (tokens/sek, input-tokens, output-tokens)
  - Alle tidligere optimeringer bevaret (torch.compile, schema-cache, warmup osv.)

Krav:
  pip install transformers torch huggingface_hub accelerate tqdm

Brug:
  # Enkelt fil
  python psg_extract_pipeline.py rapport.txt --token DIT_HF_TOKEN

  # Batch
  python psg_extract_pipeline.py --mappe forberedt/ --output-mappe resultater/ \\
      --model Qwen/Qwen2.5-14B-Instruct --device cuda --dtype bfloat16 --quant 4bit \\
      --token DIT_HF_TOKEN
"""

import os
import sys
import json
import datetime
import argparse
import time
import threading
import queue
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("FEJL: Mangler tqdm. Kør: pip install tqdm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# JSON-skema
# ---------------------------------------------------------------------------

SCHEMA = {
    "metadata": {
        "filnavn": None,
        "udfyldt_af": None
    },
    "patient": {
        "navn": None, "cpr-nummer": None, "vaegt": None,
        "hoejde": None, "BMI": None, "ESS": None
    },
    "test_oplysninger": {
        "dato": None, "henviser": None, "henvisningsdiagnose": None,
        "starttid": None, "sluttid": None, "total_optagetid": None,
        "optaget_af": None, "neurofysiologi_assisstent": None, "montage": None
    },
    "klinisk_information": {
        "klinisk_resume": None, "kommentar": None,
        "patient_oplysninger_ved_optagelse": None, "medicin": None
    },
    "soevn_opsummering": {
        "scoringens_navn": None, "analyse_start_lights_out": None,
        "analyse_afslutning_lights_on": None, "tid_i_seng_trt_min": None,
        "total_soevntid_tst_min": None, "soveperiode_min": None,
        "soevneffektivitet_procent": None, "soevnlatens_min": None,
        "rem_latens_min": None, "antal_opvaagninger": None,
        "arousals": {"total": None, "indeks": None}
    },
    "soevnstadier": {
        "vaagen":  {"latens_min": None, "varighed_min": None, "procent_af_tst": None},
        "n1":      {"latens_min": None, "varighed_min": None, "procent_af_tst": None},
        "n2":      {"latens_min": None, "varighed_min": None, "procent_af_tst": None},
        "n3":      {"latens_min": None, "varighed_min": None, "procent_af_tst": None},
        "rem":     {"latens_min": None, "varighed_min": None, "procent_af_tst": None},
        "total":   {"latens_min": None, "varighed_min": None, "procent_af_tst": None}
    },
    "oget_muskelaktivitet_rem": {"chin": None, "tib": None, "fds": None},
    "respirations_analyse": {
        "ahi_total": None, "ahi_rygleje": None, "ahi_ikke_rygleje": None,
        "andel_natten_i_rygleje_procent": None, "obstruktiv_apnoe_indeks": None,
        "mixed_apnoe_indeks": None, "central_apnoe_indeks": None,
        "hypopnoe_indeks": None,
        "oxygen_desaturationer": {"total": None, "indeks": None}
    },
    "spo2_oversigt": {
        "baseline": None,
        "minimum_procent":  {"vaagen": None, "nrem": None, "rem": None, "total": None},
        "middel_procent":   {"vaagen": None, "nrem": None, "rem": None, "total": None},
        "maksimum_procent": {"vaagen": None, "nrem": None, "rem": None, "total": None},
        "spo2_under_90_procent": {"akkumuleret_varighed": None, "akkumuleret_procent": None},
        "co2_vaerdier": {"etco2_max": None, "tcpco2_max": None}
    },
    "benbevaegelser": {
        "lm_indeks":   {"vaagen": None, "nrem": None, "rem": None, "total": None},
        "plms_indeks": {"vaagen": None, "nrem": None, "rem": None, "total": None},
        "lms_efterfulgt_af_arousals": None
    },
    "hjerte": {"middel_hjertefrekvens": None, "ekg_bemaerkninger": None},
    "sammenfatning": {
        "soevnmoenster": None, "soevn_dagtid": None, "anfald": None,
        "beskrivelse": None, "paroksystisk_aktivitet": None, "fokal": None
    },
    "konklusion_og_plan": {
        "bedoemt_af": None, "dato_for_bedoemmelse": None,
        "konklusion_tekst": None, "plan": None,
        "a_diagnose": {"kode": None, "tekst": None},
        "b_diagnose": {"kode": None, "tekst": None}
    }
}

# Caches én gang ved import — bruges i alle prompt-kald
_SCHEMA_STR = json.dumps(SCHEMA, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent

with open(PROMPTS_DIR / "system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open(PROMPTS_DIR / "user_prompt.txt", "r", encoding="utf-8") as f:
    USER_PROMPT_TEMPLATE = f.read()

# Sentinel der signalerer at producer er færdig
_QUEUE_DONE = object()


# ---------------------------------------------------------------------------
# Validering
# ---------------------------------------------------------------------------

def validate(data: dict, filename: str) -> list[str]:
    warnings = []
    def w(msg): warnings.append(f"  ⚠ {filename}: {msg}")

    def as_number(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text or text == "-":
                return None
            text = text.replace(",", ".").replace("%", "")
            try:
                match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
                if not match:
                    return None
                return float(match.group(0))
            except Exception:
                return None
        return None

    p = data.get("patient", {}) or {}
    bmi_val = as_number(p.get("BMI"))
    if bmi_val is not None and not (10 < bmi_val < 80):
        w(f"Mistænkelig BMI: {p.get('BMI')}")

    ess_val = as_number(p.get("ESS"))
    if ess_val is not None and not (0 <= ess_val <= 24):
        w(f"ESS uden for 0-24: {p.get('ESS')}")

    s = data.get("soevn_opsummering", {}) or {}
    sleep_eff = as_number(s.get("soevneffektivitet_procent"))
    if sleep_eff is not None and not (0 <= sleep_eff <= 100):
        w(f"Søvneffektivitet uden for 0-100: {s.get('soevneffektivitet_procent')}")

    tst = as_number(s.get("total_soevntid_tst_min"))
    if tst is not None and tst > 700:
        w(f"TST usædvanlig høj: {s.get('total_soevntid_tst_min')} min")

    r = data.get("respirations_analyse", {}) or {}
    ahi_total = as_number(r.get("ahi_total"))
    if ahi_total is not None and ahi_total > 150:
        w(f"AHI usædvanlig høj: {r.get('ahi_total')}")

    spo2 = data.get("spo2_oversigt", {}) or {}
    min_spo2 = (spo2.get("minimum_procent") or {}).get("total")
    min_spo2_val = as_number(min_spo2)
    if min_spo2_val is not None and min_spo2_val < 50:
        w(f"SpO2 minimum meget lav: {min_spo2}%")

    return warnings


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


def _normalize_device_name(device_value):
    if isinstance(device_value, int):
        return f"cuda:{device_value}"
    device_text = str(device_value)
    if device_text.isdigit():
        return f"cuda:{device_text}"
    return device_text


def _resolve_model_input_device(model, torch_module):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for device_value in model.hf_device_map.values():
            device_name = _normalize_device_name(device_value)
            if device_name not in {"cpu", "disk"}:
                return torch_module.device(device_name)
    if hasattr(model, "device"):
        try:
            return torch_module.device(str(model.device))
        except (TypeError, ValueError):
            pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cpu")


def load_local_model(
    token,
    model_name: str = DEFAULT_MODEL_NAME,
    device_preference: str = "auto",
    dtype_preference: str = "auto",
    quantization: str = "none",
    gpu_max_memory_gb: float | None = None,
    device_map_preference: str = "auto",
    trust_remote_code: bool = False,
    use_compile: bool = True,
):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("FEJL: Mangler pakker. Kør: pip install transformers torch accelerate")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    if device_preference == "cuda" and not cuda_available:
        raise RuntimeError("CUDA valgt, men ikke tilgængelig i denne installation.")

    use_cuda = device_preference in {"auto", "cuda"} and cuda_available

    try:
        python_exe = sys.executable
    except Exception:
        python_exe = "(ukendt)"

    print(f"Runtime: python={python_exe}  torch={getattr(torch, '__version__', '?')}  cuda={cuda_available}")
    if cuda_available:
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    if dtype_preference == "auto":
        # bfloat16 er bedre end float16 på A100 — undgår overflow og er native
        dtype = torch.bfloat16 if use_cuda else torch.float32
    elif dtype_preference == "float16":
        dtype = torch.float16
    elif dtype_preference == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    quantization_config = None
    if quantization != "none":
        if not use_cuda:
            raise RuntimeError("Kvantisering kræver CUDA.")
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            print("FEJL: pip install bitsandbytes")
            sys.exit(1)

        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            compute_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.bfloat16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            raise ValueError("Ugyldig kvantisering. Brug: none, 8bit, 4bit")

    max_memory = None
    if use_cuda and gpu_max_memory_gb is not None:
        max_memory = {0: f"{gpu_max_memory_gb}GiB", "cpu": "128GiB"}

    if device_map_preference not in {"auto", "cuda"}:
        raise ValueError("Ugyldig --device-map.")
    if device_map_preference == "cuda" and not use_cuda:
        raise RuntimeError("--device-map cuda kræver CUDA.")

    resolved_device_map = {"": 0} if device_map_preference == "cuda" else ("auto" if use_cuda else None)

    print(f"Indlæser model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=None if quantization == "8bit" else dtype,
        device_map=resolved_device_map,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    selected_devices = []
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        selected_devices = sorted({_normalize_device_name(v) for v in model.hf_device_map.values()})
    elif hasattr(model, "device"):
        selected_devices = [_normalize_device_name(model.device)]

    if selected_devices:
        print(f"Model placeret på: {', '.join(selected_devices)}")

    if use_cuda and selected_devices and all(d in {"cpu", "disk"} for d in selected_devices):
        print("ADVARSEL: Model endte på CPU trods CUDA. Prøv --device-map cuda")
        if device_preference == "cuda":
            raise RuntimeError("Bad device placement.")

    # torch.compile — ~20-30% speedup på A100
    if use_compile and use_cuda:
        try:
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_version >= (2, 0):
                print("torch.compile() aktiveret (første kald kompilerer — forvent forsinkelse)...")
                model = torch.compile(model, mode="reduce-overhead")
            else:
                print(f"torch.compile() kræver PyTorch >= 2.0 — springer over.")
        except Exception as e:
            print(f"torch.compile() fejlede ({e}) — kører uden.")
    elif use_compile and not use_cuda:
        print("torch.compile() springer over (ikke GPU).")

    print(f"Klar: {'CUDA ' + str(dtype) if use_cuda else 'CPU'}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup_model(tokenizer, model, max_new_tokens: int):
    import torch
    print("Varmer op model...")
    dummy_messages = [{"role": "user", "content": "{}"}]
    formatted = tokenizer.apply_chat_template(
        dummy_messages, tokenize=False, add_generation_prompt=True
    )
    device = _resolve_model_input_device(model, torch)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=min(32, max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup færdig.")


# ---------------------------------------------------------------------------
# Tokenisering — kører i producer-tråd på CPU
# ---------------------------------------------------------------------------

def _build_inputs_cpu(tokenizer, report_text: str, max_input_tokens: int | None) -> dict:
    """
    Bygger og returnerer tokeniserede inputs som CPU-tensors med pin_memory.
    Pin memory muliggør hurtig asynkron CPU→GPU transfer i generate-løkken.
    """
    full_prompt = (
        SYSTEM_PROMPT + "\n\n"
        + USER_PROMPT_TEMPLATE.format(text=report_text, schema=_SCHEMA_STR)
        + "\n\nStart dit svar med { og slut dit svar med }."
    )
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=max_input_tokens is not None,
        max_length=max_input_tokens,
    )
    # pin_memory gør CPU→GPU-kopieringen asynkron og hurtigere
    try:
        inputs = {k: v.pin_memory() for k, v in inputs.items()}
    except Exception:
        pass  # Virker kun for CPU-tensors — fejl er ufarlige
    return inputs


# ---------------------------------------------------------------------------
# Producer — kører i baggrundstråd og tokeniserer næste fil mens GPU arbejder
# ---------------------------------------------------------------------------

def _producer(
    files: list[Path],
    tokenizer,
    max_input_tokens: int | None,
    out_queue: queue.Queue,
):
    """
    Læser og tokeniserer filer sekventielt og lægger dem i køen.
    Sender _QUEUE_DONE som sentinel når alle filer er behandlet.
    Fejl ved enkeltfiler lægges i køen som Exception-objekter.
    """
    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8")
            cpu_inputs = _build_inputs_cpu(tokenizer, text, max_input_tokens)
            out_queue.put((filepath, cpu_inputs, len(text)))
        except Exception as e:
            out_queue.put((filepath, e, 0))

    out_queue.put(_QUEUE_DONE)


# ---------------------------------------------------------------------------
# Generate + decode — kører på GPU i main-tråd
# ---------------------------------------------------------------------------

def _generate_one(
    model,
    tokenizer,
    cpu_inputs: dict,
    max_new_tokens: int,
    use_cache: bool,
    max_retries: int,
    debug_label: str,
) -> tuple[dict, int, int, float]:
    """
    Overfører inputs til GPU (non-blocking), genererer og parser JSON.
    Returnerer (parsed_json, input_tokens, output_tokens, sekunder).
    """
    import torch

    device = _resolve_model_input_device(model, torch)
    # non_blocking=True: GPU starter transfer mens CPU fortsætter
    gpu_inputs = {k: v.to(device, non_blocking=True) for k, v in cpu_inputs.items()}
    input_len = gpu_inputs["input_ids"].shape[1]

    last_error = None
    for attempt in range(1, max_retries + 1):
        raw = ""
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **gpu_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
            elapsed = time.perf_counter() - t0
            new_tokens = outputs[0][input_len:]
            output_len = new_tokens.shape[0]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return parse_json_response(raw), input_len, output_len, elapsed

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print(f"\n[{debug_label}] OOM forsøg {attempt} — rydder VRAM...")
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(3)

        except json.JSONDecodeError as e:
            last_error = e
            stem = debug_label if not debug_label.endswith(".txt") else debug_label[:-4]
            debug_file = Path(f"failed_{stem}.txt")
            debug_file.write_text(raw, encoding="utf-8")
            print(f"\n[{debug_label}] JSON-fejl forsøg {attempt}: {e} → {debug_file}")

        except Exception as e:
            last_error = e
            print(f"\n[{debug_label}] Fejl forsøg {attempt}: {e}")

    raise RuntimeError(f"Fejlede efter {max_retries} forsøg. Sidste: {last_error}")


# ---------------------------------------------------------------------------
# Enkelt fil
# ---------------------------------------------------------------------------

def process_single_file(
    input_path: Path,
    output_path: Path,
    tokenizer,
    model,
    model_name: str,
    max_new_tokens: int = 4096,
    max_input_tokens: int | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> dict:
    text = input_path.read_text(encoding="utf-8")
    if verbose:
        print(f"  Tekst: {len(text)} tegn")

    cpu_inputs = _build_inputs_cpu(tokenizer, text, max_input_tokens)
    data, in_tok, out_tok, elapsed = _generate_one(
        model, tokenizer, cpu_inputs,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        max_retries=3,
        debug_label=input_path.name,
    )

    data["metadata"]["filnavn"]    = input_path.stem
    data["metadata"]["udfyldt_af"] = model_name

    for w in validate(data, input_path.name):
        print(w)

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    tok_per_sec = out_tok / elapsed if elapsed > 0 else 0
    print(f"  ✅ {input_path.name}: {in_tok} inp / {out_tok} out | {elapsed:.1f}s | {tok_per_sec:.1f} tok/s")
    return data


# ---------------------------------------------------------------------------
# Pipeline batch
# ---------------------------------------------------------------------------

def process_batch_pipeline(
    input_dir: Path,
    output_dir: Path,
    tokenizer,
    model,
    model_name: str,
    verbose: bool = False,
    max_filer: int | None = None,
    max_new_tokens: int = 4096,
    max_input_tokens: int | None = None,
    use_cache: bool = True,
    prefetch: int = 2,
):
    """
    Producer-consumer pipeline:

      [Producer-tråd / CPU]              [Consumer / GPU]
      læs fil N+1 fra disk     ─────►    generate() på fil N
      tokenisér fil N+1        ─────►    decode output
      læg i kø                           gem JSON

    GPU er aldrig idle mens der er filer — CPU tokeniserer parallelt.
    Prefetch-køen holder `prefetch` filer klar på forhånd.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_dir.glob("*.txt"))
    if not all_files:
        print(f"Ingen .txt filer fundet i: {input_dir}")
        return

    # Filtrer allerede behandlede filer
    todo_files = [
        f for f in all_files
        if not (output_dir / (f.stem + ".json")).exists()
    ]
    sprang_over = len(all_files) - len(todo_files)

    if max_filer:
        todo_files = todo_files[:max_filer]

    print(f"Fandt {len(all_files)} filer | {sprang_over} allerede behandlet | {len(todo_files)} tilbage")
    print(f"Pipeline prefetch: {prefetch} | Model: {model_name}")

    if not todo_files:
        print("Intet at gøre.")
        return

    # Start producer-tråd med begrænset kø (backpressure undgår RAM-overflow)
    prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch)
    producer_thread = threading.Thread(
        target=_producer,
        args=(todo_files, tokenizer, max_input_tokens, prefetch_queue),
        daemon=True,
        name="tokenizer-producer",
    )
    producer_thread.start()

    ok         = 0
    fejl       = 0
    fejl_liste = []
    total_out_tok = 0
    total_sek     = 0.0

    pbar = tqdm(total=len(todo_files), unit="fil")

    while True:
        item = prefetch_queue.get()

        if item is _QUEUE_DONE:
            break

        filepath, payload, text_len = item

        # Producer rapporterede en tokeniseringsfejl
        if isinstance(payload, Exception):
            tqdm.write(f"\n❌ Tokeniseringsfejl ved {filepath.name}: {payload}")
            fejl += 1
            fejl_liste.append(filepath.name)
            pbar.update(1)
            continue

        out_file = output_dir / (filepath.stem + ".json")
        try:
            data, in_tok, out_tok, elapsed = _generate_one(
                model, tokenizer, payload,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                max_retries=3,
                debug_label=filepath.name,
            )

            data["metadata"]["filnavn"]    = filepath.stem
            data["metadata"]["udfyldt_af"] = model_name

            for w in validate(data, filepath.name):
                tqdm.write(w)

            out_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            tok_per_sec    = out_tok / elapsed if elapsed > 0 else 0
            total_out_tok += out_tok
            total_sek     += elapsed
            ok            += 1

            if verbose:
                tqdm.write(
                    f"  ✅ {filepath.name}: {in_tok} inp / {out_tok} out | "
                    f"{elapsed:.1f}s | {tok_per_sec:.1f} tok/s | {text_len} tegn"
                )

        except Exception as e:
            tqdm.write(f"\n❌ Fejl ved {filepath.name}: {e}")
            fejl += 1
            fejl_liste.append(filepath.name)
            fejl_log = output_dir / "fejl_log.txt"
            with open(fejl_log, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {filepath.name}  {e}\n")

        pbar.update(1)

    pbar.close()
    producer_thread.join(timeout=5)

    avg_tok_s = total_out_tok / total_sek if total_sek > 0 else 0
    print(f"\n{'='*55}")
    print(f"Færdig!")
    print(f"  ✅ OK:              {ok}")
    print(f"  ⏭  Sprang over:    {sprang_over}")
    print(f"  ❌ Fejl:            {fejl}")
    print(f"  ⚡ Gns. hastighed:  {avg_tok_s:.1f} output-tok/s")
    if total_sek > 0:
        print(f"  ⏱  Total GPU-tid:  {total_sek/60:.1f} min")
    if fejl_liste:
        print(f"  Fejl-log: {output_dir / 'fejl_log.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="PSG JSON-udtræk med CPU-GPU pipeline (producer-consumer)"
    )

    input_gruppe = p.add_mutually_exclusive_group()
    input_gruppe.add_argument("input", nargs="?", help="Sti til én .txt fil")
    input_gruppe.add_argument("--mappe", "-m", help="Mappe med .txt filer (batch)")

    p.add_argument("--output",            "-o",  help="Output .json (enkelt fil)")
    p.add_argument("--output-mappe",      "-om", help="Output mappe (batch)")
    p.add_argument("--token",             "-t",  help="Hugging Face token")
    p.add_argument("--model",             default=DEFAULT_MODEL_NAME)
    p.add_argument("--device",            choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--device-map",        choices=["auto", "cuda"], default="auto")
    p.add_argument("--dtype",             choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--quant",             choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--gpu-max-memory-gb", type=float, default=None)
    p.add_argument("--max-new-tokens",    type=int, default=4096)
    p.add_argument("--max-input-tokens",  type=int, default=None)
    p.add_argument("--prefetch",          type=int, default=2,
                   help="Filer producer tokeniserer forud for GPU (standard: 2)")
    p.add_argument("--no-cache",          action="store_true")
    p.add_argument("--no-compile",        action="store_true")
    p.add_argument("--no-warmup",         action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--verbose",           "-v", action="store_true")
    p.add_argument("--print",             "-p", action="store_true", dest="do_print")
    p.add_argument("--max",               type=int, default=None, dest="max_filer")
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("FEJL: Mangler HF token. Brug --token eller sæt HF_TOKEN.")
        sys.exit(1)

    tokenizer, model = load_local_model(
        token,
        model_name=args.model,
        device_preference=args.device,
        dtype_preference=args.dtype,
        quantization=args.quant,
        gpu_max_memory_gb=args.gpu_max_memory_gb,
        device_map_preference=args.device_map,
        trust_remote_code=args.trust_remote_code,
        use_compile=not args.no_compile,
    )

    if not args.no_warmup:
        warmup_model(tokenizer, model, max_new_tokens=args.max_new_tokens)

    use_cache = not args.no_cache

    if args.input:
        input_path  = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".json")
        result = process_single_file(
            input_path, output_path, tokenizer, model,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            use_cache=use_cache,
            verbose=True,
        )
        if args.do_print:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mappe:
        input_dir  = Path(args.mappe)
        output_dir = Path(args.output_mappe) if args.output_mappe else input_dir.parent / "resultater"
        process_batch_pipeline(
            input_dir, output_dir, tokenizer, model,
            model_name=args.model,
            verbose=args.verbose,
            max_filer=args.max_filer,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            use_cache=use_cache,
            prefetch=args.prefetch,
        )
        return

    p.print_help()


if __name__ == "__main__":
    main()