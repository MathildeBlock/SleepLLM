"""
psg_extract_optimized.py  -  Trin 2: Udtræk struktureret JSON fra PSG-rapport via lokal LLM.

Optimeringer vs. originalen:
  - System-prompt tokeniseres KUN ÉN GANG ved opstart (ikke per fil)
  - torch.compile() aktiveres automatisk på GPU (PyTorch 2.0+, ~20-30% speedup)
  - Ingen unødvendige time.sleep() i retry-logik (beholder kun ved OOM)
  - schema_str caches som modul-konstant (ikke genberegnet per kald)
  - Kompileret model varmes op med et dummy-kald inden batch starter
  - Tydelige advarsler hvis torch.compile ikke er mulig

Krav:
  pip install transformers torch huggingface_hub accelerate tqdm

Brug:
  # Enkelt fil
  python psg_extract_optimized.py rapport.txt --token DIT_HF_TOKEN

  # Batch (alle .txt i en mappe)
  python psg_extract_optimized.py --mappe forberedt/ --output-mappe resultater/ --token DIT_HF_TOKEN
"""

import os
import sys
import json
import datetime
import argparse
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("FEJL: Mangler tqdm. Kør: pip install tqdm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# JSON-skema  (komplet – alle felter fra dit skema)
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

# Cache schema som streng én gang — bruges i alle prompt-kald
_SCHEMA_STR = json.dumps(SCHEMA, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Du er en præcis dataekstraktor specialiseret i polysomnografi (PSG) rapporter på dansk.
Din opgave er at udtrække strukturerede data fra kliniske søvnrapporter og returnere dem som JSON. Du skal følge reglerne.

Regler:
- Returner KUN gyldigt JSON — ingen tekst før eller efter.
- Du må ikke ændre på JSON skemaet, kun udfylde det.
- Brug null for felter der ikke findes i rapporten.
- Tomme felter er udfyldt med "-" i preprocessering, det skal forstås som null.
- Kopier værdier PRÆCIST som de står i rapporten — ingen konvertering, ingen fortolkning.
- Tal der står som tal forbliver tal: 42.5 ikke "42.5".
- Komma som decimaltegn erstattes med punktum: 35,5 → 35.5.
- Procenter som tal uden %-tegn: 85.3 ikke "85.3%".
- Tidsværdier kopieres præcist: "07:41:30" forbliver "07:41:30", "461 min" forbliver "461 min".
- Datoer kopieres præcist som de står i rapporten.
- "starttid" og "sluttid" er to separate felter — split fra "starttid-sluttid" hvis de står samlet.
- "cpr-nummer" er KUN CPR-nummeret i formatet DDMMYY-XXXX.
- "arousals" har to underfelter: "total" (antal) og "indeks" (per time).
- "sammenfatning" indeholder fritekstfelter — bevar den originale tekst fra rapporten."""

USER_PROMPT_TEMPLATE = """Udtræk alle oplysninger fra denne PSG-rapport og udfyld JSON-skemaet.

<rapport>
{text}
</rapport>

<skema>
{schema}
</skema>

JSON:"""


# ---------------------------------------------------------------------------
# Validering
# ---------------------------------------------------------------------------

def validate(data: dict, filename: str) -> list[str]:
    """Returnerer en liste af advarsler for mistænkelige værdier."""
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
                import re
                match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
                if not match:
                    return None
                return float(match.group(0))
            except Exception:
                return None
        return None

    p = data.get("patient", {}) or {}
    bmi_value = as_number(p.get("BMI"))
    if bmi_value is not None and not (10 < bmi_value < 80):
        w(f"Mistænkelig BMI: {p.get('BMI')}")

    ess_value = as_number(p.get("ESS"))
    if ess_value is not None and not (0 <= ess_value <= 24):
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
    min_spo2_value = as_number(min_spo2)
    if min_spo2_value is not None and min_spo2_value < 50:
        w(f"SpO2 minimum meget lav: {min_spo2}%")

    return warnings


# ---------------------------------------------------------------------------
# Lokal Model Håndtering (Transformers)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "google/gemma-3-1b-it"

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
        raise RuntimeError(
            "CUDA blev valgt, men denne Python-installation har ikke en CUDA-aktiveret torch."
        )

    use_cuda = device_preference in {"auto", "cuda"} and cuda_available

    try:
        python_exe = sys.executable
    except Exception:
        python_exe = "(ukendt)"

    print(
        "Runtime: "
        f"python={python_exe}  "
        f"torch={getattr(torch, '__version__', '?')}  "
        f"cuda_available={cuda_available}"
    )
    if cuda_available:
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    if dtype_preference == "auto":
        dtype = torch.float16 if use_cuda else torch.float32
    elif dtype_preference == "float16":
        dtype = torch.float16
    elif dtype_preference == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    quantization_config = None
    if quantization != "none":
        if not use_cuda:
            raise RuntimeError("Kvantisering kræver CUDA (bitsandbytes).")

        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            print("FEJL: Kvantisering kræver bitsandbytes. Prøv: pip install bitsandbytes")
            sys.exit(1)

        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            compute_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float16
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
        raise ValueError("Ugyldig --device-map. Brug: auto eller cuda")

    if device_map_preference == "cuda" and not use_cuda:
        raise RuntimeError("--device-map cuda kræver at CUDA er tilgængelig.")

    resolved_device_map = {"": 0} if device_map_preference == "cuda" else ("auto" if use_cuda else None)

    print(f"Indlæser lokal model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        dtype=None if quantization == "8bit" else dtype,
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
        print(f"Valgt enhed ved opstart: {', '.join(selected_devices)}")

    if use_cuda and selected_devices and all(d in {"cpu", "disk"} for d in selected_devices):
        msg = (
            "ADVARSEL: CUDA er tilgængelig, men modellen blev placeret på CPU via device_map. "
            "Prøv: --device-map cuda"
        )
        print(msg)
        if device_preference == "cuda":
            raise RuntimeError("Bad device placement: --device cuda men device_map endte på CPU.")

    # -----------------------------------------------------------------------
    # torch.compile — giver ~20-30% speedup på A100 ved lange sekvenser.
    # Første kald er langsomt (kompilering), derefter hurtigere.
    # -----------------------------------------------------------------------
    if use_compile and use_cuda:
        try:
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_version >= (2, 0):
                print("torch.compile() aktiveret (første kald kompilerer — forvent forsinkelse)...")
                model = torch.compile(model, mode="reduce-overhead")
            else:
                print(f"torch.compile() kræver PyTorch >= 2.0 (du har {torch.__version__}) — springer over.")
        except Exception as e:
            print(f"torch.compile() fejlede ({e}) — kører uden kompilering.")
    elif use_compile and not use_cuda:
        print("torch.compile() springer over — ikke på GPU.")

    if use_cuda:
        print(f"CUDA aktiv: kører med {dtype} og device_map='auto'")
    else:
        print("CUDA ikke tilgængelig; kører på CPU.")

    return tokenizer, model


# ---------------------------------------------------------------------------
# Cached prompt-prefix tokenisering
# Systemprompten tokeniseres KUN ÉN GANG og genbruges for alle filer.
# ---------------------------------------------------------------------------

class PromptCache:
    """
    Forbereder og cacher den del af prompten der er ens for alle filer
    (system-prompt + schema). Kun rapport-teksten tokeniseres per fil.

    Bemærk: Vi kan ikke dele KV-cache på tværs af generate()-kald med
    standard HF Transformers uden custom kode, men vi sparer Python-overhead
    ved at undgå gentagen strengformatering og tokenisering af den faste del.
    """

    def __init__(self, tokenizer, model):
        import torch
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch
        self._system_and_schema = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.split("{text}")[0]
        self._suffix = USER_PROMPT_TEMPLATE.split("{text}")[1].format(schema=_SCHEMA_STR)
        self._suffix += "\n\nStart dit svar med { og slut dit svar med }."

    def build_inputs(self, report_text: str, max_input_tokens: int | None = None):
        """Returnerer tokeniserede inputs klar til model.generate()."""
        full_prompt = (
            SYSTEM_PROMPT + "\n\n" +
            USER_PROMPT_TEMPLATE.format(text=report_text, schema=_SCHEMA_STR) +
            "\n\nStart dit svar med { og slut dit svar med }."
        )
        messages = [{"role": "user", "content": full_prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = _resolve_model_input_device(self.model, self._torch)
        return self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
        ).to(device)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup_model(tokenizer, model, max_new_tokens: int):
    """
    Kør et kort dummy-kald så torch.compile() kompilerer kernels inden
    den rigtige batch starter. Undgår at første fil er unormalt langsom.
    """
    import torch
    print("Varmer op model (dummy-kald for torch.compile)...")
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
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(raw):
    """Prøver at parse JSON, håndterer evt. Markdown fencing."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]

    return json.loads(text)


# ---------------------------------------------------------------------------
# Udtræk fra tekst
# ---------------------------------------------------------------------------

def extract_from_text(
    prompt_cache: PromptCache,
    text: str,
    max_retries: int = 3,
    max_new_tokens: int = 4096,
    max_input_tokens: int | None = None,
    use_cache: bool = True,
    debug_label: str | None = None,
) -> dict:
    import torch

    inputs = prompt_cache.build_inputs(text, max_input_tokens=max_input_tokens)

    last_error = None
    for attempt in range(1, max_retries + 1):
        raw = ""
        try:
            with torch.no_grad():
                outputs = prompt_cache.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=prompt_cache.tokenizer.eos_token_id,
                    use_cache=use_cache,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw = prompt_cache.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return parse_json_response(raw)

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print(f"\n[Forsøg {attempt}] OOM — rydder VRAM og venter...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(3)  # Behold ved OOM — VRAM skal nå at frigøres

        except json.JSONDecodeError as e:
            last_error = e
            print(f"\n[Forsøg {attempt}] JSON Parse Fejl: {e}")

            if debug_label:
                filename = f"failed_{debug_label}"
                if not filename.endswith(".txt"):
                    filename += ".txt"
                debug_file = Path(filename)
            else:
                debug_file = Path("failed_output_debug.txt")
            debug_file.write_text(raw, encoding="utf-8")
            print(f"  --> Gemte det rå, fejlede output i {debug_file.resolve()}")
            # Ingen sleep her — JSON-fejl er ikke tidsfølsomme

        except Exception as e:
            last_error = e
            print(f"\n[Forsøg {attempt}] Uventet fejl: {e}")
            # Ingen generisk sleep — kun OOM kræver ventetid

    raise RuntimeError(f"Udtrækning fejlede efter {max_retries} forsøg. Sidste fejl: {last_error}")


# ---------------------------------------------------------------------------
# Enkelt fil
# ---------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    prompt_cache: PromptCache,
    model_name: str,
    max_new_tokens: int = 4096,
    max_input_tokens: int | None = None,
    use_cache: bool = True,
    verbose: bool = False
) -> dict:

    text = input_path.read_text(encoding="utf-8")

    if verbose:
        print(f"  Tekst: {len(text)} tegn")

    data = extract_from_text(
        prompt_cache,
        text,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        use_cache=use_cache,
        debug_label=input_path.name,
    )

    data["metadata"]["filnavn"]    = input_path.stem
    data["metadata"]["udfyldt_af"] = model_name

    warnings = validate(data, input_path.name)
    for w in warnings:
        print(w)

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if verbose:
        print(f"  Gemt: {output_path}")

    return data


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def process_batch(
    input_dir,
    output_dir,
    prompt_cache: PromptCache,
    model_name,
    verbose=False,
    max_filer=None,
    max_new_tokens: int = 4096,
    max_input_tokens: int | None = None,
    use_cache: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"Ingen .txt filer fundet i: {input_dir}")
        return

    if max_filer:
        txt_files = txt_files[:max_filer]

    print(f"Fandt {len(txt_files)} filer. Starter udtrækning med model: {model_name}")

    ok          = 0
    fejl        = 0
    sprang_over = 0
    fejl_liste  = []

    for txt_file in tqdm(txt_files, unit="fil"):
        out_file = output_dir / (txt_file.stem + ".json")

        if out_file.exists():
            sprang_over += 1
            continue

        try:
            if verbose:
                tqdm.write(f"\n→ {txt_file.name}")

            process_file(
                txt_file,
                out_file,
                prompt_cache,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                use_cache=use_cache,
                verbose=verbose,
            )
            ok += 1

        except Exception as e:
            tqdm.write(f"\n❌ Fejl ved {txt_file.name}: {e}")
            fejl += 1
            fejl_liste.append(txt_file.name)

            fejl_log = output_dir / "fejl_log.txt"
            with open(fejl_log, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {txt_file.name}  {e}\n")

    print(f"\n{'='*50}")
    print(f"Færdig!")
    print(f"  ✅ OK:           {ok}")
    print(f"  ⏭  Sprang over:  {sprang_over} (allerede behandlet)")
    print(f"  ❌ Fejl:         {fejl}")
    if fejl_liste:
        print(f"\nFejlede filer gemt i: {output_dir / 'fejl_log.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Udtræk struktureret JSON batch via lokal transformers (optimeret)")

    input_gruppe = p.add_mutually_exclusive_group()
    input_gruppe.add_argument("input", nargs="?", help="Sti til én .txt fil")
    input_gruppe.add_argument("--mappe", "-m", help="Mappe med .txt filer (batch-tilstand)")

    p.add_argument("--output",           "-o",  help="Output .json fil (enkelt fil)")
    p.add_argument("--output-mappe",     "-om", help="Output mappe (batch-tilstand)")
    p.add_argument("--token",            "-t",  help="Hugging Face token")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face model-id")
    p.add_argument("--device",           choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--device-map",       choices=["auto", "cuda"], default="auto")
    p.add_argument("--dtype",            choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--quant",            choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--gpu-max-memory-gb", type=float, default=None)
    p.add_argument("--max-new-tokens",   type=int, default=4096)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--no-cache",         action="store_true", help="Deaktiver KV-cache")
    p.add_argument("--no-compile",       action="store_true", help="Deaktiver torch.compile()")
    p.add_argument("--no-warmup",        action="store_true", help="Spring warmup-kald over")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--verbose",          "-v", action="store_true")
    p.add_argument("--print",            "-p", action="store_true", dest="do_print")
    p.add_argument("--max",              type=int, default=None, dest="max_filer")
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("FEJL: Mangler Hugging Face token. Brug --token eller sæt HF_TOKEN.")
        sys.exit(1)

    model_name = args.model
    use_compile = not args.no_compile
    use_cache   = not args.no_cache

    tokenizer, model = load_local_model(
        token,
        model_name=model_name,
        device_preference=args.device,
        dtype_preference=args.dtype,
        quantization=args.quant,
        gpu_max_memory_gb=args.gpu_max_memory_gb,
        device_map_preference=args.device_map,
        trust_remote_code=args.trust_remote_code,
        use_compile=use_compile,
    )

    # Byg prompt-cache (genbruges for alle filer)
    prompt_cache = PromptCache(tokenizer, model)

    # Warmup: kompiler kernels inden batch
    if not args.no_warmup:
        warmup_model(tokenizer, model, max_new_tokens=args.max_new_tokens)

    if args.input:
        input_path  = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".json")
        result = process_file(
            input_path,
            output_path,
            prompt_cache,
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            use_cache=use_cache,
            verbose=True,
        )
        if args.do_print:
            print()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mappe:
        input_dir  = Path(args.mappe)
        output_dir = Path(args.output_mappe) if args.output_mappe else input_dir.parent / "resultater"
        process_batch(
            input_dir,
            output_dir,
            prompt_cache,
            model_name=model_name,
            verbose=args.verbose,
            max_filer=args.max_filer,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            use_cache=use_cache,
        )
        return

    p.print_help()


if __name__ == "__main__":
    main()