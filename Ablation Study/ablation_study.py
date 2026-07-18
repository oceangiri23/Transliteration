"""
Ablation study: effect of preprocessing on ByT5 romanized -> Devanagari transliteration.

Configs:
    1. Combined      : Rule-based preprocessing + PVD (dictionary) normalization -> Model
    2. Rule-based only: Rule-based preprocessing (no dictionary step)            -> Model
    3. PVD only       : Dictionary normalization on raw tokens, no other cleanup -> Model
    4. Raw / None     : No preprocessing at all                                 -> Model

Reads 500 lines from "romanized_sampled.txt" and writes 4 output files,
one line of model output per line of input, for each config.

Run locally in VS Code. Requires the following files in the same directory:
    - romanized_sampled.txt        (input lines)
    - final_byt5.pt                (local model checkpoint)
    - JSON_Final.json              (PVD/phonetic dictionary)
    - refined_english_words.txt    (english word list used by rule-based step)
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from unidecode import unidecode

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
INPUT_FILE = "Transliteration\\Qualitative Analysis\\romanized_sampled.txt"
NUM_LINES = 500
MAX_CHARS = 125  # chunking threshold, same as production pipeline

MODEL_CHECKPOINT_PATH = "final_byt5_latest.pt"
BYT5_BASE = "google/byt5-small"

JSON_DICT_PATH = "Transliteration\\Final Datasets\\Phonetic_Variant_Dictionary.json"
ENGLISH_WORDS_PATH = (
    "Transliteration\\src\\English_restoration\\refined_english_words.txt"
)

OUTPUT_FILES = {
    "combined": "Transliteration/Ablation Study/01_ablation_combined.txt",
    "rule_based": "Transliteration/Ablation Study/02_ablation_rule_based_only.txt",
    "pvd_only": "Transliteration/Ablation Study/03_ablation_pvd_only.txt",
    "raw": "Transliteration/Ablation Study/04_ablation_raw.txt",
}

# ----------------------------------------------------------------------
# 1. ENVIRONMENT & MODEL
# ----------------------------------------------------------------------
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer + base model...")
byt5_tokenizer = AutoTokenizer.from_pretrained(BYT5_BASE)
byt5_model = AutoModelForSeq2SeqLM.from_pretrained(BYT5_BASE).to(device)

print(f"Loading local checkpoint from {MODEL_CHECKPOINT_PATH} ...")
byt5_state = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
if "model" in byt5_state:
    byt5_model.load_state_dict(byt5_state["model"])
elif "state_dict" in byt5_state:
    byt5_model.load_state_dict(byt5_state["state_dict"])
else:
    byt5_model.load_state_dict(byt5_state)
byt5_model.eval()

byt5_model.config.forced_eos_token_id = None
byt5_model.config.eos_token_id = byt5_tokenizer.eos_token_id
byt5_model.config.pad_token_id = byt5_tokenizer.pad_token_id

# ----------------------------------------------------------------------
# 2. ENGLISH WORD LIST (used by rule-based x->chh / w->a rules)
# ----------------------------------------------------------------------
try:
    with open(ENGLISH_WORDS_PATH, "r", encoding="utf-8") as f:
        english_words = set(word.strip().lower() for word in f.readlines())
except FileNotFoundError:
    print(
        f"WARNING: {ENGLISH_WORDS_PATH} not found. Treating english_words as empty set."
    )
    english_words = set()

# ----------------------------------------------------------------------
# 3. PVD / PHONETIC DICTIONARY
# ----------------------------------------------------------------------
variant_to_base = {}
PHONETIC_LOADED = False
try:
    with open(JSON_DICT_PATH, "r", encoding="utf-8") as f:
        phonetic_data = json.load(f)
    for entry in phonetic_data.values():
        base = entry["base_word"].strip().lower()
        variant_to_base[base] = base
        for v in entry.get("variants", []):
            v = v.strip().lower()
            if v and v != base:
                variant_to_base[v] = base
    PHONETIC_LOADED = True
    print(f"Dictionary loaded ({len(variant_to_base)} entries)")
except FileNotFoundError:
    print(f"WARNING: {JSON_DICT_PATH} not found. PVD normalization disabled.")


# ----------------------------------------------------------------------
# 4. CHUNKING STRATEGY (unchanged from production pipeline)
# ----------------------------------------------------------------------
def chunk_text(text: str, max_chars: int = MAX_CHARS):
    if len(text) <= max_chars:
        return [text]

    chunks = []
    words = text.split()
    current_chunk = ""

    for word in words:
        test_chunk = word if not current_chunk else current_chunk + " " + word

        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(word) > max_chars:
                for i in range(0, len(word), max_chars):
                    chunks.append(word[i : i + max_chars])
                current_chunk = ""
            else:
                current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ----------------------------------------------------------------------
# 5. PREPROCESSING VARIANTS
# ----------------------------------------------------------------------
def _split_prefix_core_suffix(word: str):
    """Strip leading/trailing non-alnum chars, return (prefix, core, suffix)."""
    prefix = ""
    suffix = ""
    w = word
    while w and not w[0].isalnum():
        prefix += w[0]
        w = w[1:]
    while w and not w[-1].isalnum():
        suffix = w[-1] + suffix
        w = w[:-1]
    return prefix, w, suffix


def preprocess_rule_based(text: str) -> str:
    """
    Rule-based preprocessing only (everything EXCEPT the dictionary/PVD step):
    lowercase, accent stripping, whitespace collapse, repeated-letter collapse,
    special-char removal, x->chh, w->a.
    """
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)

    words = text.split()
    processed_words = []

    for w in words:
        prefix, core, suffix = _split_prefix_core_suffix(w)
        core = core.lower()

        if core not in english_words and "x" in core:
            core = core.replace("x", "chh")

        if core not in english_words and core.endswith("w"):
            core = core[:-1] + "a"

        processed_words.append(prefix + core + suffix)

    return " ".join(processed_words)


def preprocess_combined(text: str) -> str:
    """Rule-based preprocessing + PVD (dictionary) normalization."""
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)

    words = text.split()
    processed_words = []

    for w in words:
        prefix, core, suffix = _split_prefix_core_suffix(w)
        core = core.lower()

        if core not in english_words and "x" in core:
            core = core.replace("x", "chh")

        if core not in english_words and core.endswith("w"):
            core = core[:-1] + "a"

        if PHONETIC_LOADED and core in variant_to_base:
            core = variant_to_base[core]

        processed_words.append(prefix + core + suffix)

    return " ".join(processed_words)


def preprocess_pvd_only(text: str) -> str:
    """
    PVD/dictionary normalization applied to raw tokens, with NO other cleanup
    (no lowercasing, no accent stripping, no repeat-letter collapse, no
    special-char removal). Only leading/trailing punctuation is stripped from
    each whitespace-separated token so the alphanumeric core can be looked up.
    Since dictionary keys are lowercase, a token only matches if it is already
    lowercase in the raw input -- this is intentional for this ablation arm.
    """
    words = text.split()
    processed_words = []

    for w in words:
        prefix, core, suffix = _split_prefix_core_suffix(w)

        if PHONETIC_LOADED and core in variant_to_base:
            core = variant_to_base[core]

        processed_words.append(prefix + core + suffix)

    return " ".join(processed_words)


def preprocess_none(text: str) -> str:
    """No preprocessing at all -- raw input passed straight through."""
    return text


PREPROCESSORS = {
    "combined": preprocess_combined,
    "rule_based": preprocess_rule_based,
    "pvd_only": preprocess_pvd_only,
    "raw": preprocess_none,
}


# ----------------------------------------------------------------------
# 6. MODEL INFERENCE
# ----------------------------------------------------------------------
def transliterate_single_chunk(clean_input: str) -> str:
    if not clean_input.strip():
        return ""

    inputs = byt5_tokenizer(
        clean_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=300,
    ).to(device)

    with torch.inference_mode():
        outputs = byt5_model.generate(
            input_ids=inputs.input_ids,
            max_length=300,
            pad_token_id=byt5_tokenizer.pad_token_id,
            eos_token_id=byt5_tokenizer.eos_token_id,
        )

    return byt5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_pipeline(raw_line: str, preprocess_fn) -> str:
    clean_input = preprocess_fn(raw_line)
    chunks = chunk_text(clean_input, MAX_CHARS)
    chunk_outputs = [transliterate_single_chunk(c) for c in chunks]
    return " ".join(chunk_outputs)


# ----------------------------------------------------------------------
# 7. MAIN ABLATION LOOP
# ----------------------------------------------------------------------
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    lines = lines[:NUM_LINES]
    print(f"Loaded {len(lines)} lines from {INPUT_FILE}")

    # Open all output files up front
    file_handles = {
        name: open(path, "w", encoding="utf-8") for name, path in OUTPUT_FILES.items()
    }

    try:
        for idx, line in enumerate(lines, 1):
            for config_name, preprocess_fn in PREPROCESSORS.items():
                t0 = time.time()
                output = run_pipeline(line, preprocess_fn)
                file_handles[config_name].write(output + "\n")
                file_handles[config_name].flush()

            if idx % 10 == 0 or idx == len(lines):
                print(f"Processed {idx}/{len(lines)} lines")
    finally:
        for fh in file_handles.values():
            fh.close()

    print("Done. Outputs written to:")
    for name, path in OUTPUT_FILES.items():
        print(f"  {name:12s} -> {path}")


if __name__ == "__main__":
    main()
