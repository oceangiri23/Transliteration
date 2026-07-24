"""
Ablation study: effect of preprocessing on ByT5 and mT5 models for 
romanized -> Devanagari transliteration.

Models tested:
    1. ByT5 Original (byt5_original.pt)
    2. ByT5 Augmented (byt5_augmented.pt)
    3. mT5 Original (mt5_original.pt)
    4. mT5 Augmented (mt5_augmented.pt)

Configs per model:
    - Combined      : Rule-based preprocessing + PVD (dictionary) normalization
    - Rule-based only: Rule-based preprocessing (no dictionary step)
    - PVD only       : Dictionary normalization on raw tokens, no other cleanup
    - Raw / None     : No preprocessing at all

For each model, runs inference on 500 lines from "romanized_sampled.txt" 
and saves 4 output files per model.

Run locally in VS Code. Requires the following files:
    - romanized_sampled.txt        (input lines)
    - Models/byt5_original.pt      (ByT5 original checkpoint)
    - Models/byt5_augmented.pt     (ByT5 augmented checkpoint)
    - Models/mt5_original.pt       (mT5 original checkpoint)
    - Models/mt5_augmented.pt      (mT5 augmented checkpoint)
    - JSON_Final.json              (PVD/phonetic dictionary)
    - refined_english_words.txt    (english word list used by rule-based step)
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import re
import time
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from unidecode import unidecode

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
INPUT_FILE = "Transliteration\\Qualitative Analysis\\romanized_sampled.txt"
NUM_LINES = 500
MAX_CHARS = 125  # chunking threshold for ByT5 models

# Model configurations
MODELS_CONFIG = {
    "byt5_original": {
        "checkpoint": "Models/byt5_original.pt",
        "base_model": "google/byt5-small",
        "max_input_length": 300,
        "max_gen_length": 512,
        "use_chunking": True,
    },
    "byt5_augmented": {
        "checkpoint": "Models/byt5_augmented.pt",
        "base_model": "google/byt5-small",
        "max_input_length": 300,
        "max_gen_length": 512,
        "use_chunking": True,
    },
    "mt5_original": {
        "checkpoint": "Models/mt5_original.pt",
        "base_model": "google/mt5-small",
        "max_input_length": 300,
        "max_gen_length": 96,
        "use_chunking": False,
    },
    "mt5_augmented": {
        "checkpoint": "Models/mt5_augmented.pt",
        "base_model": "google/mt5-small",
        "max_input_length": 300,
        "max_gen_length": 512,
        "use_chunking": False,
    },
}

JSON_DICT_PATH = "Transliteration\\Final Datasets\\Phonetic_Variant_Dictionary.json"
ENGLISH_WORDS_PATH = (
    "Transliteration\\Final Datasets\\refined_english_words.txt"
)

OUTPUT_DIR = "Transliteration/Ablation Study/Outputs"

# ----------------------------------------------------------------------
# 1. ENGLISH WORD LIST (used by rule-based x->chh / w->a rules)
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
# 2. PVD / PHONETIC DICTIONARY
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
# 3. CHUNKING STRATEGY (for ByT5 models only)
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
# 4. PREPROCESSING VARIANTS
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
# 5. MODEL INFERENCE
# ----------------------------------------------------------------------
def transliterate_chunk(model, tokenizer, clean_input: str, max_gen_length: int, max_input_length: int) -> str:
    """Transliterate a single chunk."""
    if not clean_input.strip():
        return ""

    inputs = tokenizer(
        clean_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    # Check if input exceeded max length
    if inputs.input_ids.shape[1] > max_input_length:
        print(f"WARNING: Input exceeded {max_input_length} tokens (got {inputs.input_ids.shape[1]}). Truncated.")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_length=max_gen_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_pipeline(model, tokenizer, raw_line: str, preprocess_fn, use_chunking: bool, max_gen_length: int, max_input_length: int) -> str:
    """Run the full pipeline for a single line."""
    clean_input = preprocess_fn(raw_line)
    
    if use_chunking:
        chunks = chunk_text(clean_input, MAX_CHARS)
        chunk_outputs = [transliterate_chunk(model, tokenizer, c, max_gen_length, max_input_length) for c in chunks]
        return " ".join(chunk_outputs)
    else:
        return transliterate_chunk(model, tokenizer, clean_input, max_gen_length, max_input_length)


# ----------------------------------------------------------------------
# 6. MAIN ABLATION LOOP
# ----------------------------------------------------------------------
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read input lines
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    lines = lines[:NUM_LINES]
    print(f"Loaded {len(lines)} lines from {INPUT_FILE}")

    # Process each model
    for model_name, model_config in MODELS_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Load model and tokenizer
        print(f"Loading tokenizer + base model from {model_config['base_model']}...")
        tokenizer = AutoTokenizer.from_pretrained(model_config['base_model'])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_config['base_model']).to(device)
        
        print(f"Loading local checkpoint from {model_config['checkpoint']}...")
        byt5_state = torch.load(model_config['checkpoint'], map_location=device)
        if "model" in byt5_state:
            model.load_state_dict(byt5_state["model"])
        elif "state_dict" in byt5_state:
            model.load_state_dict(byt5_state["state_dict"])
        else:
            model.load_state_dict(byt5_state)
        model.eval()
        
        # Set config
        model.config.forced_eos_token_id = None
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Open output files for this model
        output_files = {}
        try:
            for config_name in PREPROCESSORS.keys():
                output_filename = f"{model_name}_{config_name}.txt"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                output_files[config_name] = open(output_path, "w", encoding="utf-8")
                print(f"  Writing to: {output_filename}")
            
            # Process each line
            for idx, line in enumerate(lines, 1):
                for config_name, preprocess_fn in PREPROCESSORS.items():
                    t0 = time.time()
                    output = run_pipeline(
                        model, 
                        tokenizer, 
                        line, 
                        preprocess_fn, 
                        model_config['use_chunking'],
                        model_config['max_gen_length'],
                        model_config['max_input_length']
                    )
                    output_files[config_name].write(output + "\n")
                    output_files[config_name].flush()
                
                if idx % 10 == 0 or idx == len(lines):
                    print(f"  Processed {idx}/{len(lines)} lines")
        
        finally:
            for fh in output_files.values():
                fh.close()
        
        print(f"Finished {model_name}")
        
        # Clean up to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "="*60)
    print("All models processed. Outputs written to:")
    print(f"  {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    # Setup device
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    main()