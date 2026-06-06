import os
import json
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from unidecode import unidecode
from two_sentence_pipeline import repair_transliteration

torch.set_num_threads(1)
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BYT5_CHECKPOINT = "google/byt5-small"  
MODEL_PATH = os.path.join(BASE_DIR, "../../Final_Models/best_model.pt")
PHONETIC_DICT_PATH = os.path.join(BASE_DIR, "../../Final Datasets/Phonetic_Variant_Dictionary .json")
ENGLISH_WORDS_PATH = os.path.join(BASE_DIR, "../../Final Datasets/English_words.txt")


with open(ENGLISH_WORDS_PATH, "r", encoding="utf-8") as f:
    english_words = set(word.strip().lower() for word in f.readlines())


variant_to_base = {}
base_words_set = set()
PHONETIC_LOADED = False

try:
    with open(PHONETIC_DICT_PATH, "r", encoding="utf-8") as f:
        phonetic_data = json.load(f)

    for entry in phonetic_data.values():
        base = entry["base_word"].strip().lower()
        base_words_set.add(base)
        variant_to_base[base] = base

        for v in entry.get("variants", []):
            v = v.strip().lower()
            if v and v != base:
                variant_to_base[v] = base

    PHONETIC_LOADED = True
    print(f"✓ Dictionary loaded ({len(base_words_set)} base words)")

except FileNotFoundError:
    print(" Phonetic dictionary file not found")

byt5_tokenizer = AutoTokenizer.from_pretrained(BYT5_CHECKPOINT)
byt5_model = AutoModelForSeq2SeqLM.from_pretrained(BYT5_CHECKPOINT).to(device)

byt5_state = torch.load(MODEL_PATH, map_location="cpu")

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

def preprocess_with_changes(text: str):
    changes = []

    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)

    words = text.split()
    processed_words = []

    for w in words:
        prefix, suffix = "", ""

        while w and not w[0].isalnum():
            prefix += w[0]
            w = w[1:]

        while w and not w[-1].isalnum():
            suffix = w[-1] + suffix
            w = w[:-1]

        core = w.lower()
        original_word = prefix + core + suffix

        # Rule 7
        if core not in english_words and "x" in core:
            new_core = core.replace("x", "chh")
            changes.append({
                "original": original_word,
                "replaced": prefix + new_core + suffix,
                "rule": "x → chh"
            })
            core = new_core

        # Rule 8
        if core not in english_words and core.endswith("w"):
            new_core = core[:-1] + "a"
            changes.append({
                "original": original_word,
                "replaced": prefix + new_core + suffix,
                "rule": "w → a"
            })
            core = new_core

        # Rule 9
        if PHONETIC_LOADED and core in variant_to_base:
            new_core = variant_to_base[core]
            changes.append({
                "original": original_word,
                "replaced": prefix + new_core + suffix,
                "rule": "dictionary mapping"
            })
            core = new_core

        processed_words.append(prefix + core + suffix)

    return " ".join(processed_words), changes


def transliterate(raw_input: str):
    if not raw_input.strip():
        return "", "", "", []

    clean_input, changes = preprocess_with_changes(raw_input)

    inputs = byt5_tokenizer(
        clean_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=300
    ).to(device)

    with torch.inference_mode():
        outputs = byt5_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=300,
            pad_token_id=byt5_tokenizer.pad_token_id,
            eos_token_id=byt5_tokenizer.eos_token_id,
        )

    raw_output = byt5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    repaired, _ = repair_transliteration(clean_input, raw_output)

    return clean_input, raw_output, repaired, changes

CUSTOM_CSS = """
.gradio-container { max-width: 850px; margin: 2rem auto; }
.header { text-align: center; margin-bottom: 1rem; }
"""

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# Romanized Nepali → Devanagari Translator (LOCAL)")

    input_text = gr.Textbox(label="Input", lines=3)

    submit = gr.Button("Transliterate")

    preprocessed_out = gr.Textbox(label="Preprocessed")
    raw_out = gr.Textbox(label="Raw Output")
    final_out = gr.Textbox(label="Final Output")
    history_out = gr.JSON(label="Changes")

    submit.click(
        transliterate,
        inputs=input_text,
        outputs=[preprocessed_out, raw_out, final_out, history_out]
    )

demo.launch()