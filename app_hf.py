import json
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
import os
from unidecode import unidecode
from two_sentence_pipeline import repair_transliteration

# ----------------------------------------------------------------------
# 1. ENVIRONMENT & MODELS
# ----------------------------------------------------------------------
torch.set_num_threads(1)
device = torch.device("cpu")
hf_token = os.getenv("HF_TOKEN")

# English list for steps 8 + repair
with open("refined_english_words.txt", "r", encoding="utf-8") as f:
    english_words = set(word.strip().lower() for word in f.readlines())

# ----------------------------------------------------------------------
# 2. PHONETIC DICTIONARY
# ----------------------------------------------------------------------
variant_to_base = {}
base_words_set = set()
PHONETIC_LOADED = False

PHONETIC_DICT_PATH = "JSON_Final.json"
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
    print(f"✓ Dictionary loaded ({len(base_words_set)} base, {len(variant_to_base)} entries)")
except FileNotFoundError:
    print("⚠️ Phonetic dictionary file not found – JSON normalization disabled")

# ByT5
byt5_checkpoint = "google/byt5-small"
byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_checkpoint)
byt5_model = AutoModelForSeq2SeqLM.from_pretrained(byt5_checkpoint).to(device)
byt5_model_path = hf_hub_download(
    repo_id="Sagar32/romanizedtransliterationmodel",
    filename="final_aug_parallel_saksham_code_best.pt",
    token=hf_token
)
byt5_state = torch.load(byt5_model_path, map_location="cpu")
# Handle both possible save formats
if "model" in byt5_state:
    byt5_model.load_state_dict(byt5_state["model"])
elif "state_dict" in byt5_state:
    byt5_model.load_state_dict(byt5_state["state_dict"])
else:
    byt5_model.load_state_dict(byt5_state)
byt5_model.eval()

# CRITICAL: Set model config for generation (already fine but kept for clarity)
byt5_model.config.forced_eos_token_id = None
byt5_model.config.eos_token_id = byt5_tokenizer.eos_token_id
byt5_model.config.pad_token_id = byt5_tokenizer.pad_token_id

# ----------------------------------------------------------------------
# 3. PREPROCESSING WITH CHANGE TRACKING (unchanged)
# ----------------------------------------------------------------------
def preprocess_with_changes(text: str):
    """
    Returns:
        processed_text (str)
        changes (list of dicts): each has {"original":..., "replaced":..., "rule":...}
    """
    changes = []

    # 1. Lowercase
    text = text.lower()

    # 2. Accents → ASCII
    text = unidecode(text)

    # 3. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 4. (other Latin variants covered by unidecode)

    # 5. Collapse repeated letters >2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 6. Remove special characters (keep alnum, spaces, common punctuation)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)

    # Split into words (with possible punctuation)
    words = text.split()
    processed_words = []

    for w in words:
        # separate leading/trailing punctuation
        prefix = ""
        suffix = ""

        while w and not w[0].isalnum():
            prefix += w[0]
            w = w[1:]

        while w and not w[-1].isalnum():
            suffix = w[-1] + suffix
            w = w[:-1]

        core = w.lower()
        original_word = prefix + core + suffix

        # --- Rule 7: x → chh (non-English only) ---
        if core not in english_words and "x" in core:
            new_core = core.replace("x", "chh")

            if new_core != core:
                changes.append({
                    "original": prefix + core + suffix,
                    "replaced": prefix + new_core + suffix,
                    "rule": "x → chh (non-English)"
                })

                core = new_core

        # --- Rule 8: w → a at end of word (non-English only) ---
        if core not in english_words and core.endswith("w"):
            new_core = core[:-1] + "a"

            if new_core != core:
                changes.append({
                    "original": prefix + core + suffix,
                    "replaced": prefix + new_core + suffix,
                    "rule": "w → a (word ending, non-English)"
                })

                core = new_core

        # --- Rule 9: JSON normalization ---
        if PHONETIC_LOADED and core in variant_to_base:
            new_core = variant_to_base[core]

            if new_core != core:
                changes.append({
                    "original": original_word,
                    "replaced": prefix + new_core + suffix,
                    "rule": "Dictionary (variant → base)"
                })

                core = new_core

        processed_words.append(prefix + core + suffix)

    final_text = " ".join(processed_words)

    return final_text, changes

# ----------------------------------------------------------------------
# 4. TRANSLITERATION (FIXED GENERATION)
# ----------------------------------------------------------------------
def transliterate(raw_input: str):
    if not raw_input.strip():
        return "", "", "", []

    clean_input, changes = preprocess_with_changes(raw_input)

    # Tokenize with proper padding and attention mask
    inputs = byt5_tokenizer(
        clean_input, 
        return_tensors="pt", 
        padding=True,
        truncation=True, 
        max_length=300
    ).to(device)
    
    with torch.inference_mode():
        # --- MATCH THE TRAINING EVALUATION GENERATION ---
        # Only max_length and default greedy decoding; no beam search or length penalties
        outputs = byt5_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,      # passed but training did not – still safe
            max_length=300,
            pad_token_id=byt5_tokenizer.pad_token_id,
            eos_token_id=byt5_tokenizer.eos_token_id,
        )
    
    raw_output = byt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debug info (optional, keep for now)
    print(f"DEBUG - Input length: {len(clean_input)} chars")
    print(f"DEBUG - Output tokens: {outputs.shape[1]}")
    print(f"DEBUG - Raw output: {raw_output[:200]}")
    
    repaired, _ = repair_transliteration(clean_input, raw_output)

    return clean_input, raw_output, repaired, changes

# ----------------------------------------------------------------------
# 5. UI (unchanged)
# ----------------------------------------------------------------------
CUSTOM_CSS = """
:root { --primary: #6C63FF; }
.gradio-container { max-width: 850px; margin: 2rem auto; }
.header { text-align: center; margin-bottom: 1.5rem; }
.header h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #6C63FF, #FF6584);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.gr-box { border-radius: 16px; border: 1px solid #eef0f5; box-shadow: 0 12px 30px rgba(0,0,0,0.04); }
button.primary { background: linear-gradient(135deg, #6C63FF, #3F3D99); border: none; border-radius: 12px; }
button.primary:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(108,99,255,0.3); }
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; margin-left: 8px; }
.status-loaded { background: #2b874b; color: #ebf5ef; }
.status-missing { background: #fee2e2; color: #991b1b; }
.footer {
    text-align: center;
    margin-top: 20px;
    color: #c9c5c5;
}
"""

status_badge = ("<span class='status-badge status-loaded'>📚 Dictionary loaded</span>" 
                if PHONETIC_LOADED else 
                "<span class='status-badge status-missing'>⚠️ Not loaded</span>")

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(f"<div class='header'><h1>⚡ Code-Mixed Romanized Nepali + English → Devanagari </h1>"
            f"<p>Preserves English words. {status_badge}</p></div>")

    with gr.Group(elem_classes="gr-box"):
        input_text = gr.Textbox(label="📝 Code‑mixed Input", lines=3,
                                placeholder="e.g., literally bro yo translator tw as expected ramro kam gariraxa tw")

    with gr.Row():
        submit_btn = gr.Button("✨ Transliterate", variant="primary")
        clear_btn = gr.Button("🗑 Clear", variant="secondary")

    gr.Markdown("### 🔍 Results")
    with gr.Row():
        preprocessed_out = gr.Textbox(label="🔧 Pre‑processed (Normalized)", interactive=False, elem_classes="gr-box")
        raw_out = gr.Textbox(label="🤖 Model Raw Output", interactive=False, elem_classes="gr-box")
    final_out = gr.Textbox(label="✅ Final Repaired Output (English restored)", interactive=False,
                           elem_classes="gr-box", lines=2)

    # Preprocessing history (JSON)
    with gr.Accordion("📜 Preprocessing History", open=False):
        history_out = gr.JSON(label="Word‑level replacements")

    gr.Examples(
        examples=[
            "digital pranali lagu garne vanxa tara digital ma kharcha hunxa nepal ma afno servier xaina ani",
            "bank ma paisa halney ani mobile banking bata use garney ho tw",
            "ekdam few manxe haru matrw extra hunxan aru sab normal life nai bachirako huxa ni",
            "yesto fake news matra aauxw aajjkal facebook kholyo ki balen ko virodh matra haha"
        ],
        inputs=input_text,
        outputs=[preprocessed_out, raw_out, final_out, history_out],
        fn=transliterate,
        cache_examples=True,
        label="💡 Try these examples:"
    )

    submit_btn.click(
        fn=transliterate,
        inputs=input_text,
        outputs=[preprocessed_out, raw_out, final_out, history_out]
    )
    clear_btn.click(
        fn=lambda: ("", "", "", []),
        outputs=[preprocessed_out, raw_out, final_out, history_out]
    )

    gr.HTML("<div class='footer'>Built using Gradio & Hugging Face</div>")

demo.launch()