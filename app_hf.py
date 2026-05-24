import json
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, T5ForConditionalGeneration
from huggingface_hub import hf_hub_download
import os


torch.set_num_threads(1)
device = torch.device("cpu")

hf_token = os.getenv("HF_TOKEN")

# ---- Load ByT5 only ----
byt5_checkpoint = "google/byt5-small"

byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_checkpoint)
byt5_model = T5ForConditionalGeneration.from_pretrained(byt5_checkpoint).to(device)

# ---- Download fine-tuned checkpoint ----
byt5_model_path = hf_hub_download(
    repo_id="Sagar32/romanizedtransliterationmodel",
    filename="augmented_dataset_byT5_epoch_8.pt",
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

byt5_state = torch.load(byt5_model_path, map_location="cpu")
#byt5_model.load_state_dict(byt5_state["model"], strict=False)
byt5_model.load_state_dict(byt5_state, strict=False)
byt5_model.eval()


def transliterate(sentence):
    if not sentence.strip():
        return "Please enter text."

    inputs = byt5_tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.inference_mode():
        outputs = byt5_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=128,
            num_beams=2,
            early_stopping=True
        )

    decoded = byt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

    submit_btn.click(
        fn=transliterate,
        inputs=input_text,
        outputs=[preprocessed_out, raw_out, final_out, history_out]
    )
    clear_btn.click(
        fn=lambda: ("", "", "", []),
        outputs=[preprocessed_out, raw_out, final_out, history_out]
    )

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("## Nepali Romanized → Devanagari Transliterator")
    gr.Markdown("Powered by fine-tuned **ByT5** model")

    input_text = gr.Textbox(
        lines=2,
        placeholder="Enter Romanized Nepali sentence..."
    )

    output_text = gr.Textbox(label="Devanagari Transliteration")

    submit_btn = gr.Button("Transliterate")

    submit_btn.click(
        transliterate,
        inputs=input_text,
        outputs=output_text
    )

demo.launch()