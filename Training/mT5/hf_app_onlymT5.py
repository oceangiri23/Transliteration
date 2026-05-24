import torch
import gradio as gr
from transformers import AutoTokenizer, T5ForConditionalGeneration
from huggingface_hub import hf_hub_download
import os


torch.set_num_threads(1)
device = torch.device("cpu")

hf_token = os.getenv("HF_TOKEN")

checkpoint = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32, 
).to(device)


model_path = hf_hub_download(
    repo_id="Sagar32/romanizedtransliterationmodel",
    filename="mt5_all_augmented_8.pt",
    token=hf_token
)

state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
model.load_state_dict(state_dict["model"], strict=False)

model.eval()

def transliterate(sentence):

    if not sentence.strip():
        return "Please enter text."

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.inference_mode():  
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=128,
            num_beams=1,  
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


with gr.Blocks() as demo:
    gr.Markdown("## Nepali Romanized → Devanagari Transliterator (mT5)")
    gr.Markdown("Fine-tuned mT5 model for Romanized Nepali transliteration.")

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