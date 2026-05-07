import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import gradio as gr
from huggingface_hub import hf_hub_download
import os 
from huggingface_hub import hf_hub_download
from pipeline_bundle.two_sentence_pipeline import repair_transliteration
hf_token = os.getenv("HF_TOKEN")

device = torch.device("cpu")

def model_init(checkpoint):
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    return model.to(device)

checkpoint = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = model_init(checkpoint)

model_path = hf_hub_download(
    repo_id="Sagar32/romanizedtransliterationmodel",  # 🔹 your repo name here
    filename="fine_tuned_byt5.pt",
    token= hf_token
)

state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
model.load_state_dict(state_dict["model"], strict=False)
model.eval()

#model.load_state_dict(torch.load('fine_tuned_byt5.pt',weights_only=False)['model'])
#model.to(device)
#model.eval()

def transliterate(sentence: str, max_length: int = 128):
    
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    
    outputs = model.generate(
        input_ids=inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_beams=5,          
        early_stopping=True
    )
    
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    repaired, _ = repair_transliteration(sentence, decoded)
    return repaired


iface = gr.Interface(
    fn=transliterate,
    inputs=gr.Textbox(lines=2, placeholder="Enter Romanized Nepali sentence..."),
    outputs=gr.Textbox(label="Devanagari Transliteration"),
    title="Nepali Romanized → Devanagari Transliterator",
    description="A fine-tuned ByT5 model that transliterates Romanized Nepali text into Devanagari script."
)

# Launch the app
iface.launch()