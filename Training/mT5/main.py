import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from nltk.metrics.distance import edit_distance
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein

torch.cuda.empty_cache()

def loadData(val_ratio: float = 0.15, max_len: int = 300, seed: int = 27):
    df = pd.read_csv("../Clean_corpus_dataset.csv")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df[["romanized", "devanagari"]]
    df.columns = ["input", "label"]
    n_val = int(len(df) * val_ratio)
    train_df = df[:-n_val].copy()
    val_df = df[-n_val:].copy()
    long_mask = train_df["input"].str.len() > max_len
    val_df = pd.concat([val_df, train_df[long_mask]], ignore_index=True)
    train_df = train_df[~long_mask].reset_index(drop=True)

    return train_df, val_df



train_df, val_df = loadData()
print("The length of train dataset and valdataset  is: ", len(train_df), len(val_df))

class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row.input, row.label

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        inputs = [input for input, _ in batch]
        labels = [label for _, label in batch]
        encodings = self.tokenizer(inputs, padding="longest", return_tensors="pt")
        label_encodings = self.tokenizer(labels, padding="longest", return_tensors="pt")

        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        labels = label_encodings.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_init(checkpoint):
    model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
    return model.to(device)

def dict_to_cuda(batch):
    return {k: v.to(device) for k, v in batch.items()}


checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = model_init(checkpoint)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train_ds = MyDataset(train_df)
val_ds = MyDataset(val_df)
data_collator = DataCollator(tokenizer)
train_loader = DataLoader(train_ds, batch_size=2, collate_fn=data_collator)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=data_collator)

max_epochs = 3
best = float("inf")
trigger = 0

@torch.no_grad()
def eval_model(model, val_loader, tokenizer):
    model.eval()
    total = 0
    count = 0
    stream = tqdm(val_loader)
    for batch in stream:
        batch = dict_to_cuda(batch)
        loss = model(**batch).loss
        total += loss.item()
        count += 1
        stream.set_description_str(desc=f"val_edit_dist {total / count:.3f}")

    model.train()
    return total / count

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
    return decoded

def evaluate(df_val):
    total_chars = 0
    total_char_errors = 0
    total_words = 0
    total_word_correct = 0
    bleu_scores = []

    smooth = SmoothingFunction().method1

    for idx, row in df_val.iterrows():
        pred = transliterate(row['input'])
        true = row['label']

        # --- Character Error Rate (CER) ---
        char_errors = Levenshtein.distance(pred, true)
        total_char_errors += char_errors
        total_chars += len(true)

        # --- Word Accuracy ---
        pred_words = pred.split()
        true_words = true.split()
        total_words += len(true_words)
        # Count exact word matches
        correct_words = sum(p==t for p,t in zip(pred_words,true_words))
        total_word_correct += correct_words

        # --- BLEU-4 (character level) ---
        # split characters for BLEU
        pred_chars = list(pred)
        true_chars = list(true)
        bleu = sentence_bleu([true_chars], pred_chars, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
        bleu_scores.append(bleu)

    cer = total_char_errors / total_chars
    word_accuracy = total_word_correct / total_words
    avg_bleu4 = sum(bleu_scores) / len(bleu_scores)

    print(f"CER: {cer:.4f}")
    print(f"Word Accuracy: {word_accuracy:.4f}")
    print(f"Average BLEU-4: {avg_bleu4:.4f}")

    return cer, word_accuracy, avg_bleu4


for epoch in range(max_epochs):
    stream = tqdm(train_loader)
    total = 0
    count = 0
    for batch in stream:
        batch = dict_to_cuda(batch)
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        count += 1
        stream.set_description_str(
            f"Epoch {epoch}/{max_epochs} | Loss {total / count : .3f}"
        )
    loss = total / count
    eval_metric = eval_model(model, val_loader, tokenizer)
    print(
        f"Epoch {epoch}/{max_epochs} | Loss {loss: .3f} | Val Loss {eval_metric: .4f}"
    )
    
    model.eval()
    evaluate(val_df)
    to_save = {"model": model.state_dict(), "optimizer": optimizer}
    name = f"mt5_all_augmented_{epoch}.pt"
    torch.save(to_save, name)

