
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import editdistance

CONFIG = {
    "csv_path": "/capstor/scratch/cscs/sagargiri/augmented_dataset.csv",
    "val_size": 1000,
    "max_input_len": 300,

    "checkpoint": "google/byt5-small",
    "checkpoint_model": "/users/sagargiri/byT5/fine_tuned_byt5_second.pt",
    "lr": 5e-5,

    "max_epochs": 1,
    "train_bs": 8,
    "val_bs": 8,
    "gradient_accumulation_steps": 2,
    "patience": 3,r
    "eval_steps": 1000,
    "grad_clip": 1.0,
    "warmup_steps": 500,
    "use_fp16": False,

    "max_gen_len": 512,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    df = pd.read_csv(CONFIG["csv_path"])
    df = df.sample(frac=1, random_state=27).reset_index(drop=True)
    df = df[["romanized", "devanaagari"]]
    df.columns = ["input", "label"]

    train_df = df[:-CONFIG["val_size"]].copy()
    val_df = df[-CONFIG["val_size"]:].copy()

    mask = train_df.input.str.len() > CONFIG["max_input_len"]
    val_df = pd.concat([val_df, train_df[mask]], ignore_index=True)
    train_df = train_df[~mask].reset_index(drop=True)

    return train_df, val_df

train_df, val_df = load_data()


class MyDataset(Dataset):
    def __init__(self, df):
        self.inputs = df["input"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input": self.inputs[idx], "label": self.labels[idx]}


tokenizer = AutoTokenizer.from_pretrained(CONFIG["checkpoint"])

def collate_fn(batch):
    inputs = [b["input"] for b in batch]
    labels = [b["label"] for b in batch]

    model_inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=CONFIG["max_input_len"],
        return_tensors="pt",
    )

    labels_tok = tokenizer(
        text_target=labels,   # <-- THIS is the new API
        padding=True,
        truncation=True,
        max_length=CONFIG["max_gen_len"],
        return_tensors="pt",
    )

    model_inputs["labels"] = labels_tok["input_ids"]

    # Important: mask padding tokens for loss
    model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100

    return model_inputs



model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["checkpoint"])
state_dict = torch.load(CONFIG["checkpoint_model"], map_location=device)
model.load_state_dict(state_dict)
model.to(device)
print("✓ Loaded fine-tuned weights")

def cer(pred, truth):
    return editdistance.eval(pred, truth) / max(len(truth), 1)

def wer(pred, truth):
    pred_words = pred.split()
    truth_words = truth.split()
    return editdistance.eval(pred_words, truth_words) / max(len(truth_words), 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_char_err = 0
    total_chars = 0
    total_word_err = 0
    total_words = 0


    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        total_loss += loss.item()

        preds = model.generate(batch["input_ids"], max_length=CONFIG["max_gen_len"])
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for p, t in zip(preds, targets):
            # CER
            total_char_err += editdistance.eval(p, t)
            total_chars += max(len(t), 1)

            # WER
            p_words = p.split()
            t_words = t.split()
            total_word_err += editdistance.eval(p_words, t_words)
            total_words += max(len(t_words), 1)


    model.train()
    cer = total_char_err / total_chars
    wer = total_word_err / total_words
    return total_loss / len(loader), cer, wer



train_loader = DataLoader(
    MyDataset(train_df),
    batch_size=CONFIG["train_bs"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=16,
    pin_memory=True,
)

val_loader = DataLoader(
    MyDataset(val_df),
    batch_size=CONFIG["val_bs"],
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=16,
    pin_memory=True,
)


optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

def lr_lambda(step):
    if step < CONFIG["warmup_steps"]:
        return max(0.01, step / CONFIG["warmup_steps"])
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda")


best_loss = float("inf")
patience_ctr = 0
global_step = 0

for epoch in range(CONFIG["max_epochs"]):
    model.train()
    running_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        if CONFIG["use_fp16"]:
            with torch.amp.autocast("cuda"):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        global_step += 1

        if global_step % CONFIG["eval_steps"] == 0:
            val_loss, val_cer, val_wer = evaluate(model, val_loader)

            print(
                f"\nStep {global_step} | "
                f"Train Loss: {running_loss / CONFIG['eval_steps']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"CER: {val_cer:.4f} | "
                f"WER: {val_wer:.4f}"
            )


            running_loss = 0

            if val_loss < best_loss:
                best_loss = val_loss
                patience_ctr = 0
                torch.save(model.state_dict(),
                           "fine_tuned_byt5_.pt")
                print("✓ Saved best model")
            else:
                patience_ctr += 1

            if patience_ctr > CONFIG["patience"]:
                print("Early stopping triggered.")
                break

print("Training complete.")