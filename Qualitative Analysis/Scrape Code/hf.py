# !pip install -q gradio_client tqdm

from gradio_client import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# HuggingFace Space client
client = Client("Sagar32/Romanized-To-Devanagari-Transliteration")
# Input text file (one sentence per line)
INPUT_FILE = "romanized_sampled.txt"
# Output text file
OUTPUT_FILE = "6_byt5_normalized.txt"
# Number of parallel workers
MAX_WORKERS = 5

# Read sentences
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

def transliterate(sentence):
    try:
        result = client.predict(
            raw_input=sentence,
            api_name="/transliterate"
        )
        # Model Raw Output
        return result[1]
    except Exception as e:
        print(f"Error processing: {sentence}")
        print(e)
        return ""

outputs = [None] * len(sentences)
# Parallel processing
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_index = {
        executor.submit(transliterate, sentence): idx
        for idx, sentence in enumerate(sentences)
    }
    for future in tqdm(as_completed(future_to_index), total=len(sentences)):
        idx = future_to_index[future]
        try:
            outputs[idx] = future.result()
        except Exception as e:
            print(f"Failed at index {idx}")
            print(e)
            outputs[idx] = ""

# Save outputs line-by-line
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in outputs:
        f.write(line + "\n")

print("Done.")