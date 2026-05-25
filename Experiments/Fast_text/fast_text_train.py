import fasttext
import re
import os

DATA_DIR = "/capstor/scratch/cscs/sagargiri/Dataset/source1" 

def preprocess(text):

    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    return text.lower().strip()

def build_corpus(data_dir, output_file="/capstor/scratch/cscs/sagargiri/Dataset/corpus.txt"):
    with open(output_file, "w", encoding="utf-8") as out_f:

        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                print(f"Processing: {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = preprocess(line)
                        if line:
                            out_f.write(line + "\n")

    print("Corpus built successfully!")

print('Building Corpus')
build_corpus(DATA_DIR)
print("Building corpus complete.")

print("starting to train the fasttext")
# Step 2: Train FastText model
model = fasttext.train_unsupervised(
    "/capstor/scratch/cscs/sagargiri/Dataset/corpus.txt",
    model="skipgram",   
    dim=100,    
    epoch=10,   
    lr=0.05,   
    wordNgrams=3,   
    minn=3,   
    maxn=6,    
    thread=200,
)

print('training complete')
model.save_model("/capstor/store/cscs/swissai/a168/sagargiri/Fast_text/fasttext_devanagari.bin")

model = fasttext.load_model("devanagari.bin")

def similar_words(word, k=5):
    neighbors = model.get_nearest_neighbors(word, k=k)
    print(f"\nTop {k} words similar to '{word}':")
    for score, w in neighbors:
        print(f"  {w:20s}  →  {score:.4f}")

# Example
similar_words("प्रशिक्षण")