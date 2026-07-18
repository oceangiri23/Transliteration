import csv
import json
import random

# --- Load JSON clusters ---
with open("C:\\Users\\N I T R O V15\\OneDrive\\Documents\\Major Project\\Transliteration\\Final Datasets\\JSON_Final.json", "r", encoding="utf-8") as f:
    clusters = json.load(f)

# Build a lookup dictionary from all variants and base_words
word_map = {}
for entry in clusters.values():
    base_word = entry['base_word']
    variants = entry['variants']
    all_words = variants + [base_word]
    for w in all_words:
        word_map[w.lower()] = all_words  # case-insensitive

# --- Function to create augmented sentences ---
def augment_sentence(sentence, max_variations=3):
    words = sentence.strip().split()
    augmented_sentences = []
    
    # Add the original sentence as the first variation
    augmented_sentences.append(sentence)

    for _ in range(max_variations):
        new_words = []
        for word in words:
            # Check if lowercase word exists in clusters
            word_lower = word.lower()
            if word_lower in word_map:
                # Randomly choose to replace or not
                if random.random() < 0.7:  # 70% chance to replace
                    new_word = random.choice(word_map[word_lower])
                else:
                    new_word = word
            else:
                new_word = word
            new_words.append(new_word)
        augmented_sentences.append(" ".join(new_words))
    
    return augmented_sentences

# --- Read CSV and create augmented CSV ---
input_csv = r"C:\\Users\\N I T R O V15\\OneDrive\\Documents\\Major Project\\Transliteration\\Final Datasets\\NEW Parallel\\final_parallel_dedup.csv"
output_csv = r"C:\\Users\\N I T R O V15\\OneDrive\\Documents\\Major Project\\Transliteration\\Final Datasets\\NEW Parallel\\final_augmented_parallel.csv"

with open(input_csv, "r", encoding="utf-8") as infile, \
     open(output_csv, "w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        romanized_text = row['romanized']
        devanagari_text = row['devanagari']
        
        # Generate 2-3 augmented sentences (plus original)
        augmented_sentences = augment_sentence(romanized_text, max_variations=3)
        
        for sentence in augmented_sentences:
            writer.writerow({
                'romanized': sentence,
                'devanagari': devanagari_text
            })

print(f"Augmented CSV saved to {output_csv}")