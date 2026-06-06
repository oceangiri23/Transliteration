import re
import string
import emoji
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import AgglomerativeClustering
import phonetics
import nltk
from nltk.corpus import words
from nepPhoneticEncoder import nepali_dmetaphone
import jellyfish
from collections import defaultdict
import json

def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def clean_text(text):
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    # Lowercase
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_unique_words(text):
    return list(set(text.split()))

def get_phonetic_codes(filtered_list):
    phonetic_map = {}
    for w in filtered_list:
        codes,a  =  nepali_dmetaphone(w)
        full_code = ''.join(c for c in codes if c is not None)
        if full_code:
            phonetic_map[w] = full_code
    return phonetic_map

def group_by_phonetic_code(phonetic_map):
    grouped_dict = defaultdict(list)
    for word, code in phonetic_map.items():
        grouped_dict[code].append(word)
    return dict(grouped_dict)




file_path = "../data_text_files/all_merged.txt"  
text = load_corpus(file_path)
clean = clean_text(text)
word_list = get_unique_words(clean)
english_vocab = set(words.words())
filtered_list = [w for w in word_list if w.lower() not in english_vocab]

phonetic_map = get_phonetic_codes(filtered_list)
grouped_dict = group_by_phonetic_code(phonetic_map)

nested_dict = {}
for code, words in grouped_dict.items():
    nested_dict[code] = {
        "base_word": ".....", 
        "variants": words
    }

with open("Fresh_Complete_phonetic_Strcuctured_group.json", "w", encoding="utf-8") as f:
    json.dump(nested_dict, f, ensure_ascii=False, indent=2)

print(" Saved nested phonetic dictionary to 'Complete_phonetic_group.json'")
