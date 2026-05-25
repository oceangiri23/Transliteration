import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]+')


def extract_words_from_file(file_path):
    """
    Extract unique Nepali words from a single file
    """
    local_words = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                words = DEVANAGARI_PATTERN.findall(line)

                for w in words:
                    if len(w) > 1:
                        local_words.add(w)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return local_words


def get_all_txt_files(root_dir):
    """
    Recursively get all .txt files
    """
    return [str(p) for p in Path(root_dir).rglob("*.txt")]


def build_dictionary(root_dir, output_file, num_workers=4):
    """
    Main pipeline to build Nepali dictionary
    """
    print(" Scanning files...")
    files = get_all_txt_files(root_dir)
    print(f" Found {len(files)} files")

    global_words = set()

    print(" Processing files in parallel...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(extract_words_from_file, files, chunksize=10)

        # tqdm progress bar
        for word_set in tqdm(results, total=len(files), desc="Processing files"):
            global_words.update(word_set)

    print(f" Total unique words: {len(global_words)}")

    print(" Saving dictionary...")

    sorted_words = sorted(global_words)

    with open(output_file, "w", encoding="utf-8") as f:
        for word in tqdm(sorted_words, desc="Writing to file"):
            f.write(word + "\n")

    print(f" Dictionary saved at: {output_file}")


if __name__ == "__main__":
    root_directory = "/capstor/scratch/cscs/sagargiri/Dataset"
    output_dictionary = "/capstor/store/cscs/swissai/a168/sagargiri/Dictionary/nepali_dictionary.txt"

    build_dictionary(
        root_dir=root_directory,
        output_file=output_dictionary,
        num_workers=200  
    )
