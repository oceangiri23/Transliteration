# Two-Sentence Pipeline Bundle

This folder is the minimal bundle needed to run the romanized-to-Devanagari repair pipeline.

## Files

- `two_sentence_pipeline.py` - main CLI and importable pipeline
- `rule.py` - romanized Nepali to Devanagari transliterator
- `nepPhoneticEncoder.py` - phonetic encoder used by the matcher
- `refined_english_words.txt` - English vocabulary used for English-word detection

## Run

Use the bundled Python environment or any Python 3.12+ environment that can run the script:

```bash
python two_sentence_pipeline.py \
  --roman "mero new job ko training suru hunchha" \
  --deva "मेरो न्यु जबको ट्रेनिङ सुरु हुन्छ" \
  --show-replacements
```

## What it does

The script takes one romanized input sentence and one Devanagari model output sentence, then:

1. detects English words using the bundled vocabulary
2. scores candidate output tokens with phonetic matching and a position window
3. preserves markers like `ko`, `ma`, and `le`, including attached forms such as `जबको`
4. prints the reconstructed sentence and the replacement log

## Importing from Python

```python
from two_sentence_pipeline import load_english_vocab, reconstruct_sentence_pair

vocab = load_english_vocab()
output, replacements = reconstruct_sentence_pair(
    "mero new job ko training suru hunchha",
    "मेरो न्यु जबको ट्रेनिङ सुरु हुन्छ",
    english_vocab=vocab,
)
print(output)
```
