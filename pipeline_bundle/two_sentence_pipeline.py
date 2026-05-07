"""Two-sentence romanized-to-Devanagari repair pipeline.

This module is a standalone version of the notebook experiments, trimmed to the
final architecture:
- no fastText
- no corpus loading
- one romanized sentence + one model-output Devanagari sentence
- refined English-word detection
- phonetic matching with a symmetric window
- length compatibility gating
- marker attachment for short clitics such as ma/ko/le

Example:

    python keep_english/two_sentence_pipeline.py \
        --roman "aaja market ma price ekdam high rahechha" \
        --deva  "आज मार्केट मा प्राइस एकदम हाइ रहेको छ"
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import exp
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
ADS_DIR = BASE_DIR
REFINED_VOCAB_PATH = BASE_DIR / "refined_english_words.txt"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ADS_DIR) not in sys.path:
    sys.path.insert(0, str(ADS_DIR))

from rule import english_to_devanagari  # noqa: E402

try:
    from nepPhoneticEncoder import nepali_dmetaphone  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback only
    def nepali_dmetaphone(source: str):
        return (source, "")


TOKEN_RE = re.compile(r"[^A-Za-z\u0900-\u097F']+")
MARKER_DEVA = {
    "ma": "मा",
    "maa": "मा",
    "le": "ले",
    "ko": "को",
    "lai": "लाई",
    "sanga": "सँग",
    "bata": "बाट",
    "pachi": "पछि",
    "samma": "सम्म",
    "bhitra": "भित्र",
    "ta": "ता",
    "nai": "नै",
}
MARKER_TOKENS = set(MARKER_DEVA)
ATTACHED_MARKER_SUFFIXES = tuple(sorted(MARKER_DEVA.items(), key=lambda item: len(item[1]), reverse=True))

EXACT_TRANSLITERATION_CANDIDATES = {
    "new": ["न्यु", "न्यू"],
    "job": ["जब", "जॉब"],
    "office": ["अफिस", "ऑफिस"],
    "phone": ["फोन"],
    "movie": ["मुभी", "मूभी"],
    "video": ["भिडियो", "विडियो"],
    "friend": ["फ्रेन्ड", "फ्रेंड"],
}

PHONETIC_MIN_SCORE = 0.35
POSITIONAL_PENALTY = 0.12
SECOND_PASS_MIN_SCORE = 0.10
LENGTH_RATIO_MIN = 0.50
LENGTH_RATIO_MAX = 2.25


@dataclass(frozen=True)
class Replacement:
    roman_word: str
    target_word: str
    score: float
    target_index: int
    kind: str = "english"


def clean_token(token: str) -> str:
    return TOKEN_RE.sub("", str(token).strip())


def load_english_vocab(vocab_path: Path = REFINED_VOCAB_PATH) -> set[str]:
    vocab: set[str] = set()
    if not vocab_path.exists():
        raise FileNotFoundError(f"English vocab file not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            word = clean_token(line).lower()
            if word:
                vocab.add(word)
    return vocab


def word_to_phonetic(word: str) -> str:
    if not word:
        return ""
    primary, alternate = nepali_dmetaphone(str(word).strip())
    return primary or alternate or str(word)[:6]


def phonetic_similarity(code1: str, code2: str) -> float:
    if not code1 or not code2:
        return 0.0
    return float(SequenceMatcher(None, code1, code2).ratio())


def is_marker_token(token: str) -> bool:
    return clean_token(token).lower() in MARKER_TOKENS


def marker_to_devanagari(token: str) -> str:
    return MARKER_DEVA.get(clean_token(token).lower(), token)


def attached_marker_token(target_word: str) -> str | None:
    """Return the roman marker token if a Devanagari target ends with it."""
    cleaned = clean_token(target_word)
    for roman_marker, deva_marker in ATTACHED_MARKER_SUFFIXES:
        if cleaned.endswith(deva_marker) and len(cleaned) > len(deva_marker):
            return roman_marker
    return None


def length_compatible(source_word: str, target_word: str, min_ratio: float = LENGTH_RATIO_MIN, max_ratio: float = LENGTH_RATIO_MAX) -> bool:
    source_len = max(1, len(clean_token(source_word)))
    target_len = max(1, len(clean_token(target_word)))
    ratio = target_len / source_len
    return min_ratio <= ratio <= max_ratio


def expected_position_score(pos: int, cand_pos: int, len_in: int, len_out: int, window_type: str) -> float:
    if len_out <= 1:
        return 1.0
    if window_type == "forward":
        distance = max(0, cand_pos - pos)
    else:
        distance = abs(cand_pos - pos)
    scale = max(1, abs(len_in - len_out) + 1)
    return exp(-distance / scale)


def get_window_bounds(window_type: str, pos: int, len_in: int, len_out: int, diff: int) -> tuple[int, int]:
    if len_out <= 0:
        return 0, -1
    if window_type == "same":
        return pos, pos
    if window_type in {"symmetric", "clipped"}:
        return max(0, pos - diff), min(len_out - 1, pos + diff)
    if window_type == "forward":
        return pos, min(len_out - 1, pos + diff)
    return max(0, pos - diff), min(len_out - 1, pos + diff)


def build_clean_positions(raw_tokens: list[str]) -> tuple[list[str], list[int]]:
    clean_tokens: list[str] = []
    clean_to_raw: list[int] = []
    for raw_index, token in enumerate(raw_tokens):
        cleaned = clean_token(token)
        if cleaned:
            clean_tokens.append(cleaned)
            clean_to_raw.append(raw_index)
    return clean_tokens, clean_to_raw


def best_target_match_for_english_word(
    english_word: str,
    target_tokens: list[str],
    target_start_idx: int,
    pos: int,
    len_in: int,
    len_out: int,
    window_type: str = "symmetric",
) -> tuple[str, str, str | None, str | None, float, float]:
    """Return the best target token for one English word.

    The scoring is phonetic-first with a positional penalty. A short exact-candidate
    second pass is used for very common loanwords.
    """

    rule_deva = english_to_devanagari(english_word)
    rule_phonetic = word_to_phonetic(rule_deva)

    best_word = None
    best_target_phonetic = None
    best_score = -1.0
    best_raw_phon = -1.0

    for local_index, token in enumerate(target_tokens):
        token_clean = clean_token(token)
        if not token_clean:
            continue
        if not length_compatible(english_word, token_clean):
            continue

        token_phonetic = word_to_phonetic(token_clean)
        raw_score = phonetic_similarity(rule_phonetic, token_phonetic)
        candidate_pos = target_start_idx + local_index
        proximity = expected_position_score(pos, candidate_pos, len_in, len_out, window_type)
        score = (1.0 - POSITIONAL_PENALTY) * raw_score + POSITIONAL_PENALTY * proximity

        if score > best_score:
            best_score = score
            best_raw_phon = raw_score
            best_word = token_clean
            best_target_phonetic = token_phonetic

    exact_candidates = EXACT_TRANSLITERATION_CANDIDATES.get(english_word.lower(), [])
    if exact_candidates:
        for local_index, token in enumerate(target_tokens):
            token_clean = clean_token(token)
            if token_clean not in exact_candidates:
                continue
            token_phonetic = word_to_phonetic(token_clean)
            raw_score = phonetic_similarity(rule_phonetic, token_phonetic)
            candidate_pos = target_start_idx + local_index
            proximity = expected_position_score(pos, candidate_pos, len_in, len_out, window_type)
            score = max(best_score, raw_score + 0.05 * proximity)
            if score >= SECOND_PASS_MIN_SCORE and score > best_score:
                best_score = score
                best_raw_phon = raw_score
                best_word = token_clean
                best_target_phonetic = token_phonetic

    return rule_deva, rule_phonetic, best_word, best_target_phonetic, float(best_score), float(best_raw_phon)


def reconstruct_sentence_pair(
    roman_sentence: str,
    deva_sentence: str,
    english_vocab: set[str],
    window_type: str = "symmetric",
    phonetic_threshold: float = PHONETIC_MIN_SCORE,
) -> tuple[str, list[Replacement]]:
    roman_raw = roman_sentence.split()
    deva_raw = deva_sentence.split()

    roman_tokens = [clean_token(token).lower() for token in roman_raw]
    roman_tokens = [token for token in roman_tokens if token]

    deva_tokens, deva_clean_to_raw = build_clean_positions(deva_raw)
    if not deva_tokens:
        return deva_sentence, []

    len_in = len(roman_tokens)
    len_out = len(deva_tokens)

    replacements_map: dict[int, str] = {}
    used_target_clean_indices: set[int] = set()
    replacements: list[Replacement] = []

    for roman_index, roman_word in enumerate(roman_tokens):
        if roman_word not in english_vocab:
            continue
        if roman_word in MARKER_TOKENS:
            continue

        if len_in == len_out and roman_index < len_out:
            start = roman_index
            end = roman_index
        else:
            diff = abs(len_in - len_out)
            start, end = get_window_bounds(window_type, roman_index, len_in, len_out, diff)

        if start > end or start >= len_out:
            continue

        window_tokens = [token for token in deva_tokens[start : end + 1]]
        if not window_tokens:
            continue

        local_tokens = [token for token in window_tokens if token]
        if not local_tokens:
            continue

        _, _, best_match, _, score, raw_score = best_target_match_for_english_word(
            roman_word,
            local_tokens,
            start,
            roman_index,
            len_in,
            len_out,
            window_type=window_type,
        )

        if best_match is None:
            continue
        if raw_score < phonetic_threshold and score < phonetic_threshold:
            continue

        try:
            match_clean_index = deva_tokens.index(best_match)
        except ValueError:
            continue
        if match_clean_index in used_target_clean_indices:
            continue

        raw_target_index = deva_clean_to_raw[match_clean_index]
        used_target_clean_indices.add(match_clean_index)

        attached_marker = attached_marker_token(best_match)
        next_roman_index = roman_index + 1
        if attached_marker and next_roman_index < len(roman_tokens) and roman_tokens[next_roman_index] == attached_marker:
            marker_deva = marker_to_devanagari(attached_marker)
            replacements_map[raw_target_index] = f"{roman_word} {marker_deva}"
            replacements.append(Replacement(roman_word, f"{best_match} + {marker_deva}", float(score), raw_target_index, kind="english+marker"))
            continue

        replacements_map[raw_target_index] = roman_word
        replacements.append(Replacement(roman_word, best_match, float(score), raw_target_index))

        if next_roman_index < len(roman_tokens):
            next_roman_word = roman_tokens[next_roman_index]
            if next_roman_word in MARKER_TOKENS:
                next_target_index = match_clean_index + 1
                if next_target_index < len(deva_tokens) and next_target_index not in used_target_clean_indices:
                    expected_marker = marker_to_devanagari(next_roman_word)
                    actual_marker = deva_tokens[next_target_index]
                    if actual_marker == expected_marker:
                        replacements_map[deva_clean_to_raw[next_target_index]] = expected_marker
                        used_target_clean_indices.add(next_target_index)
                        replacements.append(Replacement(next_roman_word, actual_marker, 0.0, deva_clean_to_raw[next_target_index], kind="marker"))

    reconstructed_tokens = [replacements_map.get(index, token) for index, token in enumerate(deva_raw)]
    return " ".join(reconstructed_tokens), replacements


def format_replacements(replacements: Iterable[Replacement]) -> str:
    lines = []
    for item in replacements:
        lines.append(f"- {item.roman_word} -> {item.target_word} (score={item.score:.4f}, index={item.target_index}, kind={item.kind})")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair English words in a model-output Devanagari sentence.")
    parser.add_argument("--roman", required=True, help="Romanized input sentence.")
    parser.add_argument("--deva", required=True, help="Devanagari model-output sentence.")
    parser.add_argument("--window-type", default="symmetric", choices=["same", "symmetric", "forward", "clipped"], help="Window strategy for uneven sentence lengths.")
    parser.add_argument("--threshold", type=float, default=PHONETIC_MIN_SCORE, help="Minimum score required to accept a match.")
    parser.add_argument("--vocab-path", type=Path, default=REFINED_VOCAB_PATH, help="Path to the refined English vocabulary file.")
    parser.add_argument("--show-replacements", action="store_true", help="Print the replacement log.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    english_vocab = load_english_vocab(args.vocab_path)
    reconstructed, replacements = reconstruct_sentence_pair(
        args.roman,
        args.deva,
        english_vocab=english_vocab,
        window_type=args.window_type,
        phonetic_threshold=args.threshold,
    )

    print("ROMANIZED   :", args.roman)
    print("DEVA        :", args.deva)
    print("RECONSTRUCT :", reconstructed)
    print("REPLACEMENTS:", len(replacements))
    if args.show_replacements and replacements:
        print(format_replacements(replacements))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())