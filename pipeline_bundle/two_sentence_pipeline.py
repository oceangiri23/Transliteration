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

    python pipeline_bundle/two_sentence_pipeline.py \
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
from functools import lru_cache
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


TOKEN_RE = re.compile(r"[^A-Za-z0-9\u0900-\u097F०-९']+")
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

# Tail words/postpositions that attach to previous words (skip when matching)
TAIL_WORDS = {
    'ma', 'maa', 'ko', 'lai', 'le', 'bata', 'sanga', 'atha', 'pani', 'ta', 'nai', 'ra', 'tah',
    'मा', 'को', 'लाई', 'ले', 'बाट', 'सँग', 'अथ', 'पनि', 'त', 'नै', 'र', 'तह',
}

EXACT_TRANSLITERATION_CANDIDATES = {
    "new": ["न्यु", "न्यू"],
    "job": ["जब", "जॉब"],
    "office": ["अफिस", "ऑफिस"],
    "phone": ["फोन"],
    "movie": ["मुभी", "मूभी"],
    "video": ["भिडियो", "विडियो"],
    "friend": ["फ्रेन्ड", "फ्रेंड"],
}

PHONETIC_MIN_SCORE = 0.30
POSITIONAL_PENALTY = 0.12
SECOND_PASS_MIN_SCORE = 0.10
LENGTH_RATIO_MIN = 0.50
LENGTH_RATIO_MAX = 2.25
CONTEXT_WEIGHT = 0.10

ARABIC_TO_DEVANAGARI_NUMERAL = str.maketrans(
    '0123456789',
    '०१२३४५६७८९'
)


@dataclass(frozen=True)
class Replacement:
    roman_word: str
    target_word: str
    score: float
    target_index: int
    kind: str = "english"
    context_score: float = 0.0


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


def context_match_score(
    roman_tokens: list[str],
    deva_tokens: list[str],
    roman_index: int,
    candidate_deva_index: int,
    context_window: int = 1,
) -> float:
    """Score how well surrounding tokens match at this position."""
    context_scores: list[float] = []
    
    # Check tokens before
    for offset in range(1, context_window + 1):
        roman_neighbor_idx = roman_index - offset
        deva_neighbor_idx = candidate_deva_index - offset
        
        if roman_neighbor_idx >= 0 and deva_neighbor_idx >= 0:
            if roman_neighbor_idx < len(roman_tokens) and deva_neighbor_idx < len(deva_tokens):
                roman_neighbor = roman_tokens[roman_neighbor_idx]
                deva_neighbor = deva_tokens[deva_neighbor_idx]
                
                if roman_neighbor and deva_neighbor:
                    roman_phon = word_to_phonetic(roman_neighbor)
                    deva_phon = word_to_phonetic(deva_neighbor)
                    score = phonetic_similarity(roman_phon, deva_phon)
                    context_scores.append(score)
    
    # Check tokens after
    for offset in range(1, context_window + 1):
        roman_neighbor_idx = roman_index + offset
        deva_neighbor_idx = candidate_deva_index + offset
        
        if roman_neighbor_idx < len(roman_tokens) and deva_neighbor_idx < len(deva_tokens):
            roman_neighbor = roman_tokens[roman_neighbor_idx]
            deva_neighbor = deva_tokens[deva_neighbor_idx]
            
            if roman_neighbor and deva_neighbor:
                roman_phon = word_to_phonetic(roman_neighbor)
                deva_phon = word_to_phonetic(deva_neighbor)
                score = phonetic_similarity(roman_phon, deva_phon)
                context_scores.append(score)
    
    # Return average context score, or 1.0 if no context available
    if context_scores:
        return sum(context_scores) / len(context_scores)
    return 1.0


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


def calculate_length_score(source_word: str, target_word: str) -> float:
    """Calculate length compatibility score (0 to 1).
    
    Returns 1.0 if lengths are perfectly compatible, lower for mismatches.
    """
    source_len = max(1, len(clean_token(source_word)))
    target_len = max(1, len(clean_token(target_word)))
    ratio = target_len / source_len
    
    if LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
        center = 1.0
        max_deviation = max(1.0 - LENGTH_RATIO_MIN, LENGTH_RATIO_MAX - 1.0)
        deviation = abs(ratio - center)
        length_score = max(0.0, 1.0 - (deviation / max_deviation))
        return length_score
    else:
        return 0.0


def element_score(roman_word: str, deva_word: str) -> float:
    """Score a single element pair combining phonetic and length.
    
    Converts romanized token to Devanagari first, then compares
    Devanagari phonetics against model output Devanagari.
    Returns 0.7 * phonetic_similarity + 0.3 * length_compatibility.
    """
    if not roman_word or not deva_word:
        return 0.0
    
    # Convert romanized word to Devanagari first
    roman_as_deva = english_to_devanagari(roman_word)
    
    # Compare Devanagari vs Devanagari phonetically
    phon_score = phonetic_similarity(
        word_to_phonetic(roman_as_deva),
        word_to_phonetic(deva_word)
    )
    len_score = calculate_length_score(roman_word, deva_word)
    
    return (0.7 * phon_score) + (0.3 * len_score)


def window_score(roman_window: list[str], deva_window: list[str], middle_weight: float = 0.333) -> float:
    """Score an entire 3-token window with equal importance to all elements.
    
    All tokens (prev, english, next) get equal weight (1/3 each).
    This ensures context and target word are equally important.
    
    Returns weighted average score of element pairs.
    """
    if len(roman_window) != len(deva_window) or len(roman_window) == 0:
        return 0.0
    
    scores = []
    
    for r_tok, d_tok in zip(roman_window, deva_window):
        score = element_score(r_tok, d_tok)
        scores.append(score)
    
    # Equal weights for all elements
    return sum(scores) / len(scores)


def is_devanagari_number(token: str) -> bool:
    """Check if token contains Devanagari numerals."""
    devanagari_numerals = '०१२३४५६७८९'
    return any(c in token for c in devanagari_numerals)


def is_roman_number(token: str) -> bool:
    """Check if token is purely Arabic numerals."""
    return token.isdigit()


def find_best_window_position(
    english_word: str,
    roman_tokens: list[str],
    deva_tokens: list[str],
    roman_index: int,
) -> tuple[str | None, float]:
    """Find best window position by sliding 3-token windows.
    
    Builds a roman window around the English word and slides it through
    the Devanagari tokens. Scores each position and returns the best match.
    
    If the next token is a tail word (like 'ma', 'ko'), uses the token
    after it instead to account for attachment in Devanagari output.
    
    Returns (best_target_token, score)
    """
    # Build roman window: [prev, english, next]
    # If next is a tail word, skip it and use the one after instead
    roman_window = []
    
    if roman_index > 0:
        roman_window.append(roman_tokens[roman_index - 1])
    
    roman_window.append(english_word)
    english_pos_in_window = len(roman_window) - 1
    
    # Add next token, but skip tail words
    next_idx = roman_index + 1
    if next_idx < len(roman_tokens):
        next_token = roman_tokens[next_idx]
        # If next is a tail word, try the one after it
        if next_token.lower() in TAIL_WORDS and next_idx + 1 < len(roman_tokens):
            roman_window.append(roman_tokens[next_idx + 1])
        else:
            roman_window.append(next_token)
    
    best_score = -1.0
    best_deva_middle = None
    best_deva_window_start_idx = -1
    
    # Calculate estimated position accounting for length difference
    # If input has more tokens than output, each input position shifts down
    length_diff = len(roman_tokens) - len(deva_tokens)
    estimated_deva_idx = roman_index - length_diff
    
    # Only test windows in tight range around estimated position
    # Range is (length_diff + 1): 2 positions down and 2 positions up
    window_range = length_diff + 1
    
    search_start = max(0, estimated_deva_idx - window_range)
    search_end = min(len(deva_tokens), estimated_deva_idx + window_range + 1)
    
    # Slide through Devanagari tokens ONLY in relevant range
    for deva_start_idx in range(search_start, search_end):
        deva_end_idx = deva_start_idx + len(roman_window)
        
        # Check if window fits
        if deva_end_idx > len(deva_tokens):
            break
        
        deva_window = deva_tokens[deva_start_idx:deva_end_idx]
        deva_middle = deva_window[english_pos_in_window]
        
        # Skip if middle token is a tail word (attach to previous instead)
        if deva_middle.lower() in TAIL_WORDS:
            continue
        
        # Score this window
        score = window_score(roman_window, deva_window)
        
        if score > best_score:
            best_score = score
            best_deva_window_start_idx = deva_start_idx
            best_deva_middle = deva_middle
    
    return best_deva_middle, best_score


def find_best_number_position(
    number_word: str,
    roman_tokens: list[str],
    deva_tokens: list[str],
    roman_index: int,
) -> tuple[str | None, float]:
    """Find best window position for a number using sliding window of context.
    
    Special case: if number is at first position, use first deva token directly.
    Otherwise, uses prev and next tokens as anchors (skip the number itself in matching).
    Builds a 2-token window [prev, next] and slides it through output,
    checking positions where center token is NOT a number or tail word.
    
    Returns (best_target_token, score)
    """
    # Special case: number at first position
    if roman_index == 0:
        if len(deva_tokens) > 0:
            return deva_tokens[0], 1.0
        return None, -1.0
    
    # Build context window: [prev, next] - skip number itself
    # For numbers, we keep ALL context including tail words (they're informative!)
    roman_window = []
    
    if roman_index > 0:
        roman_window.append(roman_tokens[roman_index - 1])
    
    # Add next token
    next_idx = roman_index + 1
    if next_idx < len(roman_tokens):
        roman_window.append(roman_tokens[next_idx])
    
    if len(roman_window) < 2:
        # Not enough context
        return None, -1.0
    
    best_score = -1.0
    best_deva_middle = None
    
    # Calculate estimated position
    length_diff = len(roman_tokens) - len(deva_tokens)
    estimated_deva_idx = roman_index - length_diff
    
    # Search range
    window_range = length_diff + 1
    search_start = max(0, estimated_deva_idx - window_range)
    search_end = min(len(deva_tokens), estimated_deva_idx + window_range + 1)
    
    # Slide through Devanagari tokens looking for good prev+next matches
    for deva_start_idx in range(search_start, search_end):
        # Test window: [center-1, center, center+1]
        # We slide center from start_idx to end_idx, checking prev and next
        
        if deva_start_idx == 0:
            # Can't have a prev token
            continue
        
        deva_prev = deva_tokens[deva_start_idx - 1]
        deva_center = deva_tokens[deva_start_idx]
        deva_next = deva_tokens[deva_start_idx + 1] if deva_start_idx + 1 < len(deva_tokens) else None
        
        # Skip if center is a number or tail word
        if is_devanagari_number(deva_center) or deva_center.lower() in TAIL_WORDS:
            continue
        
        # Score using prev and next tokens only
        score = 0.0
        
        # Score prev if available
        if len(roman_window) > 0:
            prev_score = element_score(roman_window[0], deva_prev)
            score += prev_score
        
        # Score next if available in both roman and deva
        if len(roman_window) > 1 and deva_next is not None:
            next_score = element_score(roman_window[1], deva_next)
            score += next_score
        
        # Average the scores
        num_compared = (1 if len(roman_window) > 0 else 0) + (1 if len(roman_window) > 1 and deva_next else 0)
        if num_compared > 0:
            score = score / num_compared
        
        if score > best_score:
            best_score = score
            best_deva_middle = deva_center
    
    return best_deva_middle, best_score


def find_best_offset_position(
    english_word: str,
    roman_tokens: list[str],
    deva_tokens: list[str],
    roman_index: int,
    phonetic_weight: float = 0.7,
    length_weight: float = 0.3,
) -> tuple[str | None, float]:
    """Find best output token position using multi-offset 3-token window.
    
    Combines phonetic matching (70%) and length compatibility (30%).
    Returns (best_target_token, combined_score)
    """
    mismatch = abs(len(roman_tokens) - len(deva_tokens))
    best_match = None
    best_score = -1.0
    
    for offset in range(-mismatch - 1, mismatch + 2):
        phonetic_scores = []
        length_scores = []
        valid = True
        
        if roman_index > 0:
            target_idx = roman_index - 1 + offset
            if 0 <= target_idx < len(deva_tokens):
                phon_score = phonetic_similarity(
                    word_to_phonetic(roman_tokens[roman_index - 1]),
                    word_to_phonetic(deva_tokens[target_idx])
                )
                phonetic_scores.append(phon_score)
                len_score = calculate_length_score(
                    roman_tokens[roman_index - 1],
                    deva_tokens[target_idx]
                )
                length_scores.append(len_score)
            else:
                valid = False
        
        if valid:
            target_idx = roman_index + offset
            if 0 <= target_idx < len(deva_tokens):
                word_phonetic = word_to_phonetic(english_to_devanagari(english_word))
                match_phonetic = word_to_phonetic(deva_tokens[target_idx])
                phon_score = phonetic_similarity(word_phonetic, match_phonetic)
                phonetic_scores.append(phon_score)
                len_score = calculate_length_score(english_word, deva_tokens[target_idx])
                length_scores.append(len_score)
            else:
                valid = False
        
        if roman_index < len(roman_tokens) - 1 and valid:
            target_idx = roman_index + 1 + offset
            if 0 <= target_idx < len(deva_tokens):
                phon_score = phonetic_similarity(
                    word_to_phonetic(roman_tokens[roman_index + 1]),
                    word_to_phonetic(deva_tokens[target_idx])
                )
                phonetic_scores.append(phon_score)
                len_score = calculate_length_score(
                    roman_tokens[roman_index + 1],
                    deva_tokens[target_idx]
                )
                length_scores.append(len_score)
        
        if len(phonetic_scores) > 0 and len(length_scores) > 0:
            avg_phonetic = sum(phonetic_scores) / len(phonetic_scores)
            avg_length = sum(length_scores) / len(length_scores)
            combined_score = (phonetic_weight * avg_phonetic) + (length_weight * avg_length)
            
            if combined_score > best_score:
                best_score = combined_score
                target_idx = roman_index + offset
                if 0 <= target_idx < len(deva_tokens):
                    best_match = deva_tokens[target_idx]
    
    return best_match, best_score


def best_target_match_for_english_word(
    english_word: str,
    target_tokens: list[str],
    target_start_idx: int,
    pos: int,
    len_in: int,
    len_out: int,
    window_type: str = "symmetric",
    roman_tokens: list[str] | None = None,
    roman_index: int = -1,
) -> tuple[str, str, str | None, str | None, float, float, float]:
    """Return the best target token for one English word.

    Scoring is based on: previous token match + current token match + next token match.
    Uses surrounding context as PRIMARY criteria.
    """

    rule_deva = english_to_devanagari(english_word)
    rule_phonetic = word_to_phonetic(rule_deva)

    best_word = None
    best_target_phonetic = None
    best_score = -1.0
    best_raw_phon = -1.0
    best_context_score = 0.0

    for local_index, token in enumerate(target_tokens):
        token_clean = clean_token(token)
        if not token_clean:
            continue
        if not length_compatible(english_word, token_clean):
            continue

        token_phonetic = word_to_phonetic(token_clean)
        raw_score = phonetic_similarity(rule_phonetic, token_phonetic)
        
        # Calculate 3-token context score: prev + current + next
        context_scores = []
        
        # Previous token match
        if roman_index > 0 and local_index > 0 and roman_tokens is not None:
            prev_roman = roman_tokens[roman_index - 1]
            prev_target = target_tokens[local_index - 1]
            if prev_roman and prev_target:
                prev_score = phonetic_similarity(
                    word_to_phonetic(prev_roman),
                    word_to_phonetic(prev_target)
                )
                context_scores.append(prev_score)
        
        # Current token match (the English word itself)
        context_scores.append(raw_score)
        
        # Next token match
        if roman_index < len(roman_tokens) - 1 and local_index < len(target_tokens) - 1 and roman_tokens is not None:
            next_roman = roman_tokens[roman_index + 1]
            next_target = target_tokens[local_index + 1]
            if next_roman and next_target:
                next_score = phonetic_similarity(
                    word_to_phonetic(next_roman),
                    word_to_phonetic(next_target)
                )
                context_scores.append(next_score)
        
        # Score = average of available context signals
        if context_scores:
            score = sum(context_scores) / len(context_scores)
            context_score = score
        else:
            score = raw_score
            context_score = 0.0

        if score > best_score:
            best_score = score
            best_raw_phon = raw_score
            best_word = token_clean
            best_target_phonetic = token_phonetic
            best_context_score = context_score

    exact_candidates = EXACT_TRANSLITERATION_CANDIDATES.get(english_word.lower(), [])
    if exact_candidates:
        for local_index, token in enumerate(target_tokens):
            token_clean = clean_token(token)
            if token_clean not in exact_candidates:
                continue
            token_phonetic = word_to_phonetic(token_clean)
            raw_score = phonetic_similarity(rule_phonetic, token_phonetic)
            
            # Calculate 3-token context score for exact candidates too
            context_scores = []
            
            if roman_index > 0 and local_index > 0 and roman_tokens is not None:
                prev_roman = roman_tokens[roman_index - 1]
                prev_target = target_tokens[local_index - 1]
                if prev_roman and prev_target:
                    prev_score = phonetic_similarity(
                        word_to_phonetic(prev_roman),
                        word_to_phonetic(prev_target)
                    )
                    context_scores.append(prev_score)
            
            context_scores.append(raw_score)
            
            if roman_index < len(roman_tokens) - 1 and local_index < len(target_tokens) - 1 and roman_tokens is not None:
                next_roman = roman_tokens[roman_index + 1]
                next_target = target_tokens[local_index + 1]
                if next_roman and next_target:
                    next_score = phonetic_similarity(
                        word_to_phonetic(next_roman),
                        word_to_phonetic(next_target)
                    )
                    context_scores.append(next_score)
            
            if context_scores:
                score = sum(context_scores) / len(context_scores)
                context_score = score
            else:
                score = raw_score
                context_score = 0.0
            
            if score >= SECOND_PASS_MIN_SCORE and score > best_score:
                best_score = score
                best_raw_phon = raw_score
                best_word = token_clean
                best_target_phonetic = token_phonetic
                best_context_score = context_score

    return rule_deva, rule_phonetic, best_word, best_target_phonetic, float(best_score), float(best_raw_phon), float(best_context_score)


def reconstruct_sentence_pair(
    roman_sentence: str,
    deva_sentence: str,
    english_vocab: set[str],
    window_type: str = "symmetric",
    phonetic_threshold: float = PHONETIC_MIN_SCORE,
) -> tuple[str, list[Replacement]]:
    """Two-pass repair: First numbers, then English words using updated context."""
    
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

    # FIRST PASS: Process all numbers
    for roman_index, roman_word in enumerate(roman_tokens):
        if not is_roman_number(roman_word):
            continue
        
        best_match, score = find_best_number_position(
            roman_word,
            roman_tokens,
            deva_tokens,
            roman_index,
        )
        
        if best_match is None:
            continue
        
        try:
            match_clean_index = deva_tokens.index(best_match)
        except ValueError:
            continue
        
        if match_clean_index in used_target_clean_indices:
            continue
        
        # Convert to Devanagari numerals
        deva_numerals = roman_word.translate(ARABIC_TO_DEVANAGARI_NUMERAL)
        raw_target_index = deva_clean_to_raw[match_clean_index]
        used_target_clean_indices.add(match_clean_index)
        
        replacements_map[raw_target_index] = deva_numerals
        replacements.append(Replacement(roman_word, deva_numerals, float(score), raw_target_index, kind="number"))
        
        # Update deva_tokens to reflect the replacement
        deva_tokens[match_clean_index] = deva_numerals
    
    # SECOND PASS: Process English words using updated deva_tokens
    for roman_index, roman_word in enumerate(roman_tokens):
        if roman_word not in english_vocab:
            continue
        if roman_word in MARKER_TOKENS:
            continue
        if is_roman_number(roman_word):
            continue

        best_match, combined_score = find_best_window_position(
            roman_word,
            roman_tokens,
            deva_tokens,  # Now using UPDATED tokens with numbers replaced
            roman_index,
        )

        if best_match is None:
            continue
        if combined_score < phonetic_threshold:
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
            replacements.append(Replacement(roman_word, f"{roman_word} + {marker_deva}", float(combined_score), raw_target_index, kind="english+marker", context_score=float(combined_score)))
            continue

        replacements_map[raw_target_index] = roman_word
        replacements.append(Replacement(roman_word, roman_word, float(combined_score), raw_target_index, context_score=float(combined_score)))

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


@lru_cache(maxsize=1)
def get_default_english_vocab() -> frozenset[str]:
    return frozenset(load_english_vocab())


def repair_transliteration(
    roman_sentence: str,
    deva_sentence: str,
    window_type: str = "symmetric",
    phonetic_threshold: float = PHONETIC_MIN_SCORE,
    vocab_path: Path | None = None,
) -> tuple[str, list[Replacement]]:
    english_vocab = load_english_vocab(vocab_path) if vocab_path is not None else get_default_english_vocab()
    
    # Step 1: Repair English words with context matching
    repaired_english, replacements = reconstruct_sentence_pair(
        roman_sentence,
        deva_sentence,
        english_vocab=set(english_vocab),
        window_type=window_type,
        phonetic_threshold=phonetic_threshold,
    )
    
    # Step 2: Repair numbers using context
    repaired_final, all_replacements = repair_numbers(
        roman_sentence,
        repaired_english,
        replacements,
    )
    
    return repaired_final, all_replacements


def detect_numbers_in_sentence(roman_sentence: str) -> list[tuple[int, str]]:
    """Detect digit sequences in romanized sentence.
    
    Returns list of (position, digit_string) tuples.
    """
    numbers: list[tuple[int, str]] = []
    current_number = ""
    start_pos = 0
    
    for pos, char in enumerate(roman_sentence):
        if char.isdigit():
            if not current_number:
                start_pos = pos
            current_number += char
        else:
            if current_number:
                numbers.append((start_pos, current_number))
                current_number = ""
    
    if current_number:
        numbers.append((start_pos, current_number))
    
    return numbers


def find_number_position_by_context(
    roman_sentence: str,
    deva_sentence: str,
    number_roman_index: int,
    roman_tokens: list[str],
    deva_tokens: list[str],
    deva_clean_to_raw: list[int],
) -> int | None:
    """Find position of a number in deva_sentence using surrounding context.
    
    Returns raw token index in deva_sentence, or None if not found.
    """
    # Get surrounding tokens in romanized input
    context_before = roman_tokens[number_roman_index - 1] if number_roman_index > 0 else None
    context_after = roman_tokens[number_roman_index + 1] if number_roman_index < len(roman_tokens) - 1 else None
    
    best_deva_index = None
    best_score = -1.0
    
    # Try to match surrounding tokens in output
    for deva_idx, deva_token in enumerate(deva_tokens):
        context_matches = 0
        match_score = 0.0
        
        # Check token before
        if context_before and deva_idx > 0:
            prev_deva = deva_tokens[deva_idx - 1]
            prev_phon_match = phonetic_similarity(
                word_to_phonetic(context_before),
                word_to_phonetic(prev_deva)
            )
            if prev_phon_match > PHONETIC_MIN_SCORE:
                context_matches += 1
                match_score += prev_phon_match
        
        # Check token after
        if context_after and deva_idx < len(deva_tokens) - 1:
            next_deva = deva_tokens[deva_idx + 1]
            next_phon_match = phonetic_similarity(
                word_to_phonetic(context_after),
                word_to_phonetic(next_deva)
            )
            if next_phon_match > PHONETIC_MIN_SCORE:
                context_matches += 1
                match_score += next_phon_match
        
        # Use token with best context match
        if context_matches > 0:
            avg_score = match_score / context_matches
            if avg_score > best_score:
                best_score = avg_score
                best_deva_index = deva_idx
    
    # Return raw token index
    if best_deva_index is not None and best_deva_index < len(deva_clean_to_raw):
        return deva_clean_to_raw[best_deva_index]
    return None


def repair_numbers(
    roman_sentence: str,
    deva_sentence: str,
    replacements: list[Replacement],
) -> tuple[str, list[Replacement]]:
    """Detect numbers in romanized input, find their positions in output via context, and convert to Devanagari numerals."""
    
    # Detect numbers
    numbers = detect_numbers_in_sentence(roman_sentence)
    if not numbers:
        return deva_sentence, replacements
    
    # Tokenize for context matching
    roman_tokens = [clean_token(token).lower() for token in roman_sentence.split()]
    roman_tokens = [token for token in roman_tokens if token]
    
    deva_tokens, deva_clean_to_raw = build_clean_positions(deva_sentence.split())
    deva_raw = deva_sentence.split()
    
    # Map numbers to their token positions
    replacements_map: dict[int, str] = {}
    number_replacements: list[Replacement] = []
    
    for number_char_pos, number_str in numbers:
        # Find which token this number is in (approximate via character position)
        token_count = 0
        token_pos = 0
        for i, char in enumerate(roman_sentence):
            if char.isspace():
                token_count += 1
            if i >= number_char_pos:
                token_pos = token_count
                break
        
        if token_pos >= len(roman_tokens):
            continue
        
        # Find position in deva output using context
        deva_index = find_number_position_by_context(
            roman_sentence,
            deva_sentence,
            token_pos,
            roman_tokens,
            deva_tokens,
            deva_clean_to_raw,
        )
        
        if deva_index is not None:
            # Convert number to Devanagari numerals
            deva_numerals = number_str.translate(ARABIC_TO_DEVANAGARI_NUMERAL)
            replacements_map[deva_index] = deva_numerals
            number_replacements.append(
                Replacement(number_str, deva_numerals, 1.0, deva_index, kind="number")
            )
    
    # Apply replacements
    reconstructed_tokens = [replacements_map.get(index, token) for index, token in enumerate(deva_raw)]
    repaired_sentence = " ".join(reconstructed_tokens)
    
    return repaired_sentence, replacements + number_replacements


def format_replacements(replacements: Iterable[Replacement]) -> str:
    lines = []
    for item in replacements:
        lines.append(f"- {item.roman_word} -> {item.target_word} (score={item.score:.4f}, index={item.target_index}, kind={item.kind}, context={item.context_score:.4f})")
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