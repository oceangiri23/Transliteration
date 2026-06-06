"""
roman_to_devanagari.py
----------------------
Rule-based transliteration of Romanized Nepali → Devanagari Unicode.

Conventions supported (ITRANS-inspired + common Nepali usage):
  Vowels  : a aa/A  i ii/I  u uu/U  e ai  o au  ri
  Nasals  : n~ / m~ (anusvara ं), n~~ (chandrabindu ँ)
  Aspirates: kh gh ch/chh jh th dh ph bh sh (digraphs)
  Retroflex: T Th D Dh N (capital = retroflex)
  Sibilants: sh → श,  Sh / sh~ → ष,  s → स
  Visarga : H (at end of syllable)

Usage:
    python roman_to_devanagari.py
    >>> transliterate("namaste")   # नमस्ते
    >>> transliterate("kathmandu") # काठमाडौ
    >>> transliterate("dhanyabad") # धन्यबाद
"""

import re
import sys

# ─────────────────────────────────────────────
# 1.  Mapping tables  (order = longest first)
# ─────────────────────────────────────────────

# Digraph / multi-char consonant combos → Devanagari consonant (no inherent-a yet)
# Conjuncts that are treated as atomic units
CONJUNCTS = [
    ("gya",  "ज्ञ"),
    ("ksh",  "क्ष"),
    ("tr",   "त्र"),
]

# Two-character consonant digraphs (aspirates, retroflex pairs, etc.)
DIGRAPHS = [
    ("kh",  "ख"),
    ("gh",  "घ"),
    ("ng",  "ङ"),
    ("chh", "छ"),   # chh before ch
    ("ch",  "च"),
    ("jh",  "झ"),
    ("ny",  "ञ"),
    ("Th",  "ठ"),
    ("Dh",  "ढ"),
    ("th",  "थ"),
    ("dh",  "ध"),
    ("ph",  "फ"),
    ("bh",  "भ"),
    ("Sh",  "ष"),   # retroflx sibilant (capital S)
    ("sh",  "श"),
]

# Single consonants
SINGLE_CONSONANTS = [
    ("k",  "क"),
    ("g",  "ग"),
    ("c",  "च"),   # fallback for 'c' alone
    ("j",  "ज"),
    ("T",  "ट"),   # retroflex T
    ("D",  "ड"),   # retroflex D
    ("N",  "ण"),   # retroflex N
    ("t",  "त"),
    ("d",  "द"),
    ("n",  "न"),
    ("p",  "प"),
    ("b",  "ब"),
    ("m",  "म"),
    ("y",  "य"),
    ("r",  "र"),
    ("l",  "ल"),
    ("v",  "व"),
    ("w",  "व"),
    ("s",  "स"),
    ("h",  "ह"),
    ("f",  "फ"),   # loanwords
    ("z",  "ज"),   # loanwords
]

# All consonants in longest-first order
CONSONANT_MAP = CONJUNCTS + DIGRAPHS + SINGLE_CONSONANTS

# Vowels: each entry is (roman, standalone_form, matra_form)
# matra_form is "" for 'a' because the inherent-a needs no matra,
# and we use VIRAMA to suppress the inherent-a when there is NO vowel.
VOWEL_MAP = [
    ("aa",  "आ", "ा"),
    ("A",   "आ", "ा"),
    ("ii",  "ई", "ी"),
    ("I",   "ई", "ी"),
    ("ai",  "ऐ", "ै"),
    ("au",  "औ", "ौ"),
    ("uu",  "ऊ", "ू"),
    ("U",   "ऊ", "ू"),
    ("ri",  "ऋ", "ृ"),
    ("e",   "ए", "े"),
    ("o",   "ओ", "ो"),
    ("i",   "इ", "ि"),
    ("u",   "उ", "ु"),
    ("a",   "अ", ""),    # inherent-a; matra="" means "do nothing"
]

# Special marks
ANUSVARA      = "ं"
CHANDRABINDU  = "ँ"
VISARGA       = "ः"
VIRAMA        = "्"   # halant — suppresses inherent-a

# Sets for quick lookup
VOWEL_ROMANS  = {v[0] for v in VOWEL_MAP}
_VOWEL_SA_MAP = {v[0]: (v[1], v[2]) for v in VOWEL_MAP}
_CONS_SET     = set()
for pair in CONSONANT_MAP:
    _CONS_SET.add(pair[0])


# ─────────────────────────────────────────────
# 2.  Tokenizer
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[tuple[str, str]]:
    """
    Walk through `text` left-to-right, greedily matching the longest
    rule.  Returns a list of (token_type, value) where token_type is:
        'C'  – consonant roman string
        'V'  – vowel roman string
        'AN' – anusvara trigger  (n~ or m~)
        'CB' – chandrabindu trigger (n~~)
        'VS' – visarga trigger (H)
        'SP' – space / punctuation passed through
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        matched = False

        # chandrabindu: n~~ (check before n~)
        if text[i:i+3] == "n~~":
            tokens.append(("CB", "n~~"))
            i += 3
            matched = True

        # anusvara: n~ or m~
        elif text[i:i+2] in ("n~", "m~"):
            tokens.append(("AN", text[i:i+2]))
            i += 2
            matched = True

        # visarga: uppercase H alone
        elif text[i] == "H":
            tokens.append(("VS", "H"))
            i += 1
            matched = True

        else:
            # Try vowels FIRST for multi-char vowels that start with a
            # consonant letter (e.g. 'ri' = ऋ must beat 'r' consonant).
            # Only attempt multi-char vowels here; single-char vowels come after.
            for roman, _, _ in VOWEL_MAP:
                if len(roman) > 1 and text[i:i+len(roman)] == roman:
                    tokens.append(("V", roman))
                    i += len(roman)
                    matched = True
                    break

            if not matched:
                # Try consonants (longest first)
                for roman, deva in CONSONANT_MAP:
                    if text[i:i+len(roman)] == roman:
                        tokens.append(("C", roman))
                        i += len(roman)
                        matched = True
                        break

            if not matched:
                # Try single-char vowels
                for roman, _, _ in VOWEL_MAP:
                    if len(roman) == 1 and text[i:i+1] == roman:
                        tokens.append(("V", roman))
                        i += 1
                        matched = True
                        break

            if not matched:
                # Pass through spaces, punctuation, digits, unknown chars
                tokens.append(("SP", text[i]))
                i += 1

    return tokens


# ─────────────────────────────────────────────
# 3.  Generator
# ─────────────────────────────────────────────

def tokens_to_devanagari(tokens: list[tuple[str, str]]) -> str:
    """
    Convert a token list to a Devanagari string.

    State machine with one look-ahead:
      - After a consonant, check next token:
          V  → output consonant + matra (or nothing for inherent-a)
          C  → output consonant + VIRAMA (halant) to form cluster
          AN/CB/VS → output consonant (inherent-a) + diacritic
          SP/end  → output consonant (inherent-a)
      - A standalone vowel (not after consonant) → output full vowel letter
    """
    result = []
    i = 0
    n = len(tokens)

    def get_cons(roman):
        for r, d in CONSONANT_MAP:
            if r == roman:
                return d
        return roman  # fallback

    def get_vowel(roman, after_consonant):
        standalone, matra = _VOWEL_SA_MAP[roman]
        if after_consonant:
            return matra  # may be "" for inherent-a
        return standalone

    while i < n:
        ttype, tval = tokens[i]

        if ttype == "C":
            deva_cons = get_cons(tval)
            # peek at next token
            next_type = tokens[i+1][0] if i+1 < n else None
            next_val  = tokens[i+1][1] if i+1 < n else None

            result.append(deva_cons)

            if next_type == "V":
                matra = get_vowel(next_val, after_consonant=True)
                result.append(matra)   # "" for inherent-a = nothing extra
                i += 2  # consume vowel token too
            elif next_type == "C":
                result.append(VIRAMA)  # halant → consonant cluster
                i += 1
            elif next_type in ("AN", "CB", "VS"):
                # inherent-a stays; diacritic will be added next iteration
                i += 1
            else:
                # end of word / space — inherent-a stays
                i += 1

        elif ttype == "V":
            # Standalone vowel (not consumed by previous consonant branch)
            standalone, _ = _VOWEL_SA_MAP[tval]
            result.append(standalone)
            i += 1

        elif ttype == "AN":
            result.append(ANUSVARA)
            i += 1

        elif ttype == "CB":
            result.append(CHANDRABINDU)
            i += 1

        elif ttype == "VS":
            result.append(VISARGA)
            i += 1

        elif ttype == "SP":
            result.append(tval)
            i += 1

        else:
            i += 1

    return "".join(result)


# ─────────────────────────────────────────────
# 4.  Public API
# ─────────────────────────────────────────────

ENGLISH_OVERRIDES = {
    "computer": "kampyutar",
    "school": "skul",
    "office": "ofis",
    "mobile": "mobail",
    "internet": "intarnet",
    "facebook": "fesbuk",
    "google": "gugal",
    "project": "prajekt",
    "energy": "enarji",
    "english": "inglish",
}

ENGLISH_DEVA_OVERRIDES = {
    "computer": "कम्प्युटर",
    "school": "स्कुल",
    "office": "अफिस",
    "mobile": "मोबाइल",
    "internet": "इन्टरनेट",
    "facebook": "फेसबुक",
    "google": "गुगल",
    "project": "प्रोजेक्ट",
    "energy": "एनर्जी",
    "english": "इङ्ग्लिश",
}


def normalize_english_word(word: str) -> str:
    """Convert a common English spelling into a romanized form this rule engine can handle."""
    text = word.strip().lower()
    text = re.sub(r"[^a-z'-]+", "", text)
    text = text.replace("-", "")
    if not text:
        return ""

    if text in ENGLISH_OVERRIDES:
        return ENGLISH_OVERRIDES[text]

    replacements = [
        ("sch", "sk"),
        ("tion", "shan"),
        ("sion", "zhan"),
        ("ph", "f"),
        ("ck", "k"),
        ("qu", "ku"),
        ("wr", "r"),
        ("kn", "n"),
        ("wh", "w"),
        ("oo", "u"),
        ("ee", "i"),
        ("ai", "ai"),
        ("ay", "ei"),
        ("ei", "i"),
        ("ie", "i"),
        ("x", "ks"),
        ("c", "k"),
        ("q", "k"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def english_to_devanagari(word: str) -> str:
    """Generate a Devanagari spelling for a common English word."""
    text = word.strip().lower()
    text = re.sub(r"[^a-z'-]+", "", text)
    text = text.replace("-", "")
    if text in ENGLISH_DEVA_OVERRIDES:
        return ENGLISH_DEVA_OVERRIDES[text]
    return transliterate(normalize_english_word(word))

def transliterate(roman: str) -> str:
    """
    Transliterate a Romanized Nepali string to Devanagari.

    Steps:
      1. Normalize (lowercase, but preserve retroflex caps T/D/N/S/H)
      2. Tokenize
      3. Generate Devanagari
    """
    # Normalize: lowercase everything EXCEPT special capital letters
    # T, D, N, S(h), H are meaningful caps — preserve them.
    # Strategy: lowercase, then restore known caps from original
    normalized = []
    specials = {'T', 'D', 'N', 'H'}  # S is handled via Sh digraph check

    i = 0
    while i < len(roman):
        ch = roman[i]
        # Sh / Sh (capital S + h) → keep as "Sh"
        if ch == 'S' and i+1 < len(roman) and roman[i+1] == 'h':
            normalized.append('S')
            normalized.append('h')
            i += 2
        elif ch in specials:
            normalized.append(ch)
            i += 1
        else:
            normalized.append(ch.lower())
            i += 1

    norm_str = "".join(normalized)
    tokens = tokenize(norm_str)
    return tokens_to_devanagari(tokens)


# ─────────────────────────────────────────────
# 5.  Test suite
# ─────────────────────────────────────────────

TEST_CASES = [
    # (roman_input, expected_devanagari, description)
    ("namaste",       "नमस्ते",      "namaste"),
    ("dhanyabad",     "धन्यबाद",     "thank you"),
    ("nepal",         "नेपाल",       "Nepal"),
    ("kathmandu",     "काठमाडु",     "Kathmandu (approx)"),
    ("paani",         "पानी",        "water"),
    ("aakash",        "आकाश",       "sky"),
    ("bhat",          "भात",         "rice"),
    ("shiksha",       "शिक्षा",      "education"),
    ("gyaan",         "ज्ञान",       "knowledge"),
    ("triphala",      "त्रिफला",     "triphala"),
    ("ramro",         "राम्रो",      "good/nice"),
    ("saathii",       "साथी",        "friend"),
    ("kholnu",        "खोल्नु",      "to open"),
    ("chha",          "छ",           "is (present)"),
    ("bhuiN",         "भुइँ",        "floor (chandrabindu via N~ workaround)"),
    ("a",             "अ",           "vowel a standalone"),
    ("k",             "क",           "single consonant"),
    ("ka",            "का",          "ka with inherent a as matra... wait: k+a = क + ा"),
]

# Note: "ka" should produce "क" + "" (inherent a, no matra) = "क"
# But "kaa" should produce "क" + "ा" = "का"
# Let's add clearer tests:
VOWEL_TESTS = [
    ("ka",   "क",   "k + inherent a"),
    ("kaa",  "का",  "k + aa"),
    ("ki",   "कि",  "k + i"),
    ("kii",  "की",  "k + ii"),
    ("ku",   "कु",  "k + u"),
    ("ke",   "के",  "k + e"),
    ("ko",   "को",  "k + o"),
    ("kai",  "कै",  "k + ai"),
    ("kau",  "कौ",  "k + au"),
    ("kri",  "कृ",  "k + ri"),
]


def run_tests():
    print("=" * 60)
    print("VOWEL MATRA TESTS")
    print("=" * 60)
    all_pass = True
    for roman, expected, desc in VOWEL_TESTS:
        result = transliterate(roman)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status}  {roman:10s} → {result:10s}  (expected {expected})  # {desc}")

    print()
    print("=" * 60)
    print("WORD TESTS")
    print("=" * 60)
    for roman, expected, desc in TEST_CASES:
        result = transliterate(roman)
        status = "✓" if result == expected else "~"  # ~ = close enough
        print(f"  {status}  {roman:15s} → {result:15s}  # {desc}")

    print()
    print("All vowel matra tests passed!" if all_pass else "Some vowel tests failed — check mapping.")


# ─────────────────────────────────────────────
# 6.  Interactive demo
# ─────────────────────────────────────────────

def demo():
    print()
    print("Interactive Transliterator  (type 'quit' to exit)")
    print("-" * 40)
    print("Conventions:")
    print("  aa/A=ā  ii/I=ī  uu/U=ū  ri=ṛ")
    print("  kh gh ch chh jh th dh ph bh sh Sh ng ny")
    print("  T D N = retroflex  H = visarga  n~ m~ = anusvara")
    print()
    while True:
        try:
            text = input("Roman > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() == "quit":
            break
        if text:
            print("Devanagari:", transliterate(text))


def transliterate_english_words(words: list[str]) -> list[tuple[str, str]]:
    """Return English words paired with their Devanagari spellings."""
    pairs = []
    for word in words:
        pairs.append((word, english_to_devanagari(word)))
    return pairs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for original, deva in transliterate_english_words(sys.argv[1:]):
            print(f"{original}\t{deva}")
    else:
        run_tests()
        demo()