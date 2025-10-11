import re
from typing import Tuple, List

DIGRAPHS = { 'CH', 'KH', 'GH', 'PH', 'BH', 'JH', 'TH', 'DH', 'SH', 'NG', 'X' }
VOWELS = set('aeiou')
PRIMARY_MAP = {
    'CH': 'C', 'KH': 'K', 'GH': 'G', 'PH': 'P', 'BH': 'B', 'JH': 'J', 'TH': 'T', 'DH': 'D', 'SH': 'S', 'NG': 'N', 'X': 'K',
    'k': 'K', 'q': 'K', 'c': 'K', 'g': 'G', 'j': 'J', 't': 'T', 'd': 'D', 'p': 'P', 'b': 'B',
    'm': 'M', 'n': 'N', 'y': 'Y', 'r': 'R', 'l': 'L', 's': 'S', 'h': 'H', 'v': 'B', 'w': 'W', 'f': 'F', 'z': 'S', 'x': 'K'
}
ALTERNATE_MAP = PRIMARY_MAP.copy()
ALTERNATE_MAP['X'] = 'X'

def _normalize_unicode(s: str) -> str:
    if not s:
        return ''
    s = s.strip().lower()
    s = s.replace('ā', 'a').replace('ī', 'i').replace('ū', 'u')
    s = s.replace('ṁ', 'm').replace('ṃ', 'm').replace('ñ', 'n').replace('ṙ', 'r')
    return s


def _preprocess(word: str) -> str:
    if not word:
        return ''
    
    if len(word) == 2:
        return word.upper()
    w = _normalize_unicode(word)
    w = re.sub(r'[^a-z]', '', w)
    w = re.sub(r'xo', 'cho', w)
    w = re.sub(r'xh', 'ch', w)
    w = re.sub(r'a{2,}', 'a', w)
    w = re.sub(r'e{2,}', 'e', w) 
    w = re.sub(r'i{2,}', 'i', w)
    w = re.sub(r'o{2,}', 'o', w)
    w = re.sub(r'u{2,}', 'u', w)
    w = re.sub(r'ae', 'ai', w)
    w = w.replace('chh', 'CH')
    w = w.replace('ch', 'CH')
    w = w.replace('kh', 'KH')
    w = w.replace('gh', 'GH')
    w = w.replace('ph', 'PH')
    w = w.replace('bh', 'BH')
    w = w.replace('jh', 'JH')
    w = w.replace('th', 'TH')
    w = w.replace('dh', 'DH')
    w = w.replace('sh', 'SH')
    w = w.replace('ng', 'NG')
    w = w.replace('f', 'PH')
    w = w.replace('x', 'X')
    w = re.sub(r'(.)\1{2,}', r'\1\1', w)

    return w


def _tokenize(pre: str) -> List[str]:
    tokens = []
    i = 0
    L = len(pre)
    while i < L:
        if i + 1 < L:
            two = pre[i:i+2]
            if two in DIGRAPHS:
                tokens.append(two)
                i += 2
                continue
        tokens.append(pre[i])
        i += 1
    return tokens

def _encode_tokens(tokens: List[str], mapping: dict, max_len: int = 6) -> str:
    if not tokens:
        return ''
    code = []
    last_sym = None
    i = 0
    if tokens and isinstance(tokens[0], str) and tokens[0] and tokens[0][0] in VOWELS:
        first_vowel = tokens[0].upper()
        code.append(first_vowel)
        last_sym = first_vowel
        i = 1

    while i < len(tokens) and len(code) < max_len:
        tok = tokens[i]
        if len(tok) == 1 and tok in VOWELS:
          i += 1
          continue

        sym = mapping.get(tok, None)
        if sym is None and len(tok) == 1:
            sym = mapping.get(tok.lower(), tok.upper())

        if not sym:
            i += 1
            continue

        if sym != last_sym:
            code.append(sym)
            last_sym = sym
        i += 1

    if tokens and tokens[-1] in VOWELS:
        last_vowel = tokens[-1].upper()
        if last_vowel != last_sym and len(code) < max_len:
            code.append(last_vowel)


    return ''.join(code)


def nepali_dmetaphone(source: str) -> Tuple[str, str]:
    if not source:
        return ('', '')

    pre = _preprocess(source)
    tokens = _tokenize(pre)

    primary = _encode_tokens(tokens, PRIMARY_MAP)

    alternate = _encode_tokens(tokens, ALTERNATE_MAP)

    if primary == alternate:
        alternate = ''

    return (primary, alternate)
