"""
Text perturbation:
1. Random character replacement (edit rate rho).
2. Unicode diacritic / confusable attack.
"""
import numpy as np
import unicodedata


# Visually similar confusable characters (Cyrillic, digits, etc.)
CONFUSABLE = {
    'a': '\u0430',  # Cyrillic a
    'e': '\u0435',  # Cyrillic e
    'o': '\u043E',  # Cyrillic o
    'c': '\u0441',  # Cyrillic c
    'p': '\u0440',  # Cyrillic p
    'x': '\u0445',  # Cyrillic x
    'y': '\u0443',  # Cyrillic u
    'i': '\u0456',  # Cyrillic i
    'O': '0',
    '0': 'O',
    'l': '1',
    '1': 'l',
    'S': '5',
    '5': 'S',
    'B': '8',
    '8': 'B',
    'Z': '2',
    '2': 'Z',
}

# Combining diacritics (Unicode U+0300 - U+036F)
DIACRITICS = {
    'a': 'a\u0301',  # á
    'e': 'e\u0301',  # é
    'i': 'i\u0301',  # í
    'o': 'o\u0301',  # ó
    'u': 'u\u0301',  # ú
    'n': 'n\u0303',  # ñ
    'c': 'c\u0327',  # ç
    's': 's\u0301',  # ś
}


def random_char_replacement(text, rho=0.1, rng=None):
    """
    Replace rho fraction of characters with confusables.
    No insert/delete, only replace. Keeps length.
    """
    rng = rng or np.random.default_rng()
    if len(text) == 0:
        return text

    chars = list(text)
    n_edit = max(1, int(len(chars) * rho))
    indices = rng.choice(len(chars), size=n_edit, replace=False)

    for i in indices:
        c = chars[i]
        if c in CONFUSABLE:
            chars[i] = CONFUSABLE[c]
        # else: keep as is

    return ''.join(chars)


def unicode_diacritic_attack(text, rho=0.1, rng=None):
    """
    Replace characters with diacritic/confusable variants.
    """
    rng = rng or np.random.default_rng()
    if len(text) == 0:
        return text

    chars = list(text)
    n_edit = max(1, int(len(chars) * rho))
    indices = rng.choice(len(chars), size=n_edit, replace=False)

    for i in indices:
        c = chars[i].lower()
        if c in DIACRITICS:
            chars[i] = DIACRITICS[c]
        elif c in CONFUSABLE:
            chars[i] = CONFUSABLE[c]

    return ''.join(chars)


def perturb_texts(texts, method="random", rho=0.1, rng=None):
    """
    method: 'random' | 'unicode'
    """
    rng = rng or np.random.default_rng()
    if method == "random":
        return [random_char_replacement(t, rho=rho, rng=rng) for t in texts]
    elif method == "unicode":
        return [unicode_diacritic_attack(t, rho=rho, rng=rng) for t in texts]
    else:
        raise ValueError(f"Unknown method: {method}")


def edit_rate(original, perturbed):
    """Character-level edit rate (Levenshtein / max_len)."""
    if len(original) == 0:
        return 0.0
    # Simple: Hamming-like since we don't insert/delete
    if len(original) != len(perturbed):
        # fallback to Levenshtein
        return _levenshtein(original, perturbed) / max(len(original), len(perturbed))
    diff = sum(1 for a, b in zip(original, perturbed) if a != b)
    return diff / len(original)


def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]