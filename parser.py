import re

_NUMBER_PATTERN = r"\d+(?:\.\d+)?"
_ORDINAL_DIGIT_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$", re.IGNORECASE)
_RANGE_DASH_CHARS = r"\-\u2013\u2014"
_INLINE_TAG_RE = re.compile(
    rf"\[\s*({_NUMBER_PATTERN})\s*"
    rf"(?:(?::|[{_RANGE_DASH_CHARS}]|\b(?:to|through|thru)\b)\s*({_NUMBER_PATTERN}))?\s*\]",
    re.IGNORECASE,
)
_DIGIT_RANGE_TAIL_RE = re.compile(rf"({_NUMBER_PATTERN})\s*[{_RANGE_DASH_CHARS}]\s*({_NUMBER_PATTERN})\s*$")
_BRACKETED_HEADER_TAG_RE = re.compile(
    rf"\[\s*({_NUMBER_PATTERN})\s*"
    rf"(?:(?::|[{_RANGE_DASH_CHARS}]|\b(?:to|through|thru)\b)\s*({_NUMBER_PATTERN}))?\s*\]\s*$",
    re.IGNORECASE,
)
_BODY_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+\u2013\u2014]|\d+[.)])\s+")
_RANGE_WORDS = {"to", "through", "thru"}
_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_ORDINAL_WORD_NUMBERS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}
_TENS_WORD_NUMBERS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ORDINAL_TENS_WORD_NUMBERS = {
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
}
_SIMPLE_WORD_NUMBERS = {**_WORD_NUMBERS, **_ORDINAL_WORD_NUMBERS, **_TENS_WORD_NUMBERS, **_ORDINAL_TENS_WORD_NUMBERS}


def _try_parse_simple_word_num(s):
    words = [word.lower() for word in re.split(r"[\s-]+", s.strip()) if word]
    if not words:
        return None
    if len(words) == 1:
        return _SIMPLE_WORD_NUMBERS.get(words[0])
    if len(words) == 2 and words[0] in _TENS_WORD_NUMBERS:
        ones = {**_WORD_NUMBERS, **_ORDINAL_WORD_NUMBERS}
        value = ones.get(words[1])
        if value is not None and 0 < value < 10:
            return _TENS_WORD_NUMBERS[words[0]] + value
    return None

def _try_parse_num(s):
    """Parse s as a number. Tries digit conversion first, then word2number if available."""
    s = s.strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    ordinal_match = _ORDINAL_DIGIT_RE.match(s)
    if ordinal_match:
        return float(ordinal_match.group(1))
    fallback = _try_parse_simple_word_num(s)
    if fallback is not None:
        return fallback
    try:
        from word2number import w2n
        return int(w2n.word_to_num(s))
    except Exception:
        pass
    return None


def _try_parse_word_dash_range(candidate):
    """Parse word-number ranges joined by a dash without breaking compounds.

    Examples:
        'first-third' -> (1, 3)
        'twenty-first–twenty-fourth' -> (21, 24)

    ASCII hyphens are also used inside compound ordinals, so only accept a split
    when both sides parse and the range moves forward.
    """
    for sep in ("\u2013", "\u2014", "-"):
        if sep not in candidate:
            continue
        for idx in range(1, len(candidate) - 1):
            if candidate[idx] != sep:
                continue
            left = candidate[:idx]
            right = candidate[idx + 1:]
            start = _try_parse_num(left)
            end = _try_parse_num(right)
            if start is not None and end is not None and end > start:
                return (start, end)
    return None

def _strip_header_markup(line):
    """Remove common markdown decoration around LLM block headers."""
    line = line.strip()
    line = re.sub(r"^#{1,6}\s+", "", line).strip()
    for marker in ("**", "__", "*", "_"):
        if line.startswith(marker) and line.endswith(marker) and len(line) > 2 * len(marker):
            return line[len(marker):-len(marker)].strip()
    return line


def _parse_header(line):
    """Return (start, end_or_None) if line is a block segment header, else None.

    Valid format: one or more prefix words, followed by a number (digit or word form),
    optional range, then colon at end of line. Matching is case-insensitive.

    Examples:
        'Scene 1:'          -> (1, None)
        'My Scene 3:'       -> (3, None)
        'Shot 2-4:'         -> (2, 4)
        'Part eleven:'      -> (11, None)    [requires word2number]
        'segment twenty:'   -> (20, None)   [requires word2number]
    """
    line = _strip_header_markup(line)
    if not line.endswith(':'):
        return None
    body = line[:-1].rstrip()
    tokens = body.split()
    # Need at least 2 tokens: at least one prefix word + one number token
    if len(tokens) < 2:
        return None
    # Try bracketed header markers common in LLM outputs: "Scene [0-25]:".
    m = _BRACKETED_HEADER_TAG_RE.search(body)
    if m and body[:m.start()].strip():
        start = _try_parse_num(m.group(1))
        end = _try_parse_num(m.group(2)) if m.group(2) else None
        if start is not None and (end is not None or m.group(2) is None):
            return (start, end)
    # Try digit range at end: "Scene 2-4:"
    m = _DIGIT_RANGE_TAIL_RE.search(body)
    if m and body[:m.start()].strip():
        start = _try_parse_num(m.group(1))
        end = _try_parse_num(m.group(2))
        if start is not None and end is not None:
            return (start, end)
    # Try word-number ranges at the tail: "Scene one to three:".
    # Keep at least one prefix token so bare "one to three:" is not a header.
    max_range_tokens = min(7, len(tokens) - 1)
    for n in range(max_range_tokens, 2, -1):
        candidate_tokens = tokens[-n:]
        for sep_idx, token in enumerate(candidate_tokens):
            if token.lower() not in _RANGE_WORDS:
                continue
            left = " ".join(candidate_tokens[:sep_idx])
            right = " ".join(candidate_tokens[sep_idx + 1:])
            start = _try_parse_num(left)
            end = _try_parse_num(right)
            if start is not None and end is not None:
                return (start, end)
    # Try dash-joined word ranges at the tail: "Scene first-third:" or
    # "Beat twenty-first–twenty-fourth:". Run this after explicit word
    # separators so ordinary compound ordinals still parse as single markers.
    for n in range(max_range_tokens, 0, -1):
        candidate = " ".join(tokens[-n:])
        parsed_range = _try_parse_word_dash_range(candidate)
        if parsed_range is not None:
            return parsed_range
    # Try 1..N tail tokens as word-or-digit number (longest candidate first)
    # Keep at least one prefix token so bare "1:" is not matched as a header
    max_num_tokens = min(4, len(tokens) - 1)
    for n in range(max_num_tokens, 0, -1):
        candidate = " ".join(tokens[-n:])
        val = _try_parse_num(candidate)
        if val is not None:
            return (val, None)
    return None

def _positive_range_weight(start, end):
    """Return a positive range span, or None for reversed/zero-width ranges."""
    if end is None:
        return None
    weight = end - start
    return weight if weight > 0 else None


def _extract_inline_tag(text):
    """Extract first [n] or [n-m] weight tag from text.
    Returns (clean_text, weight_or_None). Tag is stripped from text.
    """
    m = _INLINE_TAG_RE.search(text)
    if not m:
        return text.strip(), None
    val1 = float(m.group(1))
    val2 = float(m.group(2)) if m.group(2) else None
    weight = _positive_range_weight(val1, val2) if val2 is not None else val1
    clean = _INLINE_TAG_RE.sub("", text, count=1).strip()
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return clean, weight

def _strip_block_body_markers(text):
    """Strip common LLM markdown/list markers from block segment bodies.

    LLMs often format segmented prompts as:
        Scene 1:\n- camera pans left
        Scene 2:\n1. subject turns

    Headers already carry the sequence/range metadata, so leading bullets or
    numbered-list markers in the body are formatting noise rather than prompt
    text. Strip only line-leading markers to avoid changing meaningful hyphens
    inside the actual prompt.
    """
    lines = []
    for line in text.splitlines():
        lines.append(_BODY_LIST_MARKER_RE.sub("", line).strip())
    return "\n".join(line for line in lines if line).strip()

def _parse_inline_syntax(text):
    """Parse pipe-separated inline syntax with optional [n-m] weight tags.

    Syntax examples:
        'one | two | three'                       -> equal weights
        'one [0-50] | two [50-150] | three [150]' -> proportional weights
    """
    segments = []
    for part in text.split('|'):
        clean, weight = _extract_inline_tag(part)
        if clean:
            segments.append({"text": clean, "weight": weight if weight is not None else 1.0})
    return segments

def _parse_block_syntax(text):
    """Parse block header syntax where each segment is preceded by a header line.

    Header format: any words followed by a number (or word-number) and a colon
    on its own line. Optional [n-m] inline tag in body overrides header weight.

    Syntax examples:
        'Scene 1:\\ntext\\nScene 2:\\ntext'
        'My Part 3-6:\\ntext'          -> weight = 6-3 = 3
        'segment eleven:\\ntext'       -> weight = 1.0 (single number = sequence marker)
    """
    lines = text.splitlines(keepends=True)
    raw_segments = []
    current_header = None
    current_body = []
    for line in lines:
        h = _parse_header(line)
        if h is not None:
            if current_header is not None:
                raw_segments.append((current_header, "".join(current_body)))
            current_header = h
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_header is not None:
        raw_segments.append((current_header, "".join(current_body)))
    segments = []
    for header, body in raw_segments:
        body = _strip_block_body_markers(body)
        clean, inline_weight = _extract_inline_tag(body)
        if not clean:
            continue
        if inline_weight is not None:
            weight = inline_weight
        elif header is not None:
            start, end = header
            # Range header: weight is proportional span. Single number or
            # reversed/zero-width range: equal weight rather than a negative
            # or zero segment length.
            weight = _positive_range_weight(start, end) if end is not None else 1.0
            if weight is None:
                weight = 1.0
        else:
            weight = 1.0
        segments.append({"text": clean, "weight": weight})
    return segments

def parse_smart_prompt(text):
    """Parse smart_prompt text into a list of {"text": str, "weight": float} dicts.

    Detects syntax automatically:

    --- Inline (newline-agnostic) ---
    Segments separated by | with optional [n-m] proportional weight tags.
        'man walks | man runs | man jumps'
        'man walks [0-50] | man runs [50-150] | man jumps [150-200]'

    --- Block (newline-specific) ---
    Segments preceded by a header line: any words + number + colon on its own line.
    The number (or range) is stripped and used only for weight. Ordinal numbers
    like 'Scene 1:' are sequence markers (equal weight). Ranges like 'Scene 1-3:'
    assign proportional weight (3-1=2). Word-form numbers require word2number package.
        'Scene 1:\\nman walks\\nScene 2:\\nman runs'
        'My Shot 1-3:\\nman walks\\nMy Shot 3-7:\\nman runs'
        'segment eleven:\\nman walks'   (requires: pip install word2number)

    All syntax markers are fully stripped before text is returned.
    """
    lines = text.splitlines()
    has_blocks = any(_parse_header(line) is not None for line in lines)
    if has_blocks:
        return _parse_block_syntax(text)
    return _parse_inline_syntax(text)
