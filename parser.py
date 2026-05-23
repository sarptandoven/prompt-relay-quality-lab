import re

_INLINE_TAG_RE = re.compile(r"\[([\d\.]+)(?:[:\-]([\d\.]+))?\]")
_DIGIT_RANGE_TAIL_RE = re.compile(r"([\d]+(?:\.\d+)?)\s*[-\u2013]\s*([\d]+(?:\.\d+)?)\s*$")

def _try_parse_num(s):
    """Parse s as integer. Tries digit conversion first, then word2number if available."""
    s = s.strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        pass
    try:
        from word2number import w2n
        return int(w2n.word_to_num(s))
    except Exception:
        pass
    return None

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
    line = line.strip()
    if not line.endswith(':'):
        return None
    body = line[:-1].rstrip()
    tokens = body.split()
    # Need at least 2 tokens: at least one prefix word + one number token
    if len(tokens) < 2:
        return None
    # Try digit range at end: "Scene 2-4:"
    m = _DIGIT_RANGE_TAIL_RE.search(body)
    if m and body[:m.start()].strip():
        start = _try_parse_num(m.group(1))
        end = _try_parse_num(m.group(2))
        if start is not None and end is not None:
            return (start, end)
    # Try 1..N tail tokens as word-or-digit number (longest candidate first)
    # Keep at least one prefix token so bare "1:" is not matched as a header
    max_num_tokens = min(4, len(tokens) - 1)
    for n in range(max_num_tokens, 0, -1):
        candidate = " ".join(tokens[-n:])
        val = _try_parse_num(candidate)
        if val is not None:
            return (val, None)
    return None

def _extract_inline_tag(text):
    """Extract first [n] or [n-m] weight tag from text.
    Returns (clean_text, weight_or_None). Tag is stripped from text.
    """
    m = _INLINE_TAG_RE.search(text)
    if not m:
        return text.strip(), None
    val1 = float(m.group(1))
    val2 = float(m.group(2)) if m.group(2) else None
    weight = (val2 - val1) if val2 is not None else val1
    clean = _INLINE_TAG_RE.sub("", text).strip()
    return clean, weight

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
            if current_body or current_header is not None:
                raw_segments.append((current_header, "".join(current_body)))
            current_header = h
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_header is not None:
        raw_segments.append((current_header, "".join(current_body)))
    segments = []
    for header, body in raw_segments:
        clean, inline_weight = _extract_inline_tag(body)
        if not clean:
            continue
        if inline_weight is not None:
            weight = inline_weight
        elif header is not None:
            start, end = header
            # Range header: weight is proportional span. Single number: equal weight.
            weight = (end - start) if end is not None else 1.0
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
