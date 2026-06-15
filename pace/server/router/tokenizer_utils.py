# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Router-side tokenization and detokenization utilities.

All text <-> token-ID conversion lives here.  The engine operates
exclusively on token IDs; the router owns encode, decode, and
stop-string detection.

Similar to vLLM's ``vllm/entrypoints/serve/tokenize/`` package, but
kept as a single utility module since PACE does not expose separate
``/tokenize`` or ``/detokenize`` HTTP endpoints.
"""

from typing import List, Optional, Union


_tokenizer = None


def init_tokenizer(tokenizer) -> None:
    """Set the module-level tokenizer. Called once at startup from frontend.py."""
    global _tokenizer
    _tokenizer = tokenizer


def get_tokenizer():
    """Return the shared tokenizer instance."""
    return _tokenizer


def normalize_prompts(
    prompt: Union[str, List[str], List[int], List[List[int]]],
    tokenizer=None,
) -> List[List[int]]:
    """Normalize the four OpenAI prompt formats into a list of token-ID lists.

    - ``str``              -> ``[tokenizer.encode(s)]``
    - ``list[str]``        -> ``[tokenizer.encode(s) for s in list]``
    - ``list[int]``        -> ``[list]``  (single prompt already tokenized)
    - ``list[list[int]]``  -> pass through
    """
    tok = tokenizer or _tokenizer
    if tok is None:
        raise RuntimeError("Tokenizer not initialized. Call init_tokenizer() first.")

    if isinstance(prompt, str):
        return [tok.encode(prompt)]

    if isinstance(prompt, list) and len(prompt) == 0:
        return [tok.encode("")]

    first = prompt[0]

    if isinstance(first, str):
        return [tok.encode(s) for s in prompt]

    if isinstance(first, int):
        return [list(prompt)]

    if isinstance(first, list):
        return [list(ids) for ids in prompt]

    return [tok.encode(str(prompt))]


def decode_token_ids(
    token_ids: List[int],
    skip_special_tokens: bool = True,
    tokenizer=None,
) -> str:
    """Decode a list of token IDs into text."""
    tok = tokenizer or _tokenizer
    if tok is None:
        raise RuntimeError("Tokenizer not initialized. Call init_tokenizer() first.")
    return tok.decode(token_ids, skip_special_tokens=skip_special_tokens)


def check_stop_strings(
    generated_text: str,
    stop_strings: Optional[List[str]],
) -> Optional[str]:
    """Check if any stop string appears in generated text.

    Returns the matched stop string, or None if no match.
    """
    if not stop_strings:
        return None
    for s in stop_strings:
        if s in generated_text:
            return s
    return None


def truncate_at_stop_string(
    text: str,
    stop_strings: Optional[List[str]],
) -> str:
    """Truncate text at the earliest occurrence of any stop string."""
    if not stop_strings:
        return text
    earliest_pos = len(text)
    for s in stop_strings:
        pos = text.find(s)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    return text[:earliest_pos]
