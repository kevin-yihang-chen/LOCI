"""
LOCI online phrase matcher (Appendix C.1 / C.2, Algorithm 1 lines 4-10).

Implements, in order:
  N(.)                  normalization applied before matching
  CurrentClause(.)      clause extraction (delegated to scope_gate)
  LongestSuffixMatch(.) alias suffix match with optional preposed modifier

All matching is performed on detokenized text, never on raw subword IDs, so a
partial BPE piece cannot trigger before it completes a full phrase.

The machine-readable lexicon in lexicon/loci_lexicon.json is authoritative.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
LEXICON_PATH = os.path.join(_HERE, "lexicon", "loci_lexicon.json")

# Only trailing punctuation is stripped for matching: . , ; : ) ]
_TRAILING_PUNCT = ".,;:)]"
_HYPHEN_VARIANTS = dict.fromkeys(map(ord, "‐‑‒–—―-"), " ")


def normalize_surface(text: str) -> str:
    """Strip sentencepiece/BPE markers and collapse whitespace."""
    if text is None:
        return ""
    text = str(text).replace("Ġ", " ").replace("▁", " ").strip()
    return " ".join(text.split())


def decode_token_surface(tokenizer, token_id: int) -> str:
    surface = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    surface = normalize_surface(surface)
    if surface:
        return surface
    return normalize_surface(tokenizer.convert_ids_to_tokens(int(token_id)))


def normalize_match_text(text: str) -> str:
    """N(.): lowercase, NFKC, collapse whitespace, hyphens -> space, strip trailing punctuation."""
    text = unicodedata.normalize("NFKC", normalize_surface(text)).lower()
    text = text.translate(_HYPHEN_VARIANTS)
    text = " ".join(text.split())
    return text.rstrip(_TRAILING_PUNCT).strip()


def load_finding_lexicon(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the authoritative lexicon bundle (findings + modifiers + cue sets)."""
    path = path or LEXICON_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"LOCI lexicon not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):  # bare list of findings
        raw = {"findings": raw}
    if not raw.get("findings"):
        raise ValueError("LOCI lexicon contains no findings.")
    return raw


class FindingLexiconMatcher:
    """
    Does `prefix + candidate` END the current clause with a finding alias,
    optionally preceded by exactly one modifier chunk?

    Returns the canonical finding, the matched span, and the left context
    before the span (which the scope gate labels).
    """

    def __init__(self, tokenizer, lexicon=None, lookback_tokens: int = 64):
        self.tokenizer = tokenizer
        self.lookback_tokens = int(lookback_tokens)

        bundle = lexicon if isinstance(lexicon, dict) else load_finding_lexicon()
        if isinstance(lexicon, list):
            bundle = {"findings": lexicon, "modifiers": bundle.get("modifiers", [])}

        self.entries: List[Dict[str, Any]] = []
        self.flat_aliases: List[Tuple[str, str]] = []
        for item in bundle["findings"]:
            canonical = normalize_match_text(item["name"])
            if not canonical:
                continue
            aliases = {canonical}
            for a in item.get("aliases", []):
                na = normalize_match_text(a)
                if na:
                    aliases.add(na)
            aliases = sorted(aliases, key=lambda s: (-len(s), s))
            self.entries.append({"canonical": canonical, "aliases": aliases})
            for alias in aliases:
                self.flat_aliases.append((alias, canonical))
        # Longest alias first: "pleural effusion" must beat "effusion".
        self.flat_aliases.sort(key=lambda x: (-len(x[0]), x[0]))

        self.modifiers = sorted(
            {normalize_match_text(m) for m in bundle.get("modifiers", []) if m},
            key=lambda s: (-len(s), s),
        )

    # -- text assembly ------------------------------------------------
    def _decode_recent_plus_candidate(self, generated_ids, prompt_len, candidate_token_id):
        start = max(prompt_len, generated_ids.shape[1] - self.lookback_tokens)
        tail_ids = generated_ids[0, start:].detach().cpu().tolist()
        tail_ids.append(int(candidate_token_id))
        return self.tokenizer.decode(tail_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def _current_clause(self, text: str) -> str:
        from scope_gate import current_clause  # same directory
        return normalize_match_text(current_clause(text))

    # -- LongestSuffixMatch -------------------------------------------
    def _strip_one_modifier(self, left: str) -> str:
        """Remove at most ONE preposed modifier chunk from the end of the left context."""
        left = left.rstrip()
        for mod in self.modifiers:
            if left.endswith(mod):
                head = left[: len(left) - len(mod)]
                if not head or head[-1].isspace():
                    return head.rstrip()
        return left

    def longest_suffix_match(self, clause: str) -> Optional[Dict[str, Any]]:
        for alias, canonical in self.flat_aliases:
            if not clause.endswith(alias):
                continue
            begin = len(clause) - len(alias)
            if begin != 0 and not clause[begin - 1].isspace():
                continue  # must land on a word boundary
            left_raw = clause[:begin]
            return {
                "canonical": canonical,          # modifiers stripped by construction
                "matched_alias": alias,
                "span": (begin, len(clause)),
                "left_context": self._strip_one_modifier(left_raw),
            }
        return None

    def match_next(self, generated_ids, prompt_len, candidate_token_id):
        candidate_text = self._decode_recent_plus_candidate(
            generated_ids, prompt_len, candidate_token_id)
        clause = self._current_clause(candidate_text)
        if not clause:
            return None
        hit = self.longest_suffix_match(clause)
        if hit is None:
            return None
        hit["clause_with_candidate"] = clause
        hit["candidate_text"] = candidate_text
        return hit
