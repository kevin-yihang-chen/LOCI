"""
LOCI online scope gate (Appendix C.2, Algorithm 1 lines 11-15).

Assigns sigma_t(c) in {POS, NEG, UNC} to a matched finding from the LEFT
CONTEXT ONLY -- exactly what is available at phrase-completion time during
decoding. Only POS matches are intervened on.

Precedence:   POS-override/persistence  >  UNC  >  NEG  >  default POS

Sentence boundaries (. ? ! ;) and adversatives (but, however, although,
whereas, despite) reset scope. Commas and coordinators (and, or) do NOT,
because radiology reports routinely use coordinated negations such as
"no focal consolidation, edema, or pneumothorax".

Known online limitation: post-nominal cues that appear only AFTER the finding
span, e.g. "effusion is not excluded", are invisible at trigger time.
"""
from __future__ import annotations

import re
from typing import List

from phrase_matcher import load_finding_lexicon, normalize_match_text

POS, NEG, UNC = "POS", "NEG", "UNC"

_BUNDLE = load_finding_lexicon()
_CUES = _BUNDLE.get("scope_cues", {})
_BOUNDS = _BUNDLE.get("clause_boundaries", {})

POS_OVERRIDE_CUES: List[str] = list(_CUES.get("pos_override", []))
UNC_CUES: List[str] = list(_CUES.get("unc", []))
NEG_CUES: List[str] = list(_CUES.get("neg", []))

_ADVERSATIVES = _BOUNDS.get("adversative", ["but", "however", "although", "whereas", "despite"])
_SENTENCE = _BOUNDS.get("sentence", [".", "?", "!", ";"])

# CurrentClause(.): split on sentence boundaries or adversatives; keep the suffix.
CLAUSE_SPLIT_RE = re.compile(
    r"[\n" + re.escape("".join(_SENTENCE)) + r"]+|\b(?:" + "|".join(map(re.escape, _ADVERSATIVES)) + r")\b",
    re.IGNORECASE,
)


def _cue_regexes(cues):
    return [re.compile(r"\b" + re.escape(c) + r"\b", re.IGNORECASE) for c in cues]


_POS_RE = _cue_regexes(POS_OVERRIDE_CUES)
_UNC_RE = _cue_regexes(UNC_CUES)
_NEG_RE = _cue_regexes(NEG_CUES)


def current_clause(text: str) -> str:
    """Return the suffix after the most recent sentence or adversative boundary."""
    if not text:
        return ""
    parts = CLAUSE_SPLIT_RE.split(text)
    return parts[-1] if parts else text


def scope_label(left_context: str) -> str:
    """ScopeLabel(p): precedence-ordered cue matching on the normalized left context."""
    p = normalize_match_text(left_context)
    if not p:
        return POS
    if any(r.search(p) for r in _POS_RE):
        return POS
    if any(r.search(p) for r in _UNC_RE):
        return UNC
    if any(r.search(p) for r in _NEG_RE):
        return NEG
    return POS


class SimplePolarityScopeTracker:
    """
    Decoding-time wrapper: decode the recent prefix, take the current clause,
    and label it. Used when a caller wants the clause scope without having a
    matched span in hand.
    """

    CLAUSE_SPLIT_RE = CLAUSE_SPLIT_RE  # kept identical to the matcher's

    def __init__(self, tokenizer, prompt_len: int, lookback_tokens: int = 96):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.lookback_tokens = int(lookback_tokens)

    def _decode_recent_text(self, generated_ids) -> str:
        if generated_ids.shape[1] <= self.prompt_len:
            return ""
        start = max(self.prompt_len, generated_ids.shape[1] - self.lookback_tokens)
        tail_ids = generated_ids[0, start:].detach().cpu().tolist()
        return self.tokenizer.decode(tail_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def scope_of_current_clause(self, generated_ids) -> str:
        return scope_label(current_clause(self._decode_recent_text(generated_ids)))

    # backwards-compatible alias used by the backbone drivers
    def current_scope(self, generated_ids) -> str:
        return self.scope_of_current_clause(generated_ids)

    def infer(self, generated_ids) -> dict:
        """
        Label the current clause and report which cue fired.

        Precedence is the one from Appendix C.2 -- POS-override > UNC > NEG >
        default POS -- so a persistence cue such as "no change in" wins over the
        bare negation "no" inside the same clause.
        """
        recent_text = self._decode_recent_text(generated_ids)
        clause = current_clause(recent_text)
        p = normalize_match_text(clause)
        if not p:
            return {"scope": POS, "cue": None, "clause": "", "recent_text": recent_text}

        for label, regexes in ((POS, _POS_RE), (UNC, _UNC_RE), (NEG, _NEG_RE)):
            for rgx in regexes:
                m = rgx.search(p)
                if m:
                    return {"scope": label, "cue": m.group(0),
                            "clause": p, "recent_text": recent_text}

        return {"scope": POS, "cue": None, "clause": p, "recent_text": recent_text}
