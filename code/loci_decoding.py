"""
LOCI candidate scoring primitives (Section 3.1 / 3.4, Algorithm 1).

Pure text + logits observation: nothing here touches the model or the
latent intervention, so it is safe to import into any decoding loop.

    topk_decode / get_topk_candidates   base top-K candidate set C_t
    collect_topk_finding_groups         Algorithm 1: match -> scope-gate -> group
    apply_repetition_penalty            shared with standard decoding
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from phrase_matcher import FindingLexiconMatcher, decode_token_surface
from scope_gate import POS, scope_label


def topk_decode(logits: torch.Tensor, tokenizer, k: int = 5) -> list:
    """Return [{'id', 'surface', 'logit', 'prob'}, ...] for logits[0]."""
    flat = logits[0].float()
    probs = F.softmax(flat, dim=-1)
    k = min(int(k), flat.shape[-1])
    top_vals, top_ids = torch.topk(flat, k=k)
    out = []
    for tid, lv in zip(top_ids.tolist(), top_vals.tolist()):
        out.append({
            "id": int(tid),
            "surface": decode_token_surface(tokenizer, int(tid)),
            "logit": float(lv),
            "prob": float(probs[tid].item()),
        })
    return out


def get_topk_candidates(logits: torch.Tensor, tokenizer, k: int = 8) -> List[Dict[str, Any]]:
    """Return top-k candidate tokens from logits[0], ranked."""
    flat = logits[0].float()
    k = min(int(k), flat.shape[-1])
    top_vals, top_ids = torch.topk(flat, k=k)

    out = []
    for rank, (tid, lv) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
        out.append({
            "rank": rank,
            "id": int(tid),
            "surface": decode_token_surface(tokenizer, int(tid)),
            "base_logit": float(lv),
        })
    return out


def collect_topk_finding_groups(
    logits: torch.Tensor,
    tokenizer,
    generated_ids: torch.Tensor,
    prompt_len: int,
    finding_matcher: Optional["FindingLexiconMatcher"],
    candidate_topk: int,
    max_groups: Optional[int] = None,
):
    """
    Algorithm 1, lines 3-19:

    1) take the base top-K candidate set C_t
    2) for each candidate, check whether appending it completes a finding
       phrase at the end of the current clause (LongestSuffixMatch)
    3) label the left context before the matched span; SKIP anything that is
       not POS -- negated and uncertain matches are left to the base decoder
    4) de-duplicate by canonical finding, sort by best base logit, and keep
       at most `max_groups` (G_max) groups

    Returns (topk_candidates_enriched, finding_groups).
    """
    topk_candidates = get_topk_candidates(logits, tokenizer, k=candidate_topk)

    if finding_matcher is None:
        return topk_candidates, []

    groups: Dict[str, Dict[str, Any]] = {}
    enriched = []

    for cand in topk_candidates:
        match = finding_matcher.match_next(
            generated_ids=generated_ids,
            prompt_len=prompt_len,
            candidate_token_id=cand["id"],
        )

        scope = None if match is None else scope_label(match["left_context"])

        item = dict(cand)
        item["scope"] = scope
        item["matched_finding"] = None if match is None else match["canonical"]
        item["matched_alias"] = None if match is None else match["matched_alias"]
        item["clause_with_candidate"] = None if match is None else match["clause_with_candidate"]
        enriched.append(item)

        # Scope gate: only positively asserted completions are intervened on.
        if match is None or scope != POS:
            continue

        canonical = match["canonical"]
        if canonical not in groups:
            groups[canonical] = {
                "canonical": canonical,
                "matched_aliases": set(),
                "members": [],
                "best_candidate_id": cand["id"],
                "best_candidate_surface": cand["surface"],
                "best_base_logit": float(cand["base_logit"]),
                "example_clause": match["clause_with_candidate"],
            }

        g = groups[canonical]
        g["matched_aliases"].add(match["matched_alias"])
        g["members"].append({
            "rank": cand["rank"],
            "id": cand["id"],
            "surface": cand["surface"],
            "base_logit": float(cand["base_logit"]),
            "matched_alias": match["matched_alias"],
        })

        if float(cand["base_logit"]) > g["best_base_logit"]:
            g["best_candidate_id"] = cand["id"]
            g["best_candidate_surface"] = cand["surface"]
            g["best_base_logit"] = float(cand["base_logit"])
            g["example_clause"] = match["clause_with_candidate"]

    finding_groups = []
    for g in groups.values():
        g["matched_aliases"] = sorted(g["matched_aliases"])
        g["members"] = sorted(g["members"], key=lambda x: (x["rank"], x["id"]))
        finding_groups.append(g)

    finding_groups.sort(key=lambda x: (-x["best_base_logit"], x["canonical"]))
    if max_groups is not None and max_groups > 0:
        finding_groups = finding_groups[:int(max_groups)]
    return enriched, finding_groups

def apply_repetition_penalty(logits: torch.Tensor, prev_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, prev_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, prev_ids, score)
