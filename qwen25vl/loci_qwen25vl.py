"""
Qwen2.5-VL LOCI (visual-only, same-layer prototype)

Core design:
1) Mid-layer concept erasure is applied only at the visual positions of image_token; text-side hidden states are untouched.
2) Concept vector = hidden state of the trigger token's surface at the same layer (default 14).
3) Base decoding uses incremental KV-cache forwards; on a trigger an extra full forward with the hook yields the counterfactual logits.
4) Inputs are prepared with the official processor.apply_chat_template(..., tokenize=True, return_dict=True).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Data/hf_cache"))

import argparse
import datetime
import gc
import json
import sys
import traceback
from typing import Iterable, Optional, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ------------------------------------------------------------------
# Shared LOCI implementation (Appendix C.2).  <repo>/code is placed on
# sys.path directly rather than imported as a package, because a top-level
# package named `code` would shadow the stdlib module of the same name.
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, "..", "code"))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from phrase_matcher import (
    FindingLexiconMatcher, decode_token_surface, load_finding_lexicon,
    normalize_surface,
)
from scope_gate import SimplePolarityScopeTracker
from loci_decoding import (
    apply_repetition_penalty, collect_topk_finding_groups, topk_decode,
)


# =========================
# Logging
# =========================
class TeeLogger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =========================
# Utils
# =========================
def parse_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    name = dtype_name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def build_id_tensor(ids: Iterable[int], vocab_size: int, device: torch.device) -> Optional[torch.Tensor]:
    cleaned = sorted({int(x) for x in ids if 0 <= int(x) < vocab_size})
    if not cleaned:
        return None
    return torch.tensor(cleaned, device=device, dtype=torch.long)


def load_trigger_ids(path: str) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trigger file not found: {path}")
    raw = torch.load(path, map_location="cpu")

    if torch.is_tensor(raw):
        return [int(x) for x in raw.view(-1).tolist()]
    if isinstance(raw, (list, tuple, set)):
        return [int(x) for x in raw]
    raise TypeError(f"Unsupported trigger file type: {type(raw)}")




def get_bad_token_ids(processor):
    bad_words = [
        "however", "However", "AI", "language model", "consult",
        "healthcare", "professional", "disclaimer", "remember",
        "evaluate", "physician", "doctor", "ai", "an", "am",
        "important", "note", "rule", "out", "monitoring", "practitioner",
        "APPROVAL", "approve", "provider", "contact"
    ]
    bad_token_ids = set()
    for word in bad_words:
        ids_with_space = processor.tokenizer.encode(" " + word, add_special_tokens=False)
        ids_without_space = processor.tokenizer.encode(word, add_special_tokens=False)
        bad_token_ids.update(ids_with_space)
        bad_token_ids.update(ids_without_space)
    return list(bad_token_ids)


def get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError("Cannot locate text model. Expected model.model.language_model or model.language_model.")


def get_text_layers(model):
    text_model = get_text_model(model)
    if hasattr(text_model, "layers"):
        return text_model.layers
    raise AttributeError("Cannot locate decoder layers on the text model.")


def normalize_layer_idx(layer_idx: int, num_layers: int) -> int:
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"Invalid erasure_layer={layer_idx} for num_layers={num_layers}")
    return layer_idx


def get_eos_token_ids(model) -> Set[int]:
    eos = None
    if hasattr(model, "generation_config") and getattr(model.generation_config, "eos_token_id", None) is not None:
        eos = model.generation_config.eos_token_id
    elif getattr(model.config, "eos_token_id", None) is not None:
        eos = model.config.eos_token_id

    if eos is None:
        return set()
    if isinstance(eos, int):
        return {int(eos)}
    return {int(x) for x in eos}



# =========================



def build_visual_token_mask(current_input_ids: torch.Tensor, image_token_id: int, prompt_len: int) -> torch.Tensor:
    """
    Keep only the image_token positions from the original prompt; never the newly generated tokens.
    """
    mask = current_input_ids.eq(image_token_id)
    if current_input_ids.shape[1] > prompt_len:
        mask[:, prompt_len:] = False
    return mask

# =========================




# =========================
# Same-layer concept prototype
# =========================
class SameLayerConceptBank:
    def __init__(self, model, tokenizer, layer_idx: int, pooling: str = "last"):
        self.model = model
        self.tokenizer = tokenizer
        self.text_model = get_text_model(model)
        self.layers = get_text_layers(model)
        self.layer_idx = normalize_layer_idx(layer_idx, len(self.layers))
        self.layer = self.layers[self.layer_idx]
        self.pooling = pooling
        self.device = next(model.parameters()).device
        self.cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def get(self, phrase: str, fallback_token_id: Optional[int] = None) -> torch.Tensor:
        phrase = normalize_surface(phrase)
        cache_key = f"{self.layer_idx}::{self.pooling}::{phrase.lower()}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        vec = None

        if phrase:
            probe = self.tokenizer(phrase, add_special_tokens=False, return_tensors="pt")
            input_ids = probe.input_ids.to(self.device)

            if input_ids.numel() > 0:
                captured = {}

                def hook_fn(module, inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured["hidden"] = hidden.detach()

                handle = self.layer.register_forward_hook(hook_fn)
                try:
                    _ = self.text_model(input_ids=input_ids, use_cache=False)
                finally:
                    handle.remove()

                hidden = captured.get("hidden", None)
                if hidden is not None:
                    if self.pooling == "mean":
                        vec = hidden.mean(dim=1).squeeze(0)
                    else:
                        vec = hidden[:, -1, :].squeeze(0)

        if vec is None:
            if fallback_token_id is None:
                raise ValueError(f"Cannot build concept vector for phrase='{phrase}' and no fallback token id was provided.")
            vec = self.model.get_input_embeddings().weight[int(fallback_token_id)]

        vec = safe_l2_normalize(vec.float(), dim=-1)
        self.cache[cache_key] = vec
        return vec


# =========================
# Visual-only mid-layer erasure hook
# =========================
def register_visual_only_erasure_hook(
    layer,
    concept_vector: torch.Tensor,
    visual_mask: torch.Tensor,
    erasure_scale: float,
    stats_out: Optional[dict] = None,
):
    """
    Register a forward hook that subtracts the projection of layer output onto
    `concept_vector` for the visual positions only.

    If `stats_out` is provided, it is populated (in-place) with diagnostic stats
    about the erasure magnitude over the visual positions.
    """
    u = safe_l2_normalize(concept_vector.float(), dim=-1)
    visual_mask = visual_mask.bool()

    def hook_fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        tail = output[1:] if isinstance(output, tuple) else None

        h_f32 = h.float()
        local_u = u.to(h_f32.device)
        local_mask_bool = visual_mask.to(h_f32.device)
        local_mask = local_mask_bool.unsqueeze(-1).to(h_f32.dtype)

        scalar = torch.matmul(h_f32, local_u)                       # [B, L]
        proj = scalar.unsqueeze(-1) * local_u.view(1, 1, -1)         # [B, L, H]

        h_erased = h_f32 - erasure_scale * proj * local_mask
        h_erased = h_erased.to(h.dtype)

        if stats_out is not None:
            visual_proj = proj * local_mask                           # zero out non-visual
            visual_norms = visual_proj.norm(dim=-1)                   # [B, L]
            n_visual = int(local_mask_bool.sum().item())
            if n_visual > 0:
                vals = visual_norms[local_mask_bool]
                h_norms_visual = h_f32.norm(dim=-1)[local_mask_bool]
                stats_out.update({
                    "n_visual_positions": n_visual,
                    "proj_norm_mean": float(vals.mean().item()),
                    "proj_norm_max": float(vals.max().item()),
                    "proj_norm_min": float(vals.min().item()),
                    "h_norm_visual_mean": float(h_norms_visual.mean().item()),
                    "relative_erasure_ratio": float(
                        (erasure_scale * vals / h_norms_visual.clamp_min(1e-8)).mean().item()
                    ),
                })
            else:
                stats_out.update({"n_visual_positions": 0})

        return h_erased if tail is None else (h_erased,) + tail

    return layer.register_forward_hook(hook_fn)


# =========================

# =========================
# Main generation
# =========================
@torch.inference_mode()
def generate_loci_visual_only(
    model,
    processor,
    inputs,
    trigger_ids_set: Set[int],
    trigger_ids_tensor: Optional[torch.Tensor],
    concept_bank: SameLayerConceptBank,
    finding_matcher: Optional[FindingLexiconMatcher] = None,
    max_new_tokens: int = 150,
    alpha: float = 0.3,
    repetition_penalty: float = 1.2,
    alpha_apc: float = 0.3,
    bias_weight: float = 0.0,
    erasure_scale: float = 1.0,
    bad_token_ids_tensor: Optional[torch.Tensor] = None,
    candidate_topk: int = 8,
    max_findings_per_step: int = 3,
    verbose: bool = True,
    debug_log_fn=None,
):
    device = next(model.parameters()).device
    image_token_id = int(model.config.image_token_id)
    eos_token_ids = get_eos_token_ids(model)

    generated_ids = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask", torch.ones_like(generated_ids)).clone()
    prompt_len = inputs["input_ids"].shape[1]

    scope_tracker = SimplePolarityScopeTracker(
        tokenizer=processor.tokenizer,
        prompt_len=prompt_len,
        lookback_tokens=96,
    )

    fixed_mm_inputs = {}
    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]:
        if key in inputs and inputs[key] is not None:
            fixed_mm_inputs[key] = inputs[key]

    init_mm_token_type_ids = (
        inputs["mm_token_type_ids"].clone()
        if "mm_token_type_ids" in inputs and inputs["mm_token_type_ids"] is not None
        else None
    )

    def make_mm_token_type_ids(seq_len: int) -> Optional[torch.Tensor]:
        if init_mm_token_type_ids is None:
            return None
        cur_len = init_mm_token_type_ids.shape[1]
        if seq_len <= cur_len:
            return init_mm_token_type_ids[:, :seq_len]
        pad_len = seq_len - cur_len
        pad = torch.zeros(
            (init_mm_token_type_ids.shape[0], pad_len),
            device=init_mm_token_type_ids.device,
            dtype=init_mm_token_type_ids.dtype,
        )
        return torch.cat([init_mm_token_type_ids, pad], dim=-1)

    def build_full_inputs(cur_generated_ids: torch.Tensor, cur_attention_mask: torch.Tensor) -> dict:
        """
        For the counterfactual full forward:
        feed the complete prefix plus the multimodal inputs every time.
        """
        out = {
            "input_ids": cur_generated_ids,
            "attention_mask": cur_attention_mask,
            **fixed_mm_inputs,
        }
        cur_mm_type = make_mm_token_type_ids(cur_generated_ids.shape[1])
        if cur_mm_type is not None:
            out["mm_token_type_ids"] = cur_mm_type
        return out

    def run_base_step_with_cache(
        cur_generated_ids: torch.Tensor,
        cur_attention_mask: torch.Tensor,
        past_key_values=None,
        rope_deltas=None,
    ):
        """
        Base path with KV cache. Mirrors the working pattern in loci_iuxray.py:

        - Prefill (past_kv is None): full prompt + multimodal inputs +
          mm_token_type_ids. Build the cache.
        - Incremental (past_kv is not None): pass ONLY the new token and
          past_key_values. Do NOT pass attention_mask, rope_deltas,
          mm_token_type_ids, or position_ids — the model will derive them
          internally from past_kv, and passing them externally causes the
          key/value length mismatch or garbage output we saw.
        """
        if past_key_values is None:
            # Prefill — include image + mm_token_type_ids.
            model_inputs = {
                "input_ids": cur_generated_ids,
                "attention_mask": cur_attention_mask,
                **fixed_mm_inputs,
            }
            cur_mm_type = make_mm_token_type_ids(cur_generated_ids.shape[1])
            if cur_mm_type is not None:
                model_inputs["mm_token_type_ids"] = cur_mm_type
        else:
            # Incremental — minimal kwargs. Let Qwen2.5-VL handle position math.
            model_inputs = {
                "input_ids": cur_generated_ids[:, -1:],
                "past_key_values": past_key_values,
            }

        # logits_to_keep=1 trims the prefill's per-position logit tensor
        # (otherwise ~2.8 GiB at prompt_len ≈ 4800, vocab ≈ 152k).
        out = model(**model_inputs, use_cache=True, logits_to_keep=1)
        step_logits = out.logits[:, -1, :]
        next_past_key_values = out.past_key_values
        # We don't read rope_deltas because we don't pass it back; the model
        # carries that state inside past_key_values.
        del out
        return step_logits, next_past_key_values, None

    if verbose:
        print("\n🚀 Start LOCI decoding...")
        if init_mm_token_type_ids is None:
            print("⚠️ inputs has no 'mm_token_type_ids' — model will fall back to default position handling.")
        print(f"✅ Trigger mode: finding phrase top-k rerank | candidate_topk={candidate_topk} | max_findings_per_step={max_findings_per_step}")
        print("✅ Base decoding uses KV cache incremental forward.")
        print("✅ Counterfactual LOCI path still uses full forward.")
        if bias_weight > 0.0:
            print("⚠️ bias_weight is ignored in finding top-k rerank mode.")

    # -------------------------
    # 0) Initial prompt prefill
    # -------------------------
    past_key_values = None
    rope_deltas = None
    base_logits, past_key_values, rope_deltas = run_base_step_with_cache(
        cur_generated_ids=generated_ids,
        cur_attention_mask=attention_mask,
        past_key_values=past_key_values,
        rope_deltas=rope_deltas,
    )

    for step in range(max_new_tokens):
        scope_info = scope_tracker.infer(generated_ids)
        current_scope = scope_info["scope"]

        top1_id = int(torch.argmax(base_logits, dim=-1).item())
        top1_surface = decode_token_surface(processor.tokenizer, top1_id)

        # Base top-k candidates + finding grouping
        topk_candidates, finding_groups = collect_topk_finding_groups(
            logits=base_logits,
            tokenizer=processor.tokenizer,
            generated_ids=generated_ids,
            prompt_len=prompt_len,
            finding_matcher=finding_matcher,
            candidate_topk=candidate_topk,
        )

        # Start from the base logits and modify only the matched finding members
        final_logits = base_logits.clone()

        # APC is still computed on the base distribution
        base_probs = F.softmax(base_logits.float(), dim=-1)
        max_prob = base_probs.max(dim=-1, keepdim=True).values
        plausible_mask = base_probs >= (alpha_apc * max_prob)

        record = None
        if debug_log_fn is not None:
            record = {
                "step": step,
                "seq_len": int(generated_ids.shape[1]),
                "scope": current_scope,
                "scope_cue": scope_info["cue"],
                "scope_clause": scope_info["clause"],
                "scope_recent_text": scope_info["recent_text"],
                "trigger_mode": "finding_phrase_topk_rerank",
                "candidate_topk": int(candidate_topk),
                "max_findings_per_step": int(max_findings_per_step),
                "top1_id": top1_id,
                "top1_surface": top1_surface,
                "base_top5": topk_decode(base_logits, processor.tokenizer, k=5),
                "topk_candidates": topk_candidates,
                "matched_findings_topk": [
                    {
                        "canonical": g["canonical"],
                        "matched_aliases": g["matched_aliases"],
                        "best_candidate_id": g["best_candidate_id"],
                        "best_candidate_surface": g["best_candidate_surface"],
                        "best_base_logit": g["best_base_logit"],
                        "example_clause": g["example_clause"],
                        "members": g["members"],
                    }
                    for g in finding_groups
                ],
            }

        did_any_rerank = False
        reranked_findings_debug = []

        if alpha > 0.0 and finding_groups:
            if current_scope == "POS":
                groups_to_run = finding_groups[:max_findings_per_step]
                current_visual_mask = build_visual_token_mask(generated_ids, image_token_id, prompt_len)

                # Note: the counterfactual still requires a full forward
                full_base_inputs = build_full_inputs(generated_ids, attention_mask)

                for group in groups_to_run:
                    concept_phrase = group["canonical"]
                    fallback_token_id = int(group["best_candidate_id"])
                    concept_vector = concept_bank.get(concept_phrase, fallback_token_id=fallback_token_id)

                    erasure_stats: dict = {}
                    handle = register_visual_only_erasure_hook(
                        layer=concept_bank.layer,
                        concept_vector=concept_vector,
                        visual_mask=current_visual_mask,
                        erasure_scale=erasure_scale,
                        stats_out=erasure_stats,
                    )

                    try:
                        # logits_to_keep=1 avoids the full vocab × prompt_len
                        # logits tensor in the counter forward too.
                        cf_out = model(**full_base_inputs, use_cache=False, logits_to_keep=1)
                    finally:
                        handle.remove()

                    counter_logits = cf_out.logits[:, -1, :]
                    del cf_out

                    member_updates = []
                    for member in group["members"]:
                        cid = int(member["id"])
                        base_logit_c = float(base_logits[0, cid].item())
                        counter_logit_c = float(counter_logits[0, cid].item())
                        delta_c = base_logit_c - counter_logit_c
                        plausible = bool(plausible_mask[0, cid].item())

                        if plausible:
                            reranked_val_t = (1.0 + alpha) * base_logits[0, cid] - alpha * counter_logits[0, cid]
                            final_logits[0, cid] = reranked_val_t
                            reranked_val = float(reranked_val_t.item())
                        else:
                            final_logits[0, cid] = float("-inf")
                            reranked_val = float("-inf")

                        member_updates.append({
                            "rank": int(member["rank"]),
                            "id": cid,
                            "surface": member["surface"],
                            "matched_alias": member["matched_alias"],
                            "base_logit": base_logit_c,
                            "counter_logit": counter_logit_c,
                            "delta_base_minus_counter": delta_c,
                            "plausible_under_apc": plausible,
                            "reranked_logit": reranked_val,
                        })

                    did_any_rerank = True

                    if verbose:
                        proj_norm_mean = float(erasure_stats.get("proj_norm_mean", 0.0))
                        rel_erasure = float(erasure_stats.get("relative_erasure_ratio", 0.0))
                        n_visual = int(erasure_stats.get("n_visual_positions", 0))
                        member_str = ", ".join(
                            f"{m['surface']}@{m['rank']}"
                            for m in group["members"]
                        )
                        print(
                            f"\n[LOCI-RERANK] step={step} | scope={current_scope} "
                            f"| finding='{concept_phrase}' | aliases={group['matched_aliases']} "
                            f"| members=[{member_str}] | layer={concept_bank.layer_idx} "
                            f"| visual_pos={n_visual} | proj_norm_mean={proj_norm_mean:.4f} "
                            f"| rel_erasure={rel_erasure:.4f}"
                        )

                    reranked_findings_debug.append({
                        "canonical": concept_phrase,
                        "matched_aliases": group["matched_aliases"],
                        "best_candidate_id": int(group["best_candidate_id"]),
                        "best_candidate_surface": group["best_candidate_surface"],
                        "best_base_logit": float(group["best_base_logit"]),
                        "concept_vec_norm_pre_norm": float(concept_vector.norm().item()),
                        "erasure_layer": concept_bank.layer_idx,
                        "erasure_stats": erasure_stats,
                        "counter_top5": topk_decode(counter_logits, processor.tokenizer, k=5),
                        "member_updates": member_updates,
                    })

                    del counter_logits
                    # Counter forward at long prompt_len builds large transient
                    # activations; with max_findings_per_step ≥ 2 they pile up
                    # in the caching allocator and OOM on the next counter.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                if verbose:
                    blocked = ", ".join(g["canonical"] for g in finding_groups[:max_findings_per_step])
                    print(
                        f"\n[SCOPE-SKIP] step={step} | scope={current_scope} "
                        f"| cue={scope_info['cue']!r} | matched_findings_topk=[{blocked}]"
                    )

        # decoding heuristics after rerank
        if repetition_penalty != 1.0:
            final_logits = apply_repetition_penalty(final_logits, generated_ids, repetition_penalty)

        if bad_token_ids_tensor is not None and bad_token_ids_tensor.numel() > 0:
            final_logits[0, bad_token_ids_tensor] = float("-inf")

        if not torch.isfinite(final_logits).any():
            final_logits = base_logits.clone()
            if repetition_penalty != 1.0:
                final_logits = apply_repetition_penalty(final_logits, generated_ids, repetition_penalty)
            if bad_token_ids_tensor is not None and bad_token_ids_tensor.numel() > 0:
                final_logits[0, bad_token_ids_tensor] = float("-inf")

        next_token = torch.argmax(final_logits, dim=-1, keepdim=True)
        next_token_id = int(next_token.item())

        if record is not None:
            record["did_any_finding_rerank"] = bool(did_any_rerank)
            record["reranked_findings"] = reranked_findings_debug
            record["final_top5"] = topk_decode(final_logits, processor.tokenizer, k=5)
            record["selected_id"] = next_token_id
            record["selected_surface"] = decode_token_surface(processor.tokenizer, next_token_id)
            record["argmax_changed_by_loci"] = (next_token_id != top1_id) and bool(did_any_rerank)
            debug_log_fn(record)

        if next_token_id in eos_token_ids:
            if verbose:
                print("\n🛑 EOS reached.")
            break

        # append selected token
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        new_mask_col = torch.ones(
            (attention_mask.shape[0], 1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, new_mask_col], dim=-1)

        if verbose:
            token_text = processor.tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(token_text, end="", flush=True)

        # -------------------------------------
        # KV-cache incremental forward for NEXT step
        # -------------------------------------
        base_logits, past_key_values, rope_deltas = run_base_step_with_cache(
            cur_generated_ids=generated_ids,
            cur_attention_mask=attention_mask,
            past_key_values=past_key_values,
            rope_deltas=rope_deltas,
        )

        del base_probs, max_prob, plausible_mask, final_logits, next_token

    if verbose:
        print("\n✅ Decoding finished.\n")

    new_ids = generated_ids[:, prompt_len:].cpu()
    text = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Explicitly free the KV cache so the next sample starts with a clean GPU.
    # Locals would go out of scope on return, but cuda allocator is sticky;
    # explicit del + empty_cache reliably releases the largest tensors.
    del past_key_values, base_logits, generated_ids, attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


# =========================
# Args / Validation
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL LOCI (visual-only, same-layer prototype)")

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="/")
    parser.add_argument("--output-file", type=str, required=True)

    parser.add_argument("--trigger-file", type=str, default=None,
                        help="Legacy token-id whitelist. The finding-lexicon trigger path "
                             "does not consult it; kept only so old command lines keep working.")

    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--alpha-apc", type=float, default=0.3)
    parser.add_argument("--bias-weight", type=float, default=0.0)
    parser.add_argument("--erasure-scale", type=float, default=1.0)
    parser.add_argument("--erasure-layer", type=int, default=18)
    parser.add_argument("--concept-pooling", type=str, choices=["last", "mean"], default="last")

    parser.add_argument("--torch-dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug-jsonl", type=str, default=None,
                        help="If set, write per-step debug records (top-k logits, erasure stats, etc.) to this JSONL.")

    parser.add_argument("--finding-lexicon-json", type=str, default=None, help="Optional JSON lexicon for finding-level phrase triggers.")
    parser.add_argument("--finding-lookback-tokens", type=int, default=64, help="How many recently generated tokens to decode for phrase matching.")
    parser.add_argument(
        "--candidate-topk",
        type=int,
        default=8,
        help="Inspect base top-k token candidates for finding phrase completion.",
    )
    parser.add_argument(
        "--max-findings-per-step",
        type=int,
        default=3,
        help="Max distinct canonical findings to run counterfactual erasure for per decoding step.",
    )

    return parser.parse_args()


def validate_args(args):
    if args.alpha < 0:
        raise ValueError("--alpha must be >= 0")
    if args.erasure_scale < 0:
        raise ValueError("--erasure-scale must be >= 0")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty should be >= 1.0")
    if not (0.0 <= args.alpha_apc <= 1.0):
        raise ValueError("--alpha-apc should be in [0, 1]")
    if args.finding_lookback_tokens <= 0:
        raise ValueError("--finding-lookback-tokens must be > 0")
    if args.candidate_topk <= 0:
        raise ValueError("--candidate-topk must be > 0")
    if args.max_findings_per_step <= 0:
        raise ValueError("--max-findings-per-step must be > 0")


# =========================
# Main
# =========================
def main():
    args = parse_args()
    validate_args(args)

    log_file = (
        f"qwen25vl_loci_visual_only_part{args.chunk_idx}.txt"
        if args.num_chunks > 1
        else "qwen25vl_loci_visual_only.txt"
    )
    sys.stdout = TeeLogger(log_file)
    print(f"[{datetime.datetime.now()}] Starting Qwen2.5-VL LOCI (visual-only)...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = parse_torch_dtype(args.torch_dtype, device)

    print(f"Loading processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    model.eval()

    finding_lexicon = load_finding_lexicon(args.finding_lexicon_json)
    finding_matcher = FindingLexiconMatcher(
        tokenizer=processor.tokenizer,
        lexicon=finding_lexicon,
        lookback_tokens=args.finding_lookback_tokens,
    )
    print(f"Using finding-level trigger lexicon: {len(finding_matcher.entries)} entries")
    if args.finding_lexicon_json is not None:
        print(f"Finding lexicon file: {args.finding_lexicon_json}")

    layers = get_text_layers(model)
    actual_erasure_layer = normalize_layer_idx(args.erasure_layer, len(layers))
    concept_bank = SameLayerConceptBank(
        model=model,
        tokenizer=processor.tokenizer,
        layer_idx=actual_erasure_layer,
        pooling=args.concept_pooling,
    )
    print(f"Using erasure layer: {actual_erasure_layer} / {len(layers) - 1}")
    print(f"Concept pooling: {args.concept_pooling}")

    # Qwen2.5-VL stores vocab_size on config.text_config (nested) in transformers 5.x.
    # Needed for the bad-words tensor too, so it must be computed unconditionally.
    vocab_size = (
        getattr(model.config, "vocab_size", None)
        or model.config.get_text_config().vocab_size
    )

    if args.trigger_file is None:
        trigger_ids_tensor = None
        trigger_ids_set = set()
        print("No --trigger-file given; the finding-lexicon trigger path does not need one.")
    else:
        print(f"Loading triggers from {args.trigger_file}...")
        trigger_ids_list = load_trigger_ids(args.trigger_file)
        trigger_ids_tensor = build_id_tensor(trigger_ids_list, vocab_size, device)
        trigger_ids_set = set(trigger_ids_tensor.tolist()) if trigger_ids_tensor is not None else set()
        print(f"Loaded {len(trigger_ids_set)} valid trigger IDs (unused by the rerank path).")

    bad_token_ids_list = get_bad_token_ids(processor)
    bad_token_ids_tensor = build_id_tensor(bad_token_ids_list, vocab_size, device)
    print(f"Bad token IDs: {0 if bad_token_ids_tensor is None else bad_token_ids_tensor.numel()}")

    print(f"Loading dataset from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"Total samples: {len(all_data)}")

    if args.num_chunks > 1:
        data = all_data[args.chunk_idx::args.num_chunks]
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_part{args.chunk_idx}{ext}"
        print(f"Sharding: chunk {args.chunk_idx}/{args.num_chunks}, samples={len(data)}")
    else:
        data = all_data

    if args.limit is not None:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} samples for this chunk.")

    existing_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        existing_ids.add(str(obj["id"]))
                except Exception:
                    pass
        print(f"Resume mode: found {len(existing_ids)} existing results.")

    system_prompt = (
        "You are a highly experienced radiologist. "
        "Write a concise telegraphic chest X-ray report. "
        "CRITICAL INSTRUCTIONS:\n"
        "1. Use telegraphic clinical phrases.\n"
        "2. Link abnormalities to exact anatomical locations.\n"
        "3. No conversational filler, no AI disclaimer, no patient advice.\n"
        "4. Output only the report using standard section headers (FINDINGS, IMPRESSION)."
    )

    # Open optional debug JSONL writer
    debug_f = None
    if args.debug_jsonl:
        # If sharded, suffix the debug path too
        if args.num_chunks > 1:
            base_dbg, ext_dbg = os.path.splitext(args.debug_jsonl)
            args.debug_jsonl = f"{base_dbg}_part{args.chunk_idx}{ext_dbg}"
        os.makedirs(os.path.dirname(os.path.abspath(args.debug_jsonl)) or ".", exist_ok=True)
        debug_f = open(args.debug_jsonl, "a", encoding="utf-8")
        print(f"Writing per-step debug records to: {args.debug_jsonl}")

    def make_debug_log_fn(sample_id):
        if debug_f is None:
            return None
        def _log(record):
            record["sample_id"] = sample_id
            debug_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            debug_f.flush()
        return _log

    with open(args.output_file, "a", encoding="utf-8") as f_out:
        for idx, sample in tqdm(list(enumerate(data)), desc="LOCI"):
            sample_id = str(sample.get("id", idx))
            if sample_id in existing_ids:
                continue

            convs = sample.get("conversations", [])
            if not convs:
                continue

            raw_human_input = convs[0].get("value", "").replace("<image>", "").strip()
            gt_text = convs[1].get("value", "") if len(convs) > 1 else ""

            image_path = sample.get("image", None)
            if image_path is None:
                print(f"Skipping {sample_id}: missing image path.")
                continue
            if not os.path.isabs(image_path):
                image_path = os.path.join(args.image_folder, image_path)
            if not os.path.exists(image_path):
                print(f"Skipping {sample_id}: image not found at {image_path}")
                continue

            try:
                # Official multimodal chat-template path
                user_text = f"{raw_human_input}\nFINDINGS:"
                # NOTE: each message's "content" must be a list of content blocks for
                # Qwen2.5-VL apply_chat_template; a plain-string system content breaks it.
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)

                output_text = generate_loci_visual_only(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    trigger_ids_set=trigger_ids_set,
                    trigger_ids_tensor=trigger_ids_tensor,
                    concept_bank=concept_bank,
                    finding_matcher=finding_matcher,
                    max_new_tokens=args.max_tokens,
                    alpha=args.alpha,
                    repetition_penalty=args.repetition_penalty,
                    alpha_apc=args.alpha_apc,
                    bias_weight=args.bias_weight,
                    erasure_scale=args.erasure_scale,
                    bad_token_ids_tensor=bad_token_ids_tensor,
                    candidate_topk=args.candidate_topk,
                    max_findings_per_step=args.max_findings_per_step,
                    verbose=not args.quiet,
                    debug_log_fn=make_debug_log_fn(sample_id),
                )

                result = {
                    "id": sample_id,
                    "prompt": raw_human_input,
                    "text": output_text,
                    "gt": gt_text,
                    "model_id": "loci_qwen25vl_visual_only",
                    "base_model": args.model_path,
                    "meta": {
                        "erasure_layer": actual_erasure_layer,
                        "alpha": args.alpha,
                        "alpha_apc": args.alpha_apc,
                        "erasure_scale": args.erasure_scale,
                        "concept_pooling": args.concept_pooling,
                        "scope_aware_negation": True,
                        "trigger_mode": "finding_phrase_topk_rerank",
                        "finding_lexicon_entries": len(finding_matcher.entries),
                        "finding_lexicon_json": args.finding_lexicon_json,
                        "candidate_topk": args.candidate_topk,
                        "max_findings_per_step": args.max_findings_per_step,
                    }
                }

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError on sample {sample_id}: {e}")
                traceback.print_exc()

            finally:
                # Release per-sample tensors and clear CUDA fragmentation;
                # otherwise repeated full-forward passes accumulate cache and
                # cause OOM after a handful of samples (esp. on 24 GB GPUs).
                try:
                    del inputs
                except NameError:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\n✅ Done. Results saved to: {args.output_file}")
    if debug_f is not None:
        debug_f.close()
        print(f"   Debug records: {args.debug_jsonl}")


if __name__ == "__main__":
    main()