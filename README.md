# LOCI: Mitigating Hallucinations in Radiology Reports with Phrase-Triggered Counterfactual Decoding

Anonymous code release.

**LOCI** (**LO**cal **C**ounterfactual **I**ntervention) is a training-free,
scope-aware, phrase-triggered decoding method that uses concept removal as a
local counterfactual probe. It activates only when a candidate completes a
clinical finding phrase **in positive scope**. For the matched finding, LOCI
builds a text-derived direction at the intervention layer, subtracts its
projection from image-token hidden states in a temporary counterfactual branch,
and reranks only the plausible matched candidates by the induced relative change
in log probability. The edited branch is discarded after the current decoding
step; decoding is unchanged at every other step.

This repository contains the reference implementation together with the
**Qwen2.5-VL-7B-Instruct** backbone driver.

```
code/                       # the method
  phrase_matcher.py         #   N(.), LongestSuffixMatch, modifier stripping
  scope_gate.py             #   CurrentClause, ScopeLabel, POS/NEG/UNC precedence
  loci_decoding.py          #   top-K candidates, Algorithm 1 grouping, rerank primitives
  lexicon/loci_lexicon.json #   authoritative lexicon: 26 findings, modifiers, cue sets

qwen25vl/                   # backbone driver: prompt, visual mask, Eq. 1 hook, decode loop
eval/                       # RadGraph F1, CheXbert 14c/5c-F1, ROUGE-L
```

`code/` is put on `sys.path` directly rather than imported as a package: a
top-level package named `code` would shadow the standard-library module of the
same name.

## The method in three stages

**1. Phrase trigger** (`code/phrase_matcher.py`). At each step only the base
top-K candidate set is inspected. For each candidate the prefix is detokenized,
normalized, and cut to the current clause; the candidate fires only if that
clause then ends, on a word boundary, with a lexicon alias. Matching happens on
detokenized text, so a partial BPE piece never triggers before it completes a
full phrase. When several aliases match, the longest wins (`pleural effusion`
beats `effusion`); at most one preposed modifier is stripped when mapping the
surface span to its canonical finding (`left pleural effusion` → `pleural
effusion`).

**2. Scope gate** (`code/scope_gate.py`). A completion is clinically risky only
when asserted positively, so the left context of the matched span is labelled
POS / NEG / UNC with precedence `POS-override > UNC > NEG > default POS`.
Sentence boundaries and adversatives reset scope; commas and coordinators do
not, because radiology reports routinely use coordinated negations. Only POS
matches are intervened on.

| clause | label |
|---|---|
| `no focal consolidation, edema, or pneumothorax` | NEG |
| `cannot exclude small effusion` | UNC |
| `no change in pleural effusion` | POS |
| `resolved pleural effusion` | NEG |
| `decreased but persistent effusion` | POS |

*Known online limitation:* cues appearing only **after** the finding span, e.g.
`effusion is not excluded`, are invisible at trigger time.

**3. Latent intervention and rerank.** For each triggered finding group, one
counterfactual forward pass applies a visual-only hook at the intervention
layer, subtracting the projection along the matched same-layer phrase prototype
from image-token hidden states only. An adaptive plausibility constraint stops
the counterfactual branch from promoting candidates implausible under the base
model. Matched candidates are rescored by
`log p_base + alpha * (log p_base - log p_cf)`; unmatched candidates keep their
base log-probability. If no group fires, LOCI reduces exactly to standard
decoding.

## Configuration

The intervention layer is the post-block residual stream returned by decoder
block *l*, before the input normalization of block *l+1*. Values are selected on
the MIMIC-CXR validation split and then reused unchanged.

| Backbone | hook point | *l* | K_cand | G_max | alpha | lambda | alpha_apc |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL 7B | post-block residual | 18 | 30 | 3 | 0.80 | 1.0 | 0.30 |

Generation is deterministic greedy decoding with `do_sample=False`,
`max_new_tokens=150` and `repetition_penalty=1.2`, identical for standard
decoding and LOCI.

## Running

```bash
bash qwen25vl/run_loci_qwen25vl.sh           # DATA=... OUT=... to override
python eval/evaluate_reports.py --result-file <OUT>/results.jsonl
```

`ALPHA=0.0` gives the matched baseline: the same decoding loop with no rerank
and no counterfactual forward pass.

## Input data format

ShareGPT-style JSON list:

```json
[
  {
    "id": "mimic_50331901_687754ce-...",
    "image": "/abs/or/relative/path/to.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nWrite a comprehensive radiological report based on this CXR."},
      {"from": "gpt",   "value": "IMPRESSION:\n..."}
    ]
  }
]
```

`conversations[0].value` is the user query; `conversations[1].value` is the
reference report scored by `eval/evaluate_reports.py`.

## Not included here

- **Other backbones.** `code/` is backbone-agnostic; only the prompt template,
  the visual-token mask and the hook module path are backbone-specific.
- **Dataset split manifests**, which ship in the supplementary material.
- **RadFact evaluation**, run with the official RadFact toolkit.
- **Baseline decoders** (VCD, Med-VCD, OPERA, ProjectAway, VTI).
