# Evaluation

Scoring for the chest X-ray report-generation experiments. Generation runs in
each backbone's own environment; scoring runs here.

```bash
conda env create -f environment.yml && conda activate loci-eval
python evaluate_reports.py --result-file <output-dir>/results.jsonl
```

## Metrics

| Metric | Software / version | Notes |
|---|---|---|
| RadGraph F1 | `radgraph` v0.1.18 | official entity–relation F1 on generated vs. reference report text |
| CheXbert 14c-F1 / 5c-F1 | `f1chexbert` v0.0.2 | labels in {POS, NEG, UNC, BLANK}; macro-F1 over 14 or 5 observations |
| Positive precision / recall | same CheXbert outputs | POS is positive; NEG/UNC/BLANK merged into NONPOS; macro-averaged over 5 observations |
| ROUGE-L | `rouge-score` 2022-07 | lexical-overlap diagnostic only |

The 5-observation subset is Atelectasis, Cardiomegaly, Consolidation, Edema and
Pleural Effusion.

RadFact logical precision/recall on PadChest-GR is run with the official RadFact
toolkit and is **not** reproduced by this script.

## Input normalization

References and generations are Unicode-normalized to NFKC with repeated
whitespace collapsed. **No lexicon normalization is applied to evaluator
inputs** — the LOCI phrase lexicon and scope-gating rules are used for
triggering only and never touch what the evaluators see.

## Input format

One JSON object per line:

```json
{"id": "...", "image": "...", "text": "<generated report>", "gt": "<reference report>", "model_id": "..."}
```

Records whose generated text is empty are kept and replaced with a placeholder
rather than dropped, so that they receive a low score instead of silently
improving the average.

## Split manifests

The per-dataset CSV manifests that define the exact evaluated image–report pairs
are distributed in the supplementary material, not in this repository.
