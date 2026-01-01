import torch
import json
import re
import argparse
import numpy as np
import sys
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Respect shell-provided HF_HOME / proxy settings; do not hardcode stale paths
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Data/hf_cache"))

# ==========================================
# NLP metric libraries
# ==========================================
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def clean_and_normalize_report(text):
    """
    Standardised cleaning for MIMIC-CXR: flattens formatting differences so BLEU is not artificially depressed.
    """
    if not text:
        return ""
        
    text = str(text).lower()
    prefixes_to_remove = [
        r"findings:\s*",
        r"impression:\s*",
        r"final report\s*",
        r"report:\s*"
    ]
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, "", text)
    
    # Strip redundant whitespace and leading/trailing punctuation
    text = re.sub(r'\s+', ' ', text)
    return text.strip(" .:,;\n")

def calculate_nlp_metrics(refs, preds):
    """
    Compute BLEU-1,2,3,4 and ROUGE-L
    """
    print("⏳ Computing lexical metrics (BLEU, ROUGE-L)...")
    
    # Guard against empty strings breaking tokenisation (BLEU only)
    refs_clean_for_bleu = [r if r.strip() else "empty_token" for r in refs]
    preds_clean_for_bleu = [p if p.strip() else "empty_token" for p in preds]

    # Tokenise for BLEU
    refs_tokenized = [[r.split()] for r in refs_clean_for_bleu]
    preds_tokenized = [p.split() for p in preds_clean_for_bleu]
    
    chencherry = SmoothingFunction()
    
    # BLEU 1-4
    bleu_1 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1)
    bleu_2 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    bleu_3 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    bleu_4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    
    # ROUGE-L (ROUGE consumes raw text and handles empty strings safely)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(refs, preds)]
    rouge_l = np.mean(rouge_l_scores)
    
    return {
        "BLEU-1": bleu_1, 
        "BLEU-2": bleu_2, 
        "BLEU-3": bleu_3, 
        "BLEU-4": bleu_4, 
        "ROUGE-L": rouge_l
    }

def calculate_clinical_metrics(refs, preds):
    """
    Optionally run CheXbert and RadGraph for clinical factual consistency.
    """
    results = {}
    
    # ------------------------------------------
    # 1. CheXbert
    # ------------------------------------------
    try:
        from f1chexbert import F1CheXbert
        print("⏳ Computing CheXbert F1... (the first run downloads the pretrained weights)")
        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        chexbert_evaluator = F1CheXbert(device=eval_device)
        accuracy, accuracy_per_class, mac_f1, mic_f1 = chexbert_evaluator(hyps=preds, refs=refs)
        results["CheXbert Macro-F1"] = mac_f1
        results["CheXbert Micro-F1"] = mic_f1
    except ImportError:
        print("⚠️ CheXbert is not installed. Skipping. (pip install f1chexbert)")
    except Exception as e:
        print(f"⚠️ CheXbert runtime error: {e}")
        
    # ------------------------------------------
    # 2. RadGraph
    # ------------------------------------------
    try:
        from radgraph import F1RadGraph
        print("⏳ Computing RadGraph F1... (the first run downloads the weights)")
        
        # [Fix] Pin the GPU device explicitly
        radgraph_device = 0 if torch.cuda.is_available() else -1
        radgraph_evaluator = F1RadGraph(reward_level="all", cuda_device=radgraph_device)
        
        rg_score, _, _, _ = radgraph_evaluator(hyps=preds, refs=refs)
        results["RadGraph F1"] = rg_score
    except ImportError:
        print("⚠️ RadGraph is not installed. Skipping. (pip install radgraph)")
    except Exception as e:
        print(f"⚠️ RadGraph runtime error: {e}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="MIMIC-CXR radiology report evaluation")
    parser.add_argument("--result-file", type=str, required=True, help="path to the generated jsonl file")
    args = parser.parse_args()
    
    preds = []
    refs = []
    
    print(f"📂 Loading data: {args.result_file}")
    
    valid_count = 0
    skipped_count = 0
    
    try:
        with open(args.result_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    pred_text = clean_and_normalize_report(data.get('text', ''))
                    ref_text = clean_and_normalize_report(data.get('gt', ''))
                    
                    # [core patch] Keep the CheXbert/RadGraph tokenisers from crashing on empty strings
                    # Never drop these samples; substitute a meaningless placeholder so they still score low
                    if not pred_text:
                        pred_text = "empty report"

                    if ref_text:
                        preds.append(pred_text)
                        refs.append(ref_text)
                        valid_count += 1
                    else:
                        skipped_count += 1
                except json.JSONDecodeError:
                    print("⚠️ Skipping malformed JSON line")
                    continue
    except FileNotFoundError:
        print(f"❌ File not found: {args.result_file}")
        return

    print(f"✅ Loaded: {valid_count} valid samples (skipped {skipped_count} without ground truth)")
    
    if valid_count == 0:
        print("❌ No valid samples to evaluate.")
        return

    # 1. Lexical metrics
    nlp_results = calculate_nlp_metrics(refs, preds)
    print("\n" + "="*50)
    print("📊 Lexical metrics (Lexical Similarity)")
    print("="*50)
    for k, v in nlp_results.items():
        print(f"{k:<15}: {v:.4f}")
        
    # 2. Clinical factual metrics
    clinical_results = calculate_clinical_metrics(refs, preds)
    if clinical_results:
        print("\n" + "="*50)
        print("🏥 Clinical metrics (Clinical Factual Correctness)")
        print("="*50)
        for k, v in clinical_results.items():
            if isinstance(v, float):
                print(f"{k:<20}: {v:.4f}")
            else:
                print(f"{k:<20}: {v}")

if __name__ == "__main__":
    # Pre-fetch NLTK dependencies
    for pkg in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            print(f"Downloading NLTK {pkg} tokenizer...")
            nltk.download(pkg, quiet=True)
        
    main()