"""
Generate clinical trigger token IDs for Qwen2.5-VL-7B-Instruct + MIMIC-CXR.

Mirrors the reference generate_whitelist_triggers.py logic but uses the
Qwen2.5-VL tokenizer (byte-BPE with Ġ word-start marker instead of SPIECE ▁).
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Data/hf_cache"))

import json
import sys
import torch
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from trigger_dict import pathology_triggers, anatomy_device_triggers, descriptive_triggers


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

HARD_BLACKLIST_WORDS = {
    "normal", "abnormalities", "abnormality", "structures", "structure",
    "radiological", "radiograph", "radiographic", "radiographs", "image",
    "images", "x-ray", "xray", "chest", "findings", "impression", "patient",
    "appear", "appears", "show", "shows", "demonstrate", "demonstrates",
    "seen", "noted", "present", "absent", "clear", "stable", "unchanged",
}

ALLOWED_SHORT_TOKENS = {
    " ct", " mr", " pa", " ap", " tb", " xr", " iv", " ng",
    " chf", " copd", " cabg", " svc", " picc",
}

BLACKLIST_TOKENS = {
    " a", " an", " the", " is", " are", " of", " in", " on", " at", " to",
    " with", " without", " and", " or", " no", " not", " un", " im", " non",
    " radi", " tr",
}


def main():
    all_triggers = set(pathology_triggers + anatomy_device_triggers + descriptive_triggers)
    clean_triggers = sorted(
        list({str(w).strip().lower() for w in all_triggers if w} - HARD_BLACKLIST_WORDS)
    )
    print(f"Loaded {len(clean_triggers)} clean trigger words after hard filtering.")

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    trigger_token_ids = set()
    id_to_words = {}
    dropped_words = []
    trigger_sequences = {}

    print("Processing triggers with Qwen byte-BPE word-start enforcement...")
    for word in clean_triggers:
        variants = [" " + word, " " + word.capitalize()]

        unique_sequences = []
        seen_sequences = set()
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue
            key = tuple(int(x) for x in token_ids)
            if key in seen_sequences:
                continue
            seen_sequences.add(key)
            unique_sequences.append(list(key))
        if unique_sequences:
            trigger_sequences[word] = unique_sequences

        selected_tid = None
        selected_raw_token = None

        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue

            for tid in token_ids:
                raw_token = tokenizer.convert_ids_to_tokens(tid)
                decoded = tokenizer.decode([tid])

                if raw_token in {" ", "<unk>", "<s>", "</s>"}:
                    continue

                # Qwen2 tokenizer uses byte-BPE, word-start tokens prefix with Ġ
                if not raw_token.startswith("Ġ"):
                    continue

                token_text = decoded.lower()
                raw_token_lower = raw_token.lower()

                if token_text in BLACKLIST_TOKENS or raw_token_lower in BLACKLIST_TOKENS:
                    continue

                token_core = token_text.strip()
                if len(token_core) < 3 and token_text not in ALLOWED_SHORT_TOKENS:
                    continue

                selected_tid = tid
                selected_raw_token = raw_token
                break

            if selected_tid is not None:
                break

        if selected_tid is None:
            dropped_words.append(word)
            continue

        trigger_token_ids.add(selected_tid)
        id_to_words.setdefault(selected_tid, [])
        debug_info = f"{word} -> [{selected_raw_token}]"
        if debug_info not in id_to_words[selected_tid]:
            id_to_words[selected_tid].append(debug_info)

    output_dir = SCRIPT_DIR
    trigger_tensor = torch.tensor(sorted(trigger_token_ids), dtype=torch.long)

    pt_file = os.path.join(output_dir, "clinical_trigger_ids_mimic_qwen25vl.pt")
    json_file = os.path.join(output_dir, "trigger_id_map_mimic_qwen25vl.json")
    dropped_file = os.path.join(output_dir, "dropped_trigger_words_mimic_qwen25vl.json")
    sequence_file = os.path.join(output_dir, "trigger_sequences_mimic_qwen25vl.json")

    torch.save(trigger_tensor, pt_file)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(id_to_words, f, indent=2, ensure_ascii=False)
    with open(dropped_file, "w", encoding="utf-8") as f:
        json.dump(dropped_words, f, indent=2, ensure_ascii=False)
    with open(sequence_file, "w", encoding="utf-8") as f:
        json.dump(trigger_sequences, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Qwen2.5-VL MIMIC trigger generation complete.")
    print(f"Input clean words: {len(clean_triggers)}")
    print(f"Unique trigger token IDs: {len(trigger_token_ids)}")
    print(f"Dropped words: {len(dropped_words)}")
    print(f"Saved tensor: {pt_file}")
    print(f"Saved id map: {json_file}")
    print(f"Saved dropped words: {dropped_file}")
    print(f"Saved trigger sequences: {sequence_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
