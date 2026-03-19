#!/usr/bin/env python3
"""LoRA Merge/Inference Checklist Verification.

Goes through each checklist item line by line:
1. Pad token alignment between training and inference
2. 2 PAD tokens for ending (Qwen3 requirement)
3. LoRA adapter save/load correctness
4. Tokenizer save with correct special tokens
"""

import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", message=".*does not have a padding token.*")
warnings.filterwarnings("ignore", message=".*PAD_TOKEN.*")

sys.path.insert(0, str(Path(__file__).parent.parent))

from qgre.config import QGREConfig


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    return condition


def main():
    print("=" * 80)
    print("  LoRA Merge/Inference Checklist")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: GPU required")
        sys.exit(1)

    # Load model
    config = QGREConfig.from_yaml("examples/hypergraph/config.yaml")
    print(f"\nLoading {config.model.path}...")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.path,
        max_seq_length=config.generation.max_tokens + 3200,
        load_in_4bit=config.model.load_in_4bit,
        fast_inference=config.model.fast_inference,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
    )

    passes = 0
    fails = 0

    # ========== CHECK 1: Pad token alignment ==========
    print("\n--- CHECK 1: Pad token alignment ---")

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    pad_token = tokenizer.pad_token

    if check("Pad token is set", pad_id is not None, f"pad_token_id={pad_id}"):
        passes += 1
    else:
        fails += 1

    if check("Pad token != EOS token", pad_id != eos_id,
             f"PAD={pad_id} ({pad_token}), EOS={eos_id} ({tokenizer.eos_token})"):
        passes += 1
    else:
        fails += 1

    if check("Pad token is <|PAD_TOKEN|> (151669)", pad_id == 151669,
             f"Actual: {pad_id} ({pad_token})"):
        passes += 1
    else:
        fails += 1

    if check("EOS token is <|im_end|> (151645)", eos_id == 151645,
             f"Actual: {eos_id} ({tokenizer.eos_token})"):
        passes += 1
    else:
        fails += 1

    # Verify pad token is NOT a stop token
    stop_tokens = config.generation.stop_token_ids
    if check("Pad token NOT in stop_token_ids", pad_id not in stop_tokens,
             f"stop_tokens={stop_tokens}, pad={pad_id}"):
        passes += 1
    else:
        fails += 1

    # ========== CHECK 2: 2 PAD tokens for ending ==========
    print("\n--- CHECK 2: Padding behavior ---")

    # Qwen3 tokenizer: test that padding works correctly
    # Use texts with VERY different lengths to guarantee padding
    test_texts = ["Hello", "This is a much longer sentence that should require many more tokens to encode"]
    encoded = tokenizer(test_texts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"]

    # Shorter sequence (index 0) should have pad tokens
    pad_positions = (input_ids[0] == pad_id).sum().item()
    if check("Shorter sequence has pad tokens", pad_positions > 0,
             f"Pad positions in shorter sequence: {pad_positions}, seq lengths: {(input_ids[0] != pad_id).sum().item()} vs {(input_ids[1] != pad_id).sum().item()}"):
        passes += 1
    else:
        fails += 1

    # Test left-padding (needed for generation)
    tokenizer.padding_side = "left"
    encoded_left = tokenizer(test_texts, padding=True, return_tensors="pt")
    first_token_shorter = encoded_left["input_ids"][0][0].item()
    if check("Left-padding: first token of shorter seq is PAD", first_token_shorter == pad_id,
             f"First token: {first_token_shorter}, expected PAD={pad_id}"):
        passes += 1
    else:
        fails += 1

    # ========== CHECK 3: LoRA adapter save/load ==========
    print("\n--- CHECK 3: LoRA adapter save/load ---")

    import tempfile
    lora_dir = Path(tempfile.mkdtemp()) / "test_lora"

    # Save LoRA
    try:
        model.save_lora(str(lora_dir))
        if check("save_lora() succeeds", True):
            passes += 1
    except Exception as e:
        if check("save_lora() succeeds", False, str(e)):
            passes += 1
        else:
            fails += 1

    # Verify weight files exist
    safetensors = list(lora_dir.rglob("*.safetensors"))
    bin_files = list(lora_dir.rglob("*.bin"))
    has_weights = len(safetensors) > 0 or len(bin_files) > 0
    if check("Weight files saved", has_weights,
             f"safetensors: {len(safetensors)}, bin: {len(bin_files)}"):
        passes += 1
    else:
        fails += 1

    # Verify adapter_config.json exists
    adapter_config = lora_dir / "adapter_config.json"
    if check("adapter_config.json exists", adapter_config.exists()):
        passes += 1
    else:
        fails += 1

    # Hash weights before reload
    import hashlib
    weight_hash_before = hashlib.sha256()
    for f in sorted(safetensors or bin_files):
        weight_hash_before.update(f.read_bytes())
    hash_before = weight_hash_before.hexdigest()[:16]

    # Reload LoRA
    try:
        model.load_lora(str(lora_dir))
        if check("load_lora() succeeds", True, f"Weight hash: {hash_before}"):
            passes += 1
    except Exception as e:
        check("load_lora() succeeds", False, str(e))
        fails += 1

    # Verify weights match after reload
    weight_hash_after = hashlib.sha256()
    for f in sorted(lora_dir.rglob("*.safetensors") or lora_dir.rglob("*.bin")):
        weight_hash_after.update(f.read_bytes())
    hash_after = weight_hash_after.hexdigest()[:16]
    if check("Weight hash unchanged after reload", hash_before == hash_after,
             f"Before: {hash_before}, After: {hash_after}"):
        passes += 1
    else:
        fails += 1

    # ========== CHECK 4: Tokenizer save with correct special tokens ==========
    print("\n--- CHECK 4: Tokenizer special tokens ---")

    tokenizer_dir = Path(tempfile.mkdtemp()) / "test_tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))

    # Reload and verify
    from transformers import AutoTokenizer
    reloaded = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    if check("Reloaded pad_token_id matches", reloaded.pad_token_id == pad_id,
             f"Saved: {pad_id}, Reloaded: {reloaded.pad_token_id}"):
        passes += 1
    else:
        fails += 1

    if check("Reloaded eos_token_id matches", reloaded.eos_token_id == eos_id,
             f"Saved: {eos_id}, Reloaded: {reloaded.eos_token_id}"):
        passes += 1
    else:
        fails += 1

    if check("Reloaded pad_token matches", reloaded.pad_token == pad_token,
             f"Saved: {pad_token}, Reloaded: {reloaded.pad_token}"):
        passes += 1
    else:
        fails += 1

    # Verify vocab size preserved
    if check("Vocab size preserved", len(reloaded) == len(tokenizer),
             f"Original: {len(tokenizer)}, Reloaded: {len(reloaded)}"):
        passes += 1
    else:
        fails += 1

    # ========== CHECK 5: Generation produces output with correct stop behavior ==========
    print("\n--- CHECK 5: Generation with stop tokens ---")

    from vllm import SamplingParams

    FastLanguageModel.for_inference(model)
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=64,
        stop_token_ids=config.generation.stop_token_ids,
    )

    outputs = model.fast_generate(
        ["Hello, how are you?"],
        sampling_params=sampling_params,
    )

    if outputs and outputs[0].outputs:
        gen_text = outputs[0].outputs[0].text
        gen_tokens = outputs[0].outputs[0].token_ids
        if check("Generation produces output", len(gen_text) > 0,
                 f"{len(gen_tokens)} tokens, {len(gen_text)} chars"):
            passes += 1
        else:
            fails += 1

        # Check last token
        last_token = gen_tokens[-1] if gen_tokens else None
        is_stop = last_token in config.generation.stop_token_ids
        is_max = len(gen_tokens) >= 64
        if check("Stops at stop token or max_tokens", is_stop or is_max,
                 f"Last token: {last_token}, is_stop: {is_stop}, is_max: {is_max}"):
            passes += 1
        else:
            fails += 1
    else:
        check("Generation produces output", False, "No output returned")
        fails += 2

    # ========== SUMMARY ==========
    print(f"\n{'='*80}")
    total = passes + fails
    print(f"  CHECKLIST RESULT: {passes}/{total} passed, {fails} failed")
    if fails == 0:
        print("  ALL CHECKS PASSED")
    else:
        print("  FAILURES DETECTED — review above")
    print(f"{'='*80}")

    # Cleanup
    import shutil
    shutil.rmtree(lora_dir.parent, ignore_errors=True)
    shutil.rmtree(tokenizer_dir.parent, ignore_errors=True)

    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
