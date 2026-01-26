"""
Token-Level Intervention for Causal Testing (Using Pre-Generated Traces)

This script performs interventions at the token level using pre-generated traces.
It swaps computed velocity values from a source example into a base example
to test whether the model's reasoning causally depends on these intermediate values.

Workflow:
1. Load pre-generated traces from traces_metadata.json
2. Pair 250 prompts into 125 pairs (source, base)
3. For each base trace:
   - Truncate generated text at second "Question" occurrence
   - Find first occurrence of base velocity value
   - Replace with source velocity value
4. Continue generation and evaluate final answer
5. Compare intervention results with expected values

Key features:
- Uses existing traces instead of generating new ones
- Handles velocity value matching with tolerance for significant digits
- Computes expected time using base distance / source velocity
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import re
from typing import List, Tuple, Optional, Dict

# ==========================================
# CONFIGURATION
# ==========================================

# Experiment configuration
EXPERIMENT = "velocity"  # Options: 'velocity', 'current', 'radius', etc.

# Model configuration
MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"

# Traces directory
TRACES_DIR = Path("/home/wuroderi/scratch/reasoning_traces/Qwen2.5-32B/velocity")
TRACES_METADATA_FILE = TRACES_DIR / "traces_metadata.json"

# Generation configuration
MAX_TOKENS_AFTER_INTERVENTION = 256  # Max tokens to continue after intervention
TEMPERATURE = 0.0  # Use greedy decoding for consistency
TOP_P = 1.0

# Output configuration
OUTPUT_DIR = Path("/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/intervention_token_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("TOKEN-LEVEL INTERVENTION (USING PRE-GENERATED TRACES)")
print("="*70)
print(f"Experiment: {EXPERIMENT}")
print(f"Model: {MODEL_PATH}")
print(f"Traces: {TRACES_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

# ==========================================
# LOAD TRACES
# ==========================================

print("Loading traces metadata...")
with open(TRACES_METADATA_FILE, 'r') as f:
    traces = json.load(f)

print(f"Loaded {len(traces)} traces")
print(f"  Example trace keys: {list(traces[0].keys())[:10]}")
print()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Model loaded: {model.config.num_hidden_layers} layers\n")

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def truncate_at_second_question(text: str) -> str:
    """
    Truncate text at the second occurrence of 'Question'.
    This removes extra questions the model generates after answering.
    """
    parts = text.split('Question')
    if len(parts) >= 3:
        # Keep first two parts (before first Question, and between first and second Question)
        return 'Question'.join(parts[:2]) + 'Question'
    return text

def find_velocity_in_text(text: str, velocity: float, tolerance: float = 1.0) -> Optional[Tuple[int, int, str]]:
    """
    Find the first occurrence of a velocity value in text.
    Returns (start_pos, end_pos, matched_string) or None.
    
    Args:
        text: Generated text to search
        velocity: Velocity value to find (e.g., 74)
        tolerance: Matching tolerance for approximate matches
    
    Returns:
        Tuple of (start_position, end_position, matched_text) or None
    """
    # Try matching different representations
    # Pattern: number optionally followed by decimal and digits, possibly followed by m/s
    patterns = [
        # Exact integer match
        rf'\b{int(velocity)}\b(?!\.\d)',
        # With decimal point and trailing zeros
        rf'\b{int(velocity)}\.0+\b',
        # With decimal point and one digit
        rf'\b{velocity:.1f}\b',
        # With decimal point and two digits
        rf'\b{velocity:.2f}\b',
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            match = matches[0]  # Get first occurrence
            return (match.start(), match.end(), match.group())
    
    # Try fuzzy matching - look for any number close to the velocity
    number_pattern = r'\b(\d+\.?\d*)\b'
    for match in re.finditer(number_pattern, text):
        try:
            value = float(match.group(1))
            if abs(value - velocity) <= tolerance:
                return (match.start(), match.end(), match.group())
        except:
            continue
    
    return None

def extract_final_answer(text: str) -> Optional[float]:
    """Extract the final numerical answer from generated text."""
    # Look for final answer patterns
    patterns = [
        r'(?:answer|result|final|therefore).*?([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
        r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*(?:seconds?|s\b)',
        r't\s*=\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
    ]
    
    # Try to get the last number that looks like an answer
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            try:
                return float(matches[-1].group(1))
            except:
                continue
    
    # Fallback: get any number after "answer is"
    answer_pattern = r'answer is\s+([0-9]+\.?[0-9]*)'
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    
    return None

def replace_velocity_in_text(text: str, old_velocity: float, new_velocity: float) -> Optional[str]:
    """
    Replace the first occurrence of old_velocity with new_velocity in text.
    
    Returns:
        Modified text with substitution, or None if velocity not found
    """
    result = find_velocity_in_text(text, old_velocity)
    if result is None:
        return None
    
    start_pos, end_pos, matched_text = result
    
    # Format new velocity to match the style of the matched text
    if '.' in matched_text:
        # Has decimal point - preserve number of decimal places
        decimal_places = len(matched_text.split('.')[-1])
        new_velocity_str = f"{new_velocity:.{decimal_places}f}"
    else:
        # Integer format
        new_velocity_str = str(int(new_velocity))
    
    # Replace in text
    modified_text = text[:start_pos] + new_velocity_str + text[end_pos:]
    
    return modified_text

def continue_generation_from_text(model, tokenizer, text: str, max_new_tokens: int = 256) -> str:
    """
    Continue generation from a partial text string.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The text to continue from
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        Complete generated text (including input text)
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            top_p=TOP_P if TEMPERATURE > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text

# ==========================================
# INTERVENTION EXPERIMENT
# ==========================================

def run_intervention_experiment(source_trace: Dict, base_trace: Dict, 
                                model, tokenizer) -> Dict:
    """
    Run a single intervention experiment using pre-generated traces.
    
    Args:
        source_trace: Source trace with velocity to inject
        base_trace: Base trace to intervene on
        model: The language model
        tokenizer: The tokenizer
    
    Returns:
        Dictionary with results
    """
    source_velocity = source_trace['v']
    base_velocity = base_trace['v']
    base_distance = base_trace['d']
    
    # Expected time with intervention: base_distance / source_velocity
    expected_time_with_intervention = base_distance / source_velocity
    expected_time_baseline = base_trace['expected_time']
    
    result = {
        'source_id': source_trace['id'],
        'base_id': base_trace['id'],
        'source_velocity': source_velocity,
        'base_velocity': base_velocity,
        'base_distance': base_distance,
        'expected_time_baseline': expected_time_baseline,
        'expected_time_with_intervention': expected_time_with_intervention,
    }
    
    # Get base generated text
    base_text = base_trace['generated_text']
    
    # Step 1: Truncate at second "Question"
    truncated_text = truncate_at_second_question(base_text)
    result['truncated_text'] = truncated_text
    
    # Step 2: Find base velocity in truncated text
    velocity_location = find_velocity_in_text(truncated_text, base_velocity)
    
    if velocity_location is None:
        print(f"    WARNING: Could not find base velocity {base_velocity} in generated text")
        result['success'] = False
        result['error'] = 'velocity_not_found'
        return result
    
    start_pos, end_pos, matched_text = velocity_location
    print(f"    Found velocity: '{matched_text}' at position {start_pos}")
    
    # Step 3: Truncate text up to and including the velocity value
    text_up_to_velocity = truncated_text[:end_pos]
    result['text_up_to_velocity'] = text_up_to_velocity
    
    # Step 4: Replace velocity value
    modified_text = replace_velocity_in_text(text_up_to_velocity, base_velocity, source_velocity)
    
    if modified_text is None:
        print(f"    WARNING: Failed to replace velocity")
        result['success'] = False
        result['error'] = 'replacement_failed'
        return result
    
    result['text_with_intervention'] = modified_text
    print(f"    Replaced {base_velocity} → {source_velocity}")
    
    # Step 5: Continue generation
    print(f"    Continuing generation (max {MAX_TOKENS_AFTER_INTERVENTION} tokens)...")
    try:
        final_text = continue_generation_from_text(
            model, tokenizer, modified_text, MAX_TOKENS_AFTER_INTERVENTION
        )
        result['final_text'] = final_text
        
        # Extract final answer
        final_answer = extract_final_answer(final_text)
        result['final_answer'] = final_answer
        
        if final_answer is not None:
            result['error_from_intervention_expectation'] = abs(final_answer - expected_time_with_intervention)
            result['error_from_baseline_expectation'] = abs(final_answer - expected_time_baseline)
            result['moved_toward_intervention'] = (
                result['error_from_intervention_expectation'] < 
                result['error_from_baseline_expectation']
            )
        
        result['success'] = True
        
    except Exception as e:
        print(f"    ERROR during generation: {e}")
        result['success'] = False
        result['error'] = str(e)
    
    return result

# ==========================================
# PAIR TRACES AND RUN EXPERIMENTS
# ==========================================

print("Creating pairs from traces...")

# Pair traces: first 125 with second 125
n_pairs = min(125, len(traces) // 2)
pairs = []

for i in range(n_pairs):
    source_trace = traces[i]
    base_trace = traces[i + n_pairs]
    pairs.append((source_trace, base_trace))

print(f"Created {len(pairs)} pairs")
print(f"  Example: Source trace {pairs[0][0]['id']} (v={pairs[0][0]['v']}) -> "
      f"Base trace {pairs[0][1]['id']} (v={pairs[0][1]['v']})")
print()

# Run experiments
print(f"Running {len(pairs)} intervention experiments...")
print()

all_results = []

for idx, (source_trace, base_trace) in enumerate(pairs):
    print(f"[{idx+1}/{len(pairs)}] Intervention: Source v={source_trace['v']}, "
          f"Base v={base_trace['v']}, Expected time={base_trace['d']/source_trace['v']:.3f}s")
    
    result = run_intervention_experiment(source_trace, base_trace, model, tokenizer)
    all_results.append(result)
    
    if result['success']:
        print(f"    ✓ Final answer: {result.get('final_answer', 'N/A')}")
        if result.get('final_answer') is not None:
            print(f"      Error from intervention expectation: {result['error_from_intervention_expectation']:.3f}")
            print(f"      Moved toward intervention: {result['moved_toward_intervention']}")
    else:
        print(f"    ✗ Failed: {result.get('error', 'unknown')}")
    
    # Save intermediate results every 25 experiments
    if (idx + 1) % 25 == 0:
        output_file = OUTPUT_DIR / f"intervention_token_{EXPERIMENT}_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  💾 Saved intermediate results to {output_file}\n")

print()

# ==========================================
# SAVE FINAL RESULTS
# ==========================================

output_file = OUTPUT_DIR / f"intervention_token_{EXPERIMENT}_results.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print(f"Total experiments: {len(all_results)}")
print(f"Results saved to: {output_file}")

# Compute summary statistics
successful = [r for r in all_results if r.get('success', False)]
print(f"\nSuccessful interventions: {len(successful)}/{len(all_results)}")

if successful:
    with_answers = [r for r in successful if r.get('final_answer') is not None]
    print(f"Generated answers: {len(with_answers)}/{len(successful)}")
    
    if with_answers:
        moved_toward_intervention = [r for r in with_answers if r.get('moved_toward_intervention', False)]
        print(f"Answers moved toward intervention expectation: {len(moved_toward_intervention)}/{len(with_answers)}")
        
        # Average errors
        avg_error_intervention = np.mean([r['error_from_intervention_expectation'] for r in with_answers])
        avg_error_baseline = np.mean([r['error_from_baseline_expectation'] for r in with_answers])
        print(f"\nAverage error from intervention expectation: {avg_error_intervention:.3f}s")
        print(f"Average error from baseline expectation: {avg_error_baseline:.3f}s")

# Report failures
failures = [r for r in all_results if not r.get('success', False)]
if failures:
    print(f"\nFailure breakdown:")
    error_types = {}
    for r in failures:
        error = r.get('error', 'unknown')
        error_types[error] = error_types.get(error, 0) + 1
    for error, count in error_types.items():
        print(f"  {error}: {count}")

print()
