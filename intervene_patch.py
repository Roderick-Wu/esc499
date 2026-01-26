"""
Activation Patching for Latent Reasoning Features

This script performs activation patching experiments to test whether specific
layer/token positions encode latent reasoning variables (like velocity).

Methodology:
1. Generate pairs of prompts (source and base) with different hidden velocities
2. Extract activations from source prompt at specific (layer, token) positions
3. Patch those activations into base prompt
4. Generate completions and observe if the intervention changes the output
5. If patching source activations into base causes base to output source's velocity,
   this suggests that position encodes the velocity representation

Inspired by: "Unveiling LLMs: The Evolution of Latent Representations" (Bronzini et al., COLM 2024)
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import prompt_functions
from functools import partial

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B"
OUTPUT_DIR = Path("/home/wuroderi/projects/def-zhijing/wuroderi/reasoning_abstraction/intervention_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Intervention Configuration
N_PROMPT_PAIRS = 10
LAYERS_TO_TEST = [23, 31, 47]  # Middle layers
TOKENS_TO_PATCH = [24, 25, 30, 36, 41]  # Specific token positions to test
TOKEN_LABELS = ["How", "long", "travel", "Answer", ":"]  # Labels for readability

# Generation Configuration
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.1  # Low temperature for more deterministic outputs

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"="*80)
print(f"ACTIVATION PATCHING EXPERIMENT")
print(f"="*80)
print(f"Model: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Prompt pairs: {N_PROMPT_PAIRS}")
print(f"Layers: {LAYERS_TO_TEST}")
print(f"Token positions: {TOKENS_TO_PATCH} ({TOKEN_LABELS})")
print()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-32B",
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    fold_ln=False,
    center_writing_weights=False,
    fold_value_biases=False,
    move_to_device=False,
    load_state_dict=False
)

# Ensure embedding layer is on GPU
if model.embed.W_E.device.type == 'cpu':
    model.embed = model.embed.to('cuda:0')
    print("Moved embedding layer to cuda:0")

if hasattr(model, 'pos_embed') and model.pos_embed.W_pos.device.type == 'cpu':
    model.pos_embed = model.pos_embed.to('cuda:0')
    print("Moved positional embedding to cuda:0")

print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dimensions")
print()

# ==========================================
# GENERATE PROMPT PAIRS
# ==========================================

print("Generating prompt pairs...")

# Generate implicit velocity prompts (we'll use the first format for consistency)
all_prompts, all_prompt_ids, all_velocities = prompt_functions.gen_implicit_velocity(
    samples_per_prompt=N_PROMPT_PAIRS * 2  # Generate 2x to create pairs
)

# Filter to only use one prompt format for consistency
format_0_indices = [i for i, pid in enumerate(all_prompt_ids) if pid == 0]
format_0_prompts = [all_prompts[i] for i in format_0_indices[:N_PROMPT_PAIRS * 2]]
format_0_velocities = all_velocities[format_0_indices[:N_PROMPT_PAIRS * 2]]

# Create pairs: odd indices as source, even indices as base
prompt_pairs = []
for i in range(N_PROMPT_PAIRS):
    source_idx = i * 2
    base_idx = i * 2 + 1
    
    pair = {
        'source_prompt': format_0_prompts[source_idx],
        'source_velocity': format_0_velocities[source_idx],
        'base_prompt': format_0_prompts[base_idx],
        'base_velocity': format_0_velocities[base_idx],
    }
    prompt_pairs.append(pair)

print(f"Generated {len(prompt_pairs)} prompt pairs")
print(f"\nExample pair:")
print(f"  Source velocity: {prompt_pairs[0]['source_velocity']:.1f} m/s")
print(f"  Source prompt: {prompt_pairs[0]['source_prompt'][:80]}...")
print(f"  Base velocity: {prompt_pairs[0]['base_velocity']:.1f} m/s")
print(f"  Base prompt: {prompt_pairs[0]['base_prompt'][:80]}...")
print()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def generate_with_intervention(model, prompt, layer, token_pos, intervention_activation):
    """
    Generate text from prompt with an intervention at specific layer/token position.
    
    Args:
        model: HookedTransformer model
        prompt: Text prompt
        layer: Layer index to intervene at
        token_pos: Token position to intervene at
        intervention_activation: Activation tensor to patch in [d_model]
    
    Returns:
        Generated text (full completion including prompt)
    """
    # Tokenize prompt
    tokens = model.to_tokens(prompt, prepend_bos=True)
    embed_device = model.embed.W_E.device
    tokens = tokens.to(embed_device)
    
    # Define hook function to patch activation
    def patch_activation(activation, hook):
        # activation shape: [batch, seq_len, d_model]
        # We only patch at the specific token position
        if token_pos < activation.shape[1]:
            activation[0, token_pos] = intervention_activation.to(activation.device)
        return activation
    
    # Generate with hook
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            fwd_hooks=[(hook_name, patch_activation)],
            verbose=False
        )
    
    # Decode output
    generated_text = model.to_string(output[0])
    
    return generated_text

def generate_baseline(model, prompt):
    """Generate text without intervention (baseline)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    embed_device = model.embed.W_E.device
    tokens = tokens.to(embed_device)
    
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            verbose=False
        )
    
    return model.to_string(output[0])

def extract_activation(model, prompt, layer, token_pos):
    """
    Extract activation from a specific layer and token position.
    
    Returns:
        Activation tensor of shape [d_model]
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    embed_device = model.embed.W_E.device
    tokens = tokens.to(embed_device)
    
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=[hook_name]
        )
    
    # Extract activation at token position
    activation = cache[hook_name][0, token_pos]  # [d_model]
    
    return activation

# ==========================================
# RUN EXPERIMENTS
# ==========================================

print("Running activation patching experiments...")
print("="*80)

results = []

for pair_idx, pair in enumerate(prompt_pairs):
    print(f"\n{'='*80}")
    print(f"PAIR {pair_idx + 1}/{N_PROMPT_PAIRS}")
    print(f"{'='*80}")
    print(f"Source velocity: {pair['source_velocity']:.1f} m/s")
    print(f"Base velocity:   {pair['base_velocity']:.1f} m/s")
    print()
    
    # Generate baseline (no intervention) for base prompt
    print("Generating baseline (base prompt, no intervention)...")
    baseline_output = generate_baseline(model, pair['base_prompt'])
    print(f"Baseline output: {baseline_output[len(pair['base_prompt']):len(pair['base_prompt'])+100]}...")
    print()
    
    # Store baseline
    pair_results = {
        'pair_idx': pair_idx,
        'source_velocity': pair['source_velocity'],
        'base_velocity': pair['base_velocity'],
        'source_prompt': pair['source_prompt'],
        'base_prompt': pair['base_prompt'],
        'baseline_output': baseline_output,
        'interventions': []
    }
    
    # Test each intervention (layer, token) combination
    for layer in LAYERS_TO_TEST:
        for token_idx, token_pos in enumerate(TOKENS_TO_PATCH):
            token_label = TOKEN_LABELS[token_idx]
            
            print(f"Intervention: Layer {layer}, Token {token_pos} ({token_label})")
            
            # Extract activation from source prompt
            source_activation = extract_activation(
                model, 
                pair['source_prompt'], 
                layer, 
                token_pos
            )
            
            # Patch into base prompt and generate
            intervened_output = generate_with_intervention(
                model,
                pair['base_prompt'],
                layer,
                token_pos,
                source_activation
            )
            
            # Extract just the generated part (after prompt)
            generated_part = intervened_output[len(pair['base_prompt']):]
            print(f"  Generated: {generated_part[:80]}...")
            
            # Store result
            intervention_result = {
                'layer': layer,
                'token_pos': token_pos,
                'token_label': token_label,
                'intervened_output': intervened_output,
                'generated_part': generated_part
            }
            pair_results['interventions'].append(intervention_result)
            print()
    
    results.append(pair_results)

# ==========================================
# SAVE AND SUMMARIZE RESULTS
# ==========================================

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)

# Save detailed results
output_file = OUTPUT_DIR / "intervention_results.txt"
with open(output_file, 'w') as f:
    f.write("ACTIVATION PATCHING EXPERIMENT RESULTS\n")
    f.write("="*80 + "\n\n")
    
    for pair_result in results:
        f.write(f"\nPAIR {pair_result['pair_idx'] + 1}\n")
        f.write("-"*80 + "\n")
        f.write(f"Source velocity: {pair_result['source_velocity']:.1f} m/s\n")
        f.write(f"Base velocity:   {pair_result['base_velocity']:.1f} m/s\n")
        f.write(f"\nSource prompt:\n{pair_result['source_prompt']}\n")
        f.write(f"\nBase prompt:\n{pair_result['base_prompt']}\n")
        f.write(f"\nBaseline output (no intervention):\n{pair_result['baseline_output']}\n")
        f.write("\n" + "-"*80 + "\n")
        f.write("INTERVENTIONS:\n")
        f.write("-"*80 + "\n")
        
        for intervention in pair_result['interventions']:
            f.write(f"\nLayer {intervention['layer']}, Token {intervention['token_pos']} ({intervention['token_label']}):\n")
            f.write(f"{intervention['intervened_output']}\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")

print(f"\nDetailed results saved to: {output_file}")

# Print summary
print("\nSUMMARY:")
print(f"Tested {N_PROMPT_PAIRS} prompt pairs")
print(f"Tested {len(LAYERS_TO_TEST)} layers × {len(TOKENS_TO_PATCH)} token positions = {len(LAYERS_TO_TEST) * len(TOKENS_TO_PATCH)} interventions per pair")
print(f"Total interventions: {N_PROMPT_PAIRS * len(LAYERS_TO_TEST) * len(TOKENS_TO_PATCH)}")
print(f"\nLook for cases where patching source activations causes base to generate source velocity!")
print("="*80)
