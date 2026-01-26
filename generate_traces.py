"""
Generate Reasoning Traces with CoT

This script generates chain-of-thought responses for physics problems and saves:
1. Full activations for all layers and tokens
2. Token sequences
3. Prompt metadata (variables, hidden variables, expected answers)

The traces are saved to ~/scratch/reasoning_traces/<model_name>/ for later analysis.

Usage:
    python generate_traces.py --experiment velocity --n_prompts 250
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import argparse
import itertools
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='velocity', 
                    help='Experiment type: velocity, current, etc.')
parser.add_argument('--model_path', type=str, 
                    default='/home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B')
parser.add_argument('--n_prompts', type=int, default=250,
                    help='Number of prompts to generate (will be split across formats)')
parser.add_argument('--max_new_tokens', type=int, default=512,
                    help='Maximum tokens to generate per prompt')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Output directory
model_name = Path(args.model_path).name
OUTPUT_DIR = Path.home() / 'scratch' / 'reasoning_traces' / model_name / args.experiment
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*70)
print("GENERATE REASONING TRACES WITH COT")
print("="*70)
print(f"Experiment: {args.experiment}")
print(f"Model: {args.model_path}")
print(f"Output: {OUTPUT_DIR}")
print(f"Prompts to generate: {args.n_prompts}")
print(f"Max new tokens: {args.max_new_tokens}")
print()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

print("Loading HuggingFace model with device_map='auto'...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size

print(f"Model loaded: {n_layers} layers, {d_model} dimensions\n")

# ==========================================
# PROMPT GENERATION (NO DUPLICATES)
# ==========================================

def generate_unique_velocity_prompts(n_prompts):
    """
    Generate unique velocity prompts ensuring no duplicate (m, v, d) combinations.
    
    Returns:
        List of dictionaries with keys: prompt, format_id, m, ke, d, v (hidden), expected_time
    """
    objects = ["mass", "car", "train", "ball", "runner", "plane", "rocket", "vehicle"]
    
    prompt_formats = [
        "A {m} kg {obj} has {ke} Joules of kinetic energy. How long does it take to travel {d} m?",
        "Given a {m} kg {obj} with {ke} Joules of kinetic energy, calculate the duration required to cover a distance of {d} m.", 
        "The {obj} weighs {m} kg and possesses {ke} Joules of kinetic energy. What is the time needed to traverse {d} m?",
        "Mass: {m} kg, Kinetic Energy: {ke} Joules. Determine the time interval necessary for this {obj} to displace {d} m.",
        "Consider a {m} kg {obj} with {ke} Joules of kinetic energy. Find the number of seconds needed to move {d} m."
    ]
    
    # Generate unique combinations
    # For velocity: we need unique (m, v, d) tuples
    m_values = list(range(1, 21))  # 1-20 kg
    v_values = list(range(2, 101))  # 2-100 m/s
    d_values = list(range(10, 101))  # 10-100 m
    
    # Create all possible combinations
    all_combinations = list(itertools.product(m_values, v_values, d_values))
    
    # Shuffle and select n_prompts
    np.random.shuffle(all_combinations)
    selected_combinations = all_combinations[:n_prompts]
    
    prompts_data = []
    prompts_per_format = n_prompts // len(prompt_formats)
    
    for i, (m, v, d) in enumerate(selected_combinations):
        format_id = i % len(prompt_formats)
        prompt_format = prompt_formats[format_id]
        obj = np.random.choice(objects)
        
        ke = 0.5 * m * (v ** 2)
        expected_time = d / v
        
        prompt = prompt_format.format(m=m, obj=obj, ke=f"{ke:.3e}", d=d)

        prompt = "Question: " + prompt + " Answer (step-by-step): "
        
        prompts_data.append({
            'prompt': prompt,
            'format_id': format_id,
            'object': obj,
            'm': m,
            'ke': ke,
            'd': d,
            'v': v,  # Hidden variable
            'expected_time': expected_time
        })
    
    return prompts_data

def generate_unique_current_prompts(n_prompts):
    """
    Generate unique current prompts ensuring no duplicate (r, i, t) combinations.
    
    Returns:
        List of dictionaries with keys: prompt, format_id, r, p, t, i (hidden), expected_charge
    """
    objects = ["device", "computer", "appliance", "gadget", "machine", "resistor"]
    
    prompt_formats = [
        "A {obj} has {r} ohms of resistance and dissipates {p} watts of power. How much charge flows through it after {t} seconds?",
        "Given a {obj} with {r} ohms of resistance and {p} watts of power output, calculate the total charge in Coulombs that passes through over {t} seconds.", 
        "The {obj} dissipates {p} watts across a resistance of {r} ohms. Determine the magnitude of charge transferred during a {t} second interval.", 
        "Resistance: {r} ohms, Power: {p} watts. Find the net charge flow accumulated in this {obj} after {t} seconds.", 
        "Consider a {obj} rated at {r} ohms and {p} watts. How many Coulombs of charge travel through this component in {t} seconds?"
    ]
    
    # Generate unique combinations
    r_values = list(range(1, 11))  # 1-10 ohms
    i_values = list(range(2, 16))  # 2-15 A
    t_values = list(range(10, 101))  # 10-100 seconds
    
    all_combinations = list(itertools.product(r_values, i_values, t_values))
    np.random.shuffle(all_combinations)
    selected_combinations = all_combinations[:n_prompts]
    
    prompts_data = []
    
    for i, (r, current, t) in enumerate(selected_combinations):
        format_id = i % len(prompt_formats)
        prompt_format = prompt_formats[format_id]
        obj = np.random.choice(objects)
        
        p = current ** 2 * r  # P = I^2 * R
        expected_charge = current * t  # Q = I * t
        
        prompt = prompt_format.format(obj=obj, r=r, p=p, t=t)

        prompt = "Question: " + prompt + " \nAnswer (step-by-step): "
        
        prompts_data.append({
            'prompt': prompt,
            'format_id': format_id,
            'object': obj,
            'r': r,
            'p': p,
            't': t,
            'i': current,  # Hidden variable
            'expected_charge': expected_charge
        })
    
    return prompts_data

# ==========================================
# TRACE GENERATION
# ==========================================

def generate_trace_with_activations(prompt_text, model, tokenizer, max_new_tokens=256):
    """
    Generate CoT response using HuggingFace generate() with hidden states.
    Much faster than TransformerLens for pure generation.
    
    Returns:
        Dictionary with:
            - tokens: List of token IDs
            - token_strings: List of decoded token strings
            - prompt_length: Number of tokens in prompt
            - generated_text: Full generated text
            - activations: Dict mapping layer -> tensor of shape [seq_len, d_model]
    """
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    print(f"    Generating (max {max_new_tokens} tokens)...", end='', flush=True)
    
    # Generate with hidden states (greedy decoding for determinism)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding (top-1 sampling)
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract generated tokens
    generated_ids = outputs.sequences[0]
    token_ids = generated_ids.cpu().tolist()
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Extract hidden states from all layers
    # outputs.hidden_states is a tuple of tuples:
    # - Outer tuple: one per generated token
    # - Inner tuple: one per layer (including embedding)
    # Each element: [batch, seq_len, hidden_size]
    
    n_layers = model.config.num_hidden_layers
    all_layer_activations = {layer: [] for layer in range(n_layers)}
    
    # For each generated token
    for step_hidden_states in outputs.hidden_states:
        # step_hidden_states is a tuple of (n_layers + 1) tensors
        # Index 0 is embeddings, indices 1 to n_layers are transformer layers
        for layer in range(n_layers):
            # Get the last token's hidden state for this layer
            hidden_state = step_hidden_states[layer + 1][0, -1].cpu().float()  # [d_model]
            all_layer_activations[layer].append(hidden_state)
    
    print(f" [Generated {len(token_ids) - prompt_length} tokens]")
    
    # Stack activations for each layer
    stacked_activations = {}
    for layer in range(n_layers):
        if all_layer_activations[layer]:
            stacked_activations[layer] = torch.stack(all_layer_activations[layer], dim=0).numpy()
        else:
            # If no tokens were generated, use zeros
            stacked_activations[layer] = np.zeros((0, model.config.hidden_size), dtype=np.float32)
    
    return {
        'tokens': token_ids,
        'token_strings': token_strings,
        'prompt_length': prompt_length,
        'generated_text': generated_text,
        'activations': stacked_activations  # layer -> [seq_len, d_model]
    }

# ==========================================
# MAIN GENERATION LOOP
# ==========================================

print(f"Generating {args.n_prompts} prompts...")

if args.experiment == 'velocity':
    prompts_data = generate_unique_velocity_prompts(args.n_prompts)
elif args.experiment == 'current':
    prompts_data = generate_unique_current_prompts(args.n_prompts)
else:
    raise ValueError(f"Unknown experiment: {args.experiment}")

print(f"Generated {len(prompts_data)} unique prompts\n")

# Generate traces
all_traces = []

for idx, prompt_data in enumerate(tqdm(prompts_data, desc="Generating traces")):
    print(f"\n[{idx+1}/{len(prompts_data)}] Prompt: {prompt_data['prompt'][:80]}...")
    
    trace = generate_trace_with_activations(
        prompt_data['prompt'], 
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens
    )
    
    # Combine prompt metadata with trace data
    full_trace = {
        'id': idx,
        **prompt_data,  # Includes: prompt, format_id, variables, hidden variable, expected answer
        **trace  # Includes: tokens, token_strings, prompt_length, generated_text
    }
    
    # Note: activations are stored separately to avoid huge JSON file
    all_traces.append({k: v for k, v in full_trace.items() if k != 'activations'})
    
    # Save activations separately as numpy arrays
    activations_file = OUTPUT_DIR / f"trace_{idx:04d}_activations.npz"
    np.savez_compressed(
        activations_file,
        **{f"layer_{layer}": trace['activations'][layer] 
           for layer in range(n_layers)}
    )
    
    # Save intermediate metadata every 50 prompts
    if (idx + 1) % 50 == 0:
        metadata_file = OUTPUT_DIR / 'traces_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(all_traces, f, indent=2)
        print(f"\n  Saved intermediate metadata to {metadata_file}")

# ==========================================
# SAVE FINAL RESULTS
# ==========================================

metadata_file = OUTPUT_DIR / 'traces_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(all_traces, f, indent=2)

# Save generation config
config = {
    'experiment': args.experiment,
    'model_path': args.model_path,
    'model_name': model_name,
    'n_prompts': len(all_traces),
    'max_new_tokens': args.max_new_tokens,
    'temperature': args.temperature,
    'top_p': args.top_p,
    'seed': args.seed,
    'n_layers': n_layers,
    'd_model': d_model,
}

config_file = OUTPUT_DIR / 'config.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*70)
print("GENERATION COMPLETE")
print("="*70)
print(f"Generated {len(all_traces)} traces")
print(f"Metadata saved to: {metadata_file}")
print(f"Config saved to: {config_file}")
print(f"Activations saved as: trace_XXXX_activations.npz")
print()
print("Next steps:")
print("1. Run annotate_traces.py to identify hidden variable tokens")
print("2. Train probes on the activations")
print("3. Run intervention experiments")
print()
