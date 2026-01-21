"""
Annotate Reasoning Traces - Manual Token Identification

This script loads saved reasoning traces and allows manual identification of tokens
that contain the hidden variable value. The annotations are saved to a CSV file.

Usage:
    python annotate_traces.py --experiment velocity --model_name Qwen2.5-32B
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import re

# ==========================================
# CONFIGURATION
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='velocity')
parser.add_argument('--model_name', type=str, default='Qwen2.5-32B')
parser.add_argument('--start_idx', type=int, default=0,
                    help='Start from this trace index (for resuming)')
args = parser.parse_args()

# Input directory
TRACES_DIR = Path.home() / 'scratch' / 'reasoning_traces' / args.model_name / args.experiment
ANNOTATIONS_FILE = TRACES_DIR / 'annotations.csv'

print("="*70)
print("ANNOTATE REASONING TRACES")
print("="*70)
print(f"Experiment: {args.experiment}")
print(f"Traces directory: {TRACES_DIR}")
print()

# ==========================================
# LOAD TRACES
# ==========================================

metadata_file = TRACES_DIR / 'traces_metadata.json'
if not metadata_file.exists():
    print(f"ERROR: Metadata file not found: {metadata_file}")
    print("Please run generate_traces.py first.")
    exit(1)

with open(metadata_file, 'r') as f:
    traces = json.load(f)

print(f"Loaded {len(traces)} traces")
print()

# Load existing annotations if available
if ANNOTATIONS_FILE.exists():
    existing_annotations = pd.read_csv(ANNOTATIONS_FILE)
    print(f"Found {len(existing_annotations)} existing annotations")
    annotated_ids = set(existing_annotations['trace_id'].values)
else:
    existing_annotations = None
    annotated_ids = set()

# ==========================================
# ANNOTATION INTERFACE
# ==========================================

def display_trace(trace, show_full=False):
    """Display trace information for annotation."""
    print("\n" + "="*70)
    print(f"TRACE {trace['id']}")
    print("="*70)
    print(f"\nPrompt: {trace['prompt']}")
    print(f"\nHidden variable ({args.experiment}):")
    if args.experiment == 'velocity':
        print(f"  v = {trace['v']} m/s")
    elif args.experiment == 'current':
        print(f"  i = {trace['i']} A")
    
    print(f"\nGenerated text:")
    print("-"*70)
    
    if show_full:
        print(trace['generated_text'])
    else:
        # Show first 500 chars
        text = trace['generated_text']
        if len(text) > 500:
            print(text[:500] + "\n... (truncated, type 'f' to see full)")
        else:
            print(text)
    
    print("-"*70)
    print(f"\nPrompt length: {trace['prompt_length']} tokens")
    print(f"Total tokens: {len(trace['tokens'])} tokens")
    print(f"Generated tokens: {len(trace['tokens']) - trace['prompt_length']}")

def show_tokens_with_indices(trace, start=0, end=None):
    """Show tokens with their indices."""
    tokens = trace['token_strings']
    if end is None:
        end = len(tokens)
    
    print(f"\nTokens [{start}:{end}]:")
    print("-"*70)
    
    for i in range(start, min(end, len(tokens))):
        token_display = repr(tokens[i])[1:-1]  # Remove quotes
        prefix = "[PROMPT]" if i < trace['prompt_length'] else "[GEN]"
        print(f"  {i:4d} {prefix:10s} {token_display}")
    
    print("-"*70)

def find_value_in_text(trace):
    """Try to automatically find where the hidden variable appears."""
    if args.experiment == 'velocity':
        value = trace['v']
        patterns = [
            rf'velocity.*?{value}',
            rf'speed.*?{value}',
            rf'{value}\s*m/s',
            rf'v\s*=\s*{value}',
        ]
    elif args.experiment == 'current':
        value = trace['i']
        patterns = [
            rf'current.*?{value}',
            rf'{value}\s*A',
            rf'{value}\s*ampere',
            rf'I\s*=\s*{value}',
        ]
    else:
        return None
    
    text = trace['generated_text']
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Find approximate character position
            char_pos = match.start()
            
            # Estimate token position (rough)
            # Average ~4 chars per token
            est_token = trace['prompt_length'] + char_pos // 4
            
            print(f"\n[AUTO-DETECT] Found pattern '{pattern}' around token {est_token}")
            return est_token
    
    return None

def annotate_trace(trace):
    """Interactive annotation for a single trace."""
    display_trace(trace)
    
    # Try auto-detection
    auto_token = find_value_in_text(trace)
    
    while True:
        print("\n" + "="*70)
        print("ANNOTATION OPTIONS:")
        print("  [s] Show tokens (will prompt for range)")
        print("  [f] Show full generated text")
        print("  [a] Accept auto-detected position (if available)")
        print("  [m] Manual entry (specify token indices)")
        print("  [n] No hidden variable found (mark as unclear)")
        print("  [skip] Skip this trace")
        print("  [quit] Save and quit")
        print("="*70)
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice == 's':
            start = input("Start token index (default=prompt_length): ").strip()
            start = trace['prompt_length'] if not start else int(start)
            end = input(f"End token index (default={start+50}): ").strip()
            end = start + 50 if not end else int(end)
            show_tokens_with_indices(trace, start, end)
            
        elif choice == 'f':
            display_trace(trace, show_full=True)
            
        elif choice == 'a' and auto_token is not None:
            # Show context around auto-detected token
            show_tokens_with_indices(trace, max(0, auto_token - 10), auto_token + 10)
            confirm = input(f"\nAccept token {auto_token}? (y/n): ").strip().lower()
            if confirm == 'y':
                start_idx = int(input("Start token index: ").strip())
                end_idx = int(input("End token index (exclusive): ").strip())
                return {
                    'trace_id': trace['id'],
                    'hidden_var_start': start_idx,
                    'hidden_var_end': end_idx,
                    'annotation_type': 'auto-assisted'
                }
        
        elif choice == 'm':
            print("\nEnter the token indices where the hidden variable appears:")
            print("(The value might span multiple tokens)")
            try:
                start_idx = int(input("  Start token index: ").strip())
                end_idx = int(input("  End token index (exclusive): ").strip())
                
                # Show selected tokens for confirmation
                print(f"\nSelected tokens [{start_idx}:{end_idx}]:")
                for i in range(start_idx, end_idx):
                    if i < len(trace['token_strings']):
                        print(f"  {i}: {repr(trace['token_strings'][i])}")
                
                confirm = input("\nIs this correct? (y/n): ").strip().lower()
                if confirm == 'y':
                    return {
                        'trace_id': trace['id'],
                        'hidden_var_start': start_idx,
                        'hidden_var_end': end_idx,
                        'annotation_type': 'manual'
                    }
            except ValueError:
                print("Invalid input. Please enter integers.")
        
        elif choice == 'n':
            return {
                'trace_id': trace['id'],
                'hidden_var_start': -1,
                'hidden_var_end': -1,
                'annotation_type': 'not_found'
            }
        
        elif choice == 'skip':
            return None
        
        elif choice == 'quit':
            return 'QUIT'
        
        else:
            print("Invalid choice. Please try again.")

# ==========================================
# MAIN ANNOTATION LOOP
# ==========================================

annotations = []

# Load existing annotations if available
if existing_annotations is not None:
    annotations = existing_annotations.to_dict('records')

print("\nStarting annotation...")
print(f"Already annotated: {len(annotated_ids)} traces")
print(f"Remaining: {len(traces) - len(annotated_ids)} traces")
print()

for trace in traces[args.start_idx:]:
    # Skip if already annotated
    if trace['id'] in annotated_ids:
        continue
    
    result = annotate_trace(trace)
    
    if result == 'QUIT':
        print("\nQuitting...")
        break
    elif result is None:
        print("Skipped.")
        continue
    else:
        annotations.append(result)
        annotated_ids.add(trace['id'])
        
        # Save after each annotation
        df = pd.DataFrame(annotations)
        df.to_csv(ANNOTATIONS_FILE, index=False)
        
        print(f"\n✓ Annotation saved! ({len(annotations)} total)")

# ==========================================
# SAVE AND SUMMARY
# ==========================================

df = pd.DataFrame(annotations)
df.to_csv(ANNOTATIONS_FILE, index=False)

print("\n" + "="*70)
print("ANNOTATION SESSION COMPLETE")
print("="*70)
print(f"Total annotations: {len(annotations)}")
print(f"Saved to: {ANNOTATIONS_FILE}")
print()

if len(annotations) > 0:
    print("Annotation breakdown:")
    print(df['annotation_type'].value_counts())
    print()
    
    found = df[df['hidden_var_start'] >= 0]
    if len(found) > 0:
        avg_start = found['hidden_var_start'].mean()
        avg_end = found['hidden_var_end'].mean()
        print(f"Average hidden variable position: {avg_start:.1f} - {avg_end:.1f}")
        print(f"Hidden variables found: {len(found)}/{len(annotations)}")
