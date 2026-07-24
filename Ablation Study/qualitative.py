"""
Evaluation script for ablation study across all 4 models.
Calculates CER, WER, and BLEU scores for each model's ablation configurations.
"""

import os
from jiwer import cer, wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import glob

# Download required NLTK data (uncomment if running first time)
# nltk.download('punkt')

def read_file(file_path):
    """Read a file and return lines as a list, stripping whitespace."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except FileNotFoundError:
        return None

def calculate_metrics(reference_lines, hypothesis_lines):
    """
    Calculate CER, WER, and BLEU scores between reference and hypothesis.
    """
    if not hypothesis_lines:
        return None, None, None
    
    if len(reference_lines) != len(hypothesis_lines):
        print(f"  Warning: Different number of lines - Ref: {len(reference_lines)}, Hyp: {len(hypothesis_lines)}")
        # Use the minimum length
        min_len = min(len(reference_lines), len(hypothesis_lines))
        reference_lines = reference_lines[:min_len]
        hypothesis_lines = hypothesis_lines[:min_len]

    # For CER and WER, combine all lines into single strings
    reference_text = ' '.join(reference_lines)
    hypothesis_text = ' '.join(hypothesis_lines)

    # Calculate CER and WER
    cer_score = cer(reference_text, hypothesis_text)
    wer_score = wer(reference_text, hypothesis_text)

    # Calculate BLEU score (using smoothing to handle short sentences)
    smoothie = SmoothingFunction().method4
    bleu_scores = []

    for ref, hyp in zip(reference_lines, hypothesis_lines):
        # Tokenize by character for character-level BLEU
        ref_chars = list(ref)
        hyp_chars = list(hyp)

        # Handle empty strings
        if not hyp_chars or not ref_chars:
            bleu_scores.append(0.0)
            continue

        try:
            score = sentence_bleu([ref_chars], hyp_chars,
                                smoothing_function=smoothie,
                                weights=(0.25, 0.25, 0.25, 0.25))
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    return cer_score, wer_score, avg_bleu

def evaluate_model(model_name, output_dir, reference_lines):
    """
    Evaluate all ablation configurations for a single model.
    Returns a dictionary with results for each config.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Define ablation configurations
    configs = {
        "combined": "Rule-Based + PVD Normalization",
        "rule_based": "Rule-Based Normalization",
        "pvd_only": "PVD Normalization",
        "raw": "Raw (No Normalization)"
    }
    
    results = {}
    
    for config_key, config_name in configs.items():
        # Construct file path
        file_pattern = f"{model_name}_{config_key}.txt"
        file_path = os.path.join(output_dir, file_pattern)
        
        print(f"\n  Processing {config_name}...")
        
        # Read hypothesis file
        hypothesis_lines = read_file(file_path)
        if hypothesis_lines is None:
            print(f"    Error: File not found at {file_path}")
            results[config_name] = {'CER': None, 'WER': None, 'BLEU': None}
            continue
        
        print(f"    Loaded {len(hypothesis_lines)} hypothesis lines")
        
        # Calculate metrics
        cer_score, wer_score, bleu_score = calculate_metrics(reference_lines, hypothesis_lines)
        
        if cer_score is not None:
            results[config_name] = {
                'CER': cer_score,
                'WER': wer_score,
                'BLEU': bleu_score
            }
            print(f"    CER: {cer_score * 100:.4f}%")
            print(f"    WER: {wer_score * 100:.4f}%")
            print(f"    BLEU: {bleu_score * 100:.4f}")
        else:
            print(f"    Error calculating metrics")
            results[config_name] = {'CER': None, 'WER': None, 'BLEU': None}
    
    return results

def print_model_summary(model_name, results):
    """Print a summary table for a single model."""
    print(f"\n{'-'*60}")
    print(f"SUMMARY FOR: {model_name}")
    print(f"{'-'*60}")
    print(f"{'Config':<30} {'CER (%)':<12} {'WER (%)':<12} {'BLEU (0-100)':<12}")
    print("-"*60)
    
    # Sort by CER (ascending order - lower error rate first)
    sorted_results = sorted(
        results.items(),
        key=lambda item: item[1]['CER'] if item[1]['CER'] is not None else float('inf')
    )
    
    for config_name, metrics in sorted_results:
        cer_str = f"{metrics['CER'] * 100:.4f}" if metrics['CER'] is not None else "N/A"
        wer_str = f"{metrics['WER'] * 100:.4f}" if metrics['WER'] is not None else "N/A"
        bleu_str = f"{metrics['BLEU'] * 100:.4f}" if metrics['BLEU'] is not None else "N/A"
        print(f"{config_name:<30} {cer_str:<12} {wer_str:<12} {bleu_str:<12}")
    
    # Find best config for each metric
    print(f"\nBest Config for {model_name}:")
    for metric in ['CER', 'WER', 'BLEU']:
        valid_results = {k: v[metric] for k, v in results.items() if v[metric] is not None}
        if valid_results:
            if metric == 'BLEU':
                best = max(valid_results, key=valid_results.get)
                print(f"  {metric}: {best} ({valid_results[best] * 100:.4f})")
            else:
                best = min(valid_results, key=valid_results.get)
                print(f"  {metric}: {best} ({valid_results[best] * 100:.4f})")
    
    print(f"\nNote:\nCER/WER: Lower is Better.\nBLEU:    Higher is Better.")

def print_overall_comparison(all_results):
    """Print comparison of best configs across all models."""
    print("\n" + "="*80)
    print("OVERALL COMPARISON ACROSS ALL MODELS")
    print("="*80)
    
    # Collect best results for each model
    best_results = {}
    for model_name, results in all_results.items():
        best_results[model_name] = {}
        for metric in ['CER', 'WER', 'BLEU']:
            valid_results = {k: v[metric] for k, v in results.items() if v[metric] is not None}
            if valid_results:
                if metric == 'BLEU':
                    best_config = max(valid_results, key=valid_results.get)
                    best_value = valid_results[best_config]
                else:
                    best_config = min(valid_results, key=valid_results.get)
                    best_value = valid_results[best_config]
                best_results[model_name][metric] = {
                    'value': best_value,
                    'config': best_config
                }
            else:
                best_results[model_name][metric] = None
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Best Config':<30} {'CER (%)':<12} {'WER (%)':<12} {'BLEU (0-100)':<12}")
    print("-"*80)
    
    for model_name in sorted(best_results.keys()):
        cer_info = best_results[model_name].get('CER')
        wer_info = best_results[model_name].get('WER')
        bleu_info = best_results[model_name].get('BLEU')
        
        # Use the config from CER as the primary best config
        best_config = cer_info['config'] if cer_info else "N/A"
        
        cer_val = f"{cer_info['value'] * 100:.4f}" if cer_info else "N/A"
        wer_val = f"{wer_info['value'] * 100:.4f}" if wer_info else "N/A"
        bleu_val = f"{bleu_info['value'] * 100:.4f}" if bleu_info else "N/A"
        
        print(f"{model_name:<20} {best_config:<30} {cer_val:<12} {wer_val:<12} {bleu_val:<12}")
    
    # Find overall best model for each metric
    print(f"\nOVERALL BEST MODEL:")
    for metric in ['CER', 'WER', 'BLEU']:
        valid_models = {}
        for model_name, info in best_results.items():
            if info.get(metric):
                valid_models[model_name] = info[metric]['value']
        
        if valid_models:
            if metric == 'BLEU':
                best_model = max(valid_models, key=valid_models.get)
                print(f"  {metric}: {best_model} ({valid_models[best_model] * 100:.4f})")
            else:
                best_model = min(valid_models, key=valid_models.get)
                print(f"  {metric}: {best_model} ({valid_models[best_model] * 100:.4f})")

def main():
    # Define paths
    base_path = "."
    output_dir = os.path.join(base_path, "Transliteration/Ablation Study/Outputs")
    reference_file = os.path.join(base_path, "Transliteration/Ablation Study/devanagari_actual.txt")
    
    # Define models to evaluate
    models = [
        "byt5_original",
        "byt5_augmented",
        "mt5_original",
        "mt5_augmented"
    ]
    
    # Read reference file
    print("Reading reference file...")
    try:
        reference_lines = read_file(reference_file)
        if reference_lines is None:
            print(f"Error: Reference file not found at {reference_file}")
            return
        print(f"Loaded {len(reference_lines)} reference lines")
    except Exception as e:
        print(f"Error reading reference file: {e}")
        return
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found at {output_dir}")
        print(f"Please run the ablation study first to generate output files.")
        return
    
    # Evaluate each model
    all_results = {}
    for model_name in models:
        results = evaluate_model(model_name, output_dir, reference_lines)
        all_results[model_name] = results
        print_model_summary(model_name, results)
    
    # Print overall comparison
    print_overall_comparison(all_results)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

if __name__ == "__main__":
    main()