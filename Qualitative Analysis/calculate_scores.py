import os
from jiwer import cer, wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data (uncomment if running first time)
nltk.download('punkt')

def read_file(file_path):
    """Read a file and return lines as a list, stripping whitespace."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def calculate_metrics(reference_lines, hypothesis_lines):
    """
    Calculate CER, WER, and BLEU scores between reference and hypothesis.
    """
    if len(reference_lines) != len(hypothesis_lines):
        print(f"Warning: Different number of lines - Reference: {len(reference_lines)}, Hypothesis: {len(hypothesis_lines)}")
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

def main():
    # Define file paths
    base_path = os.path.join("Transliteration", "Qualitative Analysis")
    
    reference_file = os.path.join(base_path, "devanagari.txt")
    
    tool_files = {
        "Easy Nepali Typing": os.path.join(base_path, "1_easy_nepali_typing.txt"),
        "Kokil Thapa": os.path.join(base_path, "4_kokil_dot_com.txt"),
        "Nepali Unicode": os.path.join(base_path, "3_nepali_unicode_dot_com.txt"),
        "Ashesh": os.path.join(base_path, "2_ashesh_dot_com.txt"),
        "IndicXlit": os.path.join(base_path, "5_indicxlit.txt"),
        "Google Transliteration": os.path.join(base_path, "6_google_translit.txt"),
        "ByT5 (Normalized)": os.path.join(base_path, "7_byt5_normalized.txt"),
        "ByT5 (Raw)": os.path.join(base_path, "8_byt5_raw.txt")
    }
    
    # Read reference file
    print("Reading reference file...")
    try:
        reference_lines = read_file(reference_file)
        print(f"Loaded {len(reference_lines)} reference lines")
    except FileNotFoundError:
        print(f"Error: Reference file not found at {reference_file}")
        return
    except Exception as e:
        print(f"Error reading reference file: {e}")
        return
    
    # Process each tool
    results = {}
    
    for tool_name, tool_file in tool_files.items():
        print(f"\nProcessing {tool_name}...")
        try:
            hypothesis_lines = read_file(tool_file)
            print(f"  Loaded {len(hypothesis_lines)} hypothesis lines")
            
            # Calculate metrics
            cer_score, wer_score, bleu_score = calculate_metrics(reference_lines, hypothesis_lines)
            
            results[tool_name] = {
                'CER': cer_score,
                'WER': wer_score,
                'BLEU': bleu_score
            }
            
            print(f"  CER: {cer_score:.4f}")
            print(f"  WER: {wer_score:.4f}")
            print(f"  BLEU: {bleu_score:.4f}")
            
        except FileNotFoundError:
            print(f"  Error: File not found at {tool_file}")
            results[tool_name] = {'CER': None, 'WER': None, 'BLEU': None}
        except Exception as e:
            print(f"  Error processing {tool_name}: {e}")
            results[tool_name] = {'CER': None, 'WER': None, 'BLEU': None}
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Tool':<20} {'CER':<12} {'WER':<12} {'BLEU':<12}")
    print("-"*56)
    
    for tool_name, metrics in results.items():
        cer_str = f"{metrics['CER']:.4f}" if metrics['CER'] is not None else "N/A"
        wer_str = f"{metrics['WER']:.4f}" if metrics['WER'] is not None else "N/A"
        bleu_str = f"{metrics['BLEU']:.4f}" if metrics['BLEU'] is not None else "N/A"
        print(f"{tool_name:<20} {cer_str:<12} {wer_str:<12} {bleu_str:<12}")
    
    # Find best tool for each metric
    print("\n" + "="*80)
    print("BEST TOOL PER METRIC")
    print("="*80)
    
    for metric in ['CER', 'WER', 'BLEU']:
        valid_results = {k: v[metric] for k, v in results.items() if v[metric] is not None}
        if valid_results:
            if metric == 'BLEU':
                best_tool = max(valid_results, key=valid_results.get)
            else:
                best_tool = min(valid_results, key=valid_results.get)
            print(f"{metric}: {best_tool} ({valid_results[best_tool]:.4f})")

if __name__ == "__main__":
    main()