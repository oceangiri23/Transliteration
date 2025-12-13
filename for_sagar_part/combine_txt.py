#!/usr/bin/env python3
"""
Combine all .txt files in a directory into a single output file.

Usage (PowerShell):
  python .\for_sagar_part\combine_txt.py -i . -o all_merged.txt -r

Options:
  -i, --input-dir   Directory to scan for .txt files (default: current directory)
  -o, --output-file Output file path (default: all_merged.txt)
  -e, --encoding    File encoding to use (default: utf-8)
  -r, --recursive   Search subdirectories recursively
  -s, --skip        Comma-separated filename substrings to skip
"""

from pathlib import Path
import argparse


def combine_txt_files(input_dir: str = '.', output_file: str = 'all_merged.txt', encoding: str = 'utf-8', recursive: bool = False, skip_patterns=None):
    p = Path(input_dir)
    if recursive:
        txt_files = sorted([f for f in p.rglob('*.txt') if f.is_file()])
    else:
        txt_files = sorted([f for f in p.glob('*.txt') if f.is_file()])

    out_name = Path(output_file).name
    # Exclude the output file if it would be picked up
    txt_files = [f for f in txt_files if f.name != out_name]

    if skip_patterns:
        txt_files = [f for f in txt_files if not any(pat in f.name for pat in skip_patterns)]

    if not txt_files:
        print(f'No .txt files found in {input_dir} (after filters).')
        return

    total_bytes = 0
    with open(output_file, 'w', encoding=encoding) as out_f:
        for tf in txt_files:
            try:
                with open(tf, 'r', encoding=encoding, errors='replace') as in_f:
                    content = in_f.read()
                    out_f.write(content)
                    out_f.write('\n')
                    total_bytes += len(content.encode(encoding, errors='replace'))
            except Exception as e:
                print(f'Warning: failed to read {tf}: {e}')

    print(f'Combined {len(txt_files)} files ({total_bytes} bytes) into {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine all .txt files in a directory into one file')
    parser.add_argument('-i', '--input-dir', default='.', help='Directory to scan for .txt files')
    parser.add_argument('-o', '--output-file', default='all_merged.txt', help='Output file path')
    parser.add_argument('-e', '--encoding', default='utf-8', help='Encoding to use for reading/writing')
    parser.add_argument('-r', '--recursive', action='store_true', help='Search subdirectories recursively')
    parser.add_argument('-s', '--skip', default='', help='Comma-separated substrings of filenames to skip')

    args = parser.parse_args()
    skip_patterns = [s for s in (args.skip.split(',') if args.skip else []) if s]

    combine_txt_files(input_dir=args.input_dir, output_file=args.output_file, encoding=args.encoding, recursive=args.recursive, skip_patterns=skip_patterns)
