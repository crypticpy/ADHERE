#!/usr/bin/env python3

import sys
import json

def reformat_jsonl_auto(input_file, output_file):
    """
    Reads each line from input_file. If valid JSON, writes it to output_file.
    If invalid, skips it and logs the line number.

    The result is an automated pass/fail approach:
      - Good lines are kept.
      - Bad lines are discarded.
    """

    total_lines = 0
    valid_lines = 0
    invalid_lines = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, start=1):
            total_lines += 1
            stripped_line = line.strip()
            if not stripped_line:
                # If the line is empty, skip it
                invalid_lines += 1
                continue

            try:
                data = json.loads(stripped_line)
                # Re-dump as valid JSON (no extra indentation).
                # If you want pretty formatting, add indent=2 below.
                line_out = json.dumps(data, ensure_ascii=False)
                outfile.write(line_out + "\n")
                valid_lines += 1
            except json.JSONDecodeError:
                invalid_lines += 1

    print(f"Done.\n"
          f"Total lines read: {total_lines}\n"
          f"Valid lines: {valid_lines}\n"
          f"Invalid lines (skipped): {invalid_lines}\n"
          f"Reformatted file: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input_file.jsonl output_file.jsonl")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    reformat_jsonl_auto(input_path, output_path)
