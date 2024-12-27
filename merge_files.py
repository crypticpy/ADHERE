#!/usr/bin/env python3
"""
merge_files.py

A Python script to gather all .py files from a project directory (recursively by default)
and combine them into a single text file. Each file is clearly separated and wrapped 
in code blocks to make it easier for an LLM to parse.

Usage:
    python merge_files.py --src /path/to/project --out merged_project.txt

Features:
  - Recursively collects all .py files.
  - Skips virtual environments or other common ignorable directories if desired.
  - Wraps each file’s contents in a code fence along with a header for file name.
  - Creates a single text file containing the entire project’s code, suitable for sharing with LLMs.
"""

import os
import argparse


def is_ignorable_path(path: str) -> bool:
    """
    Returns True if the path is inside a directory that we want to skip.
    Edit or add patterns as needed (e.g., '.venv', '__pycache__', '.git', 'node_modules').
    """
    ignorable_dirs = {".venv", "venv", "__pycache__", ".git", "node_modules"}
    parts = set(path.split(os.sep))
    return not ignorable_dirs.isdisjoint(parts)


def gather_python_files(src_dir: str):
    """
    Recursively walks through the given directory and yields paths of all .py files
    (excluding any ignorable directories).
    """
    for root, dirs, files in os.walk(src_dir):
        # Optionally remove ignorable directories from the walk
        dirs[:] = [d for d in dirs if not is_ignorable_path(os.path.join(root, d))]

        for f in files:
            if f.endswith(".py"):
                file_path = os.path.join(root, f)
                if not is_ignorable_path(file_path):
                    yield file_path


def merge_python_files(src_dir: str, out_file: str):
    """
    Reads all .py files in src_dir (recursively),
    merges them into out_file with code fences and file headers.
    """
    py_files = list(gather_python_files(src_dir))
    py_files.sort()  # optional sort by path

    with open(out_file, "w", encoding="utf-8") as out_f:
        out_f.write("# Merged Python Project\n\n")
        out_f.write(
            "Below are all .py files in the project. Each file is wrapped in a code block.\n\n"
        )

        for file_path in py_files:
            rel_path = os.path.relpath(file_path, src_dir)
            out_f.write(f"## File: {rel_path}\n\n")
            out_f.write("```python\n")
            with open(file_path, "r", encoding="utf-8") as in_f:
                content = in_f.read()
            out_f.write(content)
            out_f.write("\n```\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge all .py files from a project into one text file for LLM consumption."
    )
    parser.add_argument(
        "--src", required=True, help="Source directory of the Python project."
    )
    parser.add_argument(
        "--out", required=True, help="Output file name (e.g. merged_project.txt)."
    )
    args = parser.parse_args()

    merge_python_files(args.src, args.out)

    print(f"Merged files saved to {args.out}")


if __name__ == "__main__":
    main()
