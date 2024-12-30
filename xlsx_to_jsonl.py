#!/usr/bin/env python3
"""
File Name: xlsx_to_jsonl.py

Description:
    This script reads an Excel (.xlsx) file (with the example columns provided)
    and writes out a JSONL file. Only the following fields are extracted:

      1) Short_Description  <- from "Short Description"
      2) Description        <- from "Description"
      3) Close_Notes        <- from "Close Notes"
      4) Category           <- from the first "Category" column in the sheet
      5) Subcategory        <- from "Subcategory"
      6) Assignment_Group   <- from "Assignment Group"

    Missing/blank cells become None (null in JSON).

Usage:
    python xlsx_to_jsonl.py input.xlsx output.jsonl

Implementation Notes:
    - We assume the Excel file has column headers matching exactly:
      "Short Description", "Description", "Close Notes", "Category",
      "Subcategory", "Assignment Group".
    - The script requires 'pandas' and 'openpyxl' to be installed.
    - All output fields are forced to strings except for blanks, which become None.

Example:
    python xlsx_to_jsonl.py incident_data.xlsx incident_data.jsonl
"""

import sys
import json
import pandas as pd

def convert_xlsx_to_jsonl(input_file: str, output_file: str) -> None:
    # Read the Excel file into a pandas DataFrame
    # Force all columns to string types so blanks will be 'NaN' if truly empty
    df = pd.read_excel(input_file, dtype=str)

    # If the columns you want are guaranteed to exist, you can proceed directly.
    # Otherwise, you might want to validate that these columns exist first.
    # We define a function to handle the blank -> None conversion for each field.
    def get_value(row, col_name):
        raw_val = row.get(col_name, "")
        if pd.isna(raw_val) or raw_val.strip() == "":
            return None
        return raw_val.strip()

    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for _, row in df.iterrows():
            record = {
                "Short_Description": get_value(row, "Short Description"),
                "Description": get_value(row, "Description"),
                "Close_Notes": get_value(row, "Close Notes"),
                "Category": get_value(row, "Category"),
                "Subcategory": get_value(row, "Subcategory"),
                "Assignment_Group": get_value(row, "Assignment Group")
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python xlsx_to_jsonl.py <input_file.xlsx> <output_file.jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_xlsx_to_jsonl(input_file, output_file)
    print(f"Conversion complete. JSONL written to: {output_file}")

if __name__ == "__main__":
    main()
