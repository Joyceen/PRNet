"""Extract NC4K dataset from HuggingFace parquet to standard Imgs/ + GT/ format."""
import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq


def extract_parquet(parquet_path: str, output_dir: str, split: str):
    os.makedirs(f"{output_dir}/Imgs", exist_ok=True)
    os.makedirs(f"{output_dir}/GT", exist_ok=True)

    table = pq.read_table(parquet_path)
    print(f"[{split}] Reading {parquet_path}: {table.num_rows} rows")

    for i in range(table.num_rows):
        row = table.slice(i, 1)
        img = row.column("image")[0].as_py()
        gt = row.column("gt")[0].as_py()

        img_path = os.path.join(output_dir, "Imgs", img["path"])
        gt_path = os.path.join(output_dir, "GT", gt["path"])

        with open(img_path, "wb") as f:
            f.write(img["bytes"])
        with open(gt_path, "wb") as f:
            f.write(gt["bytes"])

        if (i + 1) % 100 == 0:
            print(f"  [{split}] Extracted {i + 1}/{table.num_rows}")

    print(f"[{split}] Done! {table.num_rows} images + masks extracted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract NC4K from HuggingFace parquet")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.expanduser("~/Documents/Python/datasets/NC4K/data"),
                        help="NC4K parquet data directory")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/Documents/Python/datasets/NC4K_extracted"),
                        help="Output directory for extracted images")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["test"],
                        choices=["train", "test", "validation"],
                        help="Splits to extract")
    args = parser.parse_args()

    for split in args.splits:
        parquet_path = os.path.join(args.data_dir, f"{split}-00000-of-00001.parquet")
        if not os.path.exists(parquet_path):
            print(f"Warning: {parquet_path} not found, skipping {split}")
            continue
        output_dir = os.path.join(args.output_dir, split)
        extract_parquet(parquet_path, output_dir, split)


if __name__ == "__main__":
    main()
