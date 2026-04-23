#!/usr/bin/env python3
"""Convert parquet files to human-readable formats (CSV/JSONL/XLSX).

Examples:
  python scripts/parquet_to_readable.py data/chunk-000/file-000.parquet
  python scripts/parquet_to_readable.py meta --recursive --format jsonl --outdir readable
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import argcomplete
    from argcomplete.completers import DirectoriesCompleter, FilesCompleter
except ImportError:  # pragma: no cover
    argcomplete = None
    DirectoriesCompleter = None
    FilesCompleter = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert .parquet file(s) into CSV, JSONL, or XLSX for easy reading."
    )
    input_arg = parser.add_argument(
        "input_path",
        type=Path,
        help="A parquet file or a directory containing parquet files.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl", "xlsx"],
        default="csv",
        help="Output format (default: csv).",
    )
    outdir_arg = parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("readable_outputs"),
        help="Output directory (default: readable_outputs).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search parquet files when input_path is a directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert first N rows from each parquet file (for quick inspection).",
    )

    if argcomplete is not None:
        # input_path supports both parquet files and directories
        input_arg.completer = FilesCompleter(allowednames=("parquet",), directories=True)
        # outdir should complete directories
        outdir_arg.completer = DirectoriesCompleter()

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    return parser.parse_args()


def collect_parquet_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".parquet":
        return [input_path]

    if input_path.is_dir():
        pattern = "**/*.parquet" if recursive else "*.parquet"
        return sorted(input_path.glob(pattern))

    return []


def to_output_path(src: Path, root: Path, outdir: Path, out_ext: str) -> Path:
    rel = src.relative_to(root) if root.is_dir() else src.name
    rel_path = Path(rel)
    return outdir / rel_path.with_suffix(out_ext)


def convert_file(src: Path, dst: Path, fmt: str, limit: int | None) -> tuple[int, list[str]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pandas. Install with `pip install pandas pyarrow`."
        ) from exc

    df = pd.read_parquet(src)
    if limit is not None:
        df = df.head(limit)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(dst, index=False)
    elif fmt == "jsonl":
        df.to_json(dst, orient="records", lines=True, force_ascii=False)
    else:
        try:
            df.to_excel(dst, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for xlsx export: openpyxl. "
                "Install with `pip install openpyxl`."
            ) from exc

    return len(df), list(df.columns)


def main() -> int:
    args = parse_args()

    input_path = args.input_path
    outdir = args.outdir
    fmt = args.format
    out_ext_map = {"csv": ".csv", "jsonl": ".jsonl", "xlsx": ".xlsx"}
    out_ext = out_ext_map[fmt]

    parquet_files = collect_parquet_files(input_path, args.recursive)
    if not parquet_files:
        print(f"No parquet files found in: {input_path}", file=sys.stderr)
        return 1

    root = input_path if input_path.is_dir() else input_path.parent

    print(f"Found {len(parquet_files)} parquet file(s). Converting to {fmt}...")
    ok = 0
    for src in parquet_files:
        dst = to_output_path(src, root, outdir, out_ext)
        try:
            rows, cols = convert_file(src, dst, fmt, args.limit)
            print(f"[OK] {src} -> {dst} (rows={rows}, cols={len(cols)})")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {src}: {exc}", file=sys.stderr)

    print(f"Done. Success: {ok}/{len(parquet_files)}")
    return 0 if ok == len(parquet_files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
