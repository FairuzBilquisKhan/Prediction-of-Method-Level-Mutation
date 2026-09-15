"""
parse_mutants.py
Parses Major's mutants.log + kill.csv into a single per-mutant CSV.

mutants.log format (colon-separated, 7 fields):
    mutant_id:operator:original:mutated:method_signature:line:code_change

  - method_signature = "package.Class@method" (Major's format, no param types)
  - code_change can contain colons, so use maxsplit=6

kill.csv format (header + comma-separated):
    MutantNo,[...],Status,[...]
  - We need MutantNo and Status; killing test column may also exist

Usage:
    python parse_mutants.py \
        --mutation-dir mutation_outputs/Lang \
        --output mutants_parsed.csv
"""

import argparse
import csv
from pathlib import Path

import pandas as pd


# Statuses Major can emit. KILLED/TIME/EXC count as killed; LIVE is survived;
# FAIL is a compilation failure and gets excluded from the score denominator.
KILLED_STATUSES = {"KILLED", "TIME", "EXC"}
SURVIVED_STATUSES = {"LIVE"}
EXCLUDED_STATUSES = {"FAIL"}


def parse_mutants_log(path: Path) -> pd.DataFrame:
    """Parse one bug's mutants.log file. Returns a DataFrame with one row per mutant."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            # IMPORTANT: split with maxsplit=6 because the last field
            # (code_change) can contain colons in expressions like `a < b ? x : y`
            parts = line.split(":", 6)
            if len(parts) < 7:
                # Malformed line; skip but log
                print(f"  [warn] malformed mutants.log line: {line[:80]}")
                continue
            mutant_id, operator, original, mutated, method_sig, line_num, code_change = parts

            # Parse "package.Class@methodName" → (class_fqn, method_name)
            if "@" in method_sig:
                class_fqn, method_name = method_sig.rsplit("@", 1)
            else:
                class_fqn, method_name = method_sig, ""

            try:
                line_num_int = int(line_num)
            except ValueError:
                line_num_int = -1

            rows.append({
                "mutant_id": int(mutant_id),
                "operator": operator,
                "original_token": original,
                "mutated_token": mutated,
                "class_fqn": class_fqn,
                "method_name": method_name,
                "line": line_num_int,
                "code_change": code_change,
            })
    return pd.DataFrame(rows)


def parse_kill_csv(path: Path) -> pd.DataFrame:
    """Parse one bug's kill.csv. Returns a DataFrame with mutant_id and status."""
    df = pd.read_csv(path)
    # Major's column names vary slightly across versions; normalize them
    rename_map = {
        "MutantNo": "mutant_id",
        "Mutant No": "mutant_id",
        "Status": "status",
        "[Status]": "status",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "mutant_id" not in df.columns or "status" not in df.columns:
        raise ValueError(f"Unexpected kill.csv columns in {path}: {list(df.columns)}")
    df["mutant_id"] = df["mutant_id"].astype(int)
    df["status"] = df["status"].astype(str).str.strip()
    return df[["mutant_id", "status"]]


def process_one_bug(bug_dir: Path, project: str, bug_id: str) -> pd.DataFrame:
    """Process one bug's outputs. Returns a DataFrame of mutants joined with kill status."""
    mutants_log = bug_dir / "mutants.log"
    kill_csv = bug_dir / "kill.csv"

    if not mutants_log.exists():
        print(f"  [warn] missing mutants.log in {bug_dir}")
        return pd.DataFrame()
    if not kill_csv.exists():
        print(f"  [warn] missing kill.csv in {bug_dir}")
        return pd.DataFrame()

    mutants = parse_mutants_log(mutants_log)
    kills = parse_kill_csv(kill_csv)

    merged = mutants.merge(kills, on="mutant_id", how="left")
    merged["project"] = project
    merged["bug_id"] = bug_id

    # Annotate killed/survived/excluded based on status
    merged["is_killed"] = merged["status"].isin(KILLED_STATUSES)
    merged["is_survived"] = merged["status"].isin(SURVIVED_STATUSES)
    merged["is_excluded"] = merged["status"].isin(EXCLUDED_STATUSES)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutation-dir", type=Path, required=True,
                        help="Directory containing per-bug subdirectories")
    parser.add_argument("--project", type=str, default="Lang")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    all_dfs = []
    bug_dirs = sorted([d for d in args.mutation_dir.iterdir() if d.is_dir()])
    print(f"Found {len(bug_dirs)} bug directories in {args.mutation_dir}")

    for bug_dir in bug_dirs:
        bug_id = bug_dir.name
        print(f"Processing {args.project}-{bug_id} ...")
        df = process_one_bug(bug_dir, args.project, bug_id)
        if not df.empty:
            print(f"  {len(df)} mutants  "
                  f"({df['is_killed'].sum()} killed, "
                  f"{df['is_survived'].sum()} survived, "
                  f"{df['is_excluded'].sum()} excluded)")
            all_dfs.append(df)

    if not all_dfs:
        print("No data extracted. Exiting.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nWrote {len(combined)} rows to {args.output}")
    print(f"Killed: {combined['is_killed'].sum()}  "
          f"Survived: {combined['is_survived'].sum()}  "
          f"Excluded: {combined['is_excluded'].sum()}")


if __name__ == "__main__":
    main()
