"""
build_dataset.py
Final stage: join method features with mutants, aggregate mutants per method,
and compute the target column (mutation_score).

Join logic:
    For each mutant in mutants_parsed.csv, find the method in method_features.csv where:
        - project matches
        - bug_id matches
        - class_fqn matches
        - method_name matches
        - mutant.line falls within [method.start_line, method.end_line]

    The line-range check disambiguates overloaded methods.

Aggregation per method:
    num_total      = total mutants
    num_killed     = mutants with status in {KILLED, TIME, EXC}
    num_survived   = mutants with status == LIVE
    num_excluded   = mutants with status == FAIL  (dropped from denominator)
    mutation_score = num_killed / (num_killed + num_survived)
    frac_<OP>      = fraction of total mutants of each operator type

Target column:
    mutation_score (continuous, [0, 1])
    mutation_score_bin (categorical: low / medium / high)

Methods with zero mutants are dropped. Methods where num_killed + num_survived
is zero (all FAIL) are also dropped.

Usage:
    python build_dataset.py \
        --mutants mutants_parsed.csv \
        --methods method_features.csv \
        --output final_dataset.csv \
        --bin-low 0.4 --bin-high 0.8
"""

import argparse
from pathlib import Path

import pandas as pd


OPERATORS = ["ROR", "AOR", "LOR", "COR", "SOR", "ORU", "LVR", "STD"]


def join_mutants_to_methods(
    mutants: pd.DataFrame,
    methods: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match each mutant to its containing method using class+name+line-range.

    Strategy: do a regular merge on (project, bug_id, class_fqn, method_name),
    then filter rows where the mutant line falls within the method's line range.
    This handles overloaded methods (same name, different line ranges).
    """
    join_keys = ["project", "bug_id", "class_fqn", "method_name"]

    merged = mutants.merge(
        methods[join_keys + ["method_id", "start_line", "end_line"]],
        on=join_keys,
        how="left",
    )

    # Drop mutants that didn't match any method (e.g., inner-class artifacts)
    before = len(merged)
    merged = merged.dropna(subset=["start_line", "end_line"])
    after = len(merged)
    if before - after > 0:
        print(f"  [info] dropped {before - after} mutants with no matching method")

    # Line-range filter — handles overloaded methods
    merged["start_line"] = merged["start_line"].astype(int)
    merged["end_line"] = merged["end_line"].astype(int)
    in_range = (merged["line"] >= merged["start_line"]) & \
               (merged["line"] <= merged["end_line"])
    merged = merged[in_range].copy()
    print(f"  [info] {len(merged)} mutants matched to methods after line-range filter")

    return merged


def aggregate_per_method(joined: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mutants per method:
      - count total, killed, survived, excluded
      - compute mutation score
      - compute operator-type fractions
    """
    agg_rows = []
    for method_id, group in joined.groupby("method_id"):
        num_total = len(group)
        num_killed = int(group["is_killed"].sum())
        num_survived = int(group["is_survived"].sum())
        num_excluded = int(group["is_excluded"].sum())

        denom = num_killed + num_survived
        if denom == 0:
            # All mutants were FAIL; cannot compute score, skip method
            continue

        score = num_killed / denom

        # Operator distribution (fractions of total)
        op_counts = group["operator"].value_counts()
        op_fracs = {f"frac_{op}": op_counts.get(op, 0) / num_total for op in OPERATORS}

        agg_rows.append({
            "method_id": method_id,
            "num_total_mutants": num_total,
            "num_killed": num_killed,
            "num_survived": num_survived,
            "num_excluded": num_excluded,
            "mutation_score": score,
            **op_fracs,
        })

    return pd.DataFrame(agg_rows)


def bin_score(score: float, low: float, high: float) -> str:
    if score < low:
        return "low"
    elif score < high:
        return "medium"
    else:
        return "high"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants", type=Path, required=True,
                        help="Output of parse_mutants.py")
    parser.add_argument("--methods", type=Path, required=True,
                        help="Output of extract_features.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-low", type=float, default=0.4,
                        help="Boundary between low and medium")
    parser.add_argument("--bin-high", type=float, default=0.8,
                        help="Boundary between medium and high")
    args = parser.parse_args()

    print("Loading mutants ...")
    mutants = pd.read_csv(args.mutants)
    print(f"  {len(mutants)} mutants loaded")

    print("Loading method features ...")
    methods = pd.read_csv(args.methods)
    print(f"  {len(methods)} methods loaded")

    print("\nJoining mutants to methods ...")
    joined = join_mutants_to_methods(mutants, methods)

    print("\nAggregating per method ...")
    agg = aggregate_per_method(joined)
    print(f"  {len(agg)} methods with at least one non-FAIL mutant")

    print("\nMerging with method features ...")
    final = methods.merge(agg, on="method_id", how="inner")
    print(f"  {len(final)} rows in final dataset")

    # Compute the binned target
    final["mutation_score_bin"] = final["mutation_score"].apply(
        lambda s: bin_score(s, args.bin_low, args.bin_high)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"\nWrote final dataset to {args.output}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total methods: {len(final)}")
    print(f"\nMutation score distribution:")
    print(final["mutation_score"].describe())
    print(f"\nClass balance (mutation_score_bin):")
    print(final["mutation_score_bin"].value_counts())


if __name__ == "__main__":
    main()
