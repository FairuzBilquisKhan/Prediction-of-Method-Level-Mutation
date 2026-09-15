#!/bin/bash
# run_mutation.sh
# Runs `defects4j mutation` on fixed versions of the Lang project.
# Saves mutants.log and kill.csv per bug ID into ./mutation_outputs/
#
# Usage:
#   ./run_mutation.sh              # runs all bug IDs
#   ./run_mutation.sh 1 2 3        # runs specific bug IDs
#
# Prerequisites:
#   - defects4j installed and on PATH
#   - Java 8+ available
#   - At least a few GB of free disk space (each checkout is ~50MB)

set -e  # exit on error

PROJECT="Lang"
OUTPUT_DIR="$(pwd)/mutation_outputs/${PROJECT}"
WORK_DIR="/tmp/d4j_work"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORK_DIR}"

# Determine which bug IDs to run
if [ "$#" -eq 0 ]; then
    # Get the list of active (non-deprecated) bug IDs for Lang
    BUG_IDS=$(defects4j query -p "${PROJECT}" -q "bug.id" | cut -d',' -f1 | tail -n +2)
else
    BUG_IDS="$@"
fi

echo "Will process bug IDs: ${BUG_IDS}"
echo "Output directory: ${OUTPUT_DIR}"

for BUG_ID in ${BUG_IDS}; do
    echo "============================================"
    echo "Processing ${PROJECT}-${BUG_ID}f"
    echo "============================================"

    BUG_OUTPUT="${OUTPUT_DIR}/${BUG_ID}"
    if [ -f "${BUG_OUTPUT}/kill.csv" ]; then
        echo "Already processed, skipping."
        continue
    fi
    mkdir -p "${BUG_OUTPUT}"

    CHECKOUT_DIR="${WORK_DIR}/${PROJECT}_${BUG_ID}f"
    rm -rf "${CHECKOUT_DIR}"

    # Check out fixed version
    if ! defects4j checkout -p "${PROJECT}" -v "${BUG_ID}f" -w "${CHECKOUT_DIR}"; then
        echo "Checkout failed for ${PROJECT}-${BUG_ID}f, skipping."
        continue
    fi

    # Run mutation analysis
    cd "${CHECKOUT_DIR}"
    if ! defects4j mutation; then
        echo "Mutation failed for ${PROJECT}-${BUG_ID}f"
        cd - > /dev/null
        continue
    fi

    # Copy outputs to permanent location
    # Major writes outputs to the project root by default
    for f in mutants.log kill.csv summary.csv; do
        if [ -f "${f}" ]; then
            cp "${f}" "${BUG_OUTPUT}/"
        fi
    done

    cd - > /dev/null

    # Clean up the working checkout to save disk space
    rm -rf "${CHECKOUT_DIR}"

    echo "Done: ${PROJECT}-${BUG_ID}f"
done

echo "============================================"
echo "All done. Outputs in ${OUTPUT_DIR}"
echo "============================================"
