#!/usr/bin/env bash
# Generate a Synthea cohort tuned for medication reconciliation demos.
# Produces ~500 elderly patients with realistic polypharmacy in FHIR R4 Bundle
# JSON format under data/synthea_output/.
#
# Requirements:
#   - Java JDK 17+
#   - git
#   - ~2 GB disk
#
# Usage:
#   ./scripts/generate_synthea.sh              # default 500 patients
#   POP=100 ./scripts/generate_synthea.sh      # smaller cohort for dev

set -euo pipefail

POP="${POP:-500}"
SEED="${SEED:-42}"
AGE_RANGE="${AGE_RANGE:-55-90}"
STATE="${STATE:-Massachusetts}"
OUT_DIR="${OUT_DIR:-data/synthea_output}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYNTHEA_DIR="$REPO_ROOT/.synthea"

mkdir -p "$REPO_ROOT/$OUT_DIR"

if [ ! -d "$SYNTHEA_DIR" ]; then
    echo "→ Cloning Synthea into $SYNTHEA_DIR …"
    git clone --depth 1 https://github.com/synthetichealth/synthea.git "$SYNTHEA_DIR"
fi

cd "$SYNTHEA_DIR"

echo "→ Generating $POP patients, ages $AGE_RANGE, state=$STATE, seed=$SEED …"
./run_synthea \
    -s "$SEED" \
    -p "$POP" \
    -a "$AGE_RANGE" \
    --exporter.fhir.export=true \
    --exporter.fhir.bulk_data=false \
    --exporter.hospital.fhir.export=true \
    --exporter.practitioner.fhir.export=true \
    --exporter.years_of_history=10 \
    --exporter.baseDirectory="$REPO_ROOT/$OUT_DIR/" \
    "$STATE"

echo ""
echo "✓ Generation complete."
echo "  Output: $REPO_ROOT/$OUT_DIR/fhir/"
echo "  Bundle count: $(ls "$REPO_ROOT/$OUT_DIR/fhir/" 2>/dev/null | wc -l)"
echo ""
echo "Next: python scripts/upload_to_prompt_opinion.py"
