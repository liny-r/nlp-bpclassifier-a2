#!/usr/bin/env bash
# Launch the BPClassifier GUI.
# Automatically finds the Anaconda/Miniconda Streamlit that has setfit installed.
#
# Usage:
#   bash run_gui.sh
#   bash run_gui.sh --server.port 8502   # optional Streamlit flags

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI="$SCRIPT_DIR/gui.py"

# Search common Anaconda / Miniconda install locations
for candidate in \
    "$HOME/anaconda3/bin/streamlit" \
    "$HOME/miniconda3/bin/streamlit" \
    "$HOME/opt/anaconda3/bin/streamlit" \
    "$HOME/opt/miniconda3/bin/streamlit" \
    "/opt/anaconda3/bin/streamlit" \
    "/opt/miniconda3/bin/streamlit"
do
    if [ -x "$candidate" ]; then
        echo "Using: $candidate"
        exec "$candidate" run "$GUI" "$@"
    fi
done

# Fall back to 'conda run' if conda is on PATH
if command -v conda &>/dev/null; then
    echo "Using: conda run -n base streamlit"
    exec conda run -n base streamlit run "$GUI" "$@"
fi

# Last resort — whatever 'streamlit' is on PATH
echo "Warning: Anaconda not found; falling back to system streamlit."
echo "If setfit is missing, install it: pip install setfit"
exec streamlit run "$GUI" "$@"
