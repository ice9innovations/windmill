#!/bin/bash
# Install Windmill worker dependencies into windmill_venv.
# Run once before first use. Requires Python 3.11.
#
# Usage:
#   bash install.sh
#
# After install, start workers with:
#   ./windmill.sh start
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

python3.11 -m venv "$SCRIPT_DIR/windmill_venv"
source "$SCRIPT_DIR/windmill_venv/bin/activate"

pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

"$SCRIPT_DIR/windmill_venv/bin/python" -c "import nltk; nltk.download('punkt')"

echo "Done. Run ./windmill.sh start to start workers."
