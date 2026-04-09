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

rm -rf "$SCRIPT_DIR/windmill_venv"
python3.11 -m venv "$SCRIPT_DIR/windmill_venv"
source "$SCRIPT_DIR/windmill_venv/bin/activate"

pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

mkdir -p "$SCRIPT_DIR/config"

echo "Installing NLTK corpora required by consensus workers..."
"$SCRIPT_DIR/windmill_venv/bin/python" - <<'PY'
import nltk
from nltk.data import find

required = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
}

missing = []
for name, path in required.items():
    try:
        find(path)
        print(f"NLTK resource already installed: {name}")
    except LookupError:
        missing.append(name)

if missing:
    print(f"Downloading NLTK resources: {', '.join(missing)}")
    for name in missing:
        nltk.download(name)
else:
    print("All required NLTK resources already installed.")
PY

echo "Installing spaCy model required by consensus workers..."
"$SCRIPT_DIR/windmill_venv/bin/python" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("en_core_web_lg") is not None:
    print("spaCy model already installed: en_core_web_lg")
else:
    print("Downloading spaCy model: en_core_web_lg")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_lg"], check=True)
PY

if [ -f "$SCRIPT_DIR/.env" ]; then
  set +e
  source "$SCRIPT_DIR/.env"
  set -e
fi

if [ -n "$QUEUE_HOST" ] && [ -n "$QUEUE_USER" ] && [ -n "$QUEUE_PASSWORD" ] && [ -f "$SCRIPT_DIR/utils/predeclare_queues.py" ]; then
  echo "Predeclaring RabbitMQ queues from service_config.yaml..."
  "$SCRIPT_DIR/windmill_venv/bin/python" "$SCRIPT_DIR/utils/predeclare_queues.py"
else
  echo "Skipping queue predeclaration."
  echo "Requires QUEUE_HOST/QUEUE_USER/QUEUE_PASSWORD and utils/predeclare_queues.py."
  echo "When available, run: $SCRIPT_DIR/windmill_venv/bin/python $SCRIPT_DIR/utils/predeclare_queues.py"
fi

echo "Done. Run ./windmill.sh start to start workers."
