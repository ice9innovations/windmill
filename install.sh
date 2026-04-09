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

"$SCRIPT_DIR/windmill_venv/bin/python" -c "import nltk; nltk.download('punkt')"

if [ -f "$SCRIPT_DIR/.env" ]; then
  set +e
  source "$SCRIPT_DIR/.env"
  set -e
fi

if [ -n "$QUEUE_HOST" ] && [ -n "$QUEUE_USER" ] && [ -n "$QUEUE_PASSWORD" ]; then
  echo "Predeclaring RabbitMQ queues from service_config.yaml..."
  "$SCRIPT_DIR/windmill_venv/bin/python" "$SCRIPT_DIR/utils/predeclare_queues.py"
else
  echo "Skipping queue predeclaration (QUEUE_HOST/QUEUE_USER/QUEUE_PASSWORD not configured)."
  echo "After filling in .env, run: $SCRIPT_DIR/windmill_venv/bin/python $SCRIPT_DIR/utils/predeclare_queues.py"
fi

echo "Done. Run ./windmill.sh start to start workers."
