#!/bin/bash
# Install Windmill worker dependencies into windmill_venv.
# Run once before first use. Uses PYTHON if set, otherwise python3.11 when
# available, otherwise python3.
#
# Usage:
#   bash install.sh [--systemd all|workers|api|none]
#   bash install.sh --no-systemd
#
# By default, installs both API and worker-monitor systemd units rendered for
# the current checkout path/user. Set WINDMILL_SYSTEMD_MODE or pass --systemd
# to install a narrower unit set.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SYSTEMD_MODE="${WINDMILL_SYSTEMD_MODE:-all}"
STOP_RUNNING_SERVICES="${WINDMILL_STOP_RUNNING_SERVICES:-true}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --systemd)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --systemd requires one of: all, workers, api, none" >&2
        exit 1
      fi
      SYSTEMD_MODE="$2"
      shift 2
      ;;
    --no-systemd)
      SYSTEMD_MODE="none"
      shift
      ;;
    --no-stop)
      STOP_RUNNING_SERVICES="false"
      shift
      ;;
    -h|--help)
      cat <<USAGE
Usage: bash install.sh [--systemd all|workers|api|none] [--no-stop]
       bash install.sh --no-systemd

Installs Windmill dependencies into windmill_venv.

Systemd mode defaults to: \${WINDMILL_SYSTEMD_MODE:-all}
  all      Install API and worker-monitor services
  workers  Install only the worker monitor
  api      Install only the API service
  none     Skip systemd service installation

By default, running services/workers are stopped before rebuilding windmill_venv.
Use --no-stop only when you know Windmill is already stopped.

Python interpreter defaults to python3.11 when available, otherwise python3.
Override with PYTHON=/path/to/python.
USAGE
      exit 0
      ;;
    *)
      echo "ERROR: unknown option '$1'" >&2
      exit 1
      ;;
  esac
done

case "$SYSTEMD_MODE" in
  all|workers|api|none)
    ;;
  *)
    echo "ERROR: invalid systemd mode '$SYSTEMD_MODE'. Use all, workers, api, or none." >&2
    exit 1
    ;;
esac

select_python() {
  if [ -n "${PYTHON:-}" ]; then
    if command -v "$PYTHON" >/dev/null 2>&1; then
      command -v "$PYTHON"
      return
    fi
    echo "ERROR: PYTHON is set to '$PYTHON' but it is not executable." >&2
    exit 1
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  echo "ERROR: no Python interpreter found. Install python3 or set PYTHON=/path/to/python." >&2
  exit 1
}

PYTHON_BIN="$(select_python)"
echo "Using Python interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

stop_running_windmill() {
  if [ "$STOP_RUNNING_SERVICES" != "true" ]; then
    return
  fi

  echo "Stopping running Windmill services before rebuilding windmill_venv..."
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl stop windmill-workers.service 2>/dev/null || true
    sudo systemctl stop windmill.service 2>/dev/null || true
  fi

  if [ -x "$SCRIPT_DIR/windmill.sh" ]; then
    "$SCRIPT_DIR/windmill.sh" stop all || true
  fi
}

stop_running_windmill

rm -rf "$SCRIPT_DIR/windmill_venv"
"$PYTHON_BIN" -m venv "$SCRIPT_DIR/windmill_venv"
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
  if ! "$SCRIPT_DIR/windmill_venv/bin/python" "$SCRIPT_DIR/utils/predeclare_queues.py"; then
    echo "WARNING: queue predeclaration failed; core install completed but RabbitMQ bootstrap was skipped."
  fi
else
  echo "Skipping queue predeclaration."
  echo "Requires QUEUE_HOST/QUEUE_USER/QUEUE_PASSWORD and utils/predeclare_queues.py."
  echo "When available, run: $SCRIPT_DIR/windmill_venv/bin/python $SCRIPT_DIR/utils/predeclare_queues.py"
fi

if [ "$SYSTEMD_MODE" != "none" ]; then
  if command -v systemctl >/dev/null 2>&1; then
    echo "Installing systemd services (mode=$SYSTEMD_MODE)..."
    "$SCRIPT_DIR/install-systemd.sh" "$SYSTEMD_MODE"
  else
    echo "Skipping systemd service installation because systemctl is not available."
    echo "Run ./windmill.sh start to start workers manually."
  fi
else
  echo "Skipping systemd service installation."
  echo "Run ./windmill.sh start to start workers manually."
fi

echo "Done."
