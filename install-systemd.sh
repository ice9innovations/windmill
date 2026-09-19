#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"
MODE="${1:-workers}"

if [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
    cat <<USAGE
Usage: $0 [workers|api|relay|all]

Installs systemd units rendered for this checkout path and user.

  workers  Install only the Windmill worker monitor (default)
  api      Install only the Windmill API service
  relay    Install only the LAN image relay
  all      Install both services

Override WINDMILL_USER if the service should run as a different user.
USAGE
    exit 0
fi

case "$MODE" in
    workers|api|relay|all)
        ;;
    *)
        echo "ERROR: unknown mode '$MODE'. Use workers, api, or all." >&2
        exit 1
        ;;
esac

if [ -n "${WINDMILL_USER:-}" ]; then
    SERVICE_USER="$WINDMILL_USER"
elif [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    SERVICE_USER="$SUDO_USER"
else
    SERVICE_USER="$(id -un)"
fi

render_unit() {
    local template="$1"
    local output="$2"
    sed \
        -e "s#__WINDMILL_USER__#${SERVICE_USER}#g" \
        -e "s#__WINDMILL_DIR__#${SCRIPT_DIR}#g" \
        "$template" > "$output"
}

if [ "$MODE" = "relay" ]; then
    user_unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
    mkdir -p "$user_unit_dir"
    render_unit \
        "${SCRIPT_DIR}/systemd/windmill-image-relay.service.in" \
        "${user_unit_dir}/windmill-image-relay.service"
    systemctl --user daemon-reload
    systemctl --user enable --now windmill-image-relay.service
    echo "Installed Windmill image relay user service dir=${SCRIPT_DIR}"
    exit 0
fi

install_unit() {
    local unit_name="$1"
    local template="${SCRIPT_DIR}/systemd/${unit_name}.in"
    local rendered
    rendered="$(mktemp)"
    render_unit "$template" "$rendered"
    sudo install -m 0644 "$rendered" "${UNIT_DIR}/${unit_name}"
    rm -f "$rendered"
}

if [ "$MODE" = "workers" ] || [ "$MODE" = "all" ]; then
    install_unit "windmill-workers.service"
fi

if [ "$MODE" = "api" ] || [ "$MODE" = "all" ]; then
    install_unit "windmill.service"
fi

sudo systemctl daemon-reload

if [ "$MODE" = "workers" ] || [ "$MODE" = "all" ]; then
    sudo systemctl enable windmill-workers.service
    sudo systemctl restart windmill-workers.service
fi

if [ "$MODE" = "api" ] || [ "$MODE" = "all" ]; then
    sudo systemctl enable windmill.service
    sudo systemctl restart windmill.service
fi

echo "Installed Windmill systemd mode=${MODE} user=${SERVICE_USER} dir=${SCRIPT_DIR}"
