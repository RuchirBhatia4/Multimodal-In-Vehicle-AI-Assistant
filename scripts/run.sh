#!/usr/bin/env bash
# Start DriveMind at http://127.0.0.1:8000 and open the dashboard once every model has loaded.
# Models are cached by setup.sh, so it runs fully offline. Stop it with Ctrl+C.
# Set DM_NO_BROWSER=1 to skip opening the browser.
cd "$(dirname "$0")/.."
URL="http://127.0.0.1:${DM_PORT:-8000}"

open_dashboard() {
  [ -n "$DM_NO_BROWSER" ] && { echo "Dashboard ready: $URL"; return; }
  open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"  # Chrome is the tested browser
}

if curl -s --max-time 2 "$URL/health" > /dev/null; then
  echo "DriveMind is already running at $URL"
  open_dashboard
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The server keeps this terminal busy, so wait for "ready" in the background.
(
  for _ in $(seq 1 180); do
    if curl -s --max-time 2 "$URL/health" | grep -q '"router":"ready"'; then
      open_dashboard
      exit 0
    fi
    sleep 1
  done
  echo "DriveMind didn't report ready within 3 minutes; check the log above." >&2
) &
exec .venv/bin/python -m drivemind.server.app
