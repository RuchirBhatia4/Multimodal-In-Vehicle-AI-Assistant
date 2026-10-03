#!/usr/bin/env bash
# Start DriveMind at http://127.0.0.1:8000 . Models are cached, so run fully offline.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec .venv/bin/python -m drivemind.server.app
