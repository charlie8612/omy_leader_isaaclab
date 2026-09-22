#!/bin/bash
# Sim host: run teleop.py inside an Isaac Lab checkout. Usage: ISAACLAB=<path> scripts/run_teleop.sh --target omy|franka [...]
set -u
: "${ISAACLAB:?set ISAACLAB to your IsaacLab checkout}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1   # Kit exits without flushing stdout
cd "$ISAACLAB" && exec ./isaaclab.sh -p "$REPO/omy_leader_isaaclab/teleop.py" "$@"
