#!/bin/bash
# Laptop side when the sim host is only reachable over ssh: open the tunnels and the viewer.
#   video : SIM:5556   --ssh -L-->  laptop:5556                       (viewer reads it)
#   leader: L100:5555  --ssh -L-->  laptop:5555  --ssh -R-->  SIM:5555   (only if the L100 host != SIM host)
# Usage: SIM_HOST=<ssh alias> [L100_HOST=<ssh alias>] scripts/laptop_connect.sh [viewer args]
set -u
cd "$(dirname "$0")/.."
: "${SIM_HOST:?set SIM_HOST to the ssh alias of the sim machine}"
L100_HOST="${L100_HOST:-}"
SSH="ssh -N -o LogLevel=ERROR -o ServerAliveInterval=15"
PY="${PYTHON:-python}"

for p in $(pgrep -f "ssh -N .*-[LR] 555[56]:localhost"); do kill "$p" 2>/dev/null; done   # leftovers hold the ports
sleep 0.5
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null; }
trap cleanup EXIT INT TERM

$SSH -L 5556:localhost:5556 "$SIM_HOST" & PIDS+=($!)
if [ -n "$L100_HOST" ]; then
  $SSH -L 5555:localhost:5555 "$L100_HOST" & PIDS+=($!)
  $SSH -R 5555:localhost:5555 "$SIM_HOST" & PIDS+=($!)
fi
for i in $(seq 1 20); do
  $PY -c "import socket; socket.create_connection(('127.0.0.1', 5556), timeout=1).close()" 2>/dev/null && break
  sleep 0.5
done
$PY -m omy_leader_isaaclab.viewer "$@"   # no exec: the trap must run to close the tunnels
