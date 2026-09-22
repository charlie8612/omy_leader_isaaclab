#!/bin/bash
# rllab1015: run the standalone OMY leader teleop in the cyclo_lab venv (headless + video stream on 5556/5557).
#   scripts/run_teleop_1015.sh --source serial --port /dev/robotis_left --auto-zero
#   scripts/run_teleop_1015.sh --source tcp --calib omy_calib.json
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
cd ~/isaac/cyclo && source venv/bin/activate && cd IsaacLab
exec ./isaaclab.sh -p ~/isaac/cyclo/omy_leader_isaaclab_repo/omy_leader_isaaclab/teleop.py --headless --stream "$@"
