"""ROBOTIS OMY-L100 leader arm as an Isaac Lab / cyclo_lab teleop device.

    record_demos.py --task Cyclo-Pick-Place-Bottle-OMY-Leader-v0 --teleop_device omy_leader

Inside a running Isaac Sim app, importing this package registers ``OmyLeaderCfg -> OmyLeaderDevice``
in Isaac Lab's teleop device factory and the ``*-OMY-Leader-v0`` envs (joint-position variants of the
cyclo_lab OMY tasks). Outside Isaac (publisher, viewer, tests) only the pure-python modules load.

Layers:
  reader   omy_serial.py (DynamixelSDK, no lerobot needed) | link.py (TCP from a remote publisher)
  mapping  mirror.py          L100 -> OMY 1:1 joint mirror (sign / zero / clamp / rate limit / gripper)
           franka_*.py        L100 -> Franka (J3 locked, arm 1:1, fixed or ZYZ wrist), calib / monitor tools
  device   device.py          OmyLeaderCfg(target="omy"|"franka") + OmyLeaderDevice(DeviceBase)
  envs     omy_env_cfg.py     cyclo_lab OMY tasks with absolute JointPositionAction + teleop_devices
           franka_env_cfg.py  Isaac Lab Franka stack task, same treatment
"""


def register() -> bool:
    """Register device + envs. Returns False when Isaac Lab / cyclo_lab are not importable here."""
    try:
        from .device import OmyLeaderCfg, OmyLeaderDevice  # noqa: F401  (registers with the factory)
    except Exception:
        return False
    import warnings

    try:  # ROBOTIS cyclo_lab OMY tasks (Isaac Lab 2.2+)
        from .omy_env_cfg import register_envs

        register_envs()
    except Exception as e:
        warnings.warn(f"omy_leader_isaaclab: OMY envs not registered ({e})")
    try:  # Isaac Lab's own Franka stack task
        from .franka_env_cfg import register as register_franka

        register_franka()
    except Exception as e:
        warnings.warn(f"omy_leader_isaaclab: Franka envs not registered ({e})")
    return True


register()
