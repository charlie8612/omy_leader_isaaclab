"""Register the OMY-L100's Dynamixel models in lerobot's tables if they are missing.

``lerobot_teleoperator_omy`` declares XH540-W150 (arm 1-3), XC330-T288 (wrist 4-6) and
XC330-T181 (gripper). lerobot 0.4.x only ships xl330/xl430/xm430/xm540/xc430, so the bus
raises ``KeyError: Control table for model='xh540-w150' not found``. All of these are
Protocol 2.0 X-series with the same control table, 4096-tick resolution and baudrate table;
only the model numbers (used by the ping check) differ.

Model numbers from the ROBOTIS e-manual: XH540-W150 = 1110, XC330-T181 = 1210, XC330-T288 = 1220.
Import this module before constructing ``OmyLeader``. No-op if the models are already known.
"""

from __future__ import annotations

OMY_MODELS = {"xh540-w150": 1110, "xc330-t288": 1220, "xc330-t181": 1210}


def register() -> list[str]:
    from lerobot.motors.dynamixel import DynamixelMotorsBus as Bus
    from lerobot.motors.dynamixel import tables as t

    ref = "xm540-w270"
    # DynamixelMotorsBus deep-copies the module tables into class attributes at import time,
    # so patch both the module dicts and the class-level copies the bus actually reads.
    targets = [
        (t.MODEL_CONTROL_TABLE, Bus.model_ctrl_table, lambda m: t.X_SERIES_CONTROL_TABLE),
        (t.MODEL_RESOLUTION, Bus.model_resolution_table, lambda m: t.MODEL_RESOLUTION.get(ref, 4096)),
        (t.MODEL_BAUDRATE_TABLE, Bus.model_baudrate_table, lambda m: t.X_SERIES_BAUDRATE_TABLE),
        (t.MODEL_NUMBER_TABLE, Bus.model_number_table, lambda m: OMY_MODELS[m]),
        (getattr(t, "MODEL_ENCODING_TABLE", {}), getattr(Bus, "model_encoding_table", {}), lambda m: t.X_SERIES_ENCODINGS_TABLE),
    ]
    added = []
    for model in OMY_MODELS:
        if model in Bus.model_ctrl_table:
            continue
        for module_tbl, class_tbl, value in targets:
            for tbl in (module_tbl, class_tbl):
                if isinstance(tbl, dict):
                    tbl[model] = value(model)
        modes = getattr(t, "MODEL_OPERATING_MODES", None)
        if isinstance(modes, dict) and ref in modes:
            modes[model] = modes[ref]
        added.append(model)
    return added
