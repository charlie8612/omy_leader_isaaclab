# omy_leader_isaaclab

ROBOTIS **OMY-L100** leader arm as an **Isaac Lab / cyclo_lab (ROBOTIS Lab)** teleoperation device:
joint-space 1:1 mirror onto the simulated OMY-F3M, plugged into Isaac Lab's official
`create_teleop_device` factory so `record_demos.py --teleop_device omy_leader` works.

## 現況（2026-09-22）

- cyclo_lab 側整條路線用**假 leader**驗過（rllab1015，Isaac Sim 5.1.0 + Isaac Lab v2.3.0 + robotis_lab）：
  device 經官方 factory 建立 → `Cyclo-Lift-Cube-OMY-Leader-v0` 絕對 joint action → 追蹤誤差 < 0.03 rad。
- **真機還沒驗**：`omy_serial.py`（DynamixelSDK 直讀）從未接過 L100；1:1 的 sign/zero 要用 `--auto-zero` 校一次。
- 上游：投 [ROBOTIS-GIT/robotis_lab](https://github.com/ROBOTIS-GIT/robotis_lab)（他們的 OMY sim 目前只有鍵盤 IK teleop）。
  hook 只需 3 行，見 `scripts/cyclo_lab_hook.patch`。
- 姊妹專案 `../issacsim/`：同一支 L100 開 **Franka**（跨機構 retarget，腕部 ZYZ 閉式解）。

## 目錄地圖

```
omy_leader_isaaclab/
  omy_serial.py     L100 讀取：DynamixelSDK Protocol 2.0，不依賴 lerobot；夾爪彈簧（current-position mode）
  link.py           跨主機 TCP frame（leader 插在別台時用 publisher + ssh -R）
  mirror.py         L100 -> OMY 1:1：sign / zero / 限速 / 夾爪 hysteresis / auto_zero 對 OMY 預設姿
  device.py         OmyLeaderCfg(DeviceCfg) + OmyLeaderDevice(DeviceBase)；import 即註冊進 factory
  env_cfg.py        Cyclo-Lift-Cube-OMY-Leader-v0、Cyclo-Pick-Place-Bottle-OMY-Leader-v0（絕對 joint action + teleop_devices）
  teleop.py         獨立 runner（--stream 開 JPEG 相機串流）
  fake_publisher.py 無硬體測試用；viewer.py / video.py 串流
scripts/
  run_teleop_1015.sh      在 rllab1015 的 cyclo venv 跑 runner
  cyclo_lab_hook.patch    要加進 cyclo_lab/__init__.py 的 3 行
```

環境變數（給 `record_demos.py` 用，不用改它的旗標）：`OMY_LEADER_SOURCE=serial|tcp`、`OMY_LEADER_PORT`、
`OMY_LEADER_TCP_PORT`、`OMY_LEADER_CALIB`、`OMY_LEADER_AUTO_ZERO=1`。

## 明天的硬體驗證（依序）

1. **serial 讀取**（L100 插在跑 sim 的機器，或任何有 dynamixel-sdk 的機器）：
   `python -m omy_leader_isaaclab.omy_serial /dev/robotis_left` → 六軸 + 夾爪角度會刷新；捏扳機數值變大、放開回 0。
   失敗多半是 baudrate（預設 4 Mbps）或 udev 權限。
2. **1:1 mirror + auto-zero**（1015）：
   `scripts/run_teleop_1015.sh --source serial --port /dev/robotis_left --auto-zero`
   啟動後 3 秒內把 L100 擺成 OMY 預設姿（`joint2=-1.55, joint3=2.66, joint4=-1.1, joint5=1.6`，即 sim 一開始的樣子）。
   Mac：`ssh -N -L 5556:localhost:5556 rllab518_4090_2 &` 然後 `python -m omy_leader_isaaclab.viewer`。
3. **方向**：逐軸動，反的在 `omy_calib.json` 的 `omy_sign` 翻成 -1（j2/j4/j6 在 Franka 專案量到是 -1，OMY 應相同）。
   之後用 `--calib omy_calib.json` 取代 `--auto-zero`。
4. **官方腳本**（需要有畫面的機器，`record_demos.py` import `omni.ui`，headless 跑不了）：
   先把 `scripts/cyclo_lab_hook.patch` 的 3 行加進 cyclo_lab，然後
   `OMY_LEADER_SOURCE=serial python scripts/imitation_learning/isaaclab_recorder/record_demos.py --task Cyclo-Pick-Place-Bottle-OMY-Leader-v0 --teleop_device omy_leader --dataset_file datasets/omy_leader.hdf5 --num_demos 1`
5. 錄一段 30 秒影片（issue / PR 用）。

L100 插在別台時：那台跑 `python -m omy_franka_teleop.publisher`（issacsim 專案）+ `ssh -R 5555`，這邊 `--source tcp`。

## 文件索引

| 編號 | 檔案 | 狀態 |
|---|---|---|
| — | （尚無 docs/；決策與現況先記在本 README） | |

命名規則：`docs/NN_YYYY-MM-DD_<kind>_<slug>.zh.md`（見 codebox 慣例）。
