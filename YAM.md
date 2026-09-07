# Running TiPToP on the bimanual YAM

Hardware runbook. For the data-collection contract (episode format, config schema, how `embodiment:`
threads through) see `data-collection/ARCHITECTURE.md` §12.

The YAM is a **sequential-bimanual** embodiment: cuTAMP plans one kinematic chain, so an episode is
the left arm's plan followed by the right arm's, with the idle arm parked at
`NEUTRAL_Q = (0, π/4, π/2, 0, 0, 0)` — where the planning arm's cuRobo config locks it and
collision-checks it. **If a parked arm is not physically there, every plan is a statement about a
robot that does not exist.** `_assert_idle_arm_parked` refuses to plan when it drifts past 0.1 rad.

## Processes

```
yam_arm_server.py   (ZMQ 5555/5559/5557)  owns both arms over CAN     [i2rt env]
FoundationStereo    (HTTP :1234)          depth from the D435 IR pair [FS conda env]
M2T2                (HTTP :8123)          grasps                      [M2T2 conda env]
tiptop-run                                perception + cuTAMP + execution  [tiptop pixi env]
```

The arm server is a **separate process on purpose**: i2rt pins `numpy==2.2.6` and
`rerun-sdk>=0.32.2`, which this repo's cuRobo-pinned pixi environment cannot host. It is the i2rt
counterpart of `bamboo_polymetis_shim.py` and speaks the same msgpack protocol, so
`tiptop/yam/yam_client.py` is thin and everything downstream (state sampling mid-motion, queued
trajectories, gripper overlap) works the way it does on the Franka.

## One-time setup

```bash
# pyzmq + msgpack into an env that already has i2rt
/home/prism-yam/openpi/examples/yam/.venv/bin/pip install pyzmq msgpack
```

Then add the fixed camera's extrinsics. `cameras.hand.mount: world` means its calibration entry **is**
`world_from_cam` (no forward kinematics), keyed by serial, as `[x, y, z, roll, pitch, yaw]`:

```jsonc
// tiptop/config/assets/calibration_info_<workspace>.json
{ "254622075770": { "pose": [x, y, z, roll, pitch, yaw] } }
```

The frame is the URDF's `world` link (`cuTAMP/cutamp/robots/assets/yam_description/bimanual_yam.urdf`):
the base column is mounted at `(0.24, 0, 0.0551)` relative to it, with the left arm at `y = +0.24` and
the right at `y = -0.24`. So **world origin is 0.24 m in −x from the base column and 0.0551 m below
its mounting face**, +y toward the left arm. That is also the frame `robot.base_y` splits objects in.

### Measuring it: `calibrate-top-cam`

The entry shipped in `calibration_info_prism.json` is the **sim design pose** — where the camera was
placed in MolmoAct2's sim-eval, not where this one is bolted. Replace it with a measurement:

> **The board must be clamped in or taped to the moving gripper.** The whole method rests on the
> board and the arm being one rigid body. A board held by hand, standing on the table or leaning
> against something carries no information about where the camera is — the script checks this and
> refuses rather than handing back a confident wrong pose. It also has to face the **camera**: a
> board turned edge-on projects to a sliver whose corners fit to a fraction of a pixel with the pose
> metres out of place.

```bash
# 1. Arm server up (see "Per session"). Clamp the board in one gripper, roughly toward the camera.
# 2. Optional dry run: turn the wrist until the board faces the camera, then stop.
pixi run calibrate-top-cam --mode aim

# 3. Calibrate. It aims itself first, and the right arm gets parked for you.
DC_WORKSPACE=prism pixi run calibrate-top-cam --arm left --close-gripper
```

**Aiming is automatic.** The board only has to be clamped in and roughly visible — the arm turns the
wrist until it faces the camera before capturing anything. Three numbers are reported per step:
corner count, **incidence** (0° = board square-on to the camera, 90° = edge-on) and how much of the
frame the board spans. The same gates apply to every captured pose, so an edge-on frame never
reaches the solver.

Aiming looks circular — to know which way to turn you need the transform you are trying to measure —
but it only needs the *rotation*, only *approximately*, and it runs closed loop. It starts from the
stored calibration's rotation, and once the arm has turned enough it replaces that guess with a real
hand-eye estimate off the poses it just visited. In simulation it converges from 72° edge-on in one
or two moves even with the starting guess 60° wrong. What it cannot fix is a board that is too small
or too far away in frame — that is physical, and it says so rather than flailing.

`--auto-aim-steps 0` turns it off; `--aim-seconds` controls how long it keeps reporting afterwards
so you can adjust by hand.

The arm visits ~24 perturbed poses around wherever you left it; at each, the fixed camera sees the
board and the script records FK of the **measured** joints. That is eye-to-hand calibration: because
the board rides on the gripper, `world_from_cam` and the unknown `ee_from_board` come out of one
solve, so **nothing has to be measured by hand**. All five of OpenCV's hand-eye solvers run and the
one that reprojects best wins; the worst sample is dropped and re-solved until everything fits.

### Moving the board yourself: `--mode manual`

```bash
DC_WORKSPACE=prism pixi run calibrate-top-cam --mode manual --arm left --close-gripper
```

The arm's six joints go limp — gravity compensated, **gripper still clamped** — and you push the
board wherever you like. Hold still for a second somewhere new and that pose is captured; the
terminal beeps, because your hands are on the robot and your eyes are on the board. It stops at
`--num-poses` (24) or whenever you press Ctrl-C, then solves and saves exactly as `handeye` does.
The arm is put back under position control on the way out.

Use it when cuRobo will not go where the calibration wants to be — the board's clamp is not in the
collision model, and neither are you — or when you would rather choose the viewpoints than trust a
random perturbation. Nothing else changes: same board, same gates, same solve, same output.

**The board still has to be in the gripper.** "Manual" means you choose the poses, not that the
board comes off the robot. FK of the measured joints is the only thing tying a board observation to
the world frame; a board carried in your hand constrains nothing at all, however many views the
camera gets of it, and `check_rigidity` will refuse the run rather than hand back a confident wrong
pose. If you already know where the board sits in world coordinates, that is `--mode static`.

Two things are gated harder than in `handeye`, because a hand-guided pose cannot be trusted the way
a planned one can:

* **Stillness.** The joints and the frame are read milliseconds apart; at arm's length a drift
  between them is millimetres of error that no residual would reveal. A pose is captured only after
  `--still-s` (1 s) within `--still-rad` (0.01 rad), and the sample is thrown away if the arm stirs
  while the frame is being taken. If it never triggers, you are holding the arm rather than letting
  go of it.
* **Novelty.** A pose must be `--min-new-trans-mm` (30 mm) or `--min-new-rot-deg` (8°) from *every*
  pose already captured. Duplicates agree with each other by construction, so they would improve
  the reported residuals while adding nothing to the solve.

Both gates print why they are refusing, once a second, so a capture that is not happening tells you
what to change. **Turn the board as well as sliding it** — the running rotation spread is printed
after every capture, and below ~20° the solve cannot pin the camera's orientation at all.

Read the numbers it prints:

| | good | what it means |
|---|---|---|
| rigidity check | < 15% pairs | the board moved with the gripper; above 50% it refuses to solve |
| rotation diversity | > 20° | below that hand-eye is degenerate — raise `--rot-scale-deg` |
| reprojection | < ~1.5 px RMS | the whole chain (camera pose, FK, board) agrees with the pixels |
| `ee_from_board` spread | < ~2 mm | the board stayed put in the jaws |

The rigidity check needs no calibration to work: between two poses the arm's motion and the board's
motion are conjugate, so their rotation angles must be **equal**. If the arm turns 10° while the
board turns 80°, no camera pose explains the data and there is nothing to solve.

A large reprojection error with everything else healthy almost always means `--square-size` does not
match the printout — printers scale. Measure the squares before blaming the robot.

Each run writes `tiptop/.cache/top_cam_calib/<timestamp>/` with the frames, the samples and
`verification.png`, which draws every sampled gripper frame into the last image using the solved
pose. **Look at it.** If the little axes do not sit on the gripper, the calibration is wrong no
matter what the residuals said. `--replay <that dir>` re-solves offline with different settings, and
`--no-save` solves without touching the calibration file.

`--mode static --board-pose x y z r p y` is the fallback when the board sits at a pose you already
know in world coordinates: one `solvePnP` and an inversion. It is only as good as the numbers you
type, so use it to sanity-check, not to calibrate.

`DC_WORKSPACE` is required for saving — without it the result would land in the shared
`calibration_info.json`, which is the Franka rig's, and the script refuses.

## Per session

```bash
# 1. CAN up at 1 Mbps (rename from can0/can1 first if a reboot lost the udev names)
sudo ip link set can_left  up type can bitrate 1000000
sudo ip link set can_right up type can bitrate 1000000
ip -br link show type can          # expect can_left UP, can_right UP

# 2. Arm server. Gripper calibration runs at startup — THE JAWS WILL MOVE. --park additionally
#    drives both arms to the neutral posture, so clear the workspace before using it.
/home/prism-yam/openpi/examples/yam/.venv/bin/python tiptop/yam_arm_server.py --park

# 3. Perception servers (as for any tiptop run). Both are pixi projects on this workstation --
#    build once with ./build_server.sh, which compiles CUDA kernels and takes a while.
(cd ../M2T2 && pixi run python server.py --port 8123 &)
(cd ../FoundationStereo && pixi run python server.py --port 8124 \
     --ckpt-dir pretrained_models/<run>/model_best_bp2.pth &)

# 4. Collect — from the data-collection UI, picking a config with `embodiment: bimanual_yam`
#    (cfg/tamp/yam_toys.yml), or directly:
TIPTOP_CONFIG=tiptop_yam_real.yml pixi run tiptop-run \
    --output-dir ../data-collection/runs/<workspace>/tamp/yam_toys --enable-recording
```

Both arms' cuRobo solvers warm up front, so the first rollout costs roughly twice a single-arm warmup
and no rollout pays a warmup mid-episode.

## Tuning

`tiptop/config/tiptop_yam_real.yml`:

* **`robot.time_dilation_factor`** — start at 0.3 and raise toward 1.0. This is a stiff
  position-controlled arm (kp 80 on the shoulder joints, `i2rt/robots/config/yam.yml`); the Franka
  runs at 0.2.
* **`robot.arms`** — `[left, right]` for sequential bimanual, `[left]` to collect with one arm.
* **`robot.base_y`** — the midline objects are split at. `0.0` is the robot's own centreline.
* **`perception.augment_flipped_grasps`** — on by default here. A 6-DOF arm has no null space, so it
  cannot roll its wrist to whichever of the two equivalent parallel-jaw grasps M2T2 happened to emit;
  adding the flipped twin roughly doubles reachable grasps.

## Safety notes

* `execute_trajectory` **refuses** a first waypoint more than `MAX_START_JUMP_RAD` (0.25 rad) from the
  measured pose. Planned trajectories always start where the arm is; parking, which genuinely has to
  travel, goes through `move_to` instead.
* The arms come up backdrivable (i2rt gravity-comp) and engage PD by latching the pose they are
  already in, so connecting cannot itself produce motion.
* Ctrl-C / the UI's **Preempt** stops tiptop sending further waypoints and aborts the queued
  trajectory, but the arm finishes the setpoints already streamed. The physical E-stop is the only
  instant stop.
* After an abort one arm may be left away from neutral. Use the **home** robot command at the task
  prompt — it homes every configured arm through cuRobo — rather than restarting the session.
  Outside a session, `pixi run go-home-yam` does the same from a shell (and `pixi run
  gripper-open-yam` opens the jaws); both take `--arm left|right|both`, default `both`, and select
  `tiptop_yam_real.yml` unless `TIPTOP_CONFIG` already names a config. They only need the arm server
  — no perception servers, no cameras.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no reply from the YAM arm server` | `yam_arm_server.py` not running, or the wrong host/ports | Start it; check `robot.host`/`port`/`gripper_port`/`state_port` |
| `the <arm> arm is N rad from the neutral posture` | A previous rollout was aborted mid-motion | Type `home` at the task prompt, or `pixi run go-home-yam` |
| `first waypoint is N rad from the measured pose` | Planning started from a stale state | Home the arm, then retry; if it persists the state cache is stale — restart the arm server |
| `MEASURED joint trace is EMPTY` | The state socket never answered, so no episode is written | Check the server's state port (5557) and `TIPTOP_STATE_PORT` |
| `--mode manual` never captures anything | You are still holding the arm, or the pose repeats one already taken | Let go and wait a second; read the once-a-second line saying which gate is refusing |
| The board falls out of the jaws in `--mode manual` | The gripper was never closed on it | Rerun with `--close-gripper`, or tape the board to a flat face of the gripper |
| Camera `Couldn't resolve requests` | A D405 behind a USB hub drops to USB-2 | Plug it into a USB-3 root port |
| CAN named `can0`/`can1` | The udev rename applies on replug only | `sudo ip link set can0 down && sudo ip link set can0 name can_left`, then bring up |
