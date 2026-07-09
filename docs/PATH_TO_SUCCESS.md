# Five-Day Path To A Strong Demo

## Goal

Deliver a reliable autonomous marble maze demo and a defensible paper/presentation.

The target demonstration is:

1. Start the fixed-camera system.
2. Click or seed the ball.
3. Press Space.
4. The solver follows the maze path with safe servo commands.
5. Logs and analysis explain the result.

The current blocker is not path following in general. `run_autonomous.py` can already drive the maze well when tracking is stable. The highest-value work is therefore tracking reliability first, then repeatability, then speed.

## Non-Negotiables

- Do not tune controller gains while the tracker is losing the ball frequently.
- Do not increase servo limits to compensate for bad tracking.
- Every demo run must end safely: neutral on lost ball, finish, timeout, serial failure, or user abort.
- Keep the camera pose, resolution, flip settings, homography, ROI, path, and holes consistent.
- Change one variable per test run and save the log.

## Success Levels

### Level 0 - Recovery Demo

Minimum acceptable fallback if time gets tight.

- Fixed camera works.
- Ball tracker follows the ball on a short segment.
- Autonomous run completes 3 to 5 waypoints or one straight section.
- Logs show ball state, target, command, and stop reason.
- Presentation frames this as a validated closed-loop cyber-physical subsystem.

This is not the desired final demo, but it is a credible, honest result.

### Level 1 - Reliable Section Solve

Primary near-term target.

- Tracker stays locked through one meaningful maze section.
- Axis map is correct.
- The ball completes the section at least 4/5 times at conservative speed.
- `analyze_run.py` reports usable cross-track error and progress metrics.
- Annotated video or live overlay shows path, ball, target, and command.

### Level 2 - Full Maze Conservative Solve

Target for the final live demo.

- Full path runs from start toward finish without manual input after arming.
- Conservative `max_command` avoids frequent wall strikes and holes.
- At least 3/5 full-maze attempts succeed or fail only for clearly explained mechanical/path reasons.
- Evaluation table includes success rate, completion time, detection rate, and cross-track error.

### Level 3 - Speed And Polish

Impressive add-ons only after Level 2 is stable.

- Speed tiers: conservative, normal, fast.
- Per-tier metrics over repeated runs.
- Annotated debug video saved for the best run.
- Failure replay from logs.
- Presentation includes before/after tracking comparison with ROI/confusers.

## Priority Order

### P0 - Freeze Calibration Artifacts

Do this before any controller tuning.

Required artifacts:

- `calibration/board_homography.npz`
- `calibration/live_roi.json`
- `calibration/live_confusers.json`
- `configs/maze_holes.csv`
- `configs/maze_path_auto.csv`
- `calibration/axis_map.npz`

Sanity checks:

```bash
python3 scripts/check_camera.py --config configs/default.yaml
```

```bash
python3 - <<'PY'
import numpy as np

path = np.genfromtxt("configs/maze_path_auto.csv", delimiter=",", names=True)
holes = np.genfromtxt("configs/maze_holes.csv", delimiter=",", names=True)

print("path x:", path["x_mm"].min(), path["x_mm"].max())
print("path y:", path["y_mm"].min(), path["y_mm"].max())
print("holes x:", holes["x_mm"].min(), holes["x_mm"].max())
print("holes y:", holes["y_mm"].min(), holes["y_mm"].max())
PY
```

For the inside playable area, values should be roughly within `0..263` and `0..222`, with only small tolerance near edges.

### P1 - Fix Tracking Reliability

Tracking is the main risk. Treat it as the first engineering problem.

1. Finish Stage 2 ROI/confusers:

```bash
python3 scripts/select_maze_roi.py \
  --source data/raw/live_camera_20260707_234339.avi \
  --output calibration/live_roi.json \
  --overlay-output calibration/live_roi_overlay.png
```

```bash
python3 scripts/pipeline.py \
  --calibrate data/raw/live_camera_20260707_234339.avi \
  --confusers-file calibration/live_confusers.json \
  --roi-file calibration/live_roi.json
```

2. Validate tracker offline before touching servos:

```bash
python3 scripts/pipeline.py data/raw/live_camera_20260707_234339.avi \
  --auto-seed \
  --confusers-file calibration/live_confusers.json \
  --out-video data/processed/stage2_with_confusers.mp4 \
  --out-csv data/processed/stage2_with_confusers.csv
```

3. Tune `vision.min_specular`.

During `axis_check.py`, hover over the ball and the worst glare spot. Set `vision.min_specular` between those brightness values. If the ball peak is near 254 and glare/hole peaks are near 210, a value around 230 to 240 is reasonable. If the ball is dimmer, lower it carefully.

4. Prefer manual click seed for demo runs.

Auto-seed is convenient, but the demo path should use click-to-seed because it avoids false initial locks on glare.

Done when:

- Tracker does not lock onto the side ChArUco board.
- Tracker does not persistently lock onto holes or printed numbers.
- Lost-ball events are rare during a conservative run.
- When it does lose the ball, the system neutralizes instead of driving blindly.

### P2 - Validate Axis Mapping

Do this only after tracking is stable enough to measure motion.

```bash
python3 scripts/axis_check.py \
  --config configs/default.yaml \
  --homography calibration/board_homography.npz \
  --output calibration/axis_map.npz \
  --amplitude 0.4 \
  --max-amplitude 1.0 \
  --pulse-seconds 1.2
```

Procedure:

- Put the ball in an open area away from holes and walls.
- Click the ball in the window to seed the tracker.
- Press Space for each pulse.
- The script sends `+yaw`, `-yaw`, `+pitch`, `-pitch`.
- It measures board-mm displacement before and after each pulse.
- It saves a matrix that converts board-frame commands into servo commands.

Done when:

- The printed response matrix has a clear dominant movement direction for each servo axis.
- `calibration/axis_map.npz` exists.
- A small dry-run command points in the expected direction.

### P3 - Conservative Closed-Loop Runs

Start with dry-run visualization:

```bash
python3 scripts/run_autonomous.py --dry-run
```

Then run with conservative command cap:

```bash
python3 scripts/run_autonomous.py --max-command 0.25 --lookahead 12
python3 scripts/analyze_run.py data/raw/autonomous_run.csv
```

Tune in this order:

| Problem | First adjustment |
|---|---|
| Tracker loses ball | Fix ROI/confusers/min_specular/lighting before control gains |
| Ball moves wrong direction | Re-run `axis_check.py` |
| Oscillation | Increase `kd` slightly |
| Sluggish tracking of target | Increase `kp` slightly |
| Stuck while command is nonzero | Increase `stall_kick` slightly |
| Corner wall pinning | Reduce `lookahead` or adjust path around corner |
| Near-hole failures | Move path centerline away from hole |

Done when:

- One section completes repeatedly.
- Logs show reasonable cross-track error.
- Failures are explainable from video/logs.

### P4 - Full Maze And Speed Tiers

Only after P1-P3 are stable.

Conservative:

```bash
python3 scripts/run_autonomous.py --max-command 0.25 --lookahead 12
```

Normal:

```bash
python3 scripts/run_autonomous.py --max-command 0.45 --lookahead 18
```

Fast:

```bash
python3 scripts/run_autonomous.py --max-command 0.60 --lookahead 18
```

Run at least 5 attempts per tier if time allows. Record:

- success/failure
- completion time
- detection rate
- p90 cross-track error
- most common failure mode

The best paper/presentation result is not necessarily the fastest run. It is the fastest repeatable run with a clear safety story.

## Five-Day Schedule

### Day 1 - Tracking Lockdown

- Finish ROI/confuser file.
- Compare tracker with and without confusers.
- Tune `vision.min_specular`.
- Stabilize lighting and camera.
- Deliverable: annotated tracking video and selected ROI overlay.

### Day 2 - Calibration Consistency And Axis Map

- Verify homography/path/holes all use inside `263 x 222 mm` playable frame.
- Re-run holes/path if any coordinate range is inconsistent.
- Run `axis_check.py`.
- Deliverable: `axis_map.npz` and clear response matrix screenshot/output.

### Day 3 - Reliable Section Runs

- Run dry-run and live conservative section tests.
- Tune only one parameter per run.
- Fix path geometry at failing corners/holes.
- Deliverable: 4/5 or 5/5 section success plus logs.

### Day 4 - Full Maze Attempts

- Run conservative full path.
- Classify failures from logs.
- Make only necessary path/control changes.
- Establish demo command and fallback section command.
- Deliverable: best full-maze run video and evaluation table draft.

### Day 5 - Paper, Presentation, Demo Freeze

- Freeze config/artifacts.
- Record final demo videos.
- Generate final plots/tables from logs.
- Prepare presentation with architecture, calibration, tracking, control, safety, and results.
- Do not make major code changes unless the demo is broken.

## Paper And Presentation Story

Use this structure:

1. Problem: physical marble maze control under camera feedback.
2. Hardware: fixed camera, Arduino, PCA9685, two servos.
3. Perception: fixed camera convention, ROI, static confusers, motion/specular tracker.
4. Calibration: homography into playable-area millimeters.
5. Planning: annotated/traced centerline path and hole map.
6. Control: lookahead target, PID/stiction compensation, axis map.
7. Safety: conservative PWM limits, watchdog, neutral on lost ball.
8. Evaluation: detection rate, cross-track error, success rate, completion time.
9. Limitations: glare, backlash, no servo feedback, camera pose sensitivity.
10. Future work: better lighting, lens undistortion, section-specific speed limits, model predictive control.

Impressive features to highlight:

- Same tracker core works offline and live.
- ROI/confuser calibration separates static bright features from moving ball.
- Axis-map calibration learns servo sign/swap behavior instead of hardcoding.
- Logs support failure analysis and measurable tuning.
- Safety is enforced both on PC and firmware side.

## Demo Strategy

Prepare three demo modes:

1. Full maze conservative run.
2. Reliable section run if full maze has a bad reset or lighting issue.
3. Dry-run overlay showing live ball, target, path, and proposed commands.

This is not cutting corners. It is engineering a reliable presentation under real hardware uncertainty.

## Stop Conditions

Stop increasing complexity if any of these are true:

- Tracking loses the ball more than once per section.
- Axis map is unclear.
- The board hits mechanical limits.
- A faster setting reduces success rate sharply.
- Logs no longer explain the failure.

When in doubt, demo the most reliable level and use the paper to explain the measured bottleneck.
