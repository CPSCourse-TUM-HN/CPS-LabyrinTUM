# Road to a Perfect Run

Ordered plan to get from "mostly follows the path" to reliable full-maze
solves. Do the stages in order and change ONE thing per run.

## Setup on every machine (once)

```bash
git pull
pip install -e .
```

Create `configs/local.yaml` (gitignored) with YOUR machine's settings, and do
not put them in default.yaml:

```yaml
serial:
  port: "COM10"        # Windows lab PC; Mac: /dev/cu.usbmodemXXXX
camera:
  device_index: 0      # the maze camera (OV9281, reports 1280x800)
```

If the camera index is wrong you will see the laptop webcam. Indices shift
after replugging/reboot; probe 0-4 and pick the 1280x800 one.

## Golden rules

- New homography => re-run holes + path tools (their CSVs are derived from it).
- Camera physically moved => recalibrate homography (path/holes CSVs stay valid).
- After every autonomous run: `python scripts/analyze_run.py` and read it
  BEFORE changing any knob.

---

## Stage 1 - Redo the axis map (current one is corrupted)

The saved map was measured while ball tracking was broken (diagonal response,
not physical). Redo it now that tracking works:

```bash
python scripts/axis_check.py --amplitude 0.4 --max-amplitude 1.0 --pulse-seconds 1.2
```

Per pulse: place the ball in an open area, CLICK THE BALL in the window
(seeds the tracker), then SPACE. Also hover the cursor over the ball and over
the worst glare spot; set `vision.min_specular` in the config between the two
readings.

**Done when:** the printed response matrix has one clearly dominant axis per
command (e.g. +yaw -> mostly x, +pitch -> mostly y), and
`calibration/axis_map.npz` is saved.

## Stage 2 - One straight section until it is boring

```bash
# sanity check without servos: roll the ball by hand, watch the target dot
python scripts/run_autonomous.py --dry-run

# real runs on a straight stretch (click ball, SPACE to arm)
python scripts/run_autonomous.py
python scripts/analyze_run.py
```

Tune in THIS order, one knob per run (CLI overrides: `--kp --kd --ki
--stall-kick --lookahead --max-command`; defaults live in configs/default.yaml):

| Symptom | Fix |
|---|---|
| oscillates / zigzags around the line | raise `kd` (e.g. 0.006 -> 0.009) |
| sluggish, lags behind the target dot | raise `kp` slightly (0.015 -> 0.02) |
| parks / "balances" while off-target | raise `stall_kick` by 0.05 (default 0.30) |
| creeps at the very end of segments | raise `ki` slightly (0.004 -> 0.006) |

**Done when:** 5/5 traverses of the straight with p90 cross-track < ~8 mm
(analyze_run prints it).

## Stage 3 - Corners

Corner failures are geometry, not gains: the lookahead target sits across a
wall and the ball gets pinned into it.

1. Test corners with a smaller lookahead first: `--lookahead 12`
2. Still failing at a specific corner? `analyze_run` prints the path-mm where
   it stalls. Re-trace with denser corner waypoints
   (`python scripts/auto_trace_path.py`) or hand-fix just that section
   (`python scripts/annotate_path.py`), swinging wider of the inside wall.

**Done when:** each corner passes 5/5 in isolation.

## Stage 4 - Holes / traps

Where the channel forces a close pass by a hole, bias the annotated
centerline away from the hole edge and re-save. `analyze_run` shows
near-miss locations.

## Stage 5 - Mechanical honesty pass (30 min, big payoff)

- Tighten tie-rod jam nuts and horn screws. Backlash is a control ceiling no
  software removes: slop = the board angle lags the servo = randomly late
  corrections.
- Verify the board is level at neutral (`python scripts/manual_servo_test.py
  --neutral`); the integral term hides small bias but eats headroom.

## Stage 6 - Full maze + speed

Only after 1-5 are stable:

```bash
python scripts/run_autonomous.py            # full path
python scripts/analyze_run.py
```

- Raise `--max-command` gradually (0.45 -> 0.6 -> ...) while watching for
  overshoot into walls.
- Measure the demo metric: success rate + completion time over 10
  consecutive runs. That table is the Evaluation section of the report.

---

## Current tuned baseline (configs/default.yaml)

```yaml
control:
  lookahead_mm: 18.0
  kp: 0.015
  kd: 0.006
  ki: 0.004          # integral: un-sticks + absorbs non-level neutral
  integral_limit: 0.25
  stall_kick: 0.30   # min command when stalled off-target (breaks stiction)
  stall_speed_mm_s: 8.0
  stall_dist_mm: 8.0
  max_command: 0.45
```

Runner controls: click ball = seed tracker, SPACE = start, q = stop
(board returns to neutral). Firmware watchdog levels the board within 0.5 s
if anything crashes.
