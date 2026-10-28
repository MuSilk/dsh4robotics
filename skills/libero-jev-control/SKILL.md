---
name: libero-jev-control
description: Control the open LIBERO episode with depth-referenced 3-D targets. Judge objects and the final state yourself (no imajev). Verified on task 56, butter -> tray, 219/1000 steps.
---

# LIBERO manipulation skill

Working code is already in `code/` — read it before writing anything:

| file | what it gives you |
|---|---|
| `code/mcp.py` | unix-socket JSON-RPC: `connect`, `tool(ctx, "libero_step", ...)` |
| `code/geom.py` | `components(mask)` (no scipy here), `quat_to_mat`, `rotvec` |
| `code/rig.py` | depth frame, `solve_delta`, `scene_objects`, `goto`, `descend`, `close/open`, `unwind` |
| `code/pickplace.py` | `pick_at`, `place_into`, `lift_to` |
| `code/inspect.py` | run this first: pose, DELTA + checks, table height, one 4× crop per object |
| `code/scene.py` | deprecated shim — its old `T_CAM` was wrong; never unproject with a hardcoded camera |

**No imajev in the loop.** Identify the object by reading an upscaled crop yourself, and
verify the end state from depth + pixels yourself. Asking a VLM cost more wall-clock than
looking and it contradicted the geometry (it called a wooden tray floor "table").

## The one thing that will bite you: DELTA

Depth comes back in a camera frame that is *not* the base frame the arm reports. The
rotation is good — an unprojected table patch is flat to ~3 mm across the whole image — but
the translation is offset by a constant per episode:

    eef_pt = depth_pt + DELTA          # solve once, first thing, at a home-like pose

`rig.solve_delta(ctx)` does it: take the light-grey arm blob that hangs in from the top of
the picture (`rgb.mean>105 & chroma<28`, `z∈[0.25,1.05]`, blob starts at row 0, narrower
than 250 px), average the bottom-3-row pixels (the two finger tips), `DELTA = eef_pos - tip`.

Trust it only after both checks print green:
1. **finger separation ≈ 0.078 m** — the two tips are the jaws, which are 0.078 m apart open.
   This single number validates the whole unprojection (it was 0.083 m on this rig).
2. **table lands at 0.38–0.48 m** after adding `DELTA_z` (0.429 here).

Pitfalls that cost me two wasted calibrations: the *table* is the largest near-depth blob
(it is only 0.65–0.70 m from the camera, same as the gripper), and the background wall is
also light grey. Use the row-0 / width / depth-band filters above.

This episode (init_state 33): **DELTA = (-0.517, 0.015, 0.426)**. The numbers in older notes
(`-0.60, 0, +0.42`, butter at `(-0.01, 0.06)`, tray `x∈[-0.22,0]`) are for a *different
init state* — same scene, objects moved. Re-solve, don't copy.

## Manipulator

- 7-D OSC deltas in the base frame. Full scale ≈ **0.010 m/step**; the first step of a burst
  is ~4× smaller. P control `row[:3] = clip((target-p)/0.06, ±0.9)`, 5 steps per block,
  re-read the pose every block: ~4 cm per block, lands within 5 mm, never jammed a joint in
  219 steps.
- `goto` early-exits at `tol=0.008`, i.e. it can stop 5 mm short — use `tol=0.004` for a
  release height.
- Jaws close along the eef **y**. At home `R[:,1] ≈ (0,-1,0)`, already right for these
  boxes — no spin needed. `gripper_qpos` 0.034 = open (0.078 gap), **0.0197 = holding
  something**, 0.005 = air closed, 0.023 = object still brushing the pads after a release.
- Rotations only if the jaws are crosswise; then `unwind` (`[0,0,0,0,0,+1,-1]` repeated) both
  frees a jammed wrist and spins the jaws.
- `libero_observe` and `libero_step` return a **flat** dict (`eef_pos`, `eef_quat_xyzw`,
  `gripper_qpos`, `joint_pos`, `step_id`, `remaining_episode_steps`) — no `observation`
  wrapper. Check `remaining_episode_steps` for the budget (1000 here).
- A stall is the top of whatever is under the pads. **A 7×4 cm box straddled by an open jaw
  does not stall the descent** — the pads pass either side and you stall on the table. "No
  stall at the object height" is not a miss; the jaw reading after closing is the real check,
  and the wrist view at the grasp pose shows the object right between the fingers.

## Camera geometry

- agentview and wrist are 512×512, `fx=fy=618.04`, `cx=cy=256`, depth `uint16` mm, row 0 top.
  Depth pixels are aligned with the agentview RGB (verified to ~1 px), so crop and unproject
  the same pixels.
- The agentview sits ~0.65–0.70 m from the gripper and looks along **−x, ~32° below
  horizontal**: x motion barely moves the image, it mostly changes depth. You cannot eyeball
  x alignment — use the depth, then confirm under the wrist camera.
- The arm leaves the top of the agentview once the eef is above ~0.80 m or behind ~−0.20 m
  in x (it projected to v=−24 at `(−0.21, 0.05, 0.86)`). Solve DELTA with the gripper fully
  in frame.
- The wrist camera sees the fingers along the bottom edge and the grasp area just above
  them; at grasp height ≈ 38 px/cm (a 3.7 cm butter ≈ 140 px wide). Best place to read a
  label: crop it from the agentview and upscale 4×, or read it in the wrist view when the
  hand is over it.

## Task recipe (butter → tray, this episode)

1. `libero_status` → instruction, budget. `solve_delta` → DELTA, table_z.
2. `scene_objects` → clusters (`z > table+0.012`, wall/horizon blobs filtered). Crop each,
   upscale, read the label. Here: the orange box is **Breakstone's "FRESH BUTTER"** (target,
   centre `(0.063, 0.051)`, top 0.462, 7.0×3.7 cm) and the blue/white stick is **Philadelphia
   cream cheese** (decoy at `(0.118, −0.212)`); also a ketchup bottle, two cans, the wooden
   tray on the right.
3. Hover `top+0.14`, descend with the jaws open to **`top−0.012`** (grasp at eef z 0.450),
   close 8 steps → jaw 0.0197, lift 0.13, look at the wrist view.
4. Container target: `rig.floor_region(pts, z, table_z, delta, bbox=tray_bbox)` gives the
   interior floor point, the floor height and the wall ranges (here
   `x∈[−0.155,0.068]`, `y∈[0.225,0.368]`, floor 0.453). Carry at **z ≥ 0.58** (rim is
   0.48–0.52), lower to **z ≈ 0.49–0.50**, open 6 steps, retreat straight up 0.14.
5. Verify yourself before `finish`: the object's cluster xy is inside the container's floor
   `xr`/`yr` *and* its top ≈ floor + object thickness (here the butter's label top read
   0.469 after landing on the 0.453 floor). A picture of the object sitting in the tray is
   the second check; a single yes/no vision answer is not enough.
6. `finish` with no arguments.

Budget: the whole pick-and-place is ~85 steps of motion; 219 got used here only because of
two bad DELTA solves. If you are past ~150 steps and have not grasped it, stop re-exploring
and take the shortest path you can verify.

## If the grasp misses

Jaw < 0.010 after closing means air. Open, drop another 1 cm, close once more (that retry is
built into `pick_at`). If it misses twice, the xy is wrong — re-read the object's cluster
(its xy moves only if you knocked it) and compare with the wrist view before trying again.
Never translate with a saturated command for dozens of steps; check `joint_pos` if a push
stalls (a pinned `q4`/`q6` means unwind, healthy joints mean reach).
